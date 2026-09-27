import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_client_local(
    model: nn.Module,
    dataloader,
    epochs: int = 2,
    lr: float = 1e-4,
    pos_weight_val: float = 18.60,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    client_name: str = "Client",
    max_batches_per_epoch: int | None = None,
):
    """
    Executes local training on a simulated hospital client partition for E epochs.
    Returns (updated_state_dict, list_of_epoch_losses).
    """
    model.to(device)
    model.train()

    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    epoch_losses = []

    for ep in range(epochs):
        running_loss = 0.0
        total_samples = 0

        total_b = len(dataloader) if max_batches_per_epoch is None else min(len(dataloader), max_batches_per_epoch)
        pbar = tqdm(dataloader, total=total_b, desc=f"  [{client_name}] Epoch {ep + 1}/{epochs}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break

            images = batch["image"].to(device)
            clinical = batch["clinical"].to(device)
            labels = batch["label"].to(device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images, clinical)
            logits = logits.reshape(-1)
            labels = labels.reshape(-1)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.numel()
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            current_avg_loss = running_loss / max(total_samples, 1)
            pbar.set_postfix(loss=f"{current_avg_loss:.4f}")

        avg_loss = running_loss / max(total_samples, 1)
        epoch_losses.append(avg_loss)

    return model.state_dict(), epoch_losses


def evaluate_model(
    model: nn.Module,
    dataloader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    desc: str = "Evaluating",
    max_batches: int | None = None,
):
    """Evaluates model loss and accuracy on validation or test set."""
    model.to(device)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        total_b = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
        pbar = tqdm(dataloader, total=total_b, desc=f"  [{desc}]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = batch["image"].to(device)
            clinical = batch["clinical"].to(device)
            labels = batch["label"].to(device, dtype=torch.float32)

            logits = model(images, clinical)
            logits = logits.reshape(-1)
            labels = labels.reshape(-1)

            loss = criterion(logits, labels)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()

            batch_size = labels.numel()
            total_loss += loss.item() * batch_size
            total += batch_size

            pbar.set_postfix(val_loss=f"{total_loss / max(total, 1):.4f}")

    return {
        "val_loss": total_loss / max(total, 1),
        "val_accuracy": correct / max(total, 1),
    }


class ClientTrainer:
    """Train a multimodal model on one client's batches (Backwards compatibility)."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        pos_weight = torch.tensor([18.60], device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def train_epoch(self, dataloader) -> float:
        """Run one training epoch and return the sample-weighted mean loss."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            clinical = batch["clinical"].to(self.device)
            labels = batch["label"].to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images, clinical)
            logits = logits.reshape(-1)
            labels = labels.reshape(-1)
            if logits.numel() != labels.numel():
                raise ValueError(
                    "Model output and labels must contain the same number of values"
                )

            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            batch_size = labels.numel()
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples if total_samples else 0.0