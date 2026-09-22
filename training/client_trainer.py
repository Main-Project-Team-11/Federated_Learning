import torch
import torch.nn as nn
import torch.optim as optim


class ClientTrainer:
    """Train a multimodal model on one client's batches."""

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