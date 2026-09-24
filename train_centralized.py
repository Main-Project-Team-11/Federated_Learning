import torch
from data.multimodal_dataset import get_client_loader
from models.multimodal_net import MultimodalNet
from training.client_trainer import ClientTrainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loaders = [get_client_loader(client_id) for client_id in (1, 2, 3)]

    model = MultimodalNet()
    trainer = ClientTrainer(model=model, device=device)

    epochs = 10
    print(f"Starting centralized baseline training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0
        for loader in train_loaders:
            sample_count = len(loader.dataset)
            total_loss += trainer.train_epoch(loader) * sample_count
            total_samples += sample_count

        average_loss = total_loss / total_samples if total_samples else 0.0
        print(f"Epoch {epoch}/{epochs} - Average loss: {average_loss:.4f}")

    print("Centralized baseline training complete.")


if __name__ == '__main__':
    main()