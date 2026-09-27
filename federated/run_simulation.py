"""
federated/run_simulation.py
FedMed Federated Learning Simulation Orchestrator — Member 3 Deliverable.

Simulates 3 hospital clients (C1, C2, C3) training locally on disjoint data partitions,
broadcasting parameters, aggregating local updates via FedAvg, and evaluating
global validation performance across multiple communication rounds.
"""
import argparse
import json
import os
import sys
import torch

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.multimodal_dataset import get_client_loader, get_val_loader
from models.multimodal_net import MultimodalNet
from models.clinical_encoder import ClinicalEncoder
from training.client_trainer import train_client_local, evaluate_model
from federated.server import federated_averaging


def run_federated_simulation(
    num_rounds: int = 5,
    local_epochs: int = 2,
    batch_size: int = 32,
    max_batches: int | None = None,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_log_path: str = "federated_training_log.json",
    output_model_path: str = "best_global_model.pt",
):
    """
    Executes the multi-round federated simulation loop across 3 hospital clients.
    """
    print("=" * 70)
    print(f"STARTING FEDMED 30% FEDERATED SIMULATION")
    print(f"Clients: 3 | Communication Rounds: {num_rounds} | Local Epochs: {local_epochs} | Device: {device}")
    if max_batches is not None:
        print(f"Max Batches per Epoch: {max_batches} (Fast Subsampled Mode)")
    print("=" * 70)

    # 1. Load DataLoaders
    print("\n[Data] Loading client partitions & global validation DataLoader...")
    client_loaders = [
        get_client_loader(1, batch_size=batch_size, shuffle=True),
        get_client_loader(2, batch_size=batch_size, shuffle=True),
        get_client_loader(3, batch_size=batch_size, shuffle=True),
    ]
    val_loader = get_val_loader(batch_size=batch_size, shuffle=False)
    print(f"  Client 1 samples: {len(client_loaders[0].dataset):,}")
    print(f"  Client 2 samples: {len(client_loaders[1].dataset):,}")
    print(f"  Client 3 samples: {len(client_loaders[2].dataset):,}")
    print(f"  Validation samples: {len(val_loader.dataset):,}")

    # 2. Instantiate Initial Global Model
    clinical_enc = ClinicalEncoder(input_dim=2, hidden_dim=32, output_dim=64)
    global_model = MultimodalNet(
        clinical_encoder_module=clinical_enc,
        num_clinical_features=2,
    ).to(device)

    best_val_loss = float("inf")
    history = {
        "rounds": [],
        "client_train_losses": [],
        "global_val_loss": [],
        "global_val_accuracy": [],
    }

    # 3. Multi-Round Communication Loop
    for r in range(1, num_rounds + 1):
        print(f"\n" + "-" * 60)
        print(f"--- [Round {r}/{num_rounds}] Broadcasting Global Model to 3 Clients ---")
        print("-" * 60)

        client_weights = []
        client_sample_counts = []
        round_losses = []

        # Train at each simulated hospital client
        for client_idx, loader in enumerate(client_loaders, start=1):
            # Create local client model initialized with current global weights
            local_clinical_enc = ClinicalEncoder(input_dim=2, hidden_dim=32, output_dim=64)
            local_model = MultimodalNet(
                clinical_encoder_module=local_clinical_enc,
                num_clinical_features=2,
            ).to(device)
            local_model.load_state_dict(global_model.state_dict())

            # Perform local client training for E epochs
            updated_weights, epoch_losses = train_client_local(
                model=local_model,
                dataloader=loader,
                epochs=local_epochs,
                lr=lr,
                device=device,
                client_name=f"Hospital Client {client_idx}",
                max_batches_per_epoch=max_batches,
            )

            client_weights.append(updated_weights)
            client_sample_counts.append(len(loader.dataset))
            final_local_loss = epoch_losses[-1]
            round_losses.append(final_local_loss)

            print(
                f"  Hospital Client {client_idx}: Trained | Final Local Loss: {final_local_loss:.4f}"
            )

        # 4. FedAvg Aggregation on Server
        print(f"\n  [Server] Aggregating local client updates via FedAvg...")
        aggregated_weights = federated_averaging(client_weights, client_sample_counts)
        global_model.load_state_dict(aggregated_weights)

        # 5. Evaluate Aggregated Global Model on Holdout Validation Set
        val_metrics = evaluate_model(global_model, val_loader, device=device, desc=f"Val Round {r}", max_batches=max_batches)
        val_loss = val_metrics["val_loss"]
        val_acc = val_metrics["val_accuracy"]

        print(
            f"  [Global Validation] Round {r}/{num_rounds} -> "
            f"Loss: {val_loss:.4f} | Accuracy: {val_acc * 100:.2f}%"
        )

        # 6. Record History
        history["rounds"].append(r)
        history["client_train_losses"].append(round_losses)
        history["global_val_loss"].append(val_loss)
        history["global_val_accuracy"].append(val_acc)

        # Save checkpoint if best validation loss achieved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(global_model.state_dict(), output_model_path)
            print(f"  [Checkpoint] Saved best global model weights to '{output_model_path}'")

    # 7. Save Log History Artifact
    with open(output_log_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print(f"FEDERATED SIMULATION COMPLETE")
    print(f"Saved artifacts:")
    print(f"  - Training log: {os.path.abspath(output_log_path)}")
    print(f"  - Best model checkpoint: {os.path.abspath(output_model_path)}")
    print("=" * 70)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedMed 30% Federated Simulation")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated communication rounds")
    parser.add_argument("--epochs", type=int, default=2, help="Number of local client training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoaders")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches per epoch for fast CPU testing")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    run_federated_simulation(
        num_rounds=args.rounds,
        local_epochs=args.epochs,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        lr=args.lr,
    )
