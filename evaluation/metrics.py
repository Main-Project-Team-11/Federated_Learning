"""Clinical test-set metrics for centralized and federated checkpoints."""

import argparse
import json
import os

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def compute_clinical_metrics(model, test_loader, device="cpu", threshold=0.5):
    """Return diagnostic metrics, targets, and probabilities for a model."""
    model = model.to(device)
    model.eval()
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            clinical = batch["clinical"].to(device)
            labels = batch["label"].to(device, dtype=torch.float32).reshape(-1)
            logits = model(images, clinical).reshape(-1)
            if logits.numel() != labels.numel():
                raise ValueError("Model output and labels must have matching sizes")
            all_targets.extend(labels.cpu().tolist())
            all_probabilities.extend(torch.sigmoid(logits).cpu().tolist())

    if not all_targets:
        raise ValueError("The test loader produced no samples")

    predictions = [int(probability >= threshold) for probability in all_probabilities]
    tn, fp, fn, tp = confusion_matrix(
        all_targets, predictions, labels=[0, 1]
    ).ravel()
    try:
        roc_auc = float(roc_auc_score(all_targets, all_probabilities))
    except ValueError:
        roc_auc = None

    metrics = {
        "accuracy": float(accuracy_score(all_targets, predictions)),
        "sensitivity_recall": float(
            recall_score(all_targets, predictions, zero_division=0)
        ),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(
            precision_score(all_targets, predictions, zero_division=0)
        ),
        "f1_score": float(f1_score(all_targets, predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": float(threshold),
        "num_samples": len(all_targets),
    }
    return metrics, all_targets, all_probabilities


def evaluate_checkpoint(
    checkpoint_path,
    test_loader,
    device="cpu",
    threshold=0.5,
):
    """Load a MultimodalNet checkpoint and compute its test-set metrics."""
    from models.multimodal_net import MultimodalNet

    model = MultimodalNet()
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    metrics, _, _ = compute_clinical_metrics(
        model, test_loader, device=device, threshold=threshold
    )
    return metrics


def main():
    from data.multimodal_dataset import get_test_loader

    parser = argparse.ArgumentParser(
        description="Compute clinical test metrics for centralized and/or FedAvg models"
    )
    parser.add_argument("--centralized-checkpoint")
    parser.add_argument("--federated-checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.centralized_checkpoint and not args.federated_checkpoint:
        parser.error("Provide at least one checkpoint to evaluate")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader_options = {"batch_size": args.batch_size, "shuffle": False}
    if args.data_dir is not None:
        loader_options["data_dir"] = args.data_dir
    test_loader = get_test_loader(**loader_options)
    os.makedirs(args.output_dir, exist_ok=True)

    for name, checkpoint in (
        ("centralized", args.centralized_checkpoint),
        ("federated", args.federated_checkpoint),
    ):
        if checkpoint is None:
            continue
        metrics = evaluate_checkpoint(
            checkpoint, test_loader, device=device, threshold=args.threshold
        )
        output_path = os.path.join(args.output_dir, f"{name}_test_metrics.json")
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(metrics, output_file, indent=2)
        print(f"{name.capitalize()} metrics saved to {output_path}")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
