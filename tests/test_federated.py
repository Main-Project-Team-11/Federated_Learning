"""
tests/test_federated.py
Member 3 Unit Tests — FedAvg Server Aggregation & Federated Pipeline Verification.

Tests:
  1. Mathematical correctness of weighted FedAvg aggregation.
  2. state_dict layer key compatibility across ViT, Clinical, and Fusion modules.
  3. Local client training loop (train_client_local) return structure and loss tracking.
  4. Global model validation evaluation (evaluate_model) dictionary contract.
"""
import copy
import os
import sys
import torch
import torch.nn as nn

# Ensure parent directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from federated.server import federated_averaging
from models.multimodal_net import MultimodalNet
from models.clinical_encoder import ClinicalEncoder
from training.client_trainer import train_client_local, evaluate_model


def test_fedavg_mathematical_precision() -> None:
    """Verify FedAvg weighted average math: w_global = sum((n_k/N) * w_k)."""
    # Create simple mock state dicts
    w1 = {"weight": torch.full((2, 2), 1.0), "bias": torch.full((2,), 2.0), "count": torch.tensor(10)}
    w2 = {"weight": torch.full((2, 2), 5.0), "bias": torch.full((2,), 6.0), "count": torch.tensor(20)}

    client_weights = [w1, w2]
    sample_counts = [100, 300]  # Total N = 400. Weight factors: C1 = 0.25, C2 = 0.75

    aggregated = federated_averaging(client_weights, sample_counts)

    # Expected weight: 0.25 * 1.0 + 0.75 * 5.0 = 0.25 + 3.75 = 4.0
    # Expected bias:   0.25 * 2.0 + 0.75 * 6.0 = 0.50 + 4.50 = 5.0
    expected_weight = torch.full((2, 2), 4.0)
    expected_bias = torch.full((2,), 5.0)

    assert torch.allclose(aggregated["weight"], expected_weight), (
        f"FedAvg weight calculation incorrect. Got {aggregated['weight']}, expected {expected_weight}"
    )
    assert torch.allclose(aggregated["bias"], expected_bias), (
        f"FedAvg bias calculation incorrect. Got {aggregated['bias']}, expected {expected_bias}"
    )
    assert aggregated["count"].item() == 10, "Non-floating point buffer corrupted during aggregation"

    print("[PASS] Test 1 — FedAvg mathematical precision verified.")


def test_multimodal_fedavg_round_trip() -> None:
    """Verify FedAvg aggregation across 3 full MultimodalNet state dicts."""
    clin_enc1 = ClinicalEncoder(input_dim=2, hidden_dim=32, output_dim=64)
    model1 = MultimodalNet(clinical_encoder_module=clin_enc1, num_clinical_features=2, pretrained=False)

    clin_enc2 = ClinicalEncoder(input_dim=2, hidden_dim=32, output_dim=64)
    model2 = MultimodalNet(clinical_encoder_module=clin_enc2, num_clinical_features=2, pretrained=False)

    clin_enc3 = ClinicalEncoder(input_dim=2, hidden_dim=32, output_dim=64)
    model3 = MultimodalNet(clinical_encoder_module=clin_enc3, num_clinical_features=2, pretrained=False)

    client_weights = [model1.state_dict(), model2.state_dict(), model3.state_dict()]
    sample_counts = [1000, 1000, 1000]

    aggregated = federated_averaging(client_weights, sample_counts)

    # Create global model and load aggregated state dict
    global_model = MultimodalNet(num_clinical_features=2, pretrained=False)
    load_res = global_model.load_state_dict(aggregated)

    assert len(load_res.missing_keys) == 0, f"Missing keys during FedAvg load: {load_res.missing_keys}"
    assert len(load_res.unexpected_keys) == 0, f"Unexpected keys during FedAvg load: {load_res.unexpected_keys}"

    print("[PASS] Test 2 — MultimodalNet state_dict FedAvg aggregation round-trip verified.")


def test_local_trainer_and_eval() -> None:
    """Verify train_client_local and evaluate_model execution with synthetic batch DataLoader."""
    model = MultimodalNet(num_clinical_features=2, pretrained=False)

    # Create mock dataloader yielding 2 synthetic batches
    mock_batch = {
        "image": torch.randn(4, 3, 224, 224),
        "clinical": torch.randn(4, 2),
        "label": torch.tensor([[1.0], [0.0], [0.0], [1.0]]),
        "patient_id": ["p1", "p2", "p3", "p4"]
    }
    mock_loader = [mock_batch, mock_batch]

    # Test local training
    updated_state, epoch_losses = train_client_local(
        model=model,
        dataloader=mock_loader,
        epochs=2,
        lr=1e-4,
        device="cpu"
    )

    assert isinstance(updated_state, dict), "train_client_local did not return a state_dict"
    assert len(epoch_losses) == 2, f"Expected 2 epoch losses, got {len(epoch_losses)}"
    assert all(loss > 0.0 for loss in epoch_losses), "Loss should be positive"

    # Test evaluation
    eval_metrics = evaluate_model(model, mock_loader, device="cpu")

    assert "val_loss" in eval_metrics and "val_accuracy" in eval_metrics, "Missing evaluation metric keys"
    assert 0.0 <= eval_metrics["val_accuracy"] <= 1.0, f"Invalid accuracy: {eval_metrics['val_accuracy']}"

    print("[PASS] Test 3 — Local client training and evaluation routines verified.")


if __name__ == "__main__":
    print("=" * 60)
    print("MEMBER 3 — FEDERATED LEARNING PIPELINE UNIT TESTS")
    print("=" * 60)
    print()

    test_fedavg_mathematical_precision()
    print()
    test_multimodal_fedavg_round_trip()
    print()
    test_local_trainer_and_eval()

    print()
    print("=" * 60)
    print("ALL MEMBER 3 TESTS PASSED SUCCESSFULLY")
    print("=" * 60)
