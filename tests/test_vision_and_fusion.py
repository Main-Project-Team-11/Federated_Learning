"""
tests/test_vision_and_fusion.py
Member 2 Unit Tests — Vision Encoder, Fusion Classifier & MultimodalNet.

Verifies:
  1. ImageEncoder forward pass shapes, dtypes, and numerical health.
  2. FusionClassifier forward pass shapes and finite outputs.
  3. MultimodalNet end-to-end forward pass with fallback clinical encoder.
  4. Backpropagation and gradient flow through the full model.
  5. state_dict serialization and load_state_dict round-trip (FedAvg readiness).

All tests use synthetic tensors — no real dataset or pretrained weight download
is required.
"""
import copy
import os
import sys
import torch
import torch.nn as nn

# Ensure workspace root is in Python path (matches project convention)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.multimodal_net import ImageEncoder, FusionClassifier, MultimodalNet


# ===================================================================
# Test 1 — ImageEncoder forward pass
# ===================================================================

def test_vit_image_encoder() -> None:
    """Verify ViT ImageEncoder produces correct shape/dtype with no NaN/Inf."""
    encoder = ImageEncoder(embed_dim=256, vit_variant="vit_b_16", pretrained=False)
    dummy_images = torch.randn(2, 3, 224, 224)

    h_img = encoder(dummy_images)

    assert h_img.shape == (2, 256), (
        f"Expected shape (2, 256), got {h_img.shape}"
    )
    assert h_img.dtype == torch.float32, (
        f"Expected dtype float32, got {h_img.dtype}"
    )
    assert not torch.isnan(h_img).any(), "NaN detected in ImageEncoder output"
    assert not torch.isinf(h_img).any(), "Inf detected in ImageEncoder output"

    print("[PASS] Test 1 — ViT ImageEncoder forward pass verified.")
    print(f"       Output shape: {list(h_img.shape)}, dtype: {h_img.dtype}")


# ===================================================================
# Test 2 — FusionClassifier forward pass
# ===================================================================

def test_fusion_classifier() -> None:
    """Verify FusionClassifier produces [B, 1] logits with finite values."""
    fusion = FusionClassifier(
        img_embed_dim=256, clin_embed_dim=64, hidden_dim=128, dropout_rate=0.3
    )
    fusion.eval()  # Disable dropout for deterministic test

    h_img = torch.randn(4, 256)
    h_clin = torch.randn(4, 64)

    logits = fusion(h_img, h_clin)

    assert logits.shape == (4, 1), (
        f"Expected shape (4, 1), got {logits.shape}"
    )
    assert torch.isfinite(logits).all(), (
        "Non-finite values detected in FusionClassifier output"
    )

    print("[PASS] Test 2 — FusionClassifier forward pass verified.")
    print(f"       Output shape: {list(logits.shape)}")


# ===================================================================
# Test 3 — MultimodalNet end-to-end forward pass
# ===================================================================

def test_multimodal_net_forward() -> None:
    """Verify assembled MultimodalNet produces [B, 1] logits end-to-end."""
    model = MultimodalNet(
        clinical_encoder_module=None,  # Use built-in fallback
        img_embed_dim=256,
        clin_embed_dim=64,
        num_clinical_features=2,
        vit_variant="vit_b_16",
        pretrained=False,
    )
    model.eval()

    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_clinical = torch.randn(2, 2)

    logits = model(dummy_images, dummy_clinical)

    assert logits.shape == (2, 1), (
        f"Expected shape (2, 1), got {logits.shape}"
    )
    assert torch.isfinite(logits).all(), (
        "Non-finite values detected in MultimodalNet output"
    )

    print("[PASS] Test 3 — MultimodalNet end-to-end forward pass verified.")
    print(f"       Output shape: {list(logits.shape)}")


# ===================================================================
# Test 4 — Backpropagation and gradient flow
# ===================================================================

def test_backward_pass() -> None:
    """Verify loss.backward() propagates gradients through the full model."""
    model = MultimodalNet(
        clinical_encoder_module=None,
        img_embed_dim=256,
        clin_embed_dim=64,
        num_clinical_features=2,
        vit_variant="vit_b_16",
        pretrained=False,
    )
    model.train()

    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_clinical = torch.randn(2, 2)
    dummy_labels = torch.tensor([[1.0], [0.0]])

    criterion = nn.BCEWithLogitsLoss()
    logits = model(dummy_images, dummy_clinical)
    loss = criterion(logits, dummy_labels)

    loss.backward()

    # Check that at least some parameters received gradients
    params_with_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    total_params = sum(1 for _ in model.parameters())

    assert params_with_grad > 0, (
        "No model parameters received gradients after loss.backward()"
    )

    print("[PASS] Test 4 — Backpropagation verified.")
    print(f"       Loss value: {loss.item():.6f}")
    print(f"       Parameters with gradients: {params_with_grad}/{total_params}")


# ===================================================================
# Test 5 — state_dict save/load round-trip (FedAvg readiness)
# ===================================================================

def test_state_dict_round_trip() -> None:
    """Verify state_dict export and load_state_dict import without key mismatches."""
    model1 = MultimodalNet(
        clinical_encoder_module=None,
        img_embed_dim=256,
        clin_embed_dim=64,
        num_clinical_features=2,
        vit_variant="vit_b_16",
        pretrained=False,
    )

    # Export state dictionary
    state = model1.state_dict()

    assert isinstance(state, dict), "state_dict() did not return a dict"
    assert len(state) > 0, "state_dict() returned an empty dict"

    # Create a fresh model with identical architecture
    model2 = MultimodalNet(
        clinical_encoder_module=None,
        img_embed_dim=256,
        clin_embed_dim=64,
        num_clinical_features=2,
        vit_variant="vit_b_16",
        pretrained=False,
    )

    # Load the first model's state into the second model
    load_result = model2.load_state_dict(state)

    assert len(load_result.missing_keys) == 0, (
        f"Missing keys during load_state_dict: {load_result.missing_keys}"
    )
    assert len(load_result.unexpected_keys) == 0, (
        f"Unexpected keys during load_state_dict: {load_result.unexpected_keys}"
    )

    # Verify parameter values match after loading
    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_clinical = torch.randn(2, 2)

    model1.eval()
    model2.eval()

    with torch.no_grad():
        out1 = model1(dummy_images, dummy_clinical)
        out2 = model2(dummy_images, dummy_clinical)

    assert torch.equal(out1, out2), (
        "Outputs differ after state_dict round-trip"
    )

    print("[PASS] Test 5 — state_dict save/load round-trip verified.")
    print(f"       Total state_dict keys: {len(state)}")
    print(f"       Missing keys: 0, Unexpected keys: 0")
    print(f"       Output match after load: True")


# ===================================================================
# Main runner
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MEMBER 2 — VISION & MULTIMODAL ARCHITECTURE UNIT TESTS")
    print("=" * 60)
    print()

    test_vit_image_encoder()
    print()
    test_fusion_classifier()
    print()
    test_multimodal_net_forward()
    print()
    test_backward_pass()
    print()
    test_state_dict_round_trip()

    print()
    print("=" * 60)
    print("ALL MEMBER 2 TESTS PASSED SUCCESSFULLY")
    print("=" * 60)
