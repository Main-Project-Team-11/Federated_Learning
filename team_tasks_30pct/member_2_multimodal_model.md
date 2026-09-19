# Member 2: Vision & Multimodal Architecture Lead

- **Role:** Member 2 — Vision Encoder, Fusion Layer & Binary Classification
- **Milestone:** 30% Implementation Presentation
- **Primary Focus:** Constructing the core vision backbone (CNN Image Encoder), designing the Multimodal Fusion Layer, building the Binary Classification Head, and integrating with Member 4's Clinical Encoder.

---

## 1. Responsibilities & Objectives

1. **Image Encoder Backbone**:
   - Construct a convolutional vision backbone (e.g., pretrained `torchvision.models.resnet18` or a custom lightweight CNN).
   - Adapt the output head of the CNN to output an image embedding vector $h_{\text{img}} \in \mathbb{R}^{d_{\text{img}}}$ (e.g., $d_{\text{img}} = 256$).
   - Ensure the image encoder supports standard normalized input tensors: `[B, 3, 224, 224]` or `[B, 1, 224, 224]`.
2. **Multimodal Fusion Layer**:
   - Design a fusion module that combines:
     - Visual representation $h_{\text{img}}$ (from your Image Encoder)
     - Clinical representation $h_{\text{clin}}$ (from Member 4's Clinical Encoder)
   - Concatenate features: $h_{\text{fused}} = [h_{\text{img}} \,\|\, h_{\text{clin}}] \in \mathbb{R}^{d_{\text{img}} + d_{\text{clin}}}$.
   - Pass through a projection block: Linear layer $\to$ LayerNorm/BatchNorm $\to$ ReLU $\to$ Dropout ($p=0.3$).
3. **Binary Classification Head**:
   - Final projection layer mapping fused representations to a single unnormalized logit ($1$ output neuron).
   - Binary objective: Positive = `Pneumonia`, Negative = `Normal`.
   - Forward signature outputs logits suitable for `BCEWithLogitsLoss`.
4. **Integration with Member 4**:
   - Provide the complete `MultimodalNet` model class that plugs in Member 4's `ClinicalEncoder`.
   - Run forward pass unit tests with synthetic tensors to guarantee stability.

---

## 2. Interface Contract (What you receive & provide)

- **Receives from Member 1**: Image batch tensors `batch["image"]` of shape `[B, 3, 224, 224]`.
- **Receives from Member 4**: `ClinicalEncoder` class (or clinical embedding vectors $h_{\text{clin}}$ of shape `[B, 64]`).
- **Provides to Member 3 & Member 4**:
  - `models/multimodal_net.py` containing:
    - `ImageEncoder` class
    - `FusionClassifier` class
    - Assembled `MultimodalNet` class
  - Forward pass signature: `logits = model(images, clinical_features)` of shape `[B, 1]`.
  - State dictionary compatibility for FedAvg: `model.state_dict()`.

---

## 3. Step-by-Step Implementation Guide

### Step 1: Implement Image Encoder & Fusion Classifier
Create `models/multimodal_net.py`:
```python
import torch
import torch.nn as nn
from torchvision import models

class ImageEncoder(nn.Module):
    """CNN-based Image Encoder extracting radiograph visual representations."""
    def __init__(self, embed_dim=256, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        # Replace final classification head with projection to embedding dimension
        backbone.fc = nn.Linear(in_features, embed_dim)
        self.encoder = backbone
        
    def forward(self, x):
        # Input: [B, 3, 224, 224] -> Output: [B, embed_dim]
        return self.encoder(x)


class FusionClassifier(nn.Module):
    """Concatenation Fusion and Binary Classification Head."""
    def __init__(self, img_embed_dim=256, clin_embed_dim=64, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        fused_dim = img_embed_dim + clin_embed_dim
        
        self.fusion_block = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        # Binary classification output: 1 logit
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, h_img, h_clin):
        # Concatenate along feature dimension: [B, fused_dim]
        h_fused = torch.cat([h_img, h_clin], dim=1)
        features = self.fusion_block(h_fused)
        logits = self.head(features)
        return logits


class MultimodalNet(nn.Module):
    """Assembled Multimodal Architecture combining Image Encoder, Clinical Encoder & Fusion."""
    def __init__(self, clinical_encoder_module, img_embed_dim=256, clin_embed_dim=64):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=img_embed_dim)
        self.clinical_encoder = clinical_encoder_module  # Provided by Member 4
        self.fusion_classifier = FusionClassifier(
            img_embed_dim=img_embed_dim,
            clin_embed_dim=clin_embed_dim
        )

    def forward(self, images, clinical_features):
        h_img = self.image_encoder(images)
        h_clin = self.clinical_encoder(clinical_features)
        logits = self.fusion_classifier(h_img, h_clin)
        return logits
```

### Step 2: Write Forward Pass Unit Tests
Create `tests/test_vision_and_fusion.py`:
```python
import torch
import torch.nn as nn
from models.multimodal_net import ImageEncoder, FusionClassifier, MultimodalNet

def test_image_encoder():
    encoder = ImageEncoder(embed_dim=256)
    dummy_img = torch.randn(4, 3, 224, 224)
    h_img = encoder(dummy_img)
    assert h_img.shape == (4, 256), f"Expected (4, 256), got {h_img.shape}"
    print("[PASS] ImageEncoder forward pass verified.")

def test_fusion_and_head():
    fusion = FusionClassifier(img_embed_dim=256, clin_embed_dim=64)
    h_img = torch.randn(4, 256)
    h_clin = torch.randn(4, 64)
    logits = fusion(h_img, h_clin)
    assert logits.shape == (4, 1), f"Expected (4, 1), got {logits.shape}"
    print("[PASS] FusionClassifier forward pass verified.")

def test_assembled_multimodal():
    # Mock clinical encoder for testing before Member 4 integrates
    mock_clinical_encoder = nn.Linear(3, 64)
    model = MultimodalNet(
        clinical_encoder_module=mock_clinical_encoder,
        img_embed_dim=256,
        clin_embed_dim=64
    )
    
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_clin = torch.randn(2, 3)
    logits = model(dummy_img, dummy_clin)
    assert logits.shape == (2, 1), f"Expected (2, 1), got {logits.shape}"
    print("[PASS] Complete MultimodalNet end-to-end forward pass verified.")

if __name__ == "__main__":
    test_image_encoder()
    test_fusion_and_head()
    test_assembled_multimodal()
```

---

## 4. Deliverables Checklist for 30% Presentation

- [ ] `ImageEncoder` implemented with ResNet-18 backbone.
- [ ] `FusionClassifier` concatenation + projection module implemented.
- [ ] `MultimodalNet` wrapper implemented with pluggable clinical encoder.
- [ ] Forward unit test script `test_vision_and_fusion.py` passes without error.
- [ ] Deliver `models/multimodal_net.py` to **Member 4 (Clinical & Training Lead)** and **Member 3 (FL Lead)**.
