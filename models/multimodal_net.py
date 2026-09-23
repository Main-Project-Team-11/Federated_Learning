"""
models/multimodal_net.py
FedMed Multimodal Architecture — Member 2 Deliverable.

Implements the core multimodal model for binary pneumonia classification,
combining a Vision Transformer (ViT) image encoder with a clinical tabular
encoder via a fusion-and-classification head.

Architecture (Khader et al., 2023 inspired):

    Chest X-ray [B, 3, 224, 224]
            |
      ImageEncoder (ViT backbone)
            |
       h_img [B, img_embed_dim]
                                        Clinical Features [B, 2]
                                                |
                                         ClinicalEncoder (MLP)
                                                |
                                         h_clin [B, clin_embed_dim]
            |                                   |
            +------- concatenate ---------------+
            |
      FusionClassifier
            |
      logits [B, 1]  (unnormalized, for BCEWithLogitsLoss)

Exported classes:
  - ImageEncoder:      ViT-based image backbone producing visual embeddings.
  - FusionClassifier:  Concatenation-based fusion + binary classification head.
  - MultimodalNet:     Assembled end-to-end model with pluggable clinical encoder.

Interface contract:
  logits = model(images, clinical_features)
  - images:            torch.Tensor [B, 3, 224, 224], float32, ImageNet normalized
  - clinical_features: torch.Tensor [B, 2], float32 (age z-score, sex binary)
  - logits:            torch.Tensor [B, 1], float32 (unnormalized)

FedAvg compatibility:
  - model.state_dict()  returns all parameters for aggregation.
  - model.load_state_dict(weights)  loads aggregated global weights.
"""
import torch
import torch.nn as nn
from torchvision import models as tv_models


# ---------------------------------------------------------------------------
# 1. ImageEncoder — Vision Transformer backbone
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """
    Vision Transformer (ViT) Image Encoder based on Khader et al. (2023).

    Workflow:
      1. Divides input chest X-ray [B, 3, 224, 224] into non-overlapping patches.
      2. Linearly projects patches to image tokens.
      3. Adds spatial positional embeddings to preserve anatomical layout.
      4. Passes tokens through Transformer encoder layers (self-attention + MLP).
      5. Extracts the [CLS] token representation.
      6. Projects to a configurable embedding dimension for downstream fusion.

    Note:
      The ViT variant is configurable via ``vit_variant``. The default
      ``vit_b_16`` is a proposed starting point; the final variant is subject
      to team confirmation and hardware benchmarking.

    Args:
        embed_dim:   Target image embedding dimension (default: 256).
        vit_variant: Name of the torchvision ViT model function
                     (e.g. "vit_b_16", "vit_b_32", "vit_l_16").
        pretrained:  If True, loads ImageNet-pretrained weights from torchvision.
                     If False, initializes with random weights (safe for offline
                     environments and unit testing).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        vit_variant: str = "vit_b_16",
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # Resolve the requested torchvision ViT constructor
        if not hasattr(tv_models, vit_variant):
            raise ValueError(
                f"Unsupported ViT variant '{vit_variant}'. "
                f"Available torchvision ViT models include: "
                f"vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14."
            )

        vit_fn = getattr(tv_models, vit_variant)

        # Load with or without pretrained weights
        if pretrained:
            try:
                self.vit = vit_fn(weights="DEFAULT")
            except Exception:
                # Fallback for environments where weights cannot be downloaded
                print(
                    f"[ImageEncoder] WARNING: Could not load pretrained weights "
                    f"for '{vit_variant}'. Falling back to random initialization."
                )
                self.vit = vit_fn(weights=None)
        else:
            self.vit = vit_fn(weights=None)

        # Determine the ViT's native head input dimension (e.g. 768 for vit_b_*)
        vit_head_in_features: int = self.vit.heads.head.in_features

        # Replace the classification head with a projection layer to embed_dim
        self.vit.heads.head = nn.Linear(vit_head_in_features, embed_dim)

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Chest X-ray images [B, 3, 224, 224], float32, ImageNet normalized.

        Returns:
            Visual embeddings [B, embed_dim], float32.
        """
        return self.vit(x)


# ---------------------------------------------------------------------------
# 2. FusionClassifier — Multimodal fusion + binary classification head
# ---------------------------------------------------------------------------

class FusionClassifier(nn.Module):
    """
    Multimodal Fusion and Binary Classification Head.

    Combines ViT visual representations (h_img) and clinical representations
    (h_clin) via concatenation, followed by a projection block and a single
    logit output for binary pneumonia classification.

    Architecture:
        concat(h_img, h_clin) [B, img_embed_dim + clin_embed_dim]
                |
        Linear(fused_dim, hidden_dim)
        BatchNorm1d(hidden_dim)
        ReLU
        Dropout(dropout_rate)
                |
        Linear(hidden_dim, 1)
                |
        logits [B, 1]

    Args:
        img_embed_dim:  Dimension of image embeddings (default: 256).
        clin_embed_dim: Dimension of clinical embeddings (default: 64).
        hidden_dim:     Hidden layer size in the fusion block (default: 128).
        dropout_rate:   Dropout probability (default: 0.3).
    """

    def __init__(
        self,
        img_embed_dim: int = 256,
        clin_embed_dim: int = 64,
        hidden_dim: int = 128,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        fused_dim = img_embed_dim + clin_embed_dim

        self.fusion_block = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        # Binary classification output: 1 unnormalized logit
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self, h_img: torch.Tensor, h_clin: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_img:  Visual embeddings  [B, img_embed_dim],  float32.
            h_clin: Clinical embeddings [B, clin_embed_dim], float32.

        Returns:
            Unnormalized logits [B, 1], float32.
            (Directly compatible with nn.BCEWithLogitsLoss.)
        """
        h_fused = torch.cat([h_img, h_clin], dim=1)
        features = self.fusion_block(h_fused)
        logits = self.head(features)
        return logits


# ---------------------------------------------------------------------------
# 3. Default Clinical Encoder (fallback for testing before Member 4 delivers)
# ---------------------------------------------------------------------------

class _DefaultClinicalEncoder(nn.Module):
    """
    Fallback Clinical Encoder following the documented Member 4 architecture.

    Used internally by MultimodalNet when no external ClinicalEncoder module
    is provided.  This allows Member 2 and Member 3 to test the full pipeline
    before Member 4 delivers the final encoder.

    Architecture:
        Linear(num_features, 32) -> BatchNorm1d(32) -> ReLU
        Linear(32, embed_dim)    -> BatchNorm1d(embed_dim) -> ReLU

    Args:
        num_features: Number of input clinical features (default: 2).
        embed_dim:    Output embedding dimension (default: 64).
    """

    def __init__(self, num_features: int = 2, embed_dim: int = 64) -> None:
        super().__init__()
        hidden_dim = 32
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Clinical features [B, num_features], float32.

        Returns:
            Clinical embeddings [B, embed_dim], float32.
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# 4. MultimodalNet — Assembled end-to-end model
# ---------------------------------------------------------------------------

class MultimodalNet(nn.Module):
    """
    Assembled Multimodal Architecture for federated pneumonia detection.

    Combines:
      - Vision Transformer (ViT) Image Encoder   (Member 2)
      - Clinical Tabular Encoder                  (Member 4, or built-in fallback)
      - Multimodal Fusion & Classification Head    (Member 2)

    Usage with Member 4's ClinicalEncoder (when available):
        clinical_enc = ClinicalEncoder(num_features=2, embed_dim=64)
        model = MultimodalNet(clinical_encoder_module=clinical_enc)

    Usage with built-in fallback (for immediate testing):
        model = MultimodalNet()  # uses _DefaultClinicalEncoder internally

    Forward signature:
        logits = model(images, clinical_features)

    FedAvg compatibility:
        weights = model.state_dict()
        model.load_state_dict(aggregated_weights)

    Args:
        clinical_encoder_module: An nn.Module mapping [B, num_features] -> [B, clin_embed_dim].
                                 If None, a built-in fallback encoder is used.
        img_embed_dim:           Image embedding dimension (default: 256).
        clin_embed_dim:          Clinical embedding dimension (default: 64).
        num_clinical_features:   Number of input clinical features (default: 2).
                                 Only used when clinical_encoder_module is None.
        vit_variant:             Torchvision ViT model name (default: "vit_b_16").
        pretrained:              Whether to load pretrained ViT weights (default: True).
    """

    def __init__(
        self,
        clinical_encoder_module: nn.Module | None = None,
        img_embed_dim: int = 256,
        clin_embed_dim: int = 64,
        num_clinical_features: int = 2,
        vit_variant: str = "vit_b_16",
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # Image pathway (Member 2)
        self.image_encoder = ImageEncoder(
            embed_dim=img_embed_dim,
            vit_variant=vit_variant,
            pretrained=pretrained,
        )

        # Clinical pathway (Member 4 or fallback)
        if clinical_encoder_module is not None:
            self.clinical_encoder = clinical_encoder_module
        else:
            self.clinical_encoder = _DefaultClinicalEncoder(
                num_features=num_clinical_features,
                embed_dim=clin_embed_dim,
            )

        # Fusion & classification (Member 2)
        self.fusion_classifier = FusionClassifier(
            img_embed_dim=img_embed_dim,
            clin_embed_dim=clin_embed_dim,
        )

    def forward(
        self, images: torch.Tensor, clinical_features: torch.Tensor
    ) -> torch.Tensor:
        """
        End-to-end forward pass through both modalities.

        Args:
            images:            Chest X-ray images [B, 3, 224, 224], float32.
            clinical_features: Clinical features  [B, 2], float32.

        Returns:
            Unnormalized logits [B, 1], float32.
        """
        h_img = self.image_encoder(images)
        h_clin = self.clinical_encoder(clinical_features)
        logits = self.fusion_classifier(h_img, h_clin)
        return logits
