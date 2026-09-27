"""Configurable Vision Transformer and clinical-feature fusion model."""

import torch
import torch.nn as nn
from torchvision import models

from models.clinical_encoder import ClinicalEncoder


class ImageEncoder(nn.Module):
    """Project chest X-rays into a configurable visual embedding."""

    def __init__(
        self,
        embed_dim: int = 256,
        vit_variant: str = "vit_b_16",
        pretrained: bool = False,
    ):
        super().__init__()
        vit_factory = getattr(models, vit_variant, None)
        if vit_factory is None or not callable(vit_factory):
            raise ValueError(f"Unsupported torchvision ViT variant: {vit_variant}")

        weights_enum = getattr(models, f"{vit_variant}_Weights", None)
        weights = weights_enum.DEFAULT if pretrained and weights_enum else None
        self.vit = vit_factory(weights=weights)
        input_dim = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(input_dim, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.vit(images)


class FusionClassifier(nn.Module):
    """Fuse visual and clinical embeddings and produce one classification logit."""

    def __init__(
        self,
        image_dim: int = 256,
        clinical_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + clinical_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self, image_features: torch.Tensor, clinical_features: torch.Tensor
    ) -> torch.Tensor:
        fused = torch.cat((image_features, clinical_features), dim=1)
        return self.classifier(self.fusion(fused))


class MultimodalNet(nn.Module):
    """Combine a ViT image pathway and the shared clinical encoder pathway."""

    def __init__(
        self,
        clinical_encoder_module: nn.Module | None = None,
        image_dim: int = 256,
        clinical_dim: int = 64,
        hidden_dim: int = 128,
        vit_variant: str = "vit_b_16",
        pretrained: bool = False,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            embed_dim=image_dim,
            vit_variant=vit_variant,
            pretrained=pretrained,
        )
        self.clinical_encoder = clinical_encoder_module or ClinicalEncoder(
            input_dim=2,
            output_dim=clinical_dim,
        )
        self.fusion_classifier = FusionClassifier(
            image_dim=image_dim,
            clinical_dim=clinical_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self, images: torch.Tensor, clinical_features: torch.Tensor
    ) -> torch.Tensor:
        image_features = self.image_encoder(images)
        clinical_features = self.clinical_encoder(clinical_features)
        return self.fusion_classifier(image_features, clinical_features)