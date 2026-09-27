"""
models/__init__.py
FedMed Model Package.

Exports the core multimodal architecture components for federated pneumonia detection:
  - ImageEncoder: Vision Transformer (ViT) image backbone.
  - FusionClassifier: Multimodal fusion and binary classification head.
  - MultimodalNet: Assembled multimodal model combining vision and clinical pathways.
"""
from models.multimodal_net import ImageEncoder, FusionClassifier, MultimodalNet

__all__ = ["ImageEncoder", "FusionClassifier", "MultimodalNet"]
