# FedMed — Multimodal Data Pipeline & Team Integration Guide
> **Member 1 Deliverable | 30% Milestone Baseline**  
> *Module:* `data/` & `scripts/`  
> *Target Audience:* Member 2 (Vision Lead), Member 3 (FL Lead), Member 4 (Clinical & Benchmark Lead), Project Evaluators

Welcome to the **FedMed Multimodal Data Pipeline**. This module acquires, cleans, harmonizes, validates, splits, and serves paired Chest X-ray images and clinical tabular features (Age, Sex) for federated pneumonia detection across simulated hospital clients.

---

## 📑 Table of Contents
1. [What Member 1 Created (File Directory)](#1-what-member-1-created-file-directory)
2. [The Data Contract (Batch Interface Specification)](#2-the-data-contract-batch-interface-specification)
3. [Feature Guide: For Member 2 (Vision & Multimodal Architecture Lead)](#3-feature-guide-for-member-2-vision--multimodal-architecture-lead)
4. [Feature Guide: For Member 4 (Clinical Modeling & Centralized Baseline Lead)](#4-feature-guide-for-member-4-clinical-modeling--centralized-baseline-lead)
5. [Feature Guide: For Member 3 (Federated Learning Simulation Lead)](#5-feature-guide-for-member-3-federated-learning-simulation-lead)
6. [Dataset & Partition Summary](#6-dataset--partition-summary)
7. [How to Verify the Pipeline](#7-how-to-verify-the-pipeline)
8. [Critical Class Imbalance & Loss Function Guidance](#8-critical-class-imbalance--loss-function-guidance)
9. [Troubleshooting & FAQs](#9-troubleshooting--faqs)

---

## 1. What Member 1 Created (File Directory)

Below is the complete map of files delivered by Member 1 and how they connect to the project:

```text
Main_Project/
├── data/
│   ├── multimodal_dataset.py           <-- MAIN ENTRY POINT: PyTorch Dataset & DataLoaders
│   └── README.md                       <-- THIS FILE: Integration instructions for teammates
│
├── data_prep/                          <-- Partitioned Parquet & CSV tables (Local only)
│   ├── client_1.parquet / client_1.csv <-- Hospital Client 1 training data (27,593 samples)
│   ├── client_2.parquet / client_2.csv <-- Hospital Client 2 training data (27,878 samples)
│   ├── client_3.parquet / client_3.csv <-- Hospital Client 3 training data (28,225 samples)
│   ├── train.parquet / train.csv       <-- Pooled training data (83,696 samples)
│   ├── val.parquet / val.csv           <-- Held-out validation data (17,470 samples)
│   ├── test.parquet / test.csv         <-- Held-out test data (17,488 samples)
│   ├── stage5_dataset_statistics.json  <-- Exact class distributions & clinical stats
│   └── stage5_dataset_statistics.md    <-- Formatted summary report of all distributions
│
├── scripts/                            <-- 7-Stage Data Pipeline & Verification Scripts
│   ├── stage1_prepare_nih.py           <-- Stage 1: NIH metadata extraction
│   ├── stage1_prepare_chexpert.py      <-- Stage 1: CheXpert metadata extraction
│   ├── stage2_canonical_schema.py      <-- Stage 2: Schema harmonization & deduplication
│   ├── stage3_binary_labels.py         <-- Stage 3: Label policy & frontal view filtering
│   ├── stage4_data_validation_and_cleaning.py <-- Stage 4: 8 data validation & cleaning checks
│   ├── stage5_dataset_statistics.py    <-- Stage 5: Distribution & imbalance calculation
│   ├── stage6_patient_level_split.py   <-- Stage 6: Disjoint patient-level Train/Val/Test split
│   ├── stage7_client_partitions.py     <-- Stage 7: 3-Client patient-level partitioning
│   └── verify_pipeline_integration.py  <-- Automated test suite verifying all 5 DataLoaders
│
└── CREATED_FILES_TRACKER.md            <-- Comprehensive changelog & Git staging guide
```

---

## 2. The Data Contract (Batch Interface Specification)

Every DataLoader (`get_client_loader`, `get_val_loader`, `get_test_loader`) yields a Python dictionary with the following schema:

| Key | Type | Tensor Shape | Dtype | Description & Normalization |
|---|---|---|---|---|
| `batch["image"]` | `torch.Tensor` | `[B, 3, 224, 224]` | `torch.float32` | RGB Chest X-ray resized to $224 \times 224$, ImageNet normalized ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$). |
| `batch["clinical"]` | `torch.Tensor` | `[B, 2]` | `torch.float32` | **Index 0:** Patient Age $z$-score normalized ($\mu=47.70, \sigma=16.80$).<br>**Index 1:** Patient Sex binary encoded (`0.0` = Male, `1.0` = Female). |
| `batch["label"]` | `torch.Tensor` | `[B, 1]` | `torch.float32` | Binary ground truth (`1.0` = Pneumonia, `0.0` = Non-Pneumonia / Normal). |
| `batch["patient_id"]` | `list` of `str` | Length `B` | `str` | Unique patient ID (enables 1:1 clinical auditing & ensures zero leakage). |
| `batch["sample_id"]` | `list` of `str` | Length `B` | `str` | Unique scan/image identifier. |
| `batch["dataset_source"]`| `list` of `str` | Length `B` | `str` | Origin dataset (`"NIH"` or `"CheXpert"`). |

---

## 3. Feature Guide: For Member 2 (Vision & Multimodal Architecture Lead)

### 🎯 Your Goal
Build `models/multimodal_net.py` containing:
1. **`ImageEncoder` (Vision Transformer / ViT)**: Processes `batch["image"]` into visual embeddings $h_{\text{img}}$.
2. **`FusionClassifier`**: Combines $h_{\text{img}}$ with Member 4's clinical embeddings $h_{\text{clin}}$.
3. **`MultimodalNet`**: Assembled model producing unnormalized logits of shape `[B, 1]`.

### 🛠️ How to Use Member 1's Files
Import `get_client_loader` directly from `data/multimodal_dataset.py`.

### 💻 Step-by-Step Code Example for Member 2

Create a quick verification script (e.g., `test_vision_feature.py`):

```python
import torch
import torch.nn as nn
from torchvision import models
from data.multimodal_dataset import get_client_loader

# 1. Load a real batch from Member 1's Hospital Client 1 DataLoader
loader = get_client_loader(client_id=1, batch_size=16)
batch = next(iter(loader))

images = batch["image"]       # Shape: [16, 3, 224, 224], torch.float32
labels = batch["label"]       # Shape: [16, 1], torch.float32

print(f"Loaded image batch shape: {images.shape}")
print(f"Loaded label batch shape: {labels.shape}")

# 2. Define your Vision Transformer (ViT) Image Encoder
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Use torchvision ViT backbone (e.g., vit_b_16)
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.conv_proj = vit.conv_proj              # Patch embedding layer
        self.class_token = vit.class_token          # Learnable [CLS] token
        self.encoder = vit.encoder                  # Transformer blocks
        self.proj = nn.Linear(vit.heads.head.in_features, embed_dim)

    def forward(self, x):
        # x: [B, 3, 224, 224] -> patches: [B, 768, 14, 14] -> tokens: [B, 196, 768]
        tokens = self.conv_proj(x).flatten(2).transpose(1, 2)
        cls_token = self.class_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = self.encoder(tokens)
        cls_rep = tokens[:, 0]                      # Extract [CLS] token representation
        return self.proj(cls_rep)                   # Output: [B, embed_dim]

# 3. Test forward pass with real data
encoder = ImageEncoder(embed_dim=256)
visual_embeddings = encoder(images)
print(f"Visual embeddings shape: {visual_embeddings.shape}")  # Should be [16, 256]
assert visual_embeddings.shape == (16, 256), "Vision encoder output shape mismatch!"
print("SUCCESS: Member 2 vision feature verified with real data!")
```

---

## 4. Feature Guide: For Member 4 (Clinical Modeling & Centralized Baseline Lead)

### 🎯 Your Goal
1. Build `models/clinical_encoder.py`: Multi-Layer Perceptron (MLP) taking `batch["clinical"]` ($[B, 2]$) to clinical representations $h_{\text{clin}}$ ($[B, 64]$).
2. Build `training/client_trainer.py`: Local training routine `train_client_local()` and evaluation routine `evaluate_model()`.
3. Build `train_centralized.py`: Train centralized benchmark on `data_prep/train.parquet` and evaluate on `get_val_loader()` / `get_test_loader()`.
4. Implement clinical evaluation metrics: **Sensitivity**, **Specificity**, **F1-Score**, and **ROC-AUC**.

### 🛠️ How to Use Member 1's Files
- DataLoaders: `from data.multimodal_dataset import get_client_loader, get_val_loader, get_test_loader, MultimodalDataset`
- Class Imbalance Parameter: Use `pos_weight = 18.60` from `data_prep/stage5_dataset_statistics.json` in `nn.BCEWithLogitsLoss`.

### 💻 Step-by-Step Code Example for Member 4

Create your clinical encoder and test training with Member 1's loaders:

```python
import torch
import torch.nn as nn
from data.multimodal_dataset import get_client_loader, get_val_loader

# 1. Define Clinical Tabular Encoder (Input dimension = 2: Age z-score, Sex binary)
class ClinicalEncoder(nn.Module):
    def __init__(self, num_features=2, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)  # [B, 2] -> [B, 64]

# 2. Verify Clinical Encoder with real batch
loader = get_client_loader(client_id=1, batch_size=32)
batch = next(iter(loader))

clinical_data = batch["clinical"]   # Shape: [32, 2], float32
labels = batch["label"]             # Shape: [32, 1], float32

encoder = ClinicalEncoder(num_features=2, embed_dim=64)
clinical_embeds = encoder(clinical_data)
print(f"Clinical embeddings shape: {clinical_embeds.shape}")  # [32, 64]

# 3. Setup Weighted Loss for Class Imbalance (Crucial!)
# Member 1's Stage 5 analysis established the 18.60:1 non-pneumonia to pneumonia ratio
pos_weight = torch.tensor([18.60])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 4. Evaluation Loop on Held-Out Validation Set
val_loader = get_val_loader(batch_size=32)
print(f"Validation set ready with {len(val_loader.dataset)} samples across {len(val_loader)} batches.")
```

---

## 5. Feature Guide: For Member 3 (Federated Learning Simulation Lead)

### 🎯 Your Goal
1. Build `federated/server.py`: Federated Averaging (`FedAvg`) server aggregating model `state_dict` weights:
   $$w^{t+1} = \sum_{k=1}^{K} \frac{n_k}{N} w_k^{t+1}$$
2. Build `federated/run_simulation.py`: Orchestrate multi-round federated training across 3 simulated hospital clients.
3. Log communication round metrics (loss, validation accuracy, AUC) to `federated_training_log.json`.

### 🛠️ How to Use Member 1's Files
- Import `get_client_loader` for clients 1, 2, and 3.
- Import `get_val_loader` for central global validation after each FedAvg aggregation round.

### 💻 Step-by-Step Code Example for Member 3

```python
import copy
import torch
from data.multimodal_dataset import get_client_loader, get_val_loader

# 1. Instantiate the 3 Simulated Hospital Client DataLoaders
client_loaders = {
    1: get_client_loader(client_id=1, batch_size=32, shuffle=True),
    2: get_client_loader(client_id=2, batch_size=32, shuffle=True),
    3: get_client_loader(client_id=3, batch_size=32, shuffle=True)
}
val_loader = get_val_loader(batch_size=32, shuffle=False)

# 2. Extract Client Sample Counts (n_k) for FedAvg Weighting
client_sample_counts = {cid: len(loader.dataset) for cid, loader in client_loaders.items()}
total_train_samples = sum(client_sample_counts.values())

print("Hospital Client Sample Counts:")
for cid, count in client_sample_counts.items():
    print(f"  Client {cid}: {count:,} samples ({count / total_train_samples * 100:.2f}%)")
print(f"Total Federated Samples: {total_train_samples:,}")

# 3. FedAvg Aggregation Function
def federated_averaging(client_weights_list, sample_counts):
    total = sum(sample_counts)
    global_weights = copy.deepcopy(client_weights_list[0])
    
    # Zero out parameters
    for key in global_weights.keys():
        global_weights[key] = torch.zeros_like(global_weights[key], dtype=torch.float32)
        
    # Accumulate weighted averages
    for w_k, n_k in zip(client_weights_list, sample_counts):
        weight_factor = n_k / total
        for key in global_weights.keys():
            global_weights[key] += w_k[key].to(torch.float32) * weight_factor
            
    return global_weights

# 4. Skeleton of Multi-Round Communication Loop
num_rounds = 5
for round_idx in range(1, num_rounds + 1):
    print(f"\n--- Starting Communication Round {round_idx}/{num_rounds} ---")
    client_updates = []
    
    for cid, loader in client_loaders.items():
        # Member 4's local training loop:
        # updated_weights, loss = train_client_local(model, loader, epochs=2)
        # client_updates.append(updated_weights)
        pass
    
    # Server FedAvg Step:
    # global_weights = federated_averaging(client_updates, list(client_sample_counts.values()))
    # global_model.load_state_dict(global_weights)
    
    # Server Validation Step:
    # val_loss, val_auc = evaluate_model(global_model, val_loader)
```

---

## 6. Dataset & Partition Summary

- **Total Clean, Validated Records:** **118,654**
- **Total Unique Patients:** **36,324**
- **Disjoint Partitioning:** Zero patient overlap across Train, Val, Test, and between simulated hospital clients ($\text{Overlap} = \emptyset$).

| Partition / Client | Samples ($n_k$) | Unique Patients | Pneumonia (Class 1) | Non-Pneumonia (Class 0) | Imbalance Ratio |
|---|---|---|---|---|---|
| **Hospital Client 1** | **27,593** | 8,476 | 1,419 (5.14%) | 26,174 (94.86%) | 18.45 : 1 |
| **Hospital Client 2** | **27,878** | 8,475 | 1,427 (5.12%) | 26,451 (94.88%) | 18.54 : 1 |
| **Hospital Client 3** | **28,225** | 8,475 | 1,425 (5.05%) | 26,800 (94.95%) | 18.81 : 1 |
| **Full Training Set (Centralized)** | **83,696** | **25,426** | **4,271** (5.10%) | **79,425** (94.90%) | **18.60 : 1** |
| **Held-Out Validation Set** | **17,470** | **5,448** | **932** (5.33%) | **16,538** (94.67%) | **17.74 : 1** |
| **Held-Out Test Set** | **17,488** | **5,450** | **902** (5.16%) | **16,586** (94.84%) | **18.39 : 1** |

---

## 7. How to Verify the Pipeline

To verify all 5 DataLoaders, tensor shapes, float32 precision, and lazy loading without running training, execute:

```powershell
python scripts/verify_pipeline_integration.py
```

### Verification Checks Performed:
1. **DataLoader Instantiation**: Initializes Client 1, Client 2, Client 3, Validation, and Test loaders.
2. **Tensor Dimensions**: Confirms `image` is `[B, 3, 224, 224]`, `clinical` is `[B, 2]`, and `label` is `[B, 1]`.
3. **Numerical Health**: Verifies exactly 0 `NaN` and 0 `Inf` across all features and labels.
4. **Binary Ground Truth**: Asserts all label values strictly equal `0.0` or `1.0`.
5. **Lazy Loading Efficiency**: Iterates through multiple consecutive batches to verify steady RAM usage without memory accumulation.

---

## 8. Critical Class Imbalance & Loss Function Guidance

> [!WARNING]
> **Do not use unweighted loss or evaluate on accuracy alone!**  
> In medical chest radiograph screening, pneumonia represents only ~5.1% of all cases. A naive model predicting `0.0` (Non-Pneumonia) for every image will achieve 94.9% accuracy while having a **0% sensitivity**, missing every sick patient.

### Recommended Loss Setup (PyTorch):
Member 1's Stage 5 analysis calculated an exact positive weight of:
$$\text{pos\_weight} = \frac{79,425 \text{ (negative samples)}}{4,271 \text{ (positive samples)}} \approx 18.60$$

Use this in your training loops:
```python
import torch
import torch.nn as nn

pos_weight = torch.tensor([18.60])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### Recommended Evaluation Metrics:
- **Sensitivity / Recall:** $\frac{TP}{TP + FN}$ (Target: $\ge 0.75$)
- **Specificity:** $\frac{TN}{TN + FP}$ (Target: $\ge 0.85$)
- **ROC-AUC:** Area under ROC curve (Target: $\ge 0.80$)
- **F1-Score:** Harmonic mean of precision and recall

---

## 9. Troubleshooting & FAQs

### Q1: Where are the image files stored?
Images reside in local storage (e.g., `R:\FedMed_Data\NIH\images\` and `R:\FedMed_Data\CheXpert\`). If you clone this repository to a new computer, ensure the images are extracted to the path specified in your local environment, or update `DEFAULT_DATA_DIR` in `data/multimodal_dataset.py`.

### Q2: What happens if an image is missing or not yet extracted?
`MultimodalDataset` has an automatic safety fallback: if `os.path.exists(image_path)` is False, it generates a deterministic `torch.zeros(3, 224, 224)` tensor with a non-blocking log. This ensures your code doesn't crash during pipeline development even if certain image batches are pending extraction.

### Q3: Why `num_workers=0` on Windows?
On Windows PowerShell, PyTorch multiprocessing (`num_workers > 0`) requires explicit `if __name__ == '__main__':` guards. For development, `num_workers=0` is safe and fast because images are loaded lazily on demand. For faster GPU training, set `num_workers=2` or `4` inside an `if __name__ == '__main__':` block.

### Q4: How do I access individual patient IDs for auditing or error analysis?
`batch["patient_id"]` is a Python list of strings of length `B`. You can print or log patient IDs alongside model predictions:
```python
for patient_id, pred, label in zip(batch["patient_id"], predictions, batch["label"]):
    if pred != label:
        print(f"Error on Patient {patient_id}: Predicted {pred.item():.2f}, True {label.item()}")
```
