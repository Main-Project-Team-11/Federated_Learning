# FedMed — Multimodal Federated Learning for Pneumonia Detection

FedMed is a decentralized, privacy-preserving multimodal federated learning platform designed to detect pneumonia across distributed hospital institutions without sharing sensitive patient data.

---

## 📌 30% Milestone Status: Data Pipeline Ready

Member 1 has completed the **Multimodal Data Pipeline** delivering clean, patient-disjoint, model-ready data partitions and PyTorch DataLoaders for the entire team.

- **Total Clean Records:** **118,654**
- **Unique Patients:** **36,324**
- **Zero Patient Overlap:** Guaranteed mathematically across all partitions.
- **Simulated Hospital Clients:** 3 isolated partitions for federated training.
- **Modality:** Chest X-ray images ($224 \times 224$ RGB) paired with clinical tabular features (Age $z$-score, Sex binary).

---

## 🚀 For Teammates: How to Build Your Feature

Detailed step-by-step integration guides with runnable code examples are available in:
👉 **[`data/README.md`](data/README.md)**

### Quick Reference by Role:

1. **Member 2 (Vision & Multimodal Architecture Lead)**:
   - **Feature:** Build `models/multimodal_net.py` (Vision Transformer / ViT Image Encoder + Fusion Classifier).
   - **Your Input:** `from data.multimodal_dataset import get_client_loader`
   - **Image Tensor:** `batch["image"]` $\to$ shape `[B, 3, 224, 224]`, float32, ImageNet normalized.
   - **Guide & Code Example:** See [data/README.md#3-feature-guide-for-member-2-vision--multimodal-architecture-lead](data/README.md#3-feature-guide-for-member-2-vision--multimodal-architecture-lead)

2. **Member 4 (Clinical Modeling & Centralized Baseline Lead)**:
   - **Feature:** Build `models/clinical_encoder.py`, `training/client_trainer.py`, and `train_centralized.py`.
   - **Your Input:** `from data.multimodal_dataset import get_client_loader, get_val_loader, get_test_loader`
   - **Clinical Tensor:** `batch["clinical"]` $\to$ shape `[B, 2]`, float32 (Age $z$-score, Sex binary).
   - **Loss Function:** Use `pos_weight = torch.tensor([18.60])` in `nn.BCEWithLogitsLoss` due to 18.60:1 class imbalance.
   - **Guide & Code Example:** See [data/README.md#4-feature-guide-for-member-4-clinical-modeling--centralized-baseline-lead](data/README.md#4-feature-guide-for-member-4-clinical-modeling--centralized-baseline-lead)

3. **Member 3 (Federated Learning Simulation Lead)**:
   - **Feature:** Build `federated/server.py` (FedAvg) and `federated/run_simulation.py` (3-client loop).
   - **Your Input:** `get_client_loader(1)`, `get_client_loader(2)`, `get_client_loader(3)`, and `get_val_loader()`.
   - **Sample Counts for Weighting:** Client 1: 27,593 | Client 2: 27,878 | Client 3: 28,225 (Total: 83,696).
   - **Guide & Code Example:** See [data/README.md#5-feature-guide-for-member-3-federated-learning-simulation-lead](data/README.md#5-feature-guide-for-member-3-federated-learning-simulation-lead)

---

## 📂 Repository Structure

```text
Main_Project/
├── data/
│   ├── multimodal_dataset.py           # PyTorch Dataset and client/val/test DataLoaders
│   └── README.md                       # Comprehensive team integration guide
│
├── data_prep/                          # Clean Parquet/CSV partitions & distribution statistics
│   ├── client_1.parquet / .csv         # Hospital Client 1 partition
│   ├── client_2.parquet / .csv         # Hospital Client 2 partition
│   ├── client_3.parquet / .csv         # Hospital Client 3 partition
│   ├── train.parquet / .csv            # Centralized pooled training partition
│   ├── val.parquet / .csv              # Held-out validation partition
│   ├── test.parquet / .csv             # Held-out test partition
│   ├── stage5_dataset_statistics.json  # Imbalance & demographic metrics
│   └── stage5_dataset_statistics.md    # Summary report of statistics
│
├── scripts/                            # 7-stage data preparation & verification scripts
│   ├── stage1_prepare_nih.py           # Stage 1: NIH metadata extraction
│   ├── stage1_prepare_chexpert.py      # Stage 1: CheXpert metadata extraction
│   ├── stage2_canonical_schema.py      # Stage 2: Schema harmonization
│   ├── stage3_binary_labels.py         # Stage 3: Binary label & view filtering
│   ├── stage4_data_validation_and_cleaning.py # Stage 4: 8 data quality checks
│   ├── stage5_dataset_statistics.py    # Stage 5: Imbalance & statistics computation
│   ├── stage6_patient_level_split.py   # Stage 6: Disjoint patient-level train/val/test split
│   ├── stage7_client_partitions.py     # Stage 7: Disjoint 3-client hospital partitioning
│   └── verify_pipeline_integration.py  # Automated integration test suite
│
├── team_tasks_30pct/                   # Role specifications for 30% milestone
│   ├── README_30PCT_OVERVIEW.md        # Team execution plan & boundaries
│   ├── member_1_data_engineering.md    # Member 1 task documentation
│   ├── member_2_multimodal_model.md    # Member 2 task documentation
│   ├── member_3_federated_pipeline.md  # Member 3 task documentation
│   ├── member_4_benchmarks_eval.md     # Member 4 task documentation
│   └── GITHUB_PUSH_GUIDE.md            # Git branching and pull request guide
│
└── CREATED_FILES_TRACKER.md            # Full tracker of created files & git staging instructions
```

---

## ⚙️ Quick Verification

To verify all 5 DataLoaders on your machine:

```powershell
python scripts/verify_pipeline_integration.py
```
