# FedMed — 30% Milestone Team Execution Plan & Architecture

Welcome team! This folder contains the modular task assignments for delivering our **30% Milestone**: a working multimodal federated learning pipeline for pneumonia detection.

---

## 1. 30% Milestone Objective & Core Boundary Decisions

Our agreed 30% goal is:
> **Demonstrate a functional federated learning pipeline where 3 simulated hospital clients train multimodal models locally on their data partitions and contribute model updates to a central aggregation server using Federated Averaging (FedAvg), evaluated alongside a centralized pooled-data baseline.**

### Key Decisions Confirmed:
1. **Non-IID Partitioning on Hold for 30%**:
   To ensure the core federated training loop, model serialization, and FedAvg aggregation mechanics are completely verified and reproducible, complex non-IID statistical heterogeneity (Dirichlet/skew) is halted for 30%. We will use **uniform/IID partitions across 3 simulated clients** ($C_1, C_2, C_3$). Non-IID skew will be introduced immediately after this milestone baseline is proven.
2. **Dataset Strategy**:
   - Primary: **NIH ChestX-ray14** and **CheXpert**.
   - Contingency: **MIMIC-CXR** (if credentialed PhysioNet access is approved, it will be added or become the primary dataset).
3. **Multimodal Modality**:
   Chest X-ray image + paired clinical tabular features (e.g. age, gender, view position, vitals) mapped to binary classification:
   - `Class 1`: **Pneumonia**
   - `Class 0`: **Normal** / **No Finding**

---

## 2. Team Member Work Distribution & Ownership

The project is divided across our 4 members with zero overlapping confusion and clean interface contracts:

```text
+-----------------------------------------------------------------------------------+
|                                 FEDMED 30% WORKFLOW                               |
+-----------------------------------------------------------------------------------+

  [MEMBER 1: Data Lead]
    ├── Data Acquisition (NIH & CheXpert)
    ├── Run 8 Data Quality Checks (data_validation_suite/)
    ├── Clean & Preprocess (Filter non-target pathologies, fix outliers)
    ├── Disjoint Patient Split (Zero data leakage)
    └── Multimodal PyTorch Dataset & DataLoader
             │
             ├── Passes Batch: {image_tensor, tabular_tensor, label}
             │
             ▼
  [MEMBER 2: Vision & Multimodal Architecture Lead]
    ├── Image Encoder (CNN backbone: ResNet-18 or Custom CNN)
    ├── Multimodal Fusion Layer (Concatenation + Projection)
    ├── Binary Classification Head (Linear logit output: Pneumonia vs Normal)
    └── Model Parameter Serialization (state_dict get/set)
             │
             ├── Passes Image Encoder, Fusion & Classification Head
             │
             ▼
  [MEMBER 3: Federated Learning Lead]
    ├── Federated Simulation Setup (Flower flwr or Native PyTorch)
    ├── 3 Simulated Hospital Clients (Local data partition C1, C2, C3)
    ├── FedAvg Aggregation Server (Weighted parameter averaging)
    ├── Multi-Round Communication Loop (Global broadcast -> Local train -> Aggregate)
    └── Federated Metrics Logging (Round loss, Round accuracy)
             │
             ├── Passes FL Run History & Global Weights
             │
             ▼
  [MEMBER 4: Clinical Modeling, Client Training & Evaluation Lead]
    ├── Clinical Encoder (MLP: Tabular features -> clinical embedding vector)
    ├── Local Client Training Loop (Epochs, Optimizer, BCEWithLogitsLoss + pos_weight)
    ├── Centralized Pooled-Data Training Baseline (Identical architecture)
    ├── Comprehensive Diagnostic Metrics (Sensitivity, Specificity, F1, ROC-AUC)
    ├── Comparative Convergence Curves (FL Rounds vs Centralized Epochs)
    └── 30% Milestone Deliverable Presentation Tables & Visualizations
```

---

## 3. Individual Member File Directory

Each team member has a dedicated markdown document detailing their exact inputs, outputs, implementation guidelines, code skeletons, and acceptance checklist:

1. **Member 1 (Data Engineering & QA Lead)**: [`member_1_data_engineering.md`](file:///r:/VSCODE/Main_Project/team_tasks_30pct/member_1_data_engineering.md)
2. **Member 2 (Vision & Multimodal Architecture Lead)**: [`member_2_multimodal_model.md`](file:///r:/VSCODE/Main_Project/team_tasks_30pct/member_2_multimodal_model.md)
3. **Member 3 (Federated Learning Lead)**: [`member_3_federated_pipeline.md`](file:///r:/VSCODE/Main_Project/team_tasks_30pct/member_3_federated_pipeline.md)
4. **Member 4 (Clinical Modeling, Client Training & Evaluation Lead)**: [`member_4_benchmarks_eval.md`](file:///r:/VSCODE/Main_Project/team_tasks_30pct/member_4_benchmarks_eval.md)

---

## 4. Shared Interface Contracts (Critical!)

To ensure smooth integration, all members must adhere to these standard signatures:

### Multimodal Batch Format (Member 1 $\to$ Member 2 & 4):
```python
batch = {
    "image": torch.Tensor,     # Shape: [B, 1, 224, 224] or [B, 3, 224, 224] (float32 normalized)
    "clinical": torch.Tensor,  # Shape: [B, num_clinical_features] (float32 standardized)
    "label": torch.Tensor,     # Shape: [B, 1] (float32 binary: 0.0 or 1.0)
    "patient_id": list         # List of strings (for audit tracking)
}
```

### Model Forward Signature (Member 2 & 4 $\to$ Member 3):
```python
logits = model(image_tensor, clinical_tensor)  # Shape: [B, 1]
loss = criterion(logits, label_tensor)
```

### Client Parameter Transfer (Model $\to$ Member 3):
```python
# Extract weights:
weights = model.state_dict()
# Load aggregated global weights:
model.load_state_dict(global_weights)
```

---

## 5. Timeline & Integration Checklist

| Step | Responsible | Target Deliverable | Dependency |
|---|---|---|---|
| **Phase 1** | Member 1 | Clean CSVs, Data Loaders, Validation Report | Datasets downloaded |
| **Phase 2** | Member 2 & Member 4 | Member 2: Image Encoder + Fusion + Head; Member 4: Clinical Encoder + Local Trainer | Synthetic or Member 1 dummy batch |
| **Phase 3** | Member 4 | Centralized baseline training working on pooled data | Member 1 loaders + Member 2 & 4 model |
| **Phase 4** | Member 3 | 3-Client FedAvg simulation loop running 5 rounds | Member 1 partitions + Member 4 training loop |
| **Phase 5** | Member 4 | Comparative plots (FedAvg vs Centralized) & report | Member 3 FL logs + Member 4 baseline logs |
