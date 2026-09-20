# FedMed — Multimodal Federated Learning for Pneumonia Detection

FedMed is a privacy-preserving multimodal federated learning platform designed to detect pneumonia across distributed hospital institutions without centralizing sensitive patient radiographs or records.

---

## 🎯 What We Are Doing for the 30% Milestone

Our primary goal for the 30% milestone is to deliver a **functional, end-to-end multimodal federated learning pipeline** evaluated alongside a centralized pooled-data benchmark:

1. **Multimodal Data Pipeline**:
   - Curate and harmonize Chest X-ray images ($224 \times 224$ RGB) paired with clinical tabular features (Age $z$-score, Sex binary).
   - Partition data across 3 simulated hospital clients ($C_1, C_2, C_3$) with **strict patient-level separation** (zero patient overlap across clients, validation, and test sets).

2. **Multimodal Model Architecture**:
   - **Vision Backbone**: Vision Transformer (ViT) processing non-overlapping image patches into localized radiographic tokens.
   - **Clinical Backbone**: Multi-Layer Perceptron (MLP) encoding patient demographics into dense clinical embeddings.
   - **Fusion & Classification Head**: Combines visual and clinical representations to output unnormalized binary logits (Pneumonia vs. Non-Pneumonia).

3. **Federated Learning (FedAvg) Simulation**:
   - 3 simulated hospital clients training locally on their private data partitions.
   - Central server executing **Federated Averaging (FedAvg)** to aggregate client model weights over multiple communication rounds.

4. **Clinical Evaluation & Benchmark Comparison**:
   - Train a centralized pooled-data baseline with the identical model architecture.
   - Evaluate both models on a held-out test set using clinically meaningful metrics: **Sensitivity (Recall)**, **Specificity**, **ROC-AUC**, and **F1-Score**.
   - Compare convergence curves (Federated communication rounds vs. Centralized training epochs).

---

## 📊 Current Progress & Status

| Phase | Milestone Task | Assigned Role | Status | Deliverables / Notes |
|---|---|---|---|---|
| **Phase 1** | **Multimodal Data Pipeline & QA** | **Member 1 (Data Lead)** | **COMPLETED** | • **118,654** clean, model-ready records (NIH & CheXpert).<br>• **36,324** unique patients with zero data leakage.<br>• 3 hospital client DataLoaders + Val + Test loaders ready.<br>• Verified with automated integration test suite.<br>• Integration guide created in `data/README.md`. |
| **Phase 2** | **Vision Encoder & Multimodal Architecture** | **Member 2 (Vision Lead)** | **IN PROGRESS** | • Constructing ViT image encoder and patch embedding.<br>• Designing multimodal fusion layer.<br>• Assembling `MultimodalNet` binary classification model. |
| **Phase 3** | **Clinical Encoder & Centralized Baseline** | **Member 4 (Clinical & Training Lead)** | **IN PROGRESS** | • Implementing tabular `ClinicalEncoder` MLP.<br>• Writing local client training loop with weighted BCE loss (`pos_weight = 18.60`).<br>• Setting up centralized pooled-data training benchmark. |
| **Phase 4** | **Federated Simulation & FedAvg Server** | **Member 3 (FL Lead)** | **IN PROGRESS** | • Building FedAvg server weight aggregation.<br>• Orchestrating multi-round communication across 3 clients.<br>• Logging federated convergence curves. |
| **Phase 5** | **Benchmark Evaluation & Presentation** | **Team** | **UPCOMING** | • Evaluating Sensitivity, Specificity, ROC-AUC, and F1.<br>• Generating comparative performance plots (FL vs Centralized).<br>• 30% milestone review presentation. |

---

## 🚀 For Teammates: Building Your Feature

All data loaders, tensor specifications, and copy-pasteable code examples for Member 2, Member 3, and Member 4 are documented in:
👉 **[`data/README.md`](data/README.md)**
