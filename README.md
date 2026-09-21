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

## 🚀 For Teammates & AI Agents: Feature Development Workflow

> [!IMPORTANT]
> ### ⚠️ Golden Rule: Ground in Real Code — Never Build on Assumptions
> When implementing a feature (whether you are a human developer or an AI coding agent):
> 1. **Do NOT simply read `main_project/context.md` and start building with assumptions or mock interfaces.**
> 2. `main_project/context.md` outlines the **high-level project vision, domain background, and architecture**, but the **ground truth interfaces** are defined by the actual files already built by your teammates.
> 3. **Always inspect the available upstream code and contracts** (e.g., [data/multimodal_dataset.py](file:///r:/VSCODE/Main_Project/data/multimodal_dataset.py), [data/README.md](file:///r:/VSCODE/Main_Project/data/README.md), [CREATED_FILES_TRACKER.md](file:///r:/VSCODE/Main_Project/CREATED_FILES_TRACKER.md)) before writing code.
> 4. **Stay strictly within your assigned team member scope.** Do not rewrite or recreate components owned by other team members; import and consume their real deliverables.

---

### 👥 Team Member Roles & Task Ownership

Refer to the dedicated task briefs in [`team_tasks_30pct/`](file:///r:/VSCODE/Main_Project/team_tasks_30pct/) for granular deliverables:

| Role | Focus Area | Task Specification File | Key Deliverables & Interfaces |
|---|---|---|---|
| **Member 1 (Data Lead)** | Data Curation & Loaders | [`member_1_data_engineering.md`](file:///r:/VSCODE/Main_Project/team_tasks_30pct/member_1_data_engineering.md) | • [data/multimodal_dataset.py](file:///r:/VSCODE/Main_Project/data/multimodal_dataset.py)<br>• 3 Client DataLoaders + Val + Test Loaders<br>• [data/README.md](file:///r:/VSCODE/Main_Project/data/README.md) (Contract specs) |
| **Member 2 (Vision Lead)** | Vision Backbone & Fusion | [`member_2_multimodal_model.md`](file:///r:/VSCODE/Main_Project/team_tasks_30pct/member_2_multimodal_model.md) | • ViT Image Encoder (`models/vision_encoder.py`)<br>• Multimodal Fusion Layer<br>• `MultimodalNet` binary classification model |
| **Member 3 (FL Lead)** | Federated Pipeline & Server | [`member_3_federated_pipeline.md`](file:///r:/VSCODE/Main_Project/team_tasks_30pct/member_3_federated_pipeline.md) | • FedAvg Aggregation Server (`federated/server.py`)<br>• Multi-round client coordination loop<br>• Federated metrics logging & convergence curves |
| **Member 4 (Clinical & Benchmark Lead)** | Clinical MLP & Central Baseline | [`member_4_benchmarks_eval.md`](file:///r:/VSCODE/Main_Project/team_tasks_30pct/member_4_benchmarks_eval.md) | • Tabular `ClinicalEncoder` MLP (`models/clinical_encoder.py`)<br>• Local client training loop (`pos_weight = 18.60`)<br>• Centralized pooled-data baseline & eval suite |

---

### 🛠️ Step-by-Step Workflow: How to Build Your Feature

Whenever starting a new feature or handing off a task to an AI agent, follow this 5-step process:

1. **Check Your Specific Task Brief**:
   - Open your role's markdown file in [`team_tasks_30pct/`](file:///r:/VSCODE/Main_Project/team_tasks_30pct/) to identify your explicit deliverables, boundaries, and expected interfaces.
2. **Review High-Level Context**:
   - Consult [`main_project/context.md`](file:///r:/VSCODE/Main_Project/main_project/context.md) to understand overall system objectives, clinical constraints, and 30% milestone boundaries.
3. **Inspect Real Upstream Files & Contracts**:
   - Do **not** invent dummy data loaders or hallucinate tensor shapes.
   - Inspect the actual files created by other members (e.g., read [data/README.md](file:///r:/VSCODE/Main_Project/data/README.md) and [data/multimodal_dataset.py](file:///r:/VSCODE/Main_Project/data/multimodal_dataset.py)).
   - Verify batch keys and tensor shapes:
     ```python
     batch["image"]       # torch.Tensor [B, 3, 224, 224], torch.float32
     batch["clinical"]    # torch.Tensor [B, 2], torch.float32 -> [age_zscore, sex_binary]
     batch["label"]       # torch.Tensor [B, 1], torch.float32 -> 0.0 (Normal) or 1.0 (Pneumonia)
     batch["patient_id"]  # list of strings (length B)
     batch["source"]      # list of strings (length B) -> 'nih' or 'chexpert'
     ```
4. **Implement Strictly Within Your Scope**:
   - Import existing components directly (e.g., `from data.multimodal_dataset import get_client_loader, get_eval_loaders`).
   - Never write redundant placeholder scripts that bypass or duplicate existing work.
5. **Verify Against the Real Pipeline**:
   - Test your code against the actual loaders and outputs. Run verification scripts (such as [scripts/verify_pipeline_integration.py](file:///r:/VSCODE/Main_Project/scripts/verify_pipeline_integration.py)) to ensure seamless cross-member integration.

---

### 🤖 Prompt Blueprint for AI Agents

When instructing an AI coding assistant (like Antigravity) to implement a feature, use this prompt structure to prevent assumptions and enforce team grounding:

```markdown
You are working on the FedMed project as [MEMBER ROLE, e.g., Member 2: Vision & Multimodal Architecture Lead].

Please follow these strict guidelines:
1. READ CONTEXT: Refer to `main_project/context.md` for high-level project goals and architecture.
2. SCOPE TO YOUR ROLE: Read your assigned task document `team_tasks_30pct/member_X_*.md`. Follow ONLY this member's assigned scope. Do NOT touch or recreate other members' components.
3. GROUND IN REAL CODE (NO ASSUMPTIONS): Do not guess or assume data formats, shapes, or mock loaders. Inspect the real existing files created by teammates:
   - Read `data/README.md` and `data/multimodal_dataset.py` for exact data contracts and loader functions.
   - Check `CREATED_FILES_TRACKER.md` to see all active project files.
4. INTEGRATE DIRECTLY: Build your feature using the real classes, functions, and batch structures already implemented.
```

