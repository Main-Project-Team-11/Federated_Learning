# FedMed — Project Context

## 1. Project Identity

- **Project name:** FedMed
- **Project title:** Privacy-Preserving Federated Learning for Pneumonia Detection
- **Project type:** Final-year academic project developed by a team of four students.
- **Current milestone:** 30% implementation/progress presentation.
- **Domain:** Healthcare AI, multimodal deep learning, Chest X-ray classification, Federated Learning, privacy-preserving ML, and explainable AI.

## 2. Current Project Goal

The immediate goal is to demonstrate a working end-to-end **federated-learning pipeline for pneumonia detection using multimodal data**.

The proposed model will combine:

1. Chest X-ray images
2. Available and correctly paired clinical or tabular information

The output classes are:

- Pneumonia
- Normal

At the 30% milestone, the priority is to establish a stable baseline. Advanced privacy, deployment, monitoring, and explainability features will be developed later.

The 30% goal is:

> Demonstrate a working federated-learning pipeline in which three to five simulated hospital clients train locally on different data partitions and contribute model updates to a central server using Federated Averaging.

---

## 3. Problem Statement

Hospitals possess sensitive information such as Chest X-rays, demographic details, laboratory values, vital signs, diagnoses, and other clinical observations. Due to privacy requirements, institutional policies, and data-sharing restrictions, hospitals may not be able to share raw patient data.

A centralized machine-learning approach requires data to be pooled in one location, creating privacy and governance concerns. In addition, a Chest X-ray-only model cannot use potentially informative clinical information.

FedMed investigates a multimodal federated-learning approach in which:

- Chest X-rays and clinical information remain within each client’s local data partition.
- Each client trains a local multimodal model.
- Clients send model parameters or updates rather than raw patient records.
- The central server aggregates updates using Federated Averaging.
- The updated global model is sent back to the clients.

The system is an academic research prototype and is not intended to replace doctors or provide autonomous clinical diagnoses.

---

## 4. Project Objectives

1. Combine Chest X-ray images with available clinical/tabular information.
2. Build a multimodal model for Pneumonia-versus-Normal classification.
3. Divide the data into three to five simulated hospital clients.
4. Create Non-IID client partitions to represent heterogeneous hospital data.
5. Implement local training at each client.
6. Implement a central federated server.
7. Use FedAvg as the initial aggregation algorithm.
8. Compare federated training with a centralized pooled-data baseline.
9. Evaluate accuracy, loss, convergence, and other suitable metrics.
10. Add Differential Privacy in a later stage.
11. Consider Secure Aggregation as a later-stage/stretch-goal feature.
12. Add local inference, monitoring, model management, and Grad-CAM explainability in later stages.

---

## 5. 30% Implementation Scope

The 30% implementation is focused on:

- Public dataset investigation and preparation
- Image and tabular-data preprocessing
- Multimodal model construction
- Creation of simulated hospital clients
- Non-IID data partitioning
- Local client training
- Federated server setup
- FedAvg aggregation
- Multiple federated communication rounds
- Centralized pooled-data baseline
- Basic evaluation and result visualization

The 30% implementation does **not** require all final-system features.

The priority is to produce a functional and demonstrable baseline.

---

## 6. Dataset Direction

The 30% architecture currently identifies the following public dataset candidates:

- **NIH ChestX-ray14**
- **CheXpert**

The final dataset choice must depend on:

- Accessibility
- Availability of Pneumonia and Normal labels
- Availability of clinical or tabular information
- Possibility of correctly pairing images and clinical information
- Licensing and data-use conditions
- Storage and computational requirements

The team must verify whether the selected dataset actually contains suitable clinical/tabular information. If a dataset only contains images and labels, it must not be described as a complete multimodal dataset without a valid additional source of paired clinical data.

The data must not be randomly merged across patients.

A preferred multimodal sample is:

```text
Chest X-ray image
        +
Corresponding clinical/tabular features
        +
Pneumonia/Normal label
```

Possible fields include:

```text
sample_id
image_path
patient_id or valid linkage key
age
sex
other_available_clinical_features
label
```

The team must check:

- Image validity
- Patient and study identifiers
- Correct image–clinical pairing
- Label correctness
- Duplicate records
- Missing values
- Class balance
- Patient-level data leakage

---

## 7. High-Level 30% Architecture

```text
Public Dataset
(NIH ChestX-ray14 / CheXpert)
          |
          v
Dataset Preparation
          |
          v
Split into Non-IID Partitions
Across 3–5 Simulated Hospitals
          |
          +--------------------------------------------+
          |                                            |
          v                                            v
  Hospital Client 1                             Hospital Client 2 ... N
          |                                            |
      Local Data                                   Local Data
     /          \                                 /          \
  Images      Clinical                         Images      Clinical
    |            |                               |            |
    v            v                               v            v
  Image       Clinical                         Image       Clinical
 Encoder      Pathway                         Encoder      Pathway
  (ViT)     (Perceiver)                        (ViT)     (Perceiver)
    \            /                               \            /
     \          /                                 \          /
   Multimodal Fusion                            Multimodal Fusion
   (Transformer Enc)                            (Transformer Enc)
          |                                            |
      Classifier                                   Classifier
        (MLP)                                        (MLP)
          |                                            |
   Pneumonia/Normal                             Pneumonia/Normal
          |                                            |
    Local Training                               Local Training
          |                                            |
          +-------------- Model Updates ---------------+
                                 |
                                 v
                          Federated Server
                              (FedAvg)
                                 |
                         Aggregate Updates
                                 |
                            Global Model
                                 |
                      Broadcast Global Model
                         Back to Clients
```

The 30% architecture is focused on training and evaluation. The complete client and server interfaces are planned for later stages.

---

## 8. Client-Side Deliverables

### 8.1 Client-Side Role

Each simulated hospital client represents a participating healthcare institution. It maintains a local dataset partition and performs local model training.

At 30%, each client should contain:

- A local data partition
- A local multimodal model
- A local training loop
- A local validation process
- A client identifier
- A mechanism for sending model updates
- A mechanism for receiving the global model

Each client will train for a few local epochs per federated round. The exact number of epochs will be finalized during implementation.

### 8.2 Local Multimodal Model

In the selected base paper by Khader et al. (2023), the **image encoder is a Vision Transformer (ViT)**. It is responsible for processing chest X-ray images and converting them into numerical representations that can be used by the multimodal model.

The Vision Transformer divides each chest X-ray into small, non-overlapping patches. Each patch is converted into an image token, which represents the visual information contained in that particular region. Positional information is also added to preserve the spatial arrangement of the patches.

These image tokens are then processed by the Transformer layers to learn meaningful visual representations from the chest X-ray. The resulting image representations are passed to the multimodal fusion stage, where they are combined with the clinical information.

For the Grad-CAM-based explainability analysis, the image-token representations generated by the Vision Transformer can be used to identify which regions of the chest X-ray contributed to a particular disease prediction.

## Main Components of the Model

| Component | Method Used | Function |
|---|---|---|
| Image encoder | Vision Transformer (ViT) | Processes chest X-ray images and generates image tokens |
| Clinical-data pathway | Perceiver-inspired cross-attention mechanism | Processes clinical parameters using learnable latent tokens |
| Multimodal fusion | Transformer encoder | Combines image and clinical representations |
| Classification layer | Multi-layer perceptron (MLP) | Produces the final disease predictions |

> [!NOTE]
> **Parallel Dual-Pathway Processing:** The Image Encoder (ViT) and Clinical-Data Pathway (Perceiver-inspired cross-attention) process raw images and clinical tabular data **in parallel**, not sequentially. Their independently extracted visual and clinical latent representations are then fed concurrently into the Multimodal Fusion module (Transformer encoder) to be combined.

### 8.3 30% Client Backend

- Load local data
- Preprocess images and clinical features
- Load the global model
- Train locally for a few epochs
- Evaluate on local validation data
- Produce model parameters or updates
- Send updates to the server
- Receive the updated global model

### 8.4 Later Client Interface

The following are planned for later and are **not required at 30%**:

- Prediction screen for uploading a Chest X-ray
- Clinical-parameter input form
- Pneumonia/Normal result
- Confidence score
- Confidence-based specialist-review flag
- Local training-status panel
- Global model version display
- Synchronization status
- Local inference API/deployment

---

## 9. Server-Side Deliverables

### 9.1 Server Role

The central server coordinates federated training. It should not receive or store raw patient images or raw clinical records in the intended workflow.

### 9.2 30% Server Backend

The server should:

1. Register or identify participating clients.
2. Select participating clients for a round.
3. Send the current global model to selected clients.
4. Receive local model updates.
5. Aggregate updates using FedAvg.
6. Update the global model.
7. Record round-wise performance.
8. Broadcast the updated global model.
9. Continue to the next round.

### 9.3 Federated Round Workflow

```text
1. Initialize or load the global model.
2. Select participating clients.
3. Send the global model to clients.
4. Clients train locally for a few epochs.
5. Clients evaluate their local models.
6. Clients send model updates.
7. Server aggregates updates using FedAvg.
8. Server updates the global model.
9. Server records round-wise results.
10. Server broadcasts the updated global model.
11. The next round begins.
```

### 9.4 Later Server Interface

The following are planned for later and are **not required at 30%**:

- Streamlit monitoring dashboard
- Global accuracy trend chart
- Per-hospital accuracy trend chart
- Current round and connected-client display
- Round-status display
- Model registry
- Model checkpoint table
- Model restoration/rollback
- Start, pause, and resume controls
- Full database layer

For the 30% milestone, simple files or logs may be used to record round results.

---

## 10. Centralized Baseline

The architecture includes a **Centralized Baseline using pooled data**.

This is a separate reference experiment and is not part of the federated-learning workflow.

```text
Hospital 1 data ─┐
Hospital 2 data ─┼──> Pooled Dataset ───> Centralized Model
Hospital 3 data ─┘
```

The centralized baseline is used to investigate:

- How the model performs when all training data are pooled
- How federated training compares with centralized training
- Whether the federated model approaches the centralized model’s performance

The centralized baseline is not privacy-preserving because the data are pooled.

The same or equivalent multimodal model should be used for a fair comparison.

---

## 11. Non-IID Client Partitions

The training data will be divided into three to five simulated hospital clients with potentially different distributions.

For example:

```text
Hospital A: Mostly Pneumonia cases
Hospital B: Mostly Normal cases
Hospital C: More balanced distribution
```

The exact distribution will depend on the selected dataset and experimental design.

The purpose is to investigate the effect of heterogeneous client data on federated training.

At 30%, the team will use FedAvg as the initial aggregation method. The project must not claim that FedAvg completely solves the Non-IID problem.

FedProx may be evaluated later to investigate whether it improves stability under heterogeneous data.

---

## 12. Required 30% Results

The architecture identifies the following expected results.

### 12.1 Training and Validation Accuracy Curves

Generate training and validation accuracy curves for the relevant experiments.

Where possible, compare:

- FedAvg-based federated training
- Centralized baseline

### 12.2 Training Loss Curves

Generate training-loss curves for:

- Centralized training
- Federated training
- Client-level training, if useful

### 12.3 Convergence Analysis

Analyze performance across communication rounds, including:

- Whether global performance improves
- Whether training is stable
- Whether performance plateaus
- Whether client differences affect convergence

### 12.4 Final Performance Comparison

Compare centralized and federated models using actual measured values.

Possible metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- AUC, if feasible
- Loss

Also examine:

- Per-client performance
- Class imbalance
- Client-to-client differences
- Training stability
- Performance across rounds

The team must report actual results rather than predicted values.

---

## 13. Features Planned Later — Not Implemented at 30%

The architecture explicitly places the following features in the later-stage section:

1. **Differential Privacy using Opacus**
   - Gradient clipping
   - Noise addition
   - Privacy–utility analysis

2. **Secure Aggregation**
   - Protecting individual client updates from direct inspection by the server
   - Currently a later-stage or stretch-goal feature

3. **Local Inference API / Deployment**
   - Local prediction using the latest model
   - API-based client inference

4. **Monitoring Dashboard using Streamlit**
   - Global and per-client accuracy
   - Round status
   - Connected clients
   - Training progress

5. **Explainability using Grad-CAM**
   - Heatmaps for image regions contributing to the disease prediction, using image-token representations generated by the Vision Transformer (ViT)
   - Not a complete explanation of all multimodal factors

6. **Model Registry and Rollback**
   - Checkpoint records
   - Model versions
   - Round numbers
   - Timestamps
   - Evaluation metrics
   - Restoration of previous versions

These features must not be described as implemented in the 30% presentation unless they have actually been completed and tested.

---

## 14. Technology Direction

The intended technology stack is:

- Python 3.x
- PyTorch
- Torchvision
- timm (for Vision Transformer / ViT architectures)
- Pandas
- NumPy
- Pillow and/or OpenCV
- Flower or another suitable federated-learning framework
- Matplotlib
- Plotly, if required
- Opacus for later Differential Privacy experiments, if compatible
- Streamlit for the later monitoring dashboard
- File-based logs initially
- SQLite or PostgreSQL later, if required
- Visual Studio Code
- Jupyter Notebook
- Google Colab or another cloud-GPU environment, if necessary
- Git and GitHub

The stack may be revised after compatibility testing.

---

## 15. Current Algorithms and Techniques

### Initial 30% Algorithm

- **FedAvg:** Initial federated aggregation algorithm.

### Base Paper Model Architecture (Khader et al., 2023)

- **Vision Transformer (ViT):** Patch-based image encoder generating visual tokens.
- **Perceiver-inspired Cross-Attention:** Clinical-data pathway processing clinical parameters using learnable latent tokens.
- **Transformer Encoder:** Multimodal representation fusion.
- **Multi-Layer Perceptron (MLP):** Classification layer producing final disease predictions.

### Later Techniques

- **FedProx:** Investigating optimization under heterogeneous or Non-IID client data.
- **Differential Privacy:** Reducing information leakage from model updates.
- **Secure Aggregation:** Preventing direct inspection of individual client updates.
- **Grad-CAM:** Explaining influential regions of the image input using ViT image tokens.

At the 30% stage, FedAvg is the primary required federated algorithm.

---

## 16. 30% Deliverables Checklist

### Documentation

- [ ] Project context
- [ ] Requirements document
- [ ] Execution plan
- [ ] Implementation plan
- [ ] Decision log
- [ ] Progress log
- [ ] Architecture documentation

### Dataset

- [ ] NIH ChestX-ray14 and/or CheXpert investigated
- [ ] Pneumonia and Normal labels identified
- [ ] Clinical/tabular-data availability verified
- [ ] Image–clinical pairing checked
- [ ] Data limitations documented
- [ ] Data contract prepared

### Data Processing

- [ ] Image paths validated
- [ ] Invalid records handled
- [ ] Duplicates checked
- [ ] Missing values inspected
- [ ] Image preprocessing implemented
- [ ] Clinical preprocessing implemented where applicable
- [ ] Train/validation/test split created
- [ ] Data leakage checked

### Simulated Clients

- [ ] Three to five clients created
- [ ] Client partitions generated
- [ ] Non-IID strategy documented
- [ ] Client configurations prepared

### Multimodal Model

- [ ] Image encoder implemented (Vision Transformer / ViT)
- [ ] Clinical pathway implemented (Perceiver-inspired cross-attention)
- [ ] Multimodal fusion implemented (Transformer encoder)
- [ ] Binary classifier implemented (MLP)
- [ ] Forward-pass test completed
- [ ] Centralized training test completed

### Federated Learning

- [ ] Federated server created
- [ ] Local client-training loop created
- [ ] Model-update exchange implemented
- [ ] FedAvg implemented
- [ ] Multiple federated rounds completed
- [ ] Global model broadcast implemented

### Evaluation

- [ ] Training accuracy curves generated
- [ ] Validation accuracy curves generated
- [ ] Training-loss curves generated
- [ ] Convergence analyzed
- [ ] Centralized-versus-federated comparison completed
- [ ] Actual metrics documented

### Demonstration

- [ ] Client receives the global model
- [ ] Client trains locally
- [ ] Client sends an update
- [ ] Server aggregates updates
- [ ] Server creates a new global model
- [ ] Client receives the updated model
- [ ] Preliminary local prediction test completed if feasible

---

## 17. Explicitly Excluded from the 30% Milestone

The following are not required for the 30% demonstration:

- Differential Privacy implementation
- Secure Aggregation implementation
- Full client-side graphical interface
- Full server-side monitoring dashboard
- Local inference API deployment
- Grad-CAM integration
- Model registry and rollback
- REST API production deployment
- Docker deployment
- True multi-machine hospital deployment
- Real hospital deployment
- Clinical validation
- Regulatory approval
- Production-grade security
- Extensive algorithm comparisons

---

## 18. Technical Principles

1. Do not claim that a feature is implemented until it has been developed and tested.
2. Clearly distinguish implemented, planned, proposed, and stretch-goal features.
3. Do not randomly combine images and clinical information.
4. Do not assume that every sample contains every clinical feature.
5. Do not claim that FedAvg completely solves Non-IID data.
6. Do not claim that Federated Learning alone guarantees complete privacy.
7. Do not describe the centralized baseline as privacy-preserving.
8. Do not present Grad-CAM as a complete explanation of the multimodal decision.
9. Do not present confidence scores as guaranteed diagnostic correctness.
10. Do not claim clinical validation without appropriate clinical evaluation.
11. Follow all applicable dataset access and data-use conditions.
12. Do not expand the scope before the basic multimodal FedAvg pipeline works.
13. Major changes must be discussed by the team and recorded in `decisions.md`.

---

## 19. Current Confirmed Decisions

- Project name: FedMed
- Current milestone: 30% implementation
- Main task: Multimodal Pneumonia-versus-Normal detection
- Image modality: Chest X-ray
- Clinical modality: Available and correctly paired clinical/tabular data
- Current public dataset direction: NIH ChestX-ray14 and/or CheXpert
- Federated setting: Three to five simulated hospital clients
- Client distributions: Non-IID partitions
- Initial aggregation method: FedAvg
- Centralized pooled-data baseline: Required
- Base paper model (Khader et al., 2023): ViT image encoder, Perceiver-inspired cross-attention clinical pathway, Transformer encoder fusion, MLP classifier
- Local training: A few epochs per client per round
- Required 30% results: Accuracy curves, loss curves, convergence analysis, and centralized-versus-federated comparison
- Differential Privacy: Later; not part of the 30% implementation
- Secure Aggregation: Later or stretch goal
- Local inference API: Later
- Streamlit monitoring dashboard: Later
- Grad-CAM: Later
- Model registry and rollback: Later

These decisions must not be changed automatically.

---

## 20. Open Questions

1. Will the final dataset be NIH ChestX-ray14, CheXpert, or another public dataset?
2. Does the selected dataset contain valid clinical/tabular information?
3. If clinical information is absent, what valid paired source will be used?
4. How will Pneumonia and Normal labels be defined?
5. Which clinical features will be included?
6. Which image encoder will be used? — Confirmed: Vision Transformer (ViT) based on Khader et al. (2023).
7. How will missing clinical values be handled?
8. How many clients will be simulated?
9. How will Non-IID partitions be generated?
10. How many local epochs and federated rounds will be used?
11. Which federated-learning framework will be selected?
12. Which metrics will be used for the final comparison?
13. Will a preliminary local inference test be included in the 30% demonstration?

Codex must not silently answer these questions using unsupported assumptions.

---

## 21. Instructions for Coding Agent

Before creating an execution plan or implementation plan:

1. Read this entire file.
2. Inspect the existing project files and source code.
3. Identify which 30% deliverables are already completed.
4. Compare the current implementation with the 30% checklist.
5. Do not assume planned features are already implemented.
6. Do not silently change confirmed decisions.
7. Identify missing information and technical risks.
8. Ask the team for clarification where necessary.
9. Separate confirmed decisions from proposals and assumptions.
10. Prioritize a working multimodal model and FedAvg pipeline.
11. Keep the 30% milestone achievable and demonstrable.
12. Avoid unnecessary technologies and advanced features.
13. Record approved changes in `decisions.md`.
14. Propose changes to confirmed decisions before modifying them.

---

## 22. Definition of 30% Completion

The project may be considered approximately 30% complete when the team has demonstrated:

1. A documented public dataset choice or investigation.
2. A validated image and clinical/tabular data structure.
3. A working image-preprocessing pipeline.
4. A working clinical-data preprocessing pipeline where applicable.
5. Properly paired multimodal samples, or clearly documented pairing limitations.
6. A basic multimodal pneumonia-classification model.
7. A centralized pooled-data baseline.
8. Three to five simulated hospital clients.
9. Non-IID client partitions.
10. Successful local client training.
11. Successful FedAvg aggregation across multiple rounds.
12. Global-model redistribution to clients.
13. Training and validation accuracy curves.
14. Training-loss curves.
15. Convergence analysis.
16. Centralized-versus-federated performance comparison.
17. A preliminary end-to-end federated-learning demonstration.
18. Documentation of limitations and next steps.

The 30% milestone is an approximate project-management target, not a formal measure of software completeness.
