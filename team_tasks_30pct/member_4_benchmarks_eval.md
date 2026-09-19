# Member 4: Clinical Modeling, Client Training & Evaluation Lead

- **Role:** Member 4 — Clinical Tabular Encoder, Local Training Loop, Centralized Baseline & Evaluation
- **Milestone:** 30% Implementation Presentation
- **Primary Focus:** Building the Clinical Encoder (MLP), writing the PyTorch Local Client Training Loop, executing the Centralized Pooled-Data Baseline, computing clinical diagnostic metrics (Sensitivity, Specificity, ROC-AUC), and generating comparative convergence plots for the presentation.

---

## 1. Responsibilities & Objectives

1. **Clinical Tabular Encoder (`ClinicalEncoder`)**:
   - Build a Multi-Layer Perceptron (MLP) taking standardized tabular features (e.g., Age, Sex, View Position) and producing a dense clinical representation vector $h_{\text{clin}} \in \mathbb{R}^{d_{\text{clin}}}$ (e.g., $d_{\text{clin}} = 64$).
   - Hand this encoder to Member 2 for assembly into `MultimodalNet`.
2. **Local Client Training Loop (`train_client_local`)**:
   - Write the self-contained PyTorch local training function executed by each client during federated rounds:
     - Iterates over local batches for $E$ epochs (e.g., $E=2$).
     - Uses `nn.BCEWithLogitsLoss(pos_weight=...)` to handle the class imbalance measured by Member 1.
     - Performs backpropagation with Adam/SGD and returns updated model weights (`state_dict`) and training loss.
   - Write client evaluation routine `evaluate_model()`.
3. **Centralized Pooled-Data Baseline**:
   - Train a centralized model pooling all 3 clients' data into one training set.
   - Use the **exact same assembled `MultimodalNet` architecture** (Member 2's Image Encoder & Fusion + your Clinical Encoder).
   - Train for equivalent total epochs to assess the performance gap between decentralized federated learning and centralized data aggregation.
4. **Clinical Evaluation Metrics Suite**:
   - In medical diagnosis, accuracy is misleading due to high class imbalance. Implement:
     - **Sensitivity / Recall**: $\frac{TP}{TP + FN}$ (prevents missed pneumonia cases).
     - **Specificity**: $\frac{TN}{TN + FP}$ (ensures normal patients aren't falsely alarmed).
     - **Precision**: $\frac{TP}{TP + FP}$.
     - **F1-Score**: Harmonic mean of Precision and Recall.
     - **ROC-AUC**: Receiver Operating Characteristic Area Under Curve.
     - **Confusion Matrix**: TP, FP, TN, FN.
5. **Comparative Visualizations & Presentation Table**:
   - Generate high-resolution plots comparing the Centralized Baseline vs Federated FedAvg (from Member 3's logs):
     - Loss curves (Centralized epochs vs Federated rounds).
     - Validation accuracy curves.
     - Final summary comparison table for presentation slides.

---

## 2. Interface Contract (What you receive & provide)

- **Receives from Member 1**: DataLoaders yielding `batch = {"image": ..., "clinical": ..., "label": ...}`.
- **Receives from Member 2**: `ImageEncoder` and `FusionClassifier` modules to assemble `MultimodalNet`.
- **Receives from Member 3**: `federated_training_log.json` and `best_global_model.pt`.
- **Provides to Member 2**: `ClinicalEncoder` class.
- **Provides to Member 3**: `train_client_local()` training routine for client updates in FL simulation.
- **Produces for Team**:
  - `centralized_training_log.json`
  - Comparative plots: `fedmed_30pct_comparison.png`
  - Final 30% benchmark comparison table.

---

## 3. Step-by-Step Implementation Guide

### Step 1: Implement Clinical Tabular Encoder
Create `models/clinical_encoder.py`:
```python
import torch
import torch.nn as nn

class ClinicalEncoder(nn.Module):
    """Multi-Layer Perceptron encoding tabular clinical features into dense representations."""
    def __init__(self, num_features=3, embed_dim=64, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # Input shape: [B, num_features] -> Output: [B, embed_dim]
        return self.net(x)
```

### Step 2: Implement Local Client Training Loop
Create `training/client_trainer.py`:
```python
import torch
import torch.nn as nn

def train_client_local(model, dataloader, epochs=2, lr=1e-4, pos_weight_val=1.0, device="cpu"):
    """
    Executes local training on a simulated hospital client partition.
    Used by Member 3 in federated simulation rounds.
    """
    model.to(device)
    model.train()
    
    pos_weight = torch.tensor([pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    epoch_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        total_samples = 0
        
        for batch in dataloader:
            images = batch["image"].to(device)
            clinical = batch["clinical"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(images, clinical)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(labels)
            total_samples += len(labels)
            
        avg_loss = running_loss / max(total_samples, 1)
        epoch_losses.append(avg_loss)
        
    return model.state_dict(), epoch_losses


def evaluate_model(model, dataloader, device="cpu"):
    """Evaluates model loss and accuracy on validation/test set."""
    model.to(device)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            clinical = batch["clinical"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(images, clinical)
            loss = criterion(logits, labels)
            
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total_loss += loss.item() * len(labels)
            total += len(labels)
            
    return {
        "val_loss": total_loss / max(total, 1),
        "val_accuracy": correct / max(total, 1)
    }
```

### Step 3: Implement Centralized Baseline Training
Create `experiments/centralized_baseline.py`:
```python
import json
import torch
from training.client_trainer import evaluate_model

def train_centralized_baseline(
    model,
    pooled_train_loader,
    val_loader,
    epochs=10,
    lr=1e-4,
    pos_weight_val=1.0,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print("=" * 60)
    print(f"Training Centralized Pooled Baseline ({len(pooled_train_loader.dataset)} samples)")
    print("=" * 60)
    
    model.to(device)
    pos_weight = torch.tensor([pos_weight_val]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    history = {"epochs": [], "train_loss": [], "val_loss": [], "val_accuracy": []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        
        for batch in pooled_train_loader:
            images = batch["image"].to(device)
            clinical = batch["clinical"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(images, clinical)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * len(labels)
            total += len(labels)
            
        epoch_loss = running_loss / max(total, 1)
        val_metrics = evaluate_model(model, val_loader, device=device)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_accuracy']*100:.2f}%")
        
        history["epochs"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_accuracy"].append(val_metrics["val_accuracy"])
        
    with open("centralized_training_log.json", "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), "best_centralized_model.pt")
    return model, history
```

### Step 4: Implement Clinical Evaluation Metrics
Create `evaluation/metrics.py`:
```python
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

def compute_clinical_metrics(model, test_loader, device="cpu"):
    """Computes comprehensive diagnostic metrics on test set."""
    model.to(device)
    model.eval()
    
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            clinical = batch["clinical"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(images, clinical)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets = labels.cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_targets.extend(targets)
            
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    
    metrics = {
        "accuracy": float(round(accuracy_score(all_targets, all_preds), 4)),
        "sensitivity_recall": float(round(recall_score(all_targets, all_preds, zero_division=0), 4)),
        "specificity": float(round(tn / max(tn + fp, 1), 4)),
        "precision": float(round(precision_score(all_targets, all_preds, zero_division=0), 4)),
        "f1_score": float(round(f1_score(all_targets, all_preds, zero_division=0), 4)),
        "roc_auc": float(round(roc_auc_score(all_targets, all_probs), 4)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
    }
    return metrics, all_targets, all_probs
```

### Step 5: Generate Comparative Convergence Plots
Create `evaluation/visualize.py`:
```python
import json
import matplotlib.pyplot as plt

def plot_comparisons(fl_log_path="federated_training_log.json", cent_log_path="centralized_training_log.json"):
    with open(fl_log_path) as f:
        fl_log = json.load(f)
    with open(cent_log_path) as f:
        cent_log = json.load(f)
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Validation Loss Comparison
    axes[0].plot(fl_log["rounds"], fl_log["global_val_loss"], marker="o", label="Federated Global Model (FedAvg)")
    axes[0].plot(cent_log["epochs"], cent_log["val_loss"], marker="s", linestyle="--", label="Centralized Baseline")
    axes[0].set_title("Validation Loss: FedAvg vs Centralized Baseline")
    axes[0].set_xlabel("Rounds / Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Validation Accuracy Comparison
    axes[1].plot(fl_log["rounds"], [acc * 100 for acc in fl_log["global_val_accuracy"]], marker="o", label="FedAvg Global Acc (%)")
    axes[1].plot(cent_log["epochs"], [acc * 100 for acc in cent_log["val_accuracy"]], marker="s", linestyle="--", label="Centralized Acc (%)")
    axes[1].set_title("Validation Accuracy: FedAvg vs Centralized Baseline")
    axes[1].set_xlabel("Rounds / Epochs")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("fedmed_30pct_comparison.png", dpi=300)
    print("Saved comparison figure: fedmed_30pct_comparison.png")
```

---

## 4. 30% Milestone Presentation Deliverables

Prepare this table template for your presentation slides:

### Target Presentation Comparison Table:

| Metric | Centralized Pooled Baseline | Federated Baseline (FedAvg, 3 Clients) | Performance Delta ($\Delta$) |
|---|---|---|---|
| **Accuracy** | *e.g., 84.5%* | *e.g., 82.8%* | *-1.7%* |
| **Sensitivity (Recall)** | *e.g., 86.2%* | *e.g., 84.1%* | *-2.1%* |
| **Specificity** | *e.g., 83.1%* | *e.g., 81.9%* | *-1.2%* |
| **F1-Score** | *e.g., 0.824* | *e.g., 0.810* | *-0.014* |
| **ROC-AUC** | *e.g., 0.891* | *e.g., 0.875* | *-0.016* |
| **Raw Data Shared** | **100% (Pooled centrally)** | **0% (Privacy-Preserved)** | **Decentralized Success** |

### Deliverables Checklist:
- [ ] `ClinicalEncoder` implemented and delivered to Member 2.
- [ ] `train_client_local()` implemented and delivered to Member 3.
- [ ] `experiments/centralized_baseline.py` executed and saved `centralized_training_log.json`.
- [ ] `evaluation/metrics.py` executed on both models.
- [ ] `evaluation/visualize.py` generates `fedmed_30pct_comparison.png`.
- [ ] Final comparison table filled with actual experiment numbers for presentation slides.
