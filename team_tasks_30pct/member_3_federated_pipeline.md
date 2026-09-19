# Member 3: Federated Learning Simulation & Server Lead

- **Role:** Member 3 — Federated Infrastructure & FedAvg Server
- **Milestone:** 30% Implementation Presentation
- **Primary Focus:** Building the federated simulation environment, implementing the central FedAvg aggregation server, orchestrating multi-round client-server communication across 3 simulated hospital clients, and recording round convergence.

---

## 1. Responsibilities & Objectives

1. **Simulate 3 Hospital Clients**:
   - Instantiate 3 independent client entities representing participating healthcare institutions:
     - `Hospital Client 1` (using Member 1's `client_1` loader)
     - `Hospital Client 2` (using Member 1's `client_2` loader)
     - `Hospital Client 3` (using Member 1's `client_3` loader)
   - Note: Non-IID skew is **halted** for 30%; clients use uniform/IID partitions to verify pipeline stability.
2. **Implement Federated Averaging (FedAvg)**:
   - Implement the core FedAvg aggregation equation:
     $$w^{t+1} = \sum_{k=1}^{K} \frac{n_k}{N} w_k^{t+1}$$
     where $K=3$ clients, $n_k$ is the number of training samples at client $k$, $N = \sum n_k$, and $w_k^{t+1}$ is the parameter state dictionary returned by client $k$.
3. **Multi-Round Federated Communication Loop**:
   - Initialize global model weights $w^0$.
   - For each communication round $t = 1, \dots, R$ (e.g. $R=5$ or $10$ rounds):
     1. **Broadcast**: Send $w^t$ to all 3 clients.
     2. **Local Training**: Clients perform $E$ local epochs (e.g. $E=2$) using Member 2's `train_client_local`.
     3. **Upload**: Clients return updated weights $w_k^{t+1}$ and sample counts $n_k$.
     4. **Aggregate**: Server computes weighted average to produce $w^{t+1}$.
     5. **Global Validation**: Evaluate $w^{t+1}$ on the centralized validation set.
4. **Log Federated History**:
   - Record round numbers, client losses, aggregated global validation loss, and global accuracy into a structured JSON/CSV for Member 4 to visualize.

---

## 2. Interface Contract (What you receive & provide)

- **Receives from Member 1**: 3 client DataLoaders (`loader_c1`, `loader_c2`, `loader_c3`) and 1 global validation DataLoader (`loader_val`).
- **Receives from Member 2**: `MultimodalNet` model class (assembled architecture incorporating Member 2's Vision Transformer / ViT image encoder, fusion module, and Member 4's clinical encoder).
- **Receives from Member 4**: `train_client_local()` local training routine.
- **Provides to Member 4**:
  - `federated_training_log.json` containing round-by-round validation accuracy and loss curves.
  - Final aggregated global model weights: `best_global_model.pt`.

> [!NOTE]
> **Model Backbone & Parameter Serialization**: The global and local `MultimodalNet` instances incorporate Member 2's Vision Transformer (ViT) backbone. Parameter updates are serialized and aggregated across all ViT, clinical, and fusion layers using PyTorch `state_dict()` in `federated_averaging()`.

---

## 3. Step-by-Step Implementation Guide

### Step 1: Implement FedAvg Aggregation Algorithm
Create `federated/server.py`:
```python
import copy
import torch

def federated_averaging(client_weights_list, client_sample_counts):
    """
    Computes FedAvg: weighted average of model state dicts.
    w_{global} = sum( (n_k / N) * w_k )
    """
    total_samples = sum(client_sample_counts)
    assert total_samples > 0, "Total samples must be positive."
    
    # Initialize global weights with zeros matching structure of client 0
    global_weights = copy.deepcopy(client_weights_list[0])
    for key in global_weights.keys():
        global_weights[key] = torch.zeros_like(global_weights[key], dtype=torch.float32)
        
    # Accumulate weighted parameters
    for w_k, n_k in zip(client_weights_list, client_sample_counts):
        weight_factor = n_k / total_samples
        for key in global_weights.keys():
            global_weights[key] += w_k[key].to(torch.float32) * weight_factor
            
    return global_weights
```

### Step 2: Implement Federated Simulation Orchestrator
Create `federated/run_simulation.py`:
```python
import json
import torch
from models.multimodal_net import MultimodalNet
from training.client_trainer import train_client_local, evaluate_client
from federated.server import federated_averaging

def run_federated_simulation(
    client_loaders,      # List of 3 DataLoaders [c1, c2, c3]
    val_loader,          # Global holdout validation DataLoader
    num_rounds=5,
    local_epochs=2,
    lr=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print("=" * 60)
    print(f"Starting FedMed 30% Simulation: {len(client_loaders)} Clients, {num_rounds} Rounds")
    print("=" * 60)
    
    # 1. Initialize Global Model
    global_model = MultimodalNet(num_clinical_features=3).to(device)
    
    history = {
        "rounds": [],
        "client_train_losses": [],
        "global_val_loss": [],
        "global_val_accuracy": []
    }
    
    for r in range(1, num_rounds + 1):
        print(f"\n--- [Round {r}/{num_rounds}] Broadcasting Global Model ---")
        client_weights = []
        client_sample_counts = []
        round_losses = []
        
        # 2. Local Training at Each Simulated Hospital Client
        for client_idx, loader in enumerate(client_loaders):
            # Create local client model with current global weights
            local_model = MultimodalNet(num_clinical_features=3).to(device)
            local_model.load_state_dict(global_model.state_dict())
            
            # Local training for E epochs
            updated_weights, losses = train_client_local(
                model=local_model,
                dataloader=loader,
                epochs=local_epochs,
                lr=lr,
                device=device
            )
            
            client_weights.append(updated_weights)
            client_sample_counts.append(len(loader.dataset))
            round_losses.append(losses[-1])
            print(f"  Client {client_idx + 1}: Trained {len(loader.dataset)} samples, Final Loss = {losses[-1]:.4f}")
            
        # 3. Server-Side FedAvg Aggregation
        print(f"--- [Round {r}] Aggregating updates via FedAvg ---")
        aggregated_weights = federated_averaging(client_weights, client_sample_counts)
        global_model.load_state_dict(aggregated_weights)
        
        # 4. Global Validation Evaluation
        val_metrics = evaluate_client(global_model, val_loader, device=device)
        print(f"  Global Validation -> Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_accuracy']*100:.2f}%")
        
        # 5. Record History
        history["rounds"].append(r)
        history["client_train_losses"].append(round_losses)
        history["global_val_loss"].append(val_metrics["val_loss"])
        history["global_val_accuracy"].append(val_metrics["val_accuracy"])
        
    # Save results
    with open("federated_training_log.json", "w") as f:
        json.dump(history, f, indent=2)
    torch.save(global_model.state_dict(), "best_global_model.pt")
    print("\nFederated simulation complete. Artifacts saved: federated_training_log.json, best_global_model.pt")
    return history
```

---

## 4. Deliverables Checklist for 30% Presentation

- [ ] `federated/server.py` with verified `federated_averaging()` function.
- [ ] `federated/run_simulation.py` executing 5 communication rounds across 3 simulated clients.
- [ ] Output log `federated_training_log.json` saved and verified.
- [ ] Model checkpoint `best_global_model.pt` saved.
- [ ] Deliver logs to **Member 4 (Evaluation Lead)** for comparative plotting.
