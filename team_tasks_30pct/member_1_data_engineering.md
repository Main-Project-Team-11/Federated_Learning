# Member 1: Data Engineering & Quality Assurance Lead

- **Role:** Member 1 — Data Engineering & Preprocessing
- **Milestone:** 30% Implementation Presentation
- **Primary Focus:** Dataset acquisition, running the 8 data checks, binary label extraction, tabular normalization, patient-level train/val/test split (zero data leakage), and PyTorch multimodal dataset loaders.

---

## 1. Responsibilities & Objectives

1. **Dataset Setup**:
   - Organize local directory structure for **NIH ChestX-ray14** and **CheXpert** (and structure ready for **MIMIC-CXR** if PhysioNet access is approved).
2. **Execute 8 Data Quality Checks**:
   - Use [`data_validation_suite/validate_dataset.py`](file:///r:/VSCODE/Main_Project/data_validation_suite/validate_dataset.py) to audit the dataset.
   - Verify: Image validity, Patient/Study IDs, Pairing, Label correctness, Duplicates, Missing values, Class balance, and Patient leakage.
3. **Binary Label Filtering**:
   - Extract positive `Pneumonia` samples (Class 1).
   - Extract negative `Normal` / `No Finding` samples (Class 0).
   - Filter out non-target thoracic pathologies (e.g. Cardiomegaly, Atelectasis without Pneumonia) to prevent clinical ground truth contamination.
4. **Tabular Feature Pipeline**:
   - Clean tabular attributes (handle Age anomalies $> 105$, encode Sex: M=0, F=1, View Position: PA=0, AP=1).
   - Standardize numerical values (mean=0, std=1).
5. **Patient-Level Splitting & Client Partitions**:
   - Split dataset into Train (70%), Validation (15%), and Test (15%) strictly grouped by `patient_id`.
   - Divide the Train partition into **3 uniform/IID client partitions** (Hospital Client 1, 2, 3) ensuring no patient is shared across clients.
6. **PyTorch DataLoaders**:
   - Provide a clean, importable `MultimodalDataset` and `get_client_loaders()` function for Member 2, Member 3, and Member 4.

---

## 2. Interface Contract (What you provide to teammates)

### Batch Output Specification:
Your PyTorch DataLoader must yield batches structured as a dictionary:
```python
batch = {
    "image": torch.Tensor,     # Shape: [B, 3, 224, 224] (float32, normalized for Member 2's ViT encoder)
    "clinical": torch.Tensor,  # Shape: [B, num_features] (float32, e.g. Age, Sex, View)
    "label": torch.Tensor,     # Shape: [B, 1] (float32, 0.0 or 1.0)
    "patient_id": list         # List of strings [B]
}
```t

> [!NOTE]
> **Vision Backbone Compatibility**: The image tensor `batch["image"]` of shape `[B, 3, 224, 224]` directly feeds into Member 2's Vision Transformer (ViT) image encoder (which decomposes each image into non-overlapping patches and linearly projects them into visual tokens).

---

## 3. Step-by-Step Implementation Guide

### Step 1: Run the 8 Data Checks
Run the validation script located in `data_validation_suite/`:
```bash
python data_validation_suite/validate_dataset.py \
    --csv_path path/to/Data_Entry_2017.csv \
    --image_dir path/to/images/ \
    --dataset_type nih \
    --output_report data_validation_suite/nih_validation_report.json
```
Verify that all 8 checks pass. In particular, inspect:
- `age_anomalies`: Flag and remove any patients with Age $> 105$.
- `zero_byte_count`: Ensure no corrupted 0-byte images exist.
- `ghost_records`: Ensure 100% of images listed in the CSV exist on disk.

### Step 2: Create Grouped Train/Val/Test Split
```python
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def patient_level_split(df: pd.DataFrame, patient_col="Patient ID"):
    # First split: Train (70%) vs Temp (30%)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(df, groups=df[patient_col]))
    
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()
    
    # Second split: Val (15%) vs Test (15%)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[patient_col]))
    
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()
    
    # Strict assertion: Zero patient leakage
    assert len(set(train_df[patient_col]) & set(val_df[patient_col])) == 0, "Train-Val Leakage!"
    assert len(set(train_df[patient_col]) & set(test_df[patient_col])) == 0, "Train-Test Leakage!"
    assert len(set(val_df[patient_col]) & set(test_df[patient_col])) == 0, "Val-Test Leakage!"
    
    print(f"Split complete: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df
```

### Step 3: Create 3 Uniform Client Partitions
```python
def partition_for_clients(train_df: pd.DataFrame, num_clients=3, patient_col="Patient ID"):
    """Uniform/IID client partitions by grouped patients (Non-IID halted for 30%)."""
    unique_patients = np.array(train_df[patient_col].unique())
    np.random.seed(42)
    np.random.shuffle(unique_patients)
    
    patient_splits = np.array_split(unique_patients, num_clients)
    client_dfs = []
    
    for i, p_subset in enumerate(patient_splits):
        c_df = train_df[train_df[patient_col].isin(set(p_subset))].copy()
        c_df["client_id"] = f"Hospital_{i+1}"
        client_dfs.append(c_df)
        print(f"Client {i+1}: {len(c_df)} samples, {len(p_subset)} unique patients.")
        
    return client_dfs
```

### Step 4: Implement PyTorch Multimodal Dataset
```python
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MultimodalChestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, image_col="Image Index", transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_col = image_col
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # 3-channel for standard vision backbones (e.g., ViT)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.image_dir}/{row[self.image_col]}"
        
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        
        # Clinical features: e.g. [Age_norm, Sex_binary, View_binary]
        clinical_tensor = torch.tensor(row["clinical_features"], dtype=torch.float32)
        label_tensor = torch.tensor([row["binary_label"]], dtype=torch.float32)
        
        return {
            "image": image_tensor,
            "clinical": clinical_tensor,
            "label": label_tensor,
            "patient_id": str(row["Patient ID"])
        }
```

---

## 4. Deliverables Checklist for 30% Presentation

- [ ] Run `data_validation_suite/validate_dataset.py` on your dataset and save `validation_report.json`.
- [ ] Save processed split CSVs: `train_data.csv`, `val_data.csv`, `test_data.csv`.
- [ ] Save client partition CSVs: `client_1.csv`, `client_2.csv`, `client_3.csv`.
- [ ] Produce `data_loader.py` providing `get_client_loader(client_id, batch_size)`.
- [ ] Hand off `data_loader.py` to **Member 2** and **Member 4**.
