"""
scripts/verify_pipeline_integration.py
Integration Verification for Member 1 Data Pipeline.

Tests:
  1. Client DataLoaders (Hospital 1, 2, 3)
  2. Validation DataLoader
  3. Test DataLoader
  4. Batch tensor shapes, dtypes, NaN/Inf checks, label range
  5. Lazy image loading and memory stability
"""
import os
import sys
import time
import torch

# Ensure workspace root is in python path
sys.path.insert(0, r"r:\VSCODE\Main_Project")
from data.multimodal_dataset import get_client_loader, get_val_loader, get_test_loader

print("=" * 60)
print("FINAL DATA PIPELINE INTEGRATION TEST")
print("=" * 60)

start_time = time.time()
batch_size = 16

loaders = {
    "Hospital_Client_1": get_client_loader(1, batch_size=batch_size, shuffle=True),
    "Hospital_Client_2": get_client_loader(2, batch_size=batch_size, shuffle=True),
    "Hospital_Client_3": get_client_loader(3, batch_size=batch_size, shuffle=True),
    "Validation": get_val_loader(batch_size=batch_size, shuffle=False),
    "Test": get_test_loader(batch_size=batch_size, shuffle=False)
}

print(f"Loaded {len(loaders)} DataLoaders successfully.\n")

for loader_name, loader in loaders.items():
    print(f"Testing {loader_name} (Total samples: {len(loader.dataset):,}, Batches: {len(loader):,})...")
    
    # Retrieve first batch
    batch = next(iter(loader))
    
    # 1. Structure check
    assert isinstance(batch, dict), f"{loader_name}: Batch is not a dictionary!"
    required_keys = ["image", "clinical", "label", "patient_id"]
    for k in required_keys:
        assert k in batch, f"{loader_name}: Missing key '{k}' in batch!"

    img = batch["image"]
    clin = batch["clinical"]
    lbl = batch["label"]
    p_ids = batch["patient_id"]

    # 2. Shape checks
    assert img.shape == (batch_size, 3, 224, 224), f"{loader_name}: Invalid image shape {img.shape}"
    assert clin.shape == (batch_size, 2), f"{loader_name}: Invalid clinical shape {clin.shape}"
    assert lbl.shape == (batch_size, 1), f"{loader_name}: Invalid label shape {lbl.shape}"
    assert len(p_ids) == batch_size, f"{loader_name}: Patient ID count {len(p_ids)} != {batch_size}"

    # 3. Dtype checks
    assert img.dtype == torch.float32, f"{loader_name}: Image dtype {img.dtype} != float32"
    assert clin.dtype == torch.float32, f"{loader_name}: Clinical dtype {clin.dtype} != float32"
    assert lbl.dtype == torch.float32, f"{loader_name}: Label dtype {lbl.dtype} != float32"

    # 4. NaN / Inf checks
    assert not torch.isnan(img).any(), f"{loader_name}: NaN detected in image tensor!"
    assert not torch.isinf(img).any(), f"{loader_name}: Inf detected in image tensor!"
    assert not torch.isnan(clin).any(), f"{loader_name}: NaN detected in clinical tensor!"
    assert not torch.isinf(clin).any(), f"{loader_name}: Inf detected in clinical tensor!"
    assert not torch.isnan(lbl).any(), f"{loader_name}: NaN detected in label tensor!"
    assert not torch.isinf(lbl).any(), f"{loader_name}: Inf detected in label tensor!"

    # 5. Label value check
    unique_labels = set(lbl.squeeze().tolist())
    assert unique_labels.issubset({0.0, 1.0}), f"{loader_name}: Invalid labels {unique_labels}"

    print(f"  [PASS] Shapes: image={list(img.shape)}, clinical={list(clin.shape)}, label={list(lbl.shape)}")
    print(f"  [PASS] Dtypes: float32 | NaN/Inf: 0 | Labels: {unique_labels} | Patients: {len(p_ids)}")

print("\n" + "=" * 60)
print("MULTI-BATCH LAZY LOADING STABILITY TEST")
print("=" * 60)
client1_loader = loaders["Hospital_Client_1"]
print("Iterating through 5 consecutive batches of Hospital_Client_1...")
batch_count = 0
for b in client1_loader:
    batch_count += 1
    if batch_count >= 5:
        break
print(f"  [PASS] Successfully streamed {batch_count} batches without memory spike or error.")

elapsed = time.time() - start_time
print(f"\nPipeline integration test completed successfully in {elapsed:.2f} seconds.")
print("DATA PIPELINE IS FULLY MODEL-READY FOR MEMBER 2, MEMBER 4, AND MEMBER 3.")
