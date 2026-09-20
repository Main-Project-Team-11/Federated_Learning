"""
scripts/stage7_client_partitions.py
Stage 7 Training Data Client Partitioning (3 Simulated Hospital Clients).

Partitions ONLY the training data (83,696 records, 25,426 unique patients) across:
  - Hospital Client 1 (Client 1)
  - Hospital Client 2 (Client 2)
  - Hospital Client 3 (Client 3)

Strict patient-level grouping:
  - Zero patient overlap across clients: C1 ∩ C2 = ∅, C1 ∩ C3 = ∅, C2 ∩ C3 = ∅
  - Zero image duplication: All partitions reference existing local image paths on disk.

Outputs:
  data_prep/client_1.csv & data_prep/client_1.parquet
  data_prep/client_2.csv & data_prep/client_2.parquet
  data_prep/client_3.csv & data_prep/client_3.parquet
  data_prep/client_partition_report.json
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

DATA_PREP_DIR = r"r:\VSCODE\Main_Project\data_prep"
INPUT_TRAIN_PARQUET = os.path.join(DATA_PREP_DIR, "train.parquet")
OUTPUT_REPORT_JSON = os.path.join(DATA_PREP_DIR, "client_partition_report.json")

print("=" * 60)
print("STAGE 7: TRAINING DATA CLIENT PARTITIONING (3 CLIENTS)")
print("=" * 60)

start_time = time.time()

if not os.path.exists(INPUT_TRAIN_PARQUET):
    print(f"Error: {INPUT_TRAIN_PARQUET} does not exist!", file=sys.stderr)
    sys.exit(1)

print(f"Loading training metadata from {INPUT_TRAIN_PARQUET}...")
train_df = pd.read_parquet(INPUT_TRAIN_PARQUET)
total_train_records = len(train_df)
print(f"Total training records: {total_train_records:,}")

# 1. Group by patient and aggregate patient-level class
patient_labels = train_df.groupby("patient_id")["binary_label"].max().reset_index()
patient_labels.columns = ["patient_id", "has_pneu"]

total_train_patients = len(patient_labels)
print(f"Total unique training patients: {total_train_patients:,}")

# 2. Divide unique patients into 3 uniform, stratified subsets
np.random.seed(42)

pos_patients = list(patient_labels[patient_labels["has_pneu"] == 1]["patient_id"].values)
neg_patients = list(patient_labels[patient_labels["has_pneu"] == 0]["patient_id"].values)

np.random.shuffle(pos_patients)
np.random.shuffle(neg_patients)

# Split positive patients across 3 clients
pos_splits = np.array_split(pos_patients, 3)
# Split negative patients across 3 clients
neg_splits = np.array_split(neg_patients, 3)

client_patient_sets = []
for i in range(3):
    c_patients = set(pos_splits[i]) | set(neg_splits[i])
    client_patient_sets.append(c_patients)
    print(f"Client {i+1} unique patients: {len(c_patients):,} (Pos: {len(pos_splits[i]):,}, Neg: {len(neg_splits[i]):,})")

# 3. Assert Zero Patient Overlap Across Clients
print("\nVerifying Zero Patient Overlap Across Clients...")
c1_c2 = client_patient_sets[0] & client_patient_sets[1]
c1_c3 = client_patient_sets[0] & client_patient_sets[2]
c2_c3 = client_patient_sets[1] & client_patient_sets[2]

assert len(c1_c2) == 0, f"Leakage between Client 1 and 2: {len(c1_c2)} patients!"
assert len(c1_c3) == 0, f"Leakage between Client 1 and 3: {len(c1_c3)} patients!"
assert len(c2_c3) == 0, f"Leakage between Client 2 and 3: {len(c2_c3)} patients!"
assert sum(len(s) for s in client_patient_sets) == total_train_patients, "Patient count mismatch!"
print("  [PASS] Zero patient overlap verified: Client 1 & 2 = empty, Client 1 & 3 = empty, Client 2 & 3 = empty.")

# 4. Partition Training Rows & Save Files
client_reports = []
total_partitioned_records = 0

print("\n" + "=" * 60)
print("CLIENT PARTITION METRICS SUMMARY")
print("=" * 60)

for i in range(3):
    c_num = i + 1
    c_patients = client_patient_sets[i]
    c_df = train_df[train_df["patient_id"].isin(c_patients)].copy().reset_index(drop=True)
    c_df["client_id"] = f"Hospital_{c_num}"
    
    c_records = len(c_df)
    total_partitioned_records += c_records
    c_pneu = int((c_df["binary_label"] == 1).sum())
    c_non_pneu = int((c_df["binary_label"] == 0).sum())
    c_ratio = c_non_pneu / c_pneu if c_pneu > 0 else 0.0
    
    csv_path = os.path.join(DATA_PREP_DIR, f"client_{c_num}.csv")
    parquet_path = os.path.join(DATA_PREP_DIR, f"client_{c_num}.parquet")
    
    c_df.to_csv(csv_path, index=False)
    c_df.to_parquet(parquet_path, index=False)
    
    c_info = {
        "client_id": f"Hospital_{c_num}",
        "sample_count": c_records,
        "sample_pct_of_train": round((c_records / total_train_records) * 100, 2),
        "unique_patients": len(c_patients),
        "pneumonia_count": c_pneu,
        "pneumonia_pct": round((c_pneu / c_records) * 100, 2),
        "non_pneumonia_count": c_non_pneu,
        "non_pneumonia_pct": round((c_non_pneu / c_records) * 100, 2),
        "imbalance_ratio": round(c_ratio, 2),
        "csv_path": csv_path,
        "parquet_path": parquet_path
    }
    client_reports.append(c_info)
    
    print(f"[CLIENT {c_num} — Hospital_{c_num}]")
    print(f"  Samples:     {c_records:,} ({c_info['sample_pct_of_train']}%)")
    print(f"  Patients:    {len(c_patients):,}")
    print(f"  Pneumonia:   {c_pneu:,} ({c_info['pneumonia_pct']}%)")
    print(f"  Non-Pneu:    {c_non_pneu:,} ({c_info['non_pneumonia_pct']}%)")
    print(f"  Class Ratio: {c_info['imbalance_ratio']} : 1")
    print(f"  Saved to:    {csv_path} & {parquet_path}")
    print("-" * 60)

assert total_partitioned_records == total_train_records, "Total partitioned records != Total training records!"

# Save Report JSON
report_data = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "num_clients": 3,
    "total_training_records": total_train_records,
    "total_training_patients": total_train_patients,
    "clients": client_reports,
    "overlap_check": {
        "c1_c2_overlap": len(c1_c2),
        "c1_c3_overlap": len(c1_c3),
        "c2_c3_overlap": len(c2_c3),
        "status": "PASS - ZERO OVERLAP"
    }
}

with open(OUTPUT_REPORT_JSON, "w") as f:
    json.dump(report_data, f, indent=2)
print(f"Client partition report saved to {OUTPUT_REPORT_JSON}.")

elapsed = time.time() - start_time
print(f"\nStage 7 completed in {elapsed:.2f} seconds.")
