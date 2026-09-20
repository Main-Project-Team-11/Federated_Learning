"""
scripts/stage6_patient_level_split.py
Stage 6 Patient-Level Train/Validation/Test Split.

Splits the cleaned dataset (118,654 records) into:
  - Train: ~70%
  - Validation: ~15%
  - Test: ~15%

Strict grouping by patient_id:
  - Zero patient leakage: Train & Val = empty, Train & Test = empty, Val & Test = empty
  - Stratified by patient-level pneumonia presence to maintain balanced class ratios.

Outputs:
  data_prep/train.csv & data_prep/train.parquet
  data_prep/val.csv & data_prep/val.parquet
  data_prep/test.csv & data_prep/test.parquet
  data_prep/stage6_split_report.json
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

DATA_PREP_DIR = r"r:\VSCODE\Main_Project\data_prep"
INPUT_CLEANED_PARQUET = os.path.join(DATA_PREP_DIR, "stage4_cleaned_metadata.parquet")

OUTPUT_TRAIN_CSV = os.path.join(DATA_PREP_DIR, "train.csv")
OUTPUT_TRAIN_PARQUET = os.path.join(DATA_PREP_DIR, "train.parquet")
OUTPUT_VAL_CSV = os.path.join(DATA_PREP_DIR, "val.csv")
OUTPUT_VAL_PARQUET = os.path.join(DATA_PREP_DIR, "val.parquet")
OUTPUT_TEST_CSV = os.path.join(DATA_PREP_DIR, "test.csv")
OUTPUT_TEST_PARQUET = os.path.join(DATA_PREP_DIR, "test.parquet")
OUTPUT_REPORT_JSON = os.path.join(DATA_PREP_DIR, "stage6_split_report.json")

print("=" * 60)
print("STAGE 6: PATIENT-LEVEL TRAIN / VAL / TEST SPLIT")
print("=" * 60)

start_time = time.time()

if not os.path.exists(INPUT_CLEANED_PARQUET):
    print(f"Error: {INPUT_CLEANED_PARQUET} does not exist!", file=sys.stderr)
    sys.exit(1)

print(f"Loading cleaned metadata from {INPUT_CLEANED_PARQUET}...")
df = pd.read_parquet(INPUT_CLEANED_PARQUET)
total_records = len(df)
print(f"Total clean records: {total_records:,}")

# 1. Define Patient-Level Class Representation
# A patient is labeled positive (1) if they have AT LEAST ONE pneumonia scan; 0 otherwise.
print("Aggregating patient-level class representations...")
patient_labels = df.groupby("patient_id")["binary_label"].max().reset_index()
patient_labels.columns = ["patient_id", "patient_has_pneu"]

total_patients = len(patient_labels)
pos_patients = int((patient_labels["patient_has_pneu"] == 1).sum())
neg_patients = int((patient_labels["patient_has_pneu"] == 0).sum())
print(f"Total unique patients: {total_patients:,}")
print(f"  Patients with Pneumonia: {pos_patients:,} ({(pos_patients/total_patients)*100:.2f}%)")
print(f"  Patients Non-Pneumonia only: {neg_patients:,} ({(neg_patients/total_patients)*100:.2f}%)")

# 2. Stratified Patient Split (70% Train, 15% Val, 15% Test)
print("\nPerforming Stratified Patient-Level Split (70% / 15% / 15%)...")
np.random.seed(42)

# Convert to standard Python lists for reliable shuffling
pos_p_list = list(patient_labels[patient_labels["patient_has_pneu"] == 1]["patient_id"].values)
neg_p_list = list(patient_labels[patient_labels["patient_has_pneu"] == 0]["patient_id"].values)

np.random.shuffle(pos_p_list)
np.random.shuffle(neg_p_list)

# Compute split indices for positive patients
n_pos = len(pos_p_list)
pos_train_end = int(0.70 * n_pos)
pos_val_end = pos_train_end + int(0.15 * n_pos)

pos_train = pos_p_list[:pos_train_end]
pos_val = pos_p_list[pos_train_end:pos_val_end]
pos_test = pos_p_list[pos_val_end:]

# Compute split indices for negative patients
n_neg = len(neg_p_list)
neg_train_end = int(0.70 * n_neg)
neg_val_end = neg_train_end + int(0.15 * n_neg)

neg_train = neg_p_list[:neg_train_end]
neg_val = neg_p_list[neg_train_end:neg_val_end]
neg_test = neg_p_list[neg_val_end:]

train_patient_set = set(pos_train) | set(neg_train)
val_patient_set = set(pos_val) | set(neg_val)
test_patient_set = set(pos_test) | set(neg_test)

print(f"Unique patient counts:")
print(f"  Train patients: {len(train_patient_set):,} ({(len(train_patient_set)/total_patients)*100:.2f}%)")
print(f"  Val patients:   {len(val_patient_set):,} ({(len(val_patient_set)/total_patients)*100:.2f}%)")
print(f"  Test patients:  {len(test_patient_set):,} ({(len(test_patient_set)/total_patients)*100:.2f}%)")

# 3. Strict Zero-Leakage Assertions on Patients
print("\nVerifying Zero Patient Leakage...")
train_val_overlap = train_patient_set & val_patient_set
train_test_overlap = train_patient_set & test_patient_set
val_test_overlap = val_patient_set & test_patient_set

assert len(train_val_overlap) == 0, f"LEAKAGE DETECTED between Train and Val: {len(train_val_overlap)} patients!"
assert len(train_test_overlap) == 0, f"LEAKAGE DETECTED between Train and Test: {len(train_test_overlap)} patients!"
assert len(val_test_overlap) == 0, f"LEAKAGE DETECTED between Val and Test: {len(val_test_overlap)} patients!"
assert len(train_patient_set) + len(val_patient_set) + len(test_patient_set) == total_patients, "Patient count mismatch!"
print("  [PASS] Zero patient overlap verified: Train & Val = empty, Train & Test = empty, Val & Test = empty.")

# 4. Partition Full Dataset Rows by Patient
print("\nAssigning records to partitions...")
train_df = df[df["patient_id"].isin(train_patient_set)].copy().reset_index(drop=True)
val_df = df[df["patient_id"].isin(val_patient_set)].copy().reset_index(drop=True)
test_df = df[df["patient_id"].isin(test_patient_set)].copy().reset_index(drop=True)

# Row counts and integrity check
assert len(train_df) + len(val_df) + len(test_df) == total_records, "Record count mismatch!"

# 5. Compute Split Statistics
def get_partition_stats(part_df, name, total_all):
    count = len(part_df)
    pneu = int((part_df["binary_label"] == 1).sum())
    non_pneu = int((part_df["binary_label"] == 0).sum())
    patients = int(part_df["patient_id"].nunique())
    ratio = non_pneu / pneu if pneu > 0 else 0.0
    return {
        "name": name,
        "sample_count": count,
        "sample_pct": round((count / total_all) * 100, 2),
        "unique_patients": patients,
        "pneumonia_count": pneu,
        "pneumonia_pct": round((pneu / count) * 100, 2) if count > 0 else 0,
        "non_pneumonia_count": non_pneu,
        "non_pneumonia_pct": round((non_pneu / count) * 100, 2) if count > 0 else 0,
        "imbalance_ratio": round(ratio, 2)
    }

train_stats = get_partition_stats(train_df, "Train", total_records)
val_stats = get_partition_stats(val_df, "Validation", total_records)
test_stats = get_partition_stats(test_df, "Test", total_records)

print("\n" + "=" * 60)
print("STAGE 6 PARTITION METRICS SUMMARY")
print("=" * 60)
for s in [train_stats, val_stats, test_stats]:
    print(f"[{s['name'].upper()}]")
    print(f"  Samples:     {s['sample_count']:,} ({s['sample_pct']}%)")
    print(f"  Patients:    {s['unique_patients']:,}")
    print(f"  Pneumonia:   {s['pneumonia_count']:,} ({s['pneumonia_pct']}%)")
    print(f"  Non-Pneu:    {s['non_pneumonia_count']:,} ({s['non_pneumonia_pct']}%)")
    print(f"  Class Ratio: {s['imbalance_ratio']} : 1")
    print("-" * 60)

# 6. Save Partition Files
print(f"\nSaving train.csv ({len(train_df):,} records) and train.parquet...")
train_df.to_csv(OUTPUT_TRAIN_CSV, index=False)
train_df.to_parquet(OUTPUT_TRAIN_PARQUET, index=False)

print(f"Saving val.csv ({len(val_df):,} records) and val.parquet...")
val_df.to_csv(OUTPUT_VAL_CSV, index=False)
val_df.to_parquet(OUTPUT_VAL_PARQUET, index=False)

print(f"Saving test.csv ({len(test_df):,} records) and test.parquet...")
test_df.to_csv(OUTPUT_TEST_CSV, index=False)
test_df.to_parquet(OUTPUT_TEST_PARQUET, index=False)

# Save Report JSON
report_data = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "total_records": total_records,
    "total_unique_patients": total_patients,
    "partitions": {
        "train": train_stats,
        "validation": val_stats,
        "test": test_stats
    },
    "leakage_verification": {
        "train_val_overlap": len(train_val_overlap),
        "train_test_overlap": len(train_test_overlap),
        "val_test_overlap": len(val_test_overlap),
        "status": "PASS - ZERO LEAKAGE"
    }
}

with open(OUTPUT_REPORT_JSON, "w") as f:
    json.dump(report_data, f, indent=2)
print("Split report JSON saved successfully.")

elapsed = time.time() - start_time
print(f"\nStage 6 completed in {elapsed:.2f} seconds.")
