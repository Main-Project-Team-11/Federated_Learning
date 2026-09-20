"""
scripts/stage1_prepare_nih.py
Stage 1 Dataset Preparation for NIH ChestX-ray14.

Extracts and validates metadata from the 6 local Parquet shards at:
  Datasets/my_nih_chest_xr_dataset/data/train-*.parquet
Uses PyArrow column projection to read ONLY metadata columns, avoiding loading
image binary bytes into memory.

Outputs:
  data_prep/nih_prepared_meta.parquet
"""
import os
import sys
import glob
import time
import pandas as pd
import pyarrow.parquet as pq

DATASET_DIR = r"r:\VSCODE\Main_Project\Datasets\my_nih_chest_xr_dataset\data"
OUTPUT_DIR = r"r:\VSCODE\Main_Project\data_prep"
OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "nih_prepared_meta.parquet")
OUTPUT_CSV_SAMPLE = os.path.join(OUTPUT_DIR, "nih_prepared_meta_sample.csv")
LOCAL_IMAGE_DIR = r"R:\FedMed_Data\nih_png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("STAGE 1: NIH CHESTX-RAY14 DATASET PREPARATION")
print("=" * 60)

# Find all Parquet shards
shard_files = sorted(glob.glob(os.path.join(DATASET_DIR, "train-*.parquet")))
if not shard_files:
    print(f"Error: No Parquet shards found in {DATASET_DIR}!", file=sys.stderr)
    sys.exit(1)

print(f"Found {len(shard_files)} Parquet shards in {DATASET_DIR}.")

start_time = time.time()
records = []
total_rows = 0

# Columns to read (explicitly excluding 'image' struct to avoid loading image bytes into RAM)
columns_to_read = ["patient_id", "scan_id", "age", "sex", "labels"]

for idx, shard_path in enumerate(shard_files):
    shard_name = os.path.basename(shard_path)
    print(f"Reading shard {idx + 1}/{len(shard_files)}: {shard_name}...")
    
    # Read table with column projection
    table = pq.read_table(shard_path, columns=columns_to_read)
    df_shard = table.to_pandas()
    
    total_rows += len(df_shard)
    records.append(df_shard)

print(f"All shards loaded. Total raw rows: {total_rows}")
df_nih = pd.concat(records, ignore_index=True)

# 1. Standardize types
df_nih["patient_id"] = df_nih["patient_id"].astype(str)
df_nih["scan_id"] = df_nih["scan_id"].astype(int)
df_nih["age"] = pd.to_numeric(df_nih["age"], errors="coerce")
df_nih["sex"] = df_nih["sex"].astype(str).str.strip().str.upper()
df_nih["dataset_source"] = "NIH"

# 2. Canonical image file name and path
df_nih["image_filename"] = df_nih["patient_id"] + "_" + df_nih["scan_id"].astype(str) + ".png"
df_nih["image_path"] = LOCAL_IMAGE_DIR + "\\" + df_nih["image_filename"]

# 3. View type & Projection metadata
# NIH ChestX-ray14 consists exclusively of frontal chest radiographs (PA and AP projections)
df_nih["view_type"] = "Frontal"
df_nih["projection"] = "Unknown"  # Specific PA/AP sub-projection is not present in HuggingFace shard columns

# 4. Statistical Inspection & Diagnostics
print("\n--- STATISTICAL INSPECTION ---")
print(f"Total records: {len(df_nih)}")
print(f"Unique patients: {df_nih['patient_id'].nunique()}")

# Sex counts
sex_counts = df_nih["sex"].value_counts(dropna=False).to_dict()
print(f"Sex distribution: {sex_counts}")
missing_sex = df_nih["sex"].isna().sum() + (df_nih["sex"] == "").sum() + (df_nih["sex"] == "NONE").sum()
print(f"Missing / anomalous sex count: {missing_sex}")

# Age statistics
age_desc = df_nih["age"].describe().to_dict()
print(f"Age summary: min={age_desc['min']}, max={age_desc['max']}, mean={age_desc['mean']:.2f}, median={df_nih['age'].median()}")
invalid_age_count = ((df_nih["age"] < 0) | (df_nih["age"] > 105)).sum()
missing_age_count = df_nih["age"].isna().sum()
print(f"Age anomalies (< 0 or > 105): {invalid_age_count}")
print(f"Missing age count: {missing_age_count}")

# Label analysis
def contains_label(lbls, target):
    if lbls is None:
        return False
    try:
        return target in lbls
    except TypeError:
        return False

has_pneumonia = df_nih["labels"].apply(lambda lbls: contains_label(lbls, 7))
has_no_finding = df_nih["labels"].apply(lambda lbls: contains_label(lbls, 0))
has_both_0_and_7 = df_nih["labels"].apply(lambda lbls: contains_label(lbls, 0) and contains_label(lbls, 7))

pneumonia_count = has_pneumonia.sum()
no_finding_count = has_no_finding.sum()
inconsistent_count = has_both_0_and_7.sum()

print(f"Records with Pneumonia (label 7): {pneumonia_count}")
print(f"Records with No Finding (label 0): {no_finding_count}")
print(f"Records with INCONSISTENT [0, 7] label: {inconsistent_count}")

# 5. Save Intermediate Prepared Metadata
# Convert labels list to string or pyarrow list for parquet storage
print(f"\nSaving intermediate metadata to {OUTPUT_PARQUET}...")
df_nih.to_parquet(OUTPUT_PARQUET, index=False)
print("Saved successfully.")

# Save small sample CSV (first 50 rows) for inspection
df_nih.head(50).to_csv(OUTPUT_CSV_SAMPLE, index=False)
print(f"Sample CSV saved to {OUTPUT_CSV_SAMPLE}.")

elapsed = time.time() - start_time
print(f"\nStage 1 NIH preparation complete in {elapsed:.2f}s.")
