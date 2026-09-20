"""
scripts/stage2_canonical_schema.py
Stage 2 Common/Canonical Schema Harmonization.

Harmonizes NIH ChestX-ray14 and CheXpert intermediate metadata into the
unified canonical schema:
  - sample_id
  - dataset_source
  - patient_id
  - study_id
  - image_path
  - age
  - sex
  - view_type
  - projection
  - raw_label_info
  - binary_label (placeholder initialized for Stage 3 processing)

Outputs:
  data_prep/canonical_metadata.parquet
  data_prep/canonical_metadata_sample.csv
"""
import os
import sys
import time
import pandas as pd
import numpy as np

DATA_PREP_DIR = r"r:\VSCODE\Main_Project\data_prep"
NIH_PARQUET = os.path.join(DATA_PREP_DIR, "nih_prepared_meta.parquet")
CHEXPERT_PARQUET = os.path.join(DATA_PREP_DIR, "chexpert_prepared_meta.parquet")
CANONICAL_PARQUET = os.path.join(DATA_PREP_DIR, "canonical_metadata.parquet")
CANONICAL_SAMPLE_CSV = os.path.join(DATA_PREP_DIR, "canonical_metadata_sample.csv")

print("=" * 60)
print("STAGE 2: COMMON / CANONICAL SCHEMA HARMONIZATION")
print("=" * 60)

start_time = time.time()

# 1. Load NIH prepared metadata
print(f"Loading NIH metadata from {NIH_PARQUET}...")
if not os.path.exists(NIH_PARQUET):
    print(f"Error: {NIH_PARQUET} not found!", file=sys.stderr)
    sys.exit(1)
df_nih_raw = pd.read_parquet(NIH_PARQUET)
print(f"NIH rows loaded: {len(df_nih_raw)}")

# 2. Load CheXpert prepared metadata
print(f"Loading CheXpert metadata from {CHEXPERT_PARQUET}...")
if not os.path.exists(CHEXPERT_PARQUET):
    print(f"Error: {CHEXPERT_PARQUET} not found!", file=sys.stderr)
    sys.exit(1)
df_chex_raw = pd.read_parquet(CHEXPERT_PARQUET)
print(f"CheXpert rows loaded: {len(df_chex_raw)}")

# -------------------------------------------------------------
# Standardize NIH to Canonical Schema
# -------------------------------------------------------------
print("\nStandardizing NIH records to canonical schema...")
df_nih_canon = pd.DataFrame()
df_nih_canon["sample_id"] = "nih_" + df_nih_raw["patient_id"].astype(str) + "_" + df_nih_raw["scan_id"].astype(str)
df_nih_canon["dataset_source"] = "NIH"
df_nih_canon["patient_id"] = "nih_p_" + df_nih_raw["patient_id"].astype(str)
df_nih_canon["study_id"] = "scan_" + df_nih_raw["scan_id"].astype(str)
df_nih_canon["image_path"] = df_nih_raw["image_path"].astype(str)
df_nih_canon["age"] = df_nih_raw["age"].astype(np.float32)
df_nih_canon["sex"] = df_nih_raw["sex"].astype(str).str.upper()
df_nih_canon["view_type"] = "Frontal"  # NIH ChestX-ray14 is 100% frontal
df_nih_canon["projection"] = "Unknown"  # Not distinguished per-scan in HuggingFace shards

# Preserve raw labels for Stage 3 binary label mapping
# Convert numpy array / list to string representation for unified column type
df_nih_canon["raw_label_info"] = df_nih_raw["labels"].apply(
    lambda x: ",".join(map(str, x)) if isinstance(x, (list, np.ndarray)) else str(x)
)
df_nih_canon["binary_label"] = -1  # Placeholder pending Stage 3 label policy

# -------------------------------------------------------------
# Standardize CheXpert to Canonical Schema
# -------------------------------------------------------------
print("Standardizing CheXpert records to canonical schema...")
df_chex_canon = pd.DataFrame()

# Clean filename stem for sample_id
chex_filename_stems = df_chex_raw["raw_path"].apply(
    lambda p: os.path.splitext(os.path.basename(p))[0] if p else "unknown"
)
df_chex_canon["sample_id"] = "chexpert_" + df_chex_raw["patient_id"].astype(str) + "_" + df_chex_raw["study_id"].astype(str) + "_" + chex_filename_stems
df_chex_canon["dataset_source"] = "CheXpert"
df_chex_canon["patient_id"] = "chexpert_" + df_chex_raw["patient_id"].astype(str)
df_chex_canon["study_id"] = "chexpert_" + df_chex_raw["study_id"].astype(str)
df_chex_canon["image_path"] = df_chex_raw["image_path"].astype(str)
df_chex_canon["age"] = df_chex_raw["age"].astype(np.float32)

# Verified CheXpert mapping: 0=Male ('M'), 1=Female ('F')
sex_map = {0: "M", 1: "F"}
df_chex_canon["sex"] = df_chex_raw["sex"].map(sex_map).fillna("Unknown").astype(str)

df_chex_canon["view_type"] = df_chex_raw["view_type"].astype(str)

# Verified CheXpert mapping: 0=AP, 1=PA, 2=''
proj_map = {"0": "AP", "1": "PA", "0.0": "AP", "1.0": "PA", "2": "Unknown", "2.0": "Unknown", "": "Unknown"}
df_chex_canon["projection"] = df_chex_raw["projection"].astype(str).map(proj_map).fillna("Unknown")

# Preserve raw pneumonia observation code (0=Unlabeled, 1=Uncertain, 2=Absent, 3=Present)
df_chex_canon["raw_label_info"] = "pneumonia_code_" + df_chex_raw["raw_pneumonia"].astype(str)
df_chex_canon["binary_label"] = -1  # Placeholder pending Stage 3 label policy

# -------------------------------------------------------------
# Concatenate into Unified Table
# -------------------------------------------------------------
print("\nConcatenating NIH and CheXpert into unified canonical table...")
canonical_df = pd.concat([df_nih_canon, df_chex_canon], ignore_index=True)

# -------------------------------------------------------------
# Diagnostics & Verification
# -------------------------------------------------------------
print("\n=== CANONICAL SCHEMA DIAGNOSTICS ===")
print(f"Total canonical records: {len(canonical_df)}")
print(f"Dataset breakdown: {canonical_df['dataset_source'].value_counts().to_dict()}")
print(f"Unique global patients: {canonical_df['patient_id'].nunique()}")
print(f"Unique global studies: {canonical_df['study_id'].nunique()}")

print("\n--- Column Types & Missing Values ---")
for col in canonical_df.columns:
    null_count = canonical_df[col].isna().sum()
    print(f"  {col:16s} | dtype: {str(canonical_df[col].dtype):10s} | Nulls: {null_count}")

print("\n--- Categorical Distributions ---")
print(f"Sex breakdown: {canonical_df.groupby(['dataset_source', 'sex']).size().to_dict()}")
print(f"View Type breakdown: {canonical_df.groupby(['dataset_source', 'view_type']).size().to_dict()}")
print(f"Projection breakdown: {canonical_df.groupby(['dataset_source', 'projection']).size().to_dict()}")

# Check sample_id uniqueness
duplicate_sample_ids = canonical_df["sample_id"].duplicated().sum()
print(f"\nDuplicate sample_id count: {duplicate_sample_ids}")
assert duplicate_sample_ids == 0, f"Error: Found {duplicate_sample_ids} duplicate sample_ids!"

# Save Canonical Metadata
print(f"\nSaving canonical metadata to {CANONICAL_PARQUET}...")
canonical_df.to_parquet(CANONICAL_PARQUET, index=False)
print("Saved successfully.")

# Save Sample CSV (25 NIH + 25 CheXpert)
sample_csv_df = pd.concat([df_nih_canon.head(25), df_chex_canon.head(25)], ignore_index=True)
sample_csv_df.to_csv(CANONICAL_SAMPLE_CSV, index=False)
print(f"Canonical sample CSV saved to {CANONICAL_SAMPLE_CSV}.")

elapsed = time.time() - start_time
print(f"\nStage 2 Canonical Schema completed in {elapsed:.2f} seconds.")
