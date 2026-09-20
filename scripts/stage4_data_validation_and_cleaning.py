"""
scripts/stage4_data_validation_and_cleaning.py
Stage 4 Data Validation & Cleaning.

Performs the 8 essential data quality and clinical validation checks:
  1. Image Validity & Traceability
  2. Patient & Study Identifiers
  3. Image-Clinical-Label Pairing Alignment
  4. Label Correctness (strictly 0/1)
  5. Duplicate Records & Images
  6. Missing Values & Demographic Anomalies (filters Age > 105 anomalies)
  7. Class Balance Analysis
  8. Patient-Level Grouping Integrity

Outputs:
  data_prep/stage4_cleaned_metadata.parquet
  data_prep/stage4_cleaned_sample.csv
  data_prep/stage4_cleaning_report.json
"""
import os
import sys
import json
import time
import pandas as pd
import numpy as np

DATA_PREP_DIR = r"r:\VSCODE\Main_Project\data_prep"
INPUT_PARQUET = os.path.join(DATA_PREP_DIR, "stage3_filtered_metadata.parquet")
OUTPUT_CLEANED_PARQUET = os.path.join(DATA_PREP_DIR, "stage4_cleaned_metadata.parquet")
OUTPUT_CLEANED_CSV = os.path.join(DATA_PREP_DIR, "stage4_cleaned_sample.csv")
OUTPUT_REPORT_JSON = os.path.join(DATA_PREP_DIR, "stage4_cleaning_report.json")

print("=" * 60)
print("STAGE 4: DATA VALIDATION & CLEANING (8 CHECKS)")
print("=" * 60)

start_time = time.time()

if not os.path.exists(INPUT_PARQUET):
    print(f"Error: {INPUT_PARQUET} does not exist!", file=sys.stderr)
    sys.exit(1)

print(f"Loading Stage 3 filtered metadata from {INPUT_PARQUET}...")
df = pd.read_parquet(INPUT_PARQUET)
initial_count = len(df)
print(f"Initial retained records: {initial_count:,}")

validation_report = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "initial_record_count": initial_count,
    "checks": {}
}

# -------------------------------------------------------------
# Check 1: Image Validity & Traceability
# -------------------------------------------------------------
print("\n--- Check 1: Image Validity & Traceability ---")
empty_paths = df["image_path"].isna().sum() + (df["image_path"].str.strip() == "").sum()
valid_extensions = df["image_path"].apply(lambda p: p.lower().endswith((".png", ".jpg", ".jpeg"))).sum()
invalid_extensions = initial_count - valid_extensions

# Verify local CheXpert images on disk
chex_df = df[df["dataset_source"] == "CheXpert"]
chex_files_exist = chex_df["image_path"].apply(os.path.isfile).sum()
chex_missing = len(chex_df) - chex_files_exist

print(f"  Empty image paths: {empty_paths}")
print(f"  Valid image extensions (.png, .jpg, .jpeg): {valid_extensions:,} / {initial_count:,}")
print(f"  CheXpert images verified on disk: {chex_files_exist:,} / {len(chex_df):,} (Missing: {chex_missing})")

validation_report["checks"]["check_1_image_validity"] = {
    "empty_paths": int(empty_paths),
    "valid_extensions": int(valid_extensions),
    "invalid_extensions": int(invalid_extensions),
    "chexpert_images_on_disk": int(chex_files_exist),
    "chexpert_missing_on_disk": int(chex_missing),
    "status": "PASS" if empty_paths == 0 and invalid_extensions == 0 and chex_missing == 0 else "FAIL"
}

# -------------------------------------------------------------
# Check 2: Patient and Study Identifiers
# -------------------------------------------------------------
print("\n--- Check 2: Patient & Study Identifiers ---")
null_patient_ids = df["patient_id"].isna().sum() + (df["patient_id"].str.strip() == "").sum()
null_study_ids = df["study_id"].isna().sum() + (df["study_id"].str.strip() == "").sum()
unique_patients_initial = df["patient_id"].nunique()
unique_studies_initial = df["study_id"].nunique()

print(f"  Empty / null patient_id count: {null_patient_ids}")
print(f"  Empty / null study_id count: {null_study_ids}")
print(f"  Unique patient IDs: {unique_patients_initial:,}")
print(f"  Unique study IDs: {unique_studies_initial:,}")

validation_report["checks"]["check_2_identifiers"] = {
    "null_patient_ids": int(null_patient_ids),
    "null_study_ids": int(null_study_ids),
    "unique_patients": int(unique_patients_initial),
    "unique_studies": int(unique_studies_initial),
    "status": "PASS" if null_patient_ids == 0 and null_study_ids == 0 else "FAIL"
}

# -------------------------------------------------------------
# Check 3: Image-Clinical Pairing Alignment
# -------------------------------------------------------------
print("\n--- Check 3: Image-Clinical Pairing Alignment ---")
total_rows = len(df)
aligned_rows = ((~df["image_path"].isna()) & (~df["age"].isna()) & (~df["sex"].isna()) & (~df["binary_label"].isna())).sum()
pairing_discrepancies = total_rows - aligned_rows
print(f"  Complete 1:1 image-clinical-label pairs: {aligned_rows:,} / {total_rows:,}")
print(f"  Pairing discrepancies: {pairing_discrepancies}")

validation_report["checks"]["check_3_pairing_alignment"] = {
    "aligned_rows": int(aligned_rows),
    "discrepancies": int(pairing_discrepancies),
    "status": "PASS" if pairing_discrepancies == 0 else "FAIL"
}

# -------------------------------------------------------------
# Check 4: Label Correctness
# -------------------------------------------------------------
print("\n--- Check 4: Label Correctness ---")
label_set = set(df["binary_label"].unique())
invalid_labels = [x for x in label_set if x not in (0, 1)]
print(f"  Unique binary labels present: {label_set}")
print(f"  Invalid labels count: {len(invalid_labels)}")

validation_report["checks"]["check_4_label_correctness"] = {
    "unique_labels": list(map(int, label_set)),
    "invalid_labels": invalid_labels,
    "status": "PASS" if len(invalid_labels) == 0 and label_set.issubset({0, 1}) else "FAIL"
}

# -------------------------------------------------------------
# Check 5: Duplicate Records / Images
# -------------------------------------------------------------
print("\n--- Check 5: Duplicate Records & Images ---")
dup_sample_ids = df["sample_id"].duplicated().sum()
dup_image_paths = df["image_path"].duplicated().sum()
print(f"  Duplicate sample_id count: {dup_sample_ids}")
print(f"  Duplicate image_path count: {dup_image_paths}")

validation_report["checks"]["check_5_duplicates"] = {
    "duplicate_sample_ids": int(dup_sample_ids),
    "duplicate_image_paths": int(dup_image_paths),
    "status": "PASS" if dup_sample_ids == 0 and dup_image_paths == 0 else "FAIL"
}

# -------------------------------------------------------------
# Check 6: Missing Values & Demographic Anomalies (CLEANING)
# -------------------------------------------------------------
print("\n--- Check 6: Missing Values & Demographic Anomalies (Cleaning) ---")
missing_age = df["age"].isna().sum()
negative_age = (df["age"] < 0).sum()
over_105_age = (df["age"] > 105).sum()
invalid_sex = (~df["sex"].isin(["M", "F"])).sum()

print(f"  Missing age count: {missing_age}")
print(f"  Negative age count: {negative_age}")
print(f"  Age > 105 anomalies (to be removed): {over_105_age}")
print(f"  Invalid sex count (not 'M' or 'F'): {invalid_sex}")

# Apply Age Cleaning Policy: Remove records where Age > 105
age_anomaly_mask = df["age"] > 105
df_cleaned = df[~age_anomaly_mask].copy().reset_index(drop=True)
cleaned_count = len(df_cleaned)
removed_anomalies_count = initial_count - cleaned_count

print(f"\n  Removed {removed_anomalies_count} records with Age > 105.")
print(f"  Cleaned dataset count: {cleaned_count:,}")

validation_report["checks"]["check_6_demographic_cleaning"] = {
    "missing_age": int(missing_age),
    "negative_age": int(negative_age),
    "age_over_105_anomalies": int(over_105_age),
    "invalid_sex": int(invalid_sex),
    "records_removed": int(removed_anomalies_count),
    "cleaned_record_count": int(cleaned_count),
    "status": "PASS"
}

# -------------------------------------------------------------
# Check 7: Class Balance Assessment
# -------------------------------------------------------------
print("\n--- Check 7: Class Balance Assessment ---")
class_counts = df_cleaned["binary_label"].value_counts().to_dict()
pneu_clean = class_counts.get(1, 0)
non_pneu_clean = class_counts.get(0, 0)
imbalance_ratio = non_pneu_clean / pneu_clean if pneu_clean > 0 else 0.0

print(f"  Pneumonia (Class 1):     {pneu_clean:,} ({(pneu_clean/cleaned_count)*100:.2f}%)")
print(f"  Non-Pneumonia (Class 0): {non_pneu_clean:,} ({(non_pneu_clean/cleaned_count)*100:.2f}%)")
print(f"  Imbalance Ratio:         {imbalance_ratio:.2f} : 1")

validation_report["checks"]["check_7_class_balance"] = {
    "pneumonia_count": int(pneu_clean),
    "non_pneumonia_count": int(non_pneu_clean),
    "pneumonia_pct": round(float(pneu_clean / cleaned_count * 100), 2),
    "non_pneumonia_pct": round(float(non_pneu_clean / cleaned_count * 100), 2),
    "imbalance_ratio": round(float(imbalance_ratio), 2),
    "status": "PASS"
}

# -------------------------------------------------------------
# Check 8: Patient-Level Grouping Audit
# -------------------------------------------------------------
print("\n--- Check 8: Patient-Level Grouping Audit ---")
unique_patients_clean = df_cleaned["patient_id"].nunique()
patient_record_counts = df_cleaned["patient_id"].value_counts()
max_records_per_patient = patient_record_counts.max()
avg_records_per_patient = patient_record_counts.mean()

print(f"  Unique patients in clean dataset: {unique_patients_clean:,}")
print(f"  Max scans for a single patient:   {max_records_per_patient}")
print(f"  Avg scans per patient:             {avg_records_per_patient:.2f}")

validation_report["checks"]["check_8_patient_grouping"] = {
    "unique_patients": int(unique_patients_clean),
    "max_records_per_patient": int(max_records_per_patient),
    "avg_records_per_patient": round(float(avg_records_per_patient), 2),
    "status": "PASS"
}

# -------------------------------------------------------------
# Final Assertions & Output Serialization
# -------------------------------------------------------------
assert cleaned_count == 118654, f"Expected 118,654 records, got {cleaned_count:,}!"
assert df_cleaned["age"].max() <= 105.0, f"Found age > 105 after cleaning: {df_cleaned['age'].max()}"
assert df_cleaned["age"].min() >= 0.0, f"Found negative age after cleaning: {df_cleaned['age'].min()}"
assert set(df_cleaned["binary_label"].unique()) == {0, 1}, "Labels not strictly {0, 1}!"

print(f"\nSaving clean model-ready metadata to {OUTPUT_CLEANED_PARQUET}...")
df_cleaned.to_parquet(OUTPUT_CLEANED_PARQUET, index=False)
df_cleaned.head(50).to_csv(OUTPUT_CLEANED_CSV, index=False)
print("Clean metadata saved successfully.")

print(f"Saving validation report to {OUTPUT_REPORT_JSON}...")
with open(OUTPUT_REPORT_JSON, "w") as f:
    json.dump(validation_report, f, indent=2)
print("Validation report saved successfully.")

elapsed = time.time() - start_time
print(f"\nStage 4 Validation & Cleaning completed in {elapsed:.2f} seconds.")
