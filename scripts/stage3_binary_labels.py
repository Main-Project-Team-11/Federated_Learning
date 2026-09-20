"""
scripts/stage3_binary_labels.py
Stage 3 Binary Label Creation & View/Label Policy Enforcement.

Applies:
  1. NIH Label Policy:
     - Label contains 7 -> binary_label = 1 (Pneumonia)
     - Label does not contain 7 -> binary_label = 0 (Non-Pneumonia)
     - Special case: Label contains both 0 and 7 -> Flag & Exclude (NIH_Inconsistent_0_7)
  2. CheXpert View Policy:
     - Frontal -> Retain for label evaluation
     - Lateral -> Exclude (CheXpert_Lateral_View)
  3. CheXpert Label Policy (Frontal views):
     - Code 3 (Present) -> binary_label = 1 (Pneumonia)
     - Code 2 (Absent) -> binary_label = 0 (Non-Pneumonia)
     - Code 1 (Uncertain) -> Exclude (CheXpert_Pneumonia_Uncertain)
     - Code 0 (Unlabeled) -> Exclude (CheXpert_Pneumonia_Unlabeled)

Outputs:
  data_prep/stage3_filtered_metadata.parquet
  data_prep/stage3_filtered_metadata_sample.csv
  data_prep/stage3_excluded_records.parquet
  data_prep/stage3_excluded_sample.csv
"""
import os
import sys
import time
import pandas as pd
import numpy as np

DATA_PREP_DIR = r"r:\VSCODE\Main_Project\data_prep"
INPUT_PARQUET = os.path.join(DATA_PREP_DIR, "canonical_metadata.parquet")
OUTPUT_FILTERED_PARQUET = os.path.join(DATA_PREP_DIR, "stage3_filtered_metadata.parquet")
OUTPUT_FILTERED_CSV = os.path.join(DATA_PREP_DIR, "stage3_filtered_metadata_sample.csv")
OUTPUT_EXCLUDED_PARQUET = os.path.join(DATA_PREP_DIR, "stage3_excluded_records.parquet")
OUTPUT_EXCLUDED_CSV = os.path.join(DATA_PREP_DIR, "stage3_excluded_sample.csv")

print("=" * 60)
print("STAGE 3: BINARY LABEL CREATION & EXCLUSION FILTERING")
print("=" * 60)

start_time = time.time()

if not os.path.exists(INPUT_PARQUET):
    print(f"Error: {INPUT_PARQUET} does not exist!", file=sys.stderr)
    sys.exit(1)

print(f"Loading canonical metadata from {INPUT_PARQUET}...")
df = pd.read_parquet(INPUT_PARQUET)
total_raw = len(df)
print(f"Total raw canonical records loaded: {total_raw}")

# Prepare lists for retained and excluded records
retained_rows = []
excluded_rows = []

# Counters for summary
count_nih_pneu = 0
count_nih_non_pneu = 0
count_nih_inconsistent = 0

count_chex_lateral = 0
count_chex_uncertain = 0
count_chex_unlabeled = 0
count_chex_pneu = 0
count_chex_non_pneu = 0
count_other_invalid = 0

print("\nApplying NIH and CheXpert label and view policies...")

for row in df.itertuples(index=False):
    source = row.dataset_source
    row_dict = row._asdict()

    if source == "NIH":
        # Parse labels from raw_label_info (comma-separated string e.g. "2,11" or "0")
        raw_info = str(row.raw_label_info).strip()
        labels = [int(x.strip()) for x in raw_info.split(",") if x.strip().isdigit()]
        
        has_7 = 7 in labels
        has_0 = 0 in labels
        
        if has_0 and has_7:
            # Special case: Inconsistent label combination [0, 7]
            count_nih_inconsistent += 1
            row_dict["exclusion_reason"] = "NIH_Inconsistent_0_7"
            excluded_rows.append(row_dict)
        elif has_7:
            count_nih_pneu += 1
            row_dict["binary_label"] = 1
            retained_rows.append(row_dict)
        else:
            count_nih_non_pneu += 1
            row_dict["binary_label"] = 0
            retained_rows.append(row_dict)

    elif source == "CheXpert":
        # Check View Policy first: Frontal vs Lateral
        view_type = str(row.view_type).strip()
        if view_type != "Frontal":
            count_chex_lateral += 1
            row_dict["exclusion_reason"] = "CheXpert_Lateral_View"
            excluded_rows.append(row_dict)
            continue

        # For Frontal views, apply Pneumonia Label Policy
        # raw_label_info is formatted as "pneumonia_code_X"
        raw_info = str(row.raw_label_info).strip()
        code_str = raw_info.replace("pneumonia_code_", "").strip()

        if code_str == "3" or code_str == "3.0":
            # Code 3: Present -> Pneumonia (1)
            count_chex_pneu += 1
            row_dict["binary_label"] = 1
            retained_rows.append(row_dict)
        elif code_str == "2" or code_str == "2.0":
            # Code 2: Absent -> Non-Pneumonia (0)
            count_chex_non_pneu += 1
            row_dict["binary_label"] = 0
            retained_rows.append(row_dict)
        elif code_str == "1" or code_str == "1.0":
            # Code 1: Uncertain -> Exclude
            count_chex_uncertain += 1
            row_dict["exclusion_reason"] = "CheXpert_Pneumonia_Uncertain"
            excluded_rows.append(row_dict)
        elif code_str == "0" or code_str == "0.0":
            # Code 0: Unlabeled -> Exclude
            count_chex_unlabeled += 1
            row_dict["exclusion_reason"] = "CheXpert_Pneumonia_Unlabeled"
            excluded_rows.append(row_dict)
        else:
            count_other_invalid += 1
            row_dict["exclusion_reason"] = f"CheXpert_Unknown_Code_{code_str}"
            excluded_rows.append(row_dict)

    else:
        count_other_invalid += 1
        row_dict["exclusion_reason"] = f"Unknown_Dataset_{source}"
        excluded_rows.append(row_dict)

df_retained = pd.DataFrame(retained_rows)
df_excluded = pd.DataFrame(excluded_rows)

total_retained = len(df_retained)
total_excluded = len(df_excluded)
total_pneu = count_nih_pneu + count_chex_pneu
total_non_pneu = count_nih_non_pneu + count_chex_non_pneu

# -------------------------------------------------------------
# Summary Report
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STAGE 3 EXCLUSION & LABEL SUMMARY REPORT")
print("=" * 60)
print(f"Total records before filtering:        {total_raw:,}")
print(f"Total model-ready records retained:    {total_retained:,} ({(total_retained/total_raw)*100:.2f}%)")
print(f"Total records excluded:                {total_excluded:,} ({(total_excluded/total_raw)*100:.2f}%)")
print("-" * 60)
print("RETAINED CLASS BREAKDOWN:")
print(f"  Pneumonia (Class 1) retained:        {total_pneu:,}")
print(f"    - NIH Pneumonia:                   {count_nih_pneu:,}")
print(f"    - CheXpert Pneumonia (Code 3):     {count_chex_pneu:,}")
print(f"  Non-Pneumonia (Class 0) retained:    {total_non_pneu:,}")
print(f"    - NIH Non-Pneumonia:               {count_nih_non_pneu:,}")
print(f"    - CheXpert Non-Pneumonia (Code 2): {count_chex_non_pneu:,}")
print(f"  Overall Class 0 / Class 1 Ratio:     {total_non_pneu / total_pneu:.2f}:1")
print("-" * 60)
print("EXCLUDED RECORDS BREAKDOWN:")
print(f"  CheXpert Lateral views excluded:     {count_chex_lateral:,}")
print(f"  CheXpert Uncertain (Code 1) excluded:{count_chex_uncertain:,}")
print(f"  CheXpert Unlabeled (Code 0) excluded:{count_chex_unlabeled:,}")
print(f"  NIH Inconsistent [0, 7] excluded:    {count_nih_inconsistent:,}")
print(f"  Other invalid records excluded:      {count_other_invalid:,}")
print("=" * 60)

# Verifications
assert total_retained + total_excluded == total_raw, "Sanity check failed: Retained + Excluded != Total Raw!"
assert set(df_retained["binary_label"].unique()).issubset({0, 1}), f"Invalid binary labels found: {df_retained['binary_label'].unique()}"

# Save Outputs
print(f"\nSaving retained model-ready metadata to {OUTPUT_FILTERED_PARQUET}...")
df_retained.to_parquet(OUTPUT_FILTERED_PARQUET, index=False)
df_retained.head(50).to_csv(OUTPUT_FILTERED_CSV, index=False)
print("Retained metadata saved successfully.")

print(f"Saving excluded records log to {OUTPUT_EXCLUDED_PARQUET}...")
df_excluded.to_parquet(OUTPUT_EXCLUDED_PARQUET, index=False)
df_excluded.head(50).to_csv(OUTPUT_EXCLUDED_CSV, index=False)
print("Excluded records log saved successfully.")

elapsed = time.time() - start_time
print(f"\nStage 3 completed in {elapsed:.2f} seconds.")
