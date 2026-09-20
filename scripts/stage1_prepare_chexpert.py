"""
scripts/stage1_prepare_chexpert.py
Stage 1 Dataset Preparation for CheXpert.

Streams metadata directly from danjacobellis/chexpert (split='train') on Hugging Face,
consistent with the verified raw image extraction workflow.

Extracts:
  - Path -> canonical image_path at R:\\FedMed_Data\\chexpert_jpg\\{Path}
  - patient_id (e.g., patient00001)
  - study_id (e.g., study1)
  - Age
  - Sex
  - Frontal/Lateral (view_type)
  - AP/PA (projection)
  - Pneumonia (raw label: 0=Unlabeled, 1=Uncertain, 2=Absent, 3=Present)

Outputs:
  data_prep/chexpert_prepared_meta.parquet
  data_prep/chexpert_prepared_meta_sample.csv
"""
import os
import sys
import re
import time
import pandas as pd
from datasets import load_dataset

DATA_ROOT = r"R:\FedMed_Data\chexpert_jpg"
OUTPUT_DIR = r"r:\VSCODE\Main_Project\data_prep"
OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "chexpert_prepared_meta.parquet")
OUTPUT_SAMPLE_CSV = os.path.join(OUTPUT_DIR, "chexpert_prepared_meta_sample.csv")
LOG_INTERVAL = 20000

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("STAGE 1: CHEXPERT DATASET PREPARATION")
print("=" * 60)
print(f"Verified image root: {DATA_ROOT}")
print(f"Target metadata output: {OUTPUT_PARQUET}")

start_time = time.time()
records = []
total_count = 0
missing_images = 0
frontal_count = 0
lateral_count = 0
unknown_view_count = 0

# Mapping definitions for clarity
# In danjacobellis/chexpert:
# Sex: 1=Male, 2=Female (or string), let's inspect actual values
# Frontal/Lateral: 0=Frontal, 1=Lateral (or vice versa, let's extract and preserve raw + string)

print("\nConnecting to stream: danjacobellis/chexpert (split='train')...")
try:
    # Stream without decoding images for efficiency
    dataset = load_dataset("danjacobellis/chexpert", split="train", streaming=True).remove_columns(["image"])
    print("Stream connected. Extracting metadata records...")

    for sample in dataset:
        total_count += 1
        raw_path = sample.get("Path", "")
        
        # 1. Path & file presence check
        canonical_image_path = os.path.normpath(os.path.join(DATA_ROOT, raw_path)) if raw_path else ""
        
        # 2. Extract patient_id and study_id from Path
        # Format: CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
        parts = raw_path.replace("\\", "/").split("/")
        patient_id = "unknown"
        study_id = "unknown"
        for p in parts:
            if p.startswith("patient"):
                patient_id = p
            elif p.startswith("study"):
                study_id = p
                
        # 3. View Type (Frontal vs Lateral)
        # Check raw Frontal/Lateral column or filename
        raw_fl = sample.get("Frontal/Lateral")
        # In CheXpert filename: 'view1_frontal.jpg' or 'view1_lateral.jpg'
        if "frontal" in raw_path.lower():
            view_type = "Frontal"
            frontal_count += 1
        elif "lateral" in raw_path.lower():
            view_type = "Lateral"
            lateral_count += 1
        else:
            view_type = "Unknown"
            unknown_view_count += 1

        # 4. Projection (AP / PA)
        raw_ap_pa = sample.get("AP/PA")
        # In danjacobellis/chexpert, AP/PA is encoded (e.g. 0=AP, 1=PA or vice versa)
        # Filenames or metadata
        projection = str(raw_ap_pa)

        # 5. Demographics
        age = sample.get("Age")
        sex = sample.get("Sex")
        
        # 6. Pneumonia observation label (raw: 0, 1, 2, 3)
        raw_pneumonia = sample.get("Pneumonia")

        records.append({
            "sample_id": f"chexpert_{total_count}",
            "dataset_source": "CheXpert",
            "patient_id": patient_id,
            "study_id": study_id,
            "raw_path": raw_path,
            "image_path": canonical_image_path,
            "age": age,
            "sex": sex,
            "view_type": view_type,
            "raw_fl_code": raw_fl,
            "projection": projection,
            "raw_pneumonia": raw_pneumonia
        })

        if total_count % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            rate = total_count / elapsed if elapsed > 0 else 0
            print(f"Processed {total_count} records ({frontal_count} Frontal, {lateral_count} Lateral) | Rate: {rate:.1f} rec/s", flush=True)

    elapsed_stream = time.time() - start_time
    print(f"\nStreaming completed: {total_count} records in {elapsed_stream/60:.2f} minutes.")

    # Convert to DataFrame
    df_chexpert = pd.DataFrame(records)

    # Statistical Inspection
    print("\n--- STATISTICAL INSPECTION ---")
    print(f"Total CheXpert records: {len(df_chexpert)}")
    print(f"Unique patients: {df_chexpert['patient_id'].nunique()}")
    print(f"Unique studies: {df_chexpert['study_id'].nunique()}")
    print(f"View distribution: Frontal={frontal_count}, Lateral={lateral_count}, Unknown={unknown_view_count}")
    
    # Age stats
    df_chexpert["age"] = pd.to_numeric(df_chexpert["age"], errors="coerce")
    age_desc = df_chexpert["age"].describe().to_dict()
    print(f"Age summary: min={age_desc['min']}, max={age_desc['max']}, mean={age_desc['mean']:.2f}, median={df_chexpert['age'].median()}")
    invalid_age = ((df_chexpert["age"] < 0) | (df_chexpert["age"] > 105)).sum()
    missing_age = df_chexpert["age"].isna().sum()
    print(f"Age anomalies (< 0 or > 105): {invalid_age}, Missing age: {missing_age}")

    # Sex distribution
    print(f"Sex distribution: {df_chexpert['sex'].value_counts(dropna=False).to_dict()}")

    # Pneumonia label distribution
    pneu_counts = df_chexpert["raw_pneumonia"].value_counts(dropna=False).to_dict()
    print(f"Raw Pneumonia encoding distribution: {pneu_counts}")
    print("  Note: 0 = Unlabeled, 1 = Uncertain, 2 = Absent, 3 = Present")

    # Save Intermediate Parquet
    print(f"\nSaving intermediate metadata to {OUTPUT_PARQUET}...")
    df_chexpert.to_parquet(OUTPUT_PARQUET, index=False)
    print("Saved successfully.")

    # Save Sample CSV
    df_chexpert.head(50).to_csv(OUTPUT_SAMPLE_CSV, index=False)
    print(f"Sample CSV saved to {OUTPUT_SAMPLE_CSV}.")

    total_time = time.time() - start_time
    print(f"\nStage 1 CheXpert preparation complete in {total_time/60:.2f} minutes.")

except Exception as e:
    print(f"Fatal error during CheXpert preparation: {e}", file=sys.stderr)
    sys.exit(1)
