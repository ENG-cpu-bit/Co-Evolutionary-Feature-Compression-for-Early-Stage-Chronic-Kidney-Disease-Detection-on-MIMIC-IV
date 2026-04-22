"""
build_mimic_ckd_dataset.py
==========================
Builds a single, Colab-ready CKD dataset from raw MIMIC-IV files.

Usage:
    python scripts/build_mimic_ckd_dataset.py

Input:  raw MIMIC-IV CSVs in  data/raw/mimic/  (or --raw-dir)
Output: data/processed/mimic_ckd_dataset_final.csv
        data/processed/mimic_ckd_dataset_metadata.json
        data/processed/cohort_summary.csv
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42

# ICD codes that define CKD
CKD_ICD10_PREFIX = "N18"   # ICD-10
CKD_ICD9_PREFIX  = "585"   # ICD-9

# Lab features: itemid -> column name
# Selected based on clinical relevance to CKD detection
LAB_FEATURES = {
    # --- Core kidney function ---
    50912: "creatinine",
    51006: "urea_nitrogen",
    50920: "egfr",                  # Estimated GFR (MDRD)
    51007: "uric_acid",

    # --- Electrolytes ---
    50971: "potassium",
    50983: "sodium",
    50882: "bicarbonate",
    50893: "calcium_total",
    50902: "chloride",
    50960: "magnesium",

    # --- CBC / Hematology ---
    51222: "hemoglobin",
    51221: "hematocrit",
    51265: "platelet_count",
    51301: "wbc",
    51279: "rbc",
    51250: "mcv",
    51248: "mch",
    51249: "mchc",

    # --- Metabolic ---
    50931: "glucose",
    50862: "albumin",
    50885: "bilirubin_total",
    50863: "alkaline_phosphatase",
    50861: "alt",
    50878: "ast",

    # --- Iron / Anemia markers ---
    50952: "iron",
    50924: "ferritin",

    # --- Other CKD-relevant ---
    50953: "iron_binding_capacity",
    50809: "glucose_blood_gas",
}

# Admission features to extract
ADMISSION_COLS = [
    "admission_type", "admission_location", "insurance",
    "marital_status", "race", "hospital_expire_flag",
]

# Chunk size for reading labevents (memory-safe)
LABEVENTS_CHUNK_SIZE = 500_000


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_csv(path, name):
    """Load a CSV, trying .csv then .csv.gz."""
    csv_path = path / f"{name}.csv"
    gz_path  = path / f"{name}.csv.gz"

    if csv_path.exists():
        log(f"Loading {csv_path.name} ...")
        return pd.read_csv(csv_path)
    elif gz_path.exists():
        log(f"Loading {gz_path.name} (compressed) ...")
        return pd.read_csv(gz_path, compression="gzip")
    else:
        print(f"ERROR: Cannot find {name}.csv or {name}.csv.gz in {path}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD RAW TABLES
# ═══════════════════════════════════════════════════════════════════════════

def load_raw_tables(raw_dir):
    """Load patients, admissions, diagnoses_icd, d_labitems."""
    patients   = load_csv(raw_dir, "patients")
    admissions = load_csv(raw_dir, "admissions")
    diagnoses  = load_csv(raw_dir, "diagnoses_icd")
    d_labitems = load_csv(raw_dir, "d_labitems")

    log(f"  patients:   {patients.shape}")
    log(f"  admissions: {admissions.shape}")
    log(f"  diagnoses:  {diagnoses.shape}")
    log(f"  d_labitems: {d_labitems.shape}")

    return patients, admissions, diagnoses, d_labitems


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD CKD COHORT
# ═══════════════════════════════════════════════════════════════════════════

def build_ckd_cohort(diagnoses):
    """Identify CKD and non-CKD subjects from diagnosis codes."""
    log("Building CKD cohort ...")

    diagnoses["icd_code"] = diagnoses["icd_code"].astype(str).str.strip()
    diagnoses["icd_version"] = diagnoses["icd_version"].astype(int)

    # ICD-10: N18*
    icd10_mask = (
        (diagnoses["icd_version"] == 10) &
        diagnoses["icd_code"].str.startswith(CKD_ICD10_PREFIX, na=False)
    )
    # ICD-9: 585*
    icd9_mask = (
        (diagnoses["icd_version"] == 9) &
        diagnoses["icd_code"].str.startswith(CKD_ICD9_PREFIX, na=False)
    )

    ckd_rows = diagnoses[icd10_mask | icd9_mask]
    ckd_subject_ids = ckd_rows["subject_id"].unique()

    # Control group: subjects in diagnoses but NOT CKD
    all_subject_ids = diagnoses["subject_id"].unique()
    non_ckd_pool = np.setdiff1d(all_subject_ids, ckd_subject_ids)

    # Balanced control group (same size as CKD)
    rng = np.random.RandomState(RANDOM_SEED)
    n_control = min(len(ckd_subject_ids), len(non_ckd_pool))
    non_ckd_subject_ids = rng.choice(non_ckd_pool, size=n_control, replace=False)

    log(f"  CKD subjects:     {len(ckd_subject_ids):,}")
    log(f"  Non-CKD subjects: {len(non_ckd_subject_ids):,}")
    log(f"  Total cohort:     {len(ckd_subject_ids) + len(non_ckd_subject_ids):,}")

    return ckd_subject_ids, non_ckd_subject_ids


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3a — DEMOGRAPHIC FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def build_demographics(patients, selected_ids):
    """Extract age and gender for selected subjects."""
    log("Building demographic features ...")

    demo = patients[patients["subject_id"].isin(selected_ids)].copy()
    demo = demo[["subject_id", "anchor_age", "gender"]].copy()
    demo = demo.rename(columns={"anchor_age": "age"})
    demo["gender"] = (demo["gender"] == "M").astype(int)  # 1=Male, 0=Female
    demo = demo.drop_duplicates(subset="subject_id")

    log(f"  Demographics: {demo.shape}")
    return demo


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3b — ADMISSION FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def build_admission_features(admissions, selected_ids):
    """Extract admission-level features aggregated per patient."""
    log("Building admission features ...")

    adm = admissions[admissions["subject_id"].isin(selected_ids)].copy()

    # --- Length of stay ---
    adm["admittime"] = pd.to_datetime(adm["admittime"], errors="coerce")
    adm["dischtime"] = pd.to_datetime(adm["dischtime"], errors="coerce")
    adm["los_days"] = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400

    # --- Numeric aggregation per patient ---
    agg = adm.groupby("subject_id").agg(
        n_admissions        = ("hadm_id", "count"),
        avg_los_days        = ("los_days", "mean"),
        max_los_days        = ("los_days", "max"),
        hospital_expire_flag= ("hospital_expire_flag", "max"),
        n_emergency         = ("admission_type", lambda x: (x == "EMERGENCY").sum()),
        n_urgent            = ("admission_type", lambda x: (x == "URGENT").sum()),
        n_elective          = ("admission_type", lambda x: (x.str.contains("ELECTIVE", na=False)).sum()),
    ).reset_index()

    # --- Most common categorical values per patient (mode) ---
    def safe_mode(s):
        m = s.mode()
        return m.iloc[0] if len(m) > 0 else "UNKNOWN"

    cat_agg = adm.groupby("subject_id").agg(
        insurance      = ("insurance", safe_mode),
        marital_status = ("marital_status", safe_mode),
        race           = ("race", safe_mode),
    ).reset_index()

    # --- One-hot encode categoricals ---
    # Insurance
    insurance_dummies = pd.get_dummies(cat_agg["insurance"], prefix="ins").astype(int)
    # Marital status - simplify
    cat_agg["marital_status"] = cat_agg["marital_status"].fillna("UNKNOWN")
    marital_dummies = pd.get_dummies(cat_agg["marital_status"], prefix="marital").astype(int)

    cat_agg = pd.concat([cat_agg[["subject_id"]], insurance_dummies, marital_dummies], axis=1)

    # Merge numeric + categorical
    adm_features = agg.merge(cat_agg, on="subject_id", how="left")

    log(f"  Admission features: {adm_features.shape}")
    return adm_features


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3c — LAB FEATURES (MEMORY-SAFE CHUNKED PROCESSING)
# ═══════════════════════════════════════════════════════════════════════════

def build_lab_features(raw_dir, selected_ids):
    """
    Process labevents in chunks to extract lab features per patient.
    Memory-safe: never loads the full labevents file into RAM.
    """
    log("Building lab features (chunked processing) ...")

    # Determine file path
    csv_path = raw_dir / "labevents.csv"
    gz_path  = raw_dir / "labevents.csv.gz"

    if csv_path.exists():
        lab_path = csv_path
        compression = None
    elif gz_path.exists():
        lab_path = gz_path
        compression = "gzip"
    else:
        print("ERROR: Cannot find labevents.csv or labevents.csv.gz")
        sys.exit(1)

    target_itemids = set(LAB_FEATURES.keys())
    selected_ids_set = set(selected_ids)

    # We'll accumulate filtered rows, then aggregate at the end
    filtered_chunks = []
    total_rows_read = 0
    total_rows_kept = 0

    reader = pd.read_csv(
        lab_path,
        compression=compression,
        chunksize=LABEVENTS_CHUNK_SIZE,
        usecols=["subject_id", "itemid", "valuenum"],
        dtype={"subject_id": int, "itemid": int, "valuenum": float},
        low_memory=False,
    )

    for chunk_num, chunk in enumerate(reader):
        total_rows_read += len(chunk)

        # Filter: only our subjects AND our lab items AND non-null values
        mask = (
            chunk["subject_id"].isin(selected_ids_set) &
            chunk["itemid"].isin(target_itemids) &
            chunk["valuenum"].notna()
        )
        filtered = chunk[mask].copy()
        total_rows_kept += len(filtered)

        if len(filtered) > 0:
            filtered_chunks.append(filtered)

        if (chunk_num + 1) % 20 == 0:
            log(f"  ... processed {total_rows_read:,} rows, kept {total_rows_kept:,}")

    log(f"  Total rows read: {total_rows_read:,}")
    log(f"  Total rows kept: {total_rows_kept:,}")

    if not filtered_chunks:
        log("  WARNING: No lab data found for selected subjects!")
        return pd.DataFrame({"subject_id": selected_ids})

    # Combine all filtered chunks
    labs_all = pd.concat(filtered_chunks, ignore_index=True)

    # Map itemid to feature name
    labs_all["feature"] = labs_all["itemid"].map(LAB_FEATURES)

    # Aggregate per patient: median value for each lab
    labs_pivot = (
        labs_all
        .groupby(["subject_id", "feature"])["valuenum"]
        .median()
        .unstack()
        .reset_index()
    )

    log(f"  Lab features: {labs_pivot.shape}")
    log(f"  Lab columns:  {list(labs_pivot.columns[1:])}")

    return labs_pivot


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — MERGE & BUILD FINAL DATASET
# ═══════════════════════════════════════════════════════════════════════════

def build_final_dataset(demo, adm_features, lab_features, ckd_ids, non_ckd_ids):
    """Merge all features and add CKD label."""
    log("Merging all features into final dataset ...")

    # Start with demographics
    df = demo.copy()

    # Merge admission features
    df = df.merge(adm_features, on="subject_id", how="left")

    # Merge lab features
    df = df.merge(lab_features, on="subject_id", how="left")

    # Add CKD label
    df["ckd_label"] = df["subject_id"].isin(ckd_ids).astype(int)

    # Drop columns with >50% missing values (unusable for ML)
    feature_cols = [c for c in df.columns if c not in ["subject_id", "ckd_label"]]
    null_rates = df[feature_cols].isnull().mean()
    high_null_cols = null_rates[null_rates > 0.50].index.tolist()
    if high_null_cols:
        log(f"  Dropping {len(high_null_cols)} columns with >50% nulls: {high_null_cols}")
        df = df.drop(columns=high_null_cols)

    # Drop rows that are almost entirely empty (keep if at least 50% non-null)
    feature_cols = [c for c in df.columns if c not in ["subject_id", "ckd_label"]]
    min_non_null = max(3, int(len(feature_cols) * 0.5))
    before = len(df)
    df = df.dropna(subset=feature_cols, thresh=min_non_null)
    after = len(df)
    if before != after:
        log(f"  Dropped {before - after} rows with too many missing values")

    log(f"  Final shape: {df.shape}")
    log(f"  CKD count:   {df['ckd_label'].sum():,}")
    log(f"  Non-CKD:     {(df['ckd_label'] == 0).sum():,}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════

def save_outputs(df, output_dir, raw_dir):
    """Save final CSV, metadata JSON, and cohort summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Main dataset ---
    csv_path = output_dir / "mimic_ckd_dataset_final.csv"
    df.to_csv(csv_path, index=False)
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    log(f"Saved: {csv_path}  ({file_size_mb:.1f} MB)")

    # --- Metadata ---
    feature_cols = [c for c in df.columns if c not in ["subject_id", "ckd_label"]]
    metadata = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "feature_count": int(len(feature_cols)),
        "feature_names": feature_cols,
        "ckd_count": int(df["ckd_label"].sum()),
        "non_ckd_count": int((df["ckd_label"] == 0).sum()),
        "source_files": [
            "patients.csv", "admissions.csv", "diagnoses_icd.csv",
            "d_labitems.csv", "labevents.csv"
        ],
        "random_seed": RANDOM_SEED,
        "generation_timestamp": datetime.now().isoformat(),
        "file_size_mb": round(file_size_mb, 2),
        "lab_itemids_used": {str(k): v for k, v in LAB_FEATURES.items()},
    }
    meta_path = output_dir / "mimic_ckd_dataset_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"Saved: {meta_path}")

    # --- Cohort summary ---
    summary = pd.DataFrame([{
        "total_subjects": len(df),
        "ckd_subjects": int(df["ckd_label"].sum()),
        "non_ckd_subjects": int((df["ckd_label"] == 0).sum()),
        "final_rows": len(df),
        "final_columns": len(df.columns),
        "feature_columns": len(feature_cols),
        "output_file_size_mb": round(file_size_mb, 2),
    }])
    summary_path = output_dir / "cohort_summary.csv"
    summary.to_csv(summary_path, index=False)
    log(f"Saved: {summary_path}")

    return metadata


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build MIMIC-IV CKD dataset")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Path to raw MIMIC files (default: data/raw/mimic/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/processed/)"
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    repo_root  = script_dir.parent

    raw_dir    = Path(args.raw_dir) if args.raw_dir else repo_root / "data" / "raw" / "mimic"
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "data" / "processed"

    log("=" * 65)
    log("  MIMIC-IV CKD Dataset Builder")
    log("=" * 65)
    log(f"Raw dir:    {raw_dir}")
    log(f"Output dir: {output_dir}")
    log("")

    start_time = time.time()

    # ── Step 1: Load raw tables ──
    patients, admissions, diagnoses, d_labitems = load_raw_tables(raw_dir)

    # ── Step 2: Build CKD cohort ──
    ckd_ids, non_ckd_ids = build_ckd_cohort(diagnoses)
    all_selected_ids = np.concatenate([ckd_ids, non_ckd_ids])

    # ── Step 3a: Demographics ──
    demo = build_demographics(patients, all_selected_ids)

    # ── Step 3b: Admission features ──
    adm_features = build_admission_features(admissions, all_selected_ids)

    # ── Step 3c: Lab features (chunked) ──
    lab_features = build_lab_features(raw_dir, all_selected_ids)

    # ── Step 4: Merge everything ──
    df = build_final_dataset(demo, adm_features, lab_features, ckd_ids, non_ckd_ids)

    # ── Step 5: Save ──
    metadata = save_outputs(df, output_dir, raw_dir)

    elapsed = time.time() - start_time
    log("")
    log("=" * 65)
    log("  DONE!")
    log("=" * 65)
    log(f"  Rows:     {metadata['row_count']:,}")
    log(f"  Features: {metadata['feature_count']}")
    log(f"  CKD:      {metadata['ckd_count']:,}")
    log(f"  Non-CKD:  {metadata['non_ckd_count']:,}")
    log(f"  Size:     {metadata['file_size_mb']:.1f} MB")
    log(f"  Time:     {elapsed:.0f} seconds")
    log("")
    log("  Upload this file to Colab:")
    log(f"  --> {output_dir / 'mimic_ckd_dataset_final.csv'}")
    log("")


if __name__ == "__main__":
    main()
