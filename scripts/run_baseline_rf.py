"""
run_baseline_rf.py
==================
Baseline Random Forest evaluation on the processed MIMIC-IV CKD dataset.
Validates the dataset before running SGPO v2 optimization.

Usage:
    python scripts/run_baseline_rf.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    data_path = repo_root / "data" / "processed" / "mimic_ckd_dataset_final.csv"
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Baseline Random Forest — MIMIC-IV CKD Dataset")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(data_path)
    print(f"\nDataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Separate features and label
    X = df.drop(columns=["subject_id", "ckd_label"])
    y = df["ckd_label"]

    print(f"Features: {X.shape[1]}")
    print(f"CKD: {y.sum():,} | Non-CKD: {(y == 0).sum():,}")

    # Pipeline: impute + scale + RF
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # 5-fold stratified CV
    print("\nRunning 5-fold stratified cross-validation ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    sens_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="recall", n_jobs=-1)

    print(f"\nAccuracy:    {acc_scores.mean():.4f} (+/- {acc_scores.std():.4f})")
    print(f"AUC-ROC:     {auc_scores.mean():.4f} (+/- {auc_scores.std():.4f})")
    print(f"Sensitivity: {sens_scores.mean():.4f} (+/- {sens_scores.std():.4f})")

    # Full train + evaluate for detailed report
    print("\nTraining on full dataset for detailed metrics ...")
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    print(f"\nFull-data Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"Full-data AUC-ROC:   {roc_auc_score(y, y_prob):.4f}")
    print(f"Full-data Sensitivity: {recall_score(y, y_pred):.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Non-CKD", "CKD"]))

    # Feature importance
    importances = pipeline.named_steps["classifier"].feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    print("Top 15 features by importance:")
    for feat, imp in feat_imp.head(15).items():
        print(f"  {feat:30s} {imp:.4f}")

    # Save results
    results = {
        "model": "RandomForest (baseline)",
        "n_estimators": 100,
        "cv_folds": 5,
        "cv_accuracy_mean": round(float(acc_scores.mean()), 4),
        "cv_accuracy_std": round(float(acc_scores.std()), 4),
        "cv_auc_mean": round(float(auc_scores.mean()), 4),
        "cv_auc_std": round(float(auc_scores.std()), 4),
        "cv_sensitivity_mean": round(float(sens_scores.mean()), 4),
        "cv_sensitivity_std": round(float(sens_scores.std()), 4),
        "dataset_rows": int(len(df)),
        "dataset_features": int(X.shape[1]),
        "top_features": dict(feat_imp.head(15).round(4)),
        "timestamp": datetime.now().isoformat(),
    }

    results_path = results_dir / "baseline_rf_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
