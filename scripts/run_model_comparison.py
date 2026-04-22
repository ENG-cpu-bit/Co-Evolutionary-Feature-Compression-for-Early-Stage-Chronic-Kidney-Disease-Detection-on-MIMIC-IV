"""
run_model_comparison.py
=======================
Compare multiple classifiers on the MIMIC-IV CKD dataset using
the same evaluation protocol: 10-fold stratified CV.

Models compared:
  1. Logistic Regression (all 42 features)
  2. Random Forest baseline (all 42 features)
  3. XGBoost (all 42 features)
  4. SGPO v2 optimized RF (8 selected features, tuned HP)

Outputs:
  - results/model_comparison_results.json
  - results/tables/model_comparison.csv
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

sys.stdout.reconfigure(encoding="utf-8")

repo_root = Path(__file__).resolve().parent.parent
data_path = repo_root / "data" / "processed" / "mimic_ckd_dataset_final.csv"
results_dir = repo_root / "results"
tables_dir = results_dir / "tables"
tables_dir.mkdir(parents=True, exist_ok=True)


def evaluate_model(name, pipeline, X, y, n_folds=10, seed=42):
    """Run n-fold stratified CV and return metrics dict."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    acc, auc, sens, spec, f1 = [], [], [], [], []
    all_y_true, all_y_pred = [], []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)
        y_prob = pipeline.predict_proba(X_te)[:, 1]

        acc.append(accuracy_score(y_te, y_pred))
        auc.append(roc_auc_score(y_te, y_prob))
        sens.append(recall_score(y_te, y_pred))
        tn = ((y_te == 0) & (y_pred == 0)).sum()
        fp = ((y_te == 0) & (y_pred == 1)).sum()
        spec.append(tn / max(1, tn + fp))
        f1.append(f1_score(y_te, y_pred))

        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())

    return {
        "model": name,
        "accuracy_mean": round(np.mean(acc), 4),
        "accuracy_std": round(np.std(acc), 4),
        "auc_mean": round(np.mean(auc), 4),
        "auc_std": round(np.std(auc), 4),
        "sensitivity_mean": round(np.mean(sens), 4),
        "sensitivity_std": round(np.std(sens), 4),
        "specificity_mean": round(np.mean(spec), 4),
        "specificity_std": round(np.std(spec), 4),
        "f1_mean": round(np.mean(f1), 4),
        "f1_std": round(np.std(f1), 4),
    }


def main():
    print("Loading dataset ...")
    df = pd.read_csv(data_path)
    X_all = df.drop(columns=["subject_id", "ckd_label"])
    y = df["ckd_label"]
    print(f"Dataset: {len(df):,} rows, {X_all.shape[1]} features")

    # Load SGPO v2 best solution
    sgpo_results_path = results_dir / "sgpo_v2_results.json"
    with open(sgpo_results_path) as f:
        sgpo = json.load(f)

    sgpo_features = sgpo["optimization_results"]["selected_features"]
    sgpo_hp = sgpo["optimization_results"]["best_hyperparameters"]
    X_sgpo = X_all[sgpo_features]

    # Define models
    models = []

    # 1. Logistic Regression
    models.append((
        "Logistic Regression",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        X_all,
        42,
    ))

    # 2. Random Forest baseline
    models.append((
        "Random Forest (baseline)",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]),
        X_all,
        42,
    ))

    # 3. XGBoost
    if HAS_XGB:
        models.append((
            "XGBoost",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    random_state=42, n_jobs=-1, eval_metric="logloss",
                    verbosity=0,
                )),
            ]),
            X_all,
            42,
        ))
    else:
        # Fallback: Gradient Boosting
        models.append((
            "Gradient Boosting",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    random_state=42,
                )),
            ]),
            X_all,
            42,
        ))

    # 4. SGPO v2 optimized RF
    models.append((
        "SGPO v2 (optimized RF)",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=sgpo_hp["n_estimators"],
                max_depth=sgpo_hp["max_depth"],
                min_samples_split=sgpo_hp["min_samples_split"],
                min_samples_leaf=sgpo_hp["min_samples_leaf"],
                random_state=42, n_jobs=-1,
            )),
        ]),
        X_sgpo,
        len(sgpo_features),
    ))

    # Run evaluations
    all_results = []
    print(f"\nRunning 10-fold CV for {len(models)} models ...\n")

    for name, pipe, X_data, n_feat in models:
        print(f"  Evaluating: {name} ({n_feat} features) ...", end=" ", flush=True)
        metrics = evaluate_model(name, pipe, X_data, y)
        metrics["n_features"] = n_feat
        all_results.append(metrics)
        print(f"AUC={metrics['auc_mean']:.4f}, Sens={metrics['sensitivity_mean']:.4f}")

    # Print comparison table
    print("\n" + "=" * 90)
    print("  MODEL COMPARISON — 10-fold Stratified CV")
    print("=" * 90)
    header = f"{'Model':<28} {'Feat':>4} {'Accuracy':>12} {'AUC-ROC':>12} {'Sensitivity':>12} {'F1':>12}"
    print(header)
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['model']:<28} {r['n_features']:>4} "
            f"{r['accuracy_mean']:.4f}+/-{r['accuracy_std']:.4f} "
            f"{r['auc_mean']:.4f}+/-{r['auc_std']:.4f} "
            f"{r['sensitivity_mean']:.4f}+/-{r['sensitivity_std']:.4f} "
            f"{r['f1_mean']:.4f}+/-{r['f1_std']:.4f}"
        )

    # Save JSON
    output = {
        "experiment": "model_comparison",
        "cv_folds": 10,
        "dataset_rows": len(df),
        "models": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    json_path = results_dir / "model_comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Save CSV table
    table_df = pd.DataFrame(all_results)
    csv_path = tables_dir / "model_comparison.csv"
    table_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
