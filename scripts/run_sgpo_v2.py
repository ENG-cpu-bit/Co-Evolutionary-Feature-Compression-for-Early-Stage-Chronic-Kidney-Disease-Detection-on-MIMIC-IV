"""
run_sgpo_v2.py
==============
Run SGPO v2 co-evolutionary optimization on the MIMIC-IV CKD dataset.
Performs simultaneous feature selection + hyperparameter tuning.

Usage:
    python scripts/run_sgpo_v2.py
    python scripts/run_sgpo_v2.py --generations 50 --sfoa-pop 12 --doa-pop 12
    python scripts/run_sgpo_v2.py --quick   (fast test run)
"""

import argparse
import json
import sys
import time
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add project root to path
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from src.optimizers.sgpo_v2 import SGPOv2


def final_evaluation(X, y, best_mask, best_hp, n_outer_folds=10):
    """
    Final evaluation using OUTER nested CV (never seen by optimizer).
    This is the honest, unbiased performance estimate.
    """
    print("\n" + "=" * 60)
    print("  Final Evaluation — 10-fold Outer CV (unseen by optimizer)")
    print("=" * 60)

    selected_idx = np.where(np.array(best_mask) == 1)[0]
    X_sel = X.iloc[:, selected_idx]

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=best_hp.get("n_estimators", 100),
            max_depth=best_hp.get("max_depth", None),
            min_samples_split=best_hp.get("min_samples_split", 2),
            min_samples_leaf=best_hp.get("min_samples_leaf", 1),
            random_state=42,
            n_jobs=-1,
        )),
    ])

    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=42)

    auc_scores = []
    acc_scores = []
    sens_scores = []
    spec_scores = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_sel, y)):
        X_train, X_test = X_sel.iloc[train_idx], X_sel.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc_scores.append(accuracy_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_prob))
        sens_scores.append(recall_score(y_test, y_pred))

        # Specificity = TN / (TN + FP)
        tn = ((y_test == 0) & (y_pred == 0)).sum()
        fp = ((y_test == 0) & (y_pred == 1)).sum()
        spec = tn / max(1, tn + fp)
        spec_scores.append(spec)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    print(f"\n  Accuracy:    {np.mean(acc_scores):.4f} (+/- {np.std(acc_scores):.4f})")
    print(f"  AUC-ROC:     {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")
    print(f"  Sensitivity: {np.mean(sens_scores):.4f} (+/- {np.std(sens_scores):.4f})")
    print(f"  Specificity: {np.mean(spec_scores):.4f} (+/- {np.std(spec_scores):.4f})")

    print(f"\n  Confusion Matrix (aggregated):")
    print(confusion_matrix(all_y_true, all_y_pred))

    print(f"\n  Classification Report:")
    print(classification_report(all_y_true, all_y_pred, target_names=["Non-CKD", "CKD"]))

    return {
        "accuracy_mean": round(float(np.mean(acc_scores)), 4),
        "accuracy_std": round(float(np.std(acc_scores)), 4),
        "auc_mean": round(float(np.mean(auc_scores)), 4),
        "auc_std": round(float(np.std(auc_scores)), 4),
        "sensitivity_mean": round(float(np.mean(sens_scores)), 4),
        "sensitivity_std": round(float(np.std(sens_scores)), 4),
        "specificity_mean": round(float(np.mean(spec_scores)), 4),
        "specificity_std": round(float(np.std(spec_scores)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Run SGPO v2 on MIMIC-IV CKD dataset")
    parser.add_argument("--generations", type=int, default=30, help="Number of generations")
    parser.add_argument("--sfoa-pop", type=int, default=10, help="SFOA population size")
    parser.add_argument("--doa-pop", type=int, default=10, help="DOA population size")
    parser.add_argument("--inner-folds", type=int, default=3, help="Inner CV folds")
    parser.add_argument("--outer-folds", type=int, default=10, help="Outer CV folds for final eval")
    parser.add_argument("--strategy", type=str, default="diagonal",
                        choices=["full", "diagonal"],
                        help="Evaluation strategy (full=NxM, diagonal=faster)")
    parser.add_argument("--smote", action="store_true", default=True, help="Use SMOTE")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run (5 gen, pop 5)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Use a random sample of N rows (for testing)")
    args = parser.parse_args()

    # Quick mode override
    if args.quick:
        args.generations = 5
        args.sfoa_pop = 5
        args.doa_pop = 5
        args.sample = 5000

    use_smote = args.smote and not args.no_smote

    # Load dataset
    data_path = repo_root / "data" / "processed" / "mimic_ckd_dataset_final.csv"
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset ...")
    df = pd.read_csv(data_path)

    if args.sample:
        print(f"Sampling {args.sample} rows for faster execution ...")
        df = df.sample(n=min(args.sample, len(df)), random_state=42).reset_index(drop=True)

    X = df.drop(columns=["subject_id", "ckd_label"])
    y = df["ckd_label"]

    print(f"Dataset: {len(df):,} rows, {X.shape[1]} features")
    print(f"CKD: {y.sum():,} | Non-CKD: {(y == 0).sum():,}")

    # Run SGPO v2
    optimizer = SGPOv2(
        n_features=X.shape[1],
        sfoa_pop_size=args.sfoa_pop,
        doa_pop_size=args.doa_pop,
        n_generations=args.generations,
        n_inner_folds=args.inner_folds,
        use_smote=use_smote,
        sample_strategy=args.strategy,
        random_state=42,
        verbose=True,
    )

    results = optimizer.run(X, y)

    # Final evaluation with outer CV
    final_metrics = final_evaluation(
        X, y,
        results["best_mask"],
        results["best_hp"],
        n_outer_folds=args.outer_folds,
    )

    # Save all results
    output = {
        "optimizer": "SGPO_v2",
        "components": ["SFOA", "DOA", "FungalGrowth"],
        "config": {
            "sfoa_pop_size": args.sfoa_pop,
            "doa_pop_size": args.doa_pop,
            "n_generations": args.generations,
            "inner_cv_folds": args.inner_folds,
            "outer_cv_folds": args.outer_folds,
            "strategy": args.strategy,
            "smote": use_smote,
            "dataset_rows": len(df),
            "dataset_features": X.shape[1],
        },
        "optimization_results": {
            "best_fitness": results["best_fitness"],
            "best_auc": results["best_auc"],
            "best_sensitivity": results["best_sensitivity"],
            "best_n_features": results["best_n_features"],
            "selected_features": results["selected_features"],
            "best_hyperparameters": results["best_hp"],
            "total_time_seconds": results["total_time_seconds"],
        },
        "final_evaluation": final_metrics,
        "timestamp": datetime.now().isoformat(),
    }

    # Save main results
    results_path = results_dir / "sgpo_v2_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Save convergence history
    history_df = pd.DataFrame(results["history"])
    history_path = results_dir / "convergence_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Convergence history saved: {history_path}")

    # Save selected features
    features_path = results_dir / "best_features.json"
    with open(features_path, "w") as f:
        json.dump({
            "selected_features": results["selected_features"],
            "feature_mask": results["best_mask"],
            "n_selected": results["best_n_features"],
            "n_total": X.shape[1],
        }, f, indent=2)
    print(f"Best features saved: {features_path}")

    # Save best hyperparameters
    hp_path = results_dir / "best_hyperparameters.json"
    with open(hp_path, "w") as f:
        json.dump(results["best_hp"], f, indent=2)
    print(f"Best hyperparameters saved: {hp_path}")

    # Print comparison with baseline
    print("\n" + "=" * 60)
    print("  COMPARISON: Baseline vs SGPO v2")
    print("=" * 60)

    baseline_path = results_dir / "baseline_rf_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        print(f"{'Metric':<20} {'Baseline':>12} {'SGPO v2':>12} {'Change':>12}")
        print("-" * 56)
        print(f"{'AUC-ROC':<20} {baseline['cv_auc_mean']:>12.4f} {final_metrics['auc_mean']:>12.4f} {final_metrics['auc_mean'] - baseline['cv_auc_mean']:>+12.4f}")
        print(f"{'Sensitivity':<20} {baseline['cv_sensitivity_mean']:>12.4f} {final_metrics['sensitivity_mean']:>12.4f} {final_metrics['sensitivity_mean'] - baseline['cv_sensitivity_mean']:>+12.4f}")
        print(f"{'Accuracy':<20} {baseline['cv_accuracy_mean']:>12.4f} {final_metrics['accuracy_mean']:>12.4f} {final_metrics['accuracy_mean'] - baseline['cv_accuracy_mean']:>+12.4f}")
        print(f"{'Features':<20} {baseline['dataset_features']:>12d} {results['best_n_features']:>12d} {results['best_n_features'] - baseline['dataset_features']:>+12d}")
    else:
        print("  (baseline results not found for comparison)")

    print("\nDone!")


if __name__ == "__main__":
    main()
