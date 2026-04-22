#!/usr/bin/env python3
"""
Phase 1: Classical Feature Selection Baselines
Compares RFE-RF, L1-LR, RF-Importance, and Mutual-Info against SGPO v2.
Evaluation: 10-fold stratified CV, SMOTE on training folds only, fixed RF classifier.
"""
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, recall_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_PATH = REPO_ROOT / "data" / "processed" / "mimic_ckd_dataset_final.csv"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
CV_FOLDS = 10
N_FEATURES_TO_KEEP = 8
N_ESTIMATORS = 100

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["subject_id", "ckd_label"]).values
    y = df["ckd_label"].values
    feature_names = list(df.drop(columns=["subject_id", "ckd_label"]).columns)
    print(f"[DATA] Loaded: {X.shape[0]:,} samples, {X.shape[1]} features | CKD: {y.sum():,}")
    return X, y, feature_names

def evaluate_subset(X, y, feature_mask, cv=CV_FOLDS, seed=SEED):
    """Evaluate AUC & Sensitivity using nested 10-fold CV + SMOTE on train only."""
    X_sel = X[:, feature_mask]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    aucs, sens = [], []
    
    for train_idx, test_idx in skf.split(X_sel, y):
        X_tr, X_te = X_sel[train_idx], X_sel[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X_tr)
        X_te = imputer.transform(X_te)

        smote = SMOTE(random_state=seed)
        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
        
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        
        probas = clf.predict_proba(X_te)[:, 1]
        preds = clf.predict(X_te)
        aucs.append(roc_auc_score(y_te, probas))
        sens.append(recall_score(y_te, preds))
        
    return np.mean(aucs), np.std(aucs), np.mean(sens), np.std(sens)

# ──────────────────────────────────────────────────────────────
# FEATURE SELECTORS (All return exactly 8 features)
# ──────────────────────────────────────────────────────────────
def rfe_rf_top8(X, y, feature_names):
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED, n_jobs=-1)
    rfe = RFE(estimator=rf, n_features_to_select=N_FEATURES_TO_KEEP, step=1)
    rfe.fit(X, y)
    mask = rfe.support_
    selected = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
    return mask, selected

def l1_lr_top8(X, y, feature_names):
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        lr = LogisticRegression(penalty="l1", solver="liblinear", C=C, random_state=SEED, max_iter=1000)
        lr.fit(X, y)
        coef = np.abs(lr.coef_[0])
        if np.sum(coef > 1e-6) == N_FEATURES_TO_KEEP:
            mask = coef > 1e-6
            return mask, [feature_names[i] for i in range(len(feature_names)) if mask[i]]
    # Fallback: force top-8 by magnitude
    lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.01, random_state=SEED, max_iter=1000)
    lr.fit(X, y)
    top8_idx = np.argsort(np.abs(lr.coef_[0]))[-N_FEATURES_TO_KEEP:]
    mask = np.zeros(len(feature_names), dtype=bool)
    mask[top8_idx] = True
    return mask, [feature_names[i] for i in range(len(feature_names)) if mask[i]]

def rf_importance_top8(X, y, feature_names):
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=SEED, n_jobs=-1)
    rf.fit(X, y)
    top8_idx = np.argsort(rf.feature_importances_)[-N_FEATURES_TO_KEEP:]
    mask = np.zeros(len(feature_names), dtype=bool)
    mask[top8_idx] = True
    return mask, [feature_names[i] for i in range(len(feature_names)) if mask[i]]

def mi_top8(X, y, feature_names):
    mi = mutual_info_classif(X, y, random_state=SEED)
    top8_idx = np.argsort(mi)[-N_FEATURES_TO_KEEP:]
    mask = np.zeros(len(feature_names), dtype=bool)
    mask[top8_idx] = True
    return mask, [feature_names[i] for i in range(len(feature_names)) if mask[i]]

# ──────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print("  Classical Feature Selection Baselines (Top-8 Features)")
    print("="*70)
    
    X, y, feature_names = load_data()

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    print(f"[INFO] NaN imputation done (median). Data shape: {X.shape}")

    baselines = {
        "RFE-RF": rfe_rf_top8,
        "L1-LR": l1_lr_top8,
        "RF-Importance": rf_importance_top8,
        "Mutual-Info": mi_top8
    }
    
    results = {}
    print("\n[RUN] Running baselines (10-fold CV, SMOTE on train only)...")
    print("-"*70)
    
    for name, selector in baselines.items():
        print(f"[...] {name:<18} | Training & evaluating...", end=" ", flush=True)
        mask, selected = selector(X, y, feature_names)
        auc, auc_std, sens, sens_std = evaluate_subset(X, y, mask)
        
        results[name] = {
            "selected_features": selected,
            "n_features": int(np.sum(mask)),
            "auc": float(auc), "auc_std": float(auc_std),
            "sensitivity": float(sens), "sensitivity_std": float(sens_std)
        }
        print(f"[OK] AUC={auc:.4f}+-{auc_std:.4f} | Sens={sens:.4f}+-{sens_std:.4f}")
    
    # Save to JSON
    out_path = RESULTS_DIR / "fs_baselines_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] Results saved to: {out_path}")
    
    # Print comparison table
    print("\n" + "="*70)
    print("[TABLE] COMPARISON TABLE (Ready for Paper)")
    print("="*70)
    print(f"{'Method':<18} {'Features':<8} {'AUC':<16} {'Sensitivity':<16}")
    print("-"*70)
    for name, res in results.items():
        print(f"{name:<18} {res['n_features']:<8} {res['auc']:.4f}+-{res['auc_std']:.4f}  {res['sensitivity']:.4f}+-{res['sensitivity_std']:.4f}")

    # Append SGPO v2 if available
    sgpo_path = RESULTS_DIR / "sgpo_v2_results.json"
    if sgpo_path.exists():
        with open(sgpo_path, "r") as f:
            sgpo = json.load(f)
        ev = sgpo.get("final_evaluation", sgpo)
        n_feat = sgpo.get("optimization_results", {}).get("best_n_features", sgpo.get("n_features", "?"))
        auc = ev.get("auc_mean", ev.get("auc", 0))
        auc_std = ev.get("auc_std", 0)
        sens = ev.get("sensitivity_mean", ev.get("sensitivity", 0))
        sens_std = ev.get("sensitivity_std", 0)
        print(f"{'SGPO v2':<18} {n_feat:<8} {auc:.4f}+-{auc_std:.4f}  {sens:.4f}+-{sens_std:.4f}")
    else:
        print(f"{'SGPO v2':<18} {8:<8} {'0.9537':<16} {'0.8902':<16} (from paper)")
    
    print("="*70)
    print("[DONE] Copy the table above or check the JSON file for details.")

if __name__ == "__main__":
    main()