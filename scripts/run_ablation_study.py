"""
run_ablation_study.py
=====================
Ablation study for SGPO v2: isolate the contribution of each component.

Variants tested:
  1. Full SGPO v2 (SFOA + DOA + FGO) — use saved results
  2. No Fungal Growth — SFOA + DOA only, no perturbation
  3. No DOA — SFOA + default HP, no HP tuning
  4. No SFOA — all features + DOA HP tuning only
  5. Feature selection only — SFOA + default HP, no DOA, no FGO
  6. HP tuning only — all features + DOA, no SFOA, no FGO

Each ablation runs a reduced SGPO v2 with 15 generations, pop 8.
The full SGPO v2 result is loaded from saved outputs.

Outputs:
  - results/ablation_results.json
  - results/tables/ablation_results.csv
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

from src.optimizers.sfoa import SFOA
from src.optimizers.doa import DOA
from src.optimizers.fungal_growth import FungalGrowthOptimizer
from src.evaluation.fitness import evaluate_solution

data_path = repo_root / "data" / "processed" / "mimic_ckd_dataset_final.csv"
results_dir = repo_root / "results"
tables_dir = results_dir / "tables"
tables_dir.mkdir(parents=True, exist_ok=True)

# Ablation config — shorter runs to keep runtime reasonable
ABL_GENS = 15
ABL_POP = 8
ABL_INNER_FOLDS = 3
ABL_SEED = 42
DEFAULT_HP = {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}


def final_eval(X, y, mask, hp, n_folds=10, seed=42):
    """10-fold outer CV evaluation."""
    selected_idx = np.where(np.array(mask) == 1)[0]
    if len(selected_idx) < 2:
        return {"accuracy_mean": 0, "auc_mean": 0, "sensitivity_mean": 0, "specificity_mean": 0}

    X_sel = X.iloc[:, selected_idx]
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", None),
            min_samples_split=hp.get("min_samples_split", 2),
            min_samples_leaf=hp.get("min_samples_leaf", 1),
            random_state=seed, n_jobs=-1,
        )),
    ])

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    acc, auc_s, sens, spec = [], [], [], []

    for train_idx, test_idx in cv.split(X_sel, y):
        X_tr, X_te = X_sel.iloc[train_idx], X_sel.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        acc.append(accuracy_score(y_te, y_pred))
        auc_s.append(roc_auc_score(y_te, y_prob))
        sens.append(recall_score(y_te, y_pred))
        tn = ((y_te == 0) & (y_pred == 0)).sum()
        fp = ((y_te == 0) & (y_pred == 1)).sum()
        spec.append(tn / max(1, tn + fp))

    return {
        "accuracy_mean": round(np.mean(acc), 4),
        "accuracy_std": round(np.std(acc), 4),
        "auc_mean": round(np.mean(auc_s), 4),
        "auc_std": round(np.std(auc_s), 4),
        "sensitivity_mean": round(np.mean(sens), 4),
        "sensitivity_std": round(np.std(sens), 4),
        "specificity_mean": round(np.mean(spec), 4),
        "specificity_std": round(np.std(spec), 4),
    }


def run_ablation_variant(X, y, name, use_sfoa=True, use_doa=True, use_fgo=True):
    """
    Run an ablation variant of SGPO v2.

    When use_sfoa=False: all features are used (mask = all 1s)
    When use_doa=False: default HP are used
    When use_fgo=False: no fungal growth perturbation
    """
    n_features = X.shape[1]
    start_time = time.time()

    # Initialize components
    sfoa = SFOA(n_features=n_features, pop_size=ABL_POP, random_state=ABL_SEED) if use_sfoa else None
    doa = DOA(pop_size=ABL_POP, random_state=ABL_SEED) if use_doa else None
    fgo = FungalGrowthOptimizer(stagnation_threshold=3, random_state=ABL_SEED) if use_fgo else None

    global_best_fitness = -np.inf
    global_best_mask = None
    global_best_hp = None

    for gen in range(ABL_GENS):
        # Get populations
        if use_sfoa:
            masks = sfoa.get_population()
        else:
            masks = [np.ones(n_features, dtype=int)]

        if use_doa:
            hps = doa.get_hp_dicts()
        else:
            hps = [DEFAULT_HP]

        # Evaluate pairs (diagonal-like)
        sfoa_best_fit = np.full(len(masks), -np.inf)
        doa_best_fit = np.full(len(hps), -np.inf)

        for i in range(len(masks)):
            # Pair each mask with one random HP + best HP
            j_indices = [i % len(hps)]
            if use_doa and doa.g_best_fit > -np.inf:
                best_j = int(np.argmax(doa.fitness))
                if best_j not in j_indices:
                    j_indices.append(best_j)

            for j in j_indices:
                fitness, auc_val, sens_val, n_feat = evaluate_solution(
                    X, y, masks[i], hps[j],
                    n_inner_folds=ABL_INNER_FOLDS,
                    use_smote=True,
                    random_state=ABL_SEED + gen,
                )

                if fitness > sfoa_best_fit[i]:
                    sfoa_best_fit[i] = fitness

                if fitness > doa_best_fit[j]:
                    doa_best_fit[j] = fitness

                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_mask = masks[i].copy()
                    global_best_hp = hps[j].copy()

        # Update fitness
        if use_sfoa:
            for i in range(len(masks)):
                if sfoa_best_fit[i] > -np.inf:
                    sfoa.update_fitness(i, sfoa_best_fit[i])

        if use_doa:
            for j in range(len(hps)):
                if doa_best_fit[j] > -np.inf:
                    doa.update_fitness(j, doa_best_fit[j])

        # Apply FGO
        if use_fgo and use_sfoa and use_doa:
            fgo.apply(sfoa, doa, global_best_fitness)
        elif use_fgo and use_sfoa:
            # FGO on SFOA only
            is_stag = fgo.check_stagnation(global_best_fitness)
            if is_stag:
                sfoa.population, _ = fgo.spore_dispersal_binary(sfoa.population, n_features)
                fgo.stagnation_count = 0
            else:
                sfoa.population = fgo.hyphal_growth_binary(sfoa.population)
        elif use_fgo and use_doa:
            # FGO on DOA only
            is_stag = fgo.check_stagnation(global_best_fitness)
            if is_stag:
                doa.positions, _ = fgo.spore_dispersal_continuous(doa.positions)
                fgo.stagnation_count = 0
            else:
                doa.positions = fgo.hyphal_growth_continuous(doa.positions)

        # Evolve
        if use_sfoa:
            sfoa.evolve()
        if use_doa:
            doa.evolve(gen, ABL_GENS)

        print(f"    Gen {gen:2d} | fitness={global_best_fitness:.5f} | feat={int(global_best_mask.sum())}", flush=True)

    runtime = time.time() - start_time

    # Final evaluation
    n_feat = int(global_best_mask.sum())
    feature_names = list(X.columns)
    selected = [feature_names[i] for i in range(n_features) if global_best_mask[i] == 1]

    print(f"    Optimization done ({runtime:.0f}s). Running 10-fold outer CV ...")
    metrics = final_eval(X, y, global_best_mask.tolist(), global_best_hp)

    return {
        "variant": name,
        "use_sfoa": use_sfoa,
        "use_doa": use_doa,
        "use_fgo": use_fgo,
        "best_fitness": round(global_best_fitness, 5),
        "n_features": n_feat,
        "selected_features": selected,
        "best_hp": global_best_hp,
        "runtime_seconds": round(runtime, 1),
        **metrics,
    }


def main():
    print("Loading dataset ...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=["subject_id", "ckd_label"])
    y = df["ckd_label"]
    print(f"Dataset: {len(df):,} rows, {X.shape[1]} features\n")

    # Load full SGPO v2 results (no need to rerun)
    sgpo_path = results_dir / "sgpo_v2_results.json"
    with open(sgpo_path) as f:
        sgpo = json.load(f)

    full_sgpo = {
        "variant": "Full SGPO v2 (SFOA+DOA+FGO)",
        "use_sfoa": True,
        "use_doa": True,
        "use_fgo": True,
        "best_fitness": sgpo["optimization_results"]["best_fitness"],
        "n_features": sgpo["optimization_results"]["best_n_features"],
        "selected_features": sgpo["optimization_results"]["selected_features"],
        "best_hp": sgpo["optimization_results"]["best_hyperparameters"],
        "runtime_seconds": sgpo["optimization_results"]["total_time_seconds"],
        "accuracy_mean": sgpo["final_evaluation"]["accuracy_mean"],
        "accuracy_std": sgpo["final_evaluation"]["accuracy_std"],
        "auc_mean": sgpo["final_evaluation"]["auc_mean"],
        "auc_std": sgpo["final_evaluation"]["auc_std"],
        "sensitivity_mean": sgpo["final_evaluation"]["sensitivity_mean"],
        "sensitivity_std": sgpo["final_evaluation"]["sensitivity_std"],
        "specificity_mean": sgpo["final_evaluation"]["specificity_mean"],
        "specificity_std": sgpo["final_evaluation"]["specificity_std"],
    }

    all_results = [full_sgpo]

    # Ablation variants
    variants = [
        ("No FGO (SFOA+DOA only)", True, True, False),
        ("No DOA (SFOA+FGO, default HP)", True, False, True),
        ("No SFOA (all features, DOA+FGO)", False, True, True),
        ("Feature selection only (SFOA)", True, False, False),
        ("HP tuning only (DOA)", False, True, False),
    ]

    for name, use_sfoa, use_doa, use_fgo in variants:
        print(f"Running: {name}")
        result = run_ablation_variant(X, y, name, use_sfoa, use_doa, use_fgo)
        all_results.append(result)
        print(f"  -> AUC={result['auc_mean']:.4f}, Sens={result['sensitivity_mean']:.4f}, "
              f"Feat={result['n_features']}, Time={result['runtime_seconds']:.0f}s\n")

    # Print summary
    print("\n" + "=" * 100)
    print("  ABLATION STUDY RESULTS")
    print("=" * 100)
    header = f"{'Variant':<38} {'Feat':>4} {'AUC-ROC':>12} {'Sensitivity':>12} {'Accuracy':>12}"
    print(header)
    print("-" * 100)
    for r in all_results:
        auc_str = f"{r['auc_mean']:.4f}+/-{r.get('auc_std', 0):.4f}"
        sens_str = f"{r['sensitivity_mean']:.4f}+/-{r.get('sensitivity_std', 0):.4f}"
        acc_str = f"{r['accuracy_mean']:.4f}+/-{r.get('accuracy_std', 0):.4f}"
        print(f"{r['variant']:<38} {r['n_features']:>4} {auc_str:>12} {sens_str:>12} {acc_str:>12}")

    # Save JSON
    output = {
        "experiment": "ablation_study",
        "ablation_config": {
            "generations": ABL_GENS,
            "pop_size": ABL_POP,
            "inner_folds": ABL_INNER_FOLDS,
            "outer_folds": 10,
            "note": "Full SGPO v2 uses saved 30-gen results; ablations run 15-gen for efficiency",
        },
        "variants": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    json_path = results_dir / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # Save CSV
    table_rows = []
    for r in all_results:
        table_rows.append({
            "variant": r["variant"],
            "sfoa": r["use_sfoa"],
            "doa": r["use_doa"],
            "fgo": r["use_fgo"],
            "n_features": r["n_features"],
            "accuracy_mean": r["accuracy_mean"],
            "accuracy_std": r.get("accuracy_std", 0),
            "auc_mean": r["auc_mean"],
            "auc_std": r.get("auc_std", 0),
            "sensitivity_mean": r["sensitivity_mean"],
            "sensitivity_std": r.get("sensitivity_std", 0),
            "specificity_mean": r["specificity_mean"],
            "specificity_std": r.get("specificity_std", 0),
            "runtime_seconds": r["runtime_seconds"],
        })
    csv_path = tables_dir / "ablation_results.csv"
    pd.DataFrame(table_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
