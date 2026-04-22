# Co-Evolutionary Feature Selection and Hyperparameter Optimization for Early-Stage CKD Detection

> **Simultaneous Group-based Pareto Optimization v2** — a hybrid nature-inspired framework for clinical machine learning on large-scale MIMIC-IV data.

---

## Abstract

Chronic Kidney Disease (CKD) affects over 800 million people globally, yet early-stage detection remains challenging due to high-dimensional, noisy clinical data. We present **SGPO v2**, a co-evolutionary optimization framework that simultaneously performs feature selection and hyperparameter tuning using three 2025 nature-inspired algorithms: Starfish Optimization Algorithm (SFOA), Dream Optimization Algorithm (DOA), and Fungal Growth Optimizer (FGO).

Applied to 57,875 MIMIC-IV patients, SGPO v2 reduces 42 clinical features to **8 features** (81% reduction) while preserving **99.75% of the baseline AUC-ROC** (0.9537 vs. 0.9561). Critically, only **1 lab test** (serum creatinine) is required alongside 7 readily available administrative features — enabling low-cost early CKD screening.

---

## Key Results

| Metric | Baseline RF (42 features) | SGPO v2 (8 features) | Change |
|---|---|---|---|
| AUC-ROC | 0.9561 | 0.9537 | −0.25% |
| Sensitivity | 0.8904 | 0.8902 | −0.02% |
| Accuracy | 0.8897 | 0.8867 | −0.34% |
| Features | 42 | **8** | **−81%** |

**Selected features:** `creatinine`, `age`, `n_admissions`, `avg_los_days`, `ins_UNKNOWN`, `marital_SINGLE`, `marital_UNKNOWN`, `marital_WIDOWED`

**Optimized hyperparameters:** `n_estimators=278`, `max_depth=15`, `min_samples_split=3`, `min_samples_leaf=8`

---

## SGPO v2 Framework

Three optimizers run **co-evolutionarily in parallel** each generation, sharing fitness feedback:

| Component | Algorithm | Role |
|---|---|---|
| Feature Selection | SFOA (Starfish Optimization) | Binary mask over 42 features |
| HP Tuning | DOA (Dream Optimization) | Continuous HP parameter search |
| Noise Handling | FGO (Fungal Growth Optimizer) | Perturbation to escape local optima |

**Fitness Function:**
```
Fitness = 0.50 × AUC − 0.20 × (|selected_features| / 42) + 0.30 × Sensitivity
```

**Validation:** Nested CV — outer 10-fold (test never seen by optimizer), inner 3-fold (fitness evaluation). SMOTE applied to training folds only.

---

## Installation

```bash
git clone https://github.com/<your-username>/SGPO-v2-Clinical-ML-Repo.git
cd SGPO-v2-Clinical-ML-Repo
pip install -r requirements.txt
```

**Python 3.9+** required.

---

## Dataset

This repository does **not** include the raw dataset. Access requires MIMIC-IV approval:

1. Complete CITI training at [physionet.org](https://physionet.org/content/mimiciv/)
2. Download MIMIC-IV v2.2
3. Place files in `data/raw/`
4. Run `python scripts/build_mimic_ckd_dataset.py` to generate `data/processed/mimic_ckd_dataset_final.csv`

The processed dataset contains 57,875 patients × 42 features (10.9 MB).

---

## Usage

### 1. Run Baseline Random Forest
```bash
python scripts/run_baseline_rf.py
# Output: results/baseline_rf_results.json
```

### 2. Run SGPO v2 Optimization
```bash
python scripts/run_sgpo_v2.py
# Output: results/sgpo_v2_results.json, results/best_features.json, results/best_hyperparameters.json
```

### 3. Run Model Comparison (4 classifiers)
```bash
python scripts/run_model_comparison.py
# Output: results/model_comparison_results.json
```

### 4. Run Ablation Study (6 variants)
```bash
python scripts/run_ablation_study.py
# Output: results/ablation_results.json
```

### 5. Generate Publication Figures
```bash
python scripts/generate_figures.py
# Output: results/figures/ (7 publication-ready figures)
```

### 6. Analysis Notebooks
Open notebooks in order:
```
notebooks/01_baseline_analysis.ipynb       # EDA + baseline
notebooks/02_sgpo_results.ipynb            # SGPO v2 convergence & selection
notebooks/03_model_comparison.ipynb        # 4-model comparison
notebooks/04_ablation_study.ipynb          # Component ablation
notebooks/05_full_summary.ipynb            # Complete results summary
```

---

## Repository Structure

```
SGPO-v2-Clinical-ML-Repo/
├── src/
│   ├── __init__.py
│   ├── optimizers/
│   │   ├── sfoa.py              # Starfish Optimization Algorithm
│   │   ├── doa.py               # Dream Optimization Algorithm
│   │   ├── fungal_growth.py     # Fungal Growth Optimizer
│   │   └── sgpo_v2.py           # Co-evolutionary SGPO v2 framework
│   └── evaluation/
│       └── fitness.py           # Shared fitness function
├── scripts/
│   ├── build_mimic_ckd_dataset.py
│   ├── run_baseline_rf.py
│   ├── run_sgpo_v2.py
│   ├── run_model_comparison.py
│   ├── run_ablation_study.py
│   ├── 16_run_fs_baselines.py
│   └── generate_figures.py
├── notebooks/
│   ├── 01_baseline_analysis.ipynb
│   ├── 02_sgpo_results.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_ablation_study.ipynb
│   └── 05_full_summary.ipynb
├── paper/
│   ├── paper.md                 # Full IEEE-format manuscript
│   └── references/              # 6 reference PDFs
├── results/
│   ├── figures/                 # 15 publication-ready figures
│   ├── tables/                  # CSV comparison tables
│   └── *.json                   # Machine-readable experiment results
├── data/
│   ├── raw/                     # Place MIMIC-IV files here (not tracked)
│   └── processed/               # Generated dataset (not tracked)
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Ablation Study Summary

| Variant | Features | AUC-ROC | Sensitivity |
|---|---|---|---|
| **Full SGPO v2** | **8** | **0.9537** | **0.8902** |
| No FGO | 9 | 0.9541 | 0.8949 |
| No DOA | 12 | 0.9502 | 0.8841 |
| No SFOA | 42 | 0.9573 | 0.8946 |
| SFOA only | 7 | 0.9480 | 0.8848 |
| DOA only | 42 | 0.9573 | 0.8946 |

---

## Citation

If you use this work, please cite:

```bibtex
@article{sgpo_v2_ckd_2025,
  title   = {Co-Evolutionary Feature Selection and Hyperparameter Optimization
             for Early-Stage Chronic Kidney Disease Detection Using MIMIC-IV},
  author  = {Mustafa, Muhammed Abdel-Hamid Shawki and Al-Bili, Abdul Rahman Ismat},
  year    = {2025},
  institution = {New Mansoura University, Computer Science \& Engineering, CSE015},
  note    = {Supervised by Dr. Ibrahim}
}
```

---

## Authors

- **Muhammed Abdel-Hamid Shawki Mustafa** (224100686) — New Mansoura University
- **Abdul Rahman Al-Bili Ismat Al-Bili** (222101372) — New Mansoura University
- **Supervisor:** Dr. Ibrahim | Department of Computer Science & Engineering | CSE015

---

## License

This project is released for academic and research use. The MIMIC-IV dataset is subject to its own [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciv/view-license/2.2/).
