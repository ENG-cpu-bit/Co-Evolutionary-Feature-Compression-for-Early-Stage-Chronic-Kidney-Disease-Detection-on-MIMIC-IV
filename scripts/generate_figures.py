"""
generate_figures.py
===================
Generates two publication-ready figures:
  Fig 1: Classical FS baselines vs SGPO v2 AUC bar chart
  Fig 2: Ablation convergence curves (real per-gen fitness from equal-budget runs)
"""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
with open(RESULTS / "fs_baselines_results.json") as f:
    baselines = json.load(f)

with open(RESULTS / "sgpo_v2_results.json") as f:
    sgpo_raw = json.load(f)

sgpo_auc     = sgpo_raw["final_evaluation"]["auc_mean"]
sgpo_auc_std = sgpo_raw["final_evaluation"]["auc_std"]
sgpo_best_fitness = sgpo_raw["optimization_results"]["best_fitness"]  # 0.7056

# ── Figure 1: Baseline AUC comparison ─────────────────────────────────────
method_order   = ["L1-LR", "Mutual-Info", "RFE-RF", "RF-Importance", "SGPO v2"]
auc_vals       = [baselines["L1-LR"]["auc"],
                  baselines["Mutual-Info"]["auc"],
                  baselines["RFE-RF"]["auc"],
                  baselines["RF-Importance"]["auc"],
                  sgpo_auc]
auc_stds       = [baselines["L1-LR"]["auc_std"],
                  baselines["Mutual-Info"]["auc_std"],
                  baselines["RFE-RF"]["auc_std"],
                  baselines["RF-Importance"]["auc_std"],
                  sgpo_auc_std]
bar_colors     = ["#777777"] * 4 + ["#0072BD"]
edge_colors    = ["#444444"] * 4 + ["red"]
edge_widths    = [1.0] * 4 + [2.5]

fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
x = range(len(method_order))
bars = ax1.bar(x, auc_vals, color=bar_colors,
               edgecolor=edge_colors, linewidth=edge_widths,
               yerr=auc_stds, capsize=4,
               error_kw={"elinewidth": 1.5, "ecolor": "#333333"})

ax1.annotate(
    f"Best: {sgpo_auc:.4f}",
    xy=(4, sgpo_auc + sgpo_auc_std + 0.0003),
    ha="center", va="bottom", fontsize=8.5, color="red", fontweight="bold"
)

ax1.set_xticks(list(x))
ax1.set_xticklabels(method_order, rotation=20, ha="right", fontsize=8)
ax1.set_ylabel("AUC-ROC (Mean +/- Std)", fontsize=9)
ax1.set_title("Classical FS vs. SGPO v2 (8 Features, 10-fold CV)", fontsize=10, fontweight="bold")
ax1.set_ylim(0.93, 0.96)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
fig1.tight_layout()
out1 = FIGURES / "fig_baseline_comparison_final.png"
fig1.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig1)
print(f"[OK] Figure 1 saved: {out1}")

# ── Figure 2: Convergence curves (real data from ablation_run.log) ─────────
# Per-generation best fitness extracted from equal-budget ablation run
no_fgo = [
    0.66521, 0.67092, 0.67487, 0.67499, 0.67975, 0.68049, 0.68246, 0.68246,
    0.68726, 0.69171, 0.69249, 0.69249, 0.69249, 0.69249, 0.69622, 0.69675,
    0.69675, 0.69758, 0.70097, 0.70097, 0.70103, 0.70103, 0.70103, 0.70175,
    0.70344, 0.70350, 0.70350, 0.70350, 0.70350, 0.70350,
]
no_doa = [
    0.66355, 0.66401, 0.67485, 0.67485, 0.67609, 0.67647, 0.68106, 0.68750,
    0.68750, 0.68750, 0.68800, 0.68800, 0.68800, 0.68800, 0.68800, 0.68800,
    0.68800, 0.69152, 0.69152, 0.69162, 0.69162, 0.69187, 0.69187, 0.69187,
    0.69187, 0.69187, 0.69187, 0.69187, 0.69187, 0.69187,
]
sfoa_only = [
    0.66355, 0.66996, 0.67442, 0.67442, 0.67863, 0.67894, 0.68183, 0.68204,
    0.68513, 0.68966, 0.69010, 0.69010, 0.69010, 0.69010, 0.69447, 0.69518,
    0.69518, 0.69518, 0.69821, 0.69821, 0.69821, 0.69836, 0.69836, 0.69836,
    0.70022, 0.70022, 0.70022, 0.70022, 0.70022, 0.70022,
]
gens = list(range(1, 31))

fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=300)
ax2.plot(gens, no_fgo,    label="no-fgo  (SFOA+DOA)",  color="#D95319", linewidth=1.8, linestyle="--")
ax2.plot(gens, no_doa,    label="no-doa  (SFOA+FGO)",  color="#EDB120", linewidth=1.8, linestyle="-.")
ax2.plot(gens, sfoa_only, label="sfoa-only",            color="#7E2F8E", linewidth=1.8, linestyle=":")
ax2.axhline(sgpo_best_fitness, color="#0072BD", linewidth=2.0,
            linestyle="-", label=f"Full SGPO v2 (final={sgpo_best_fitness:.4f})")

ax2.set_xlabel("Generation", fontsize=9)
ax2.set_ylabel("Best Fitness (inner CV)", fontsize=9)
ax2.set_title("Convergence: Equal Budget Ablation (gen=30, pop=10)", fontsize=10, fontweight="bold")
ax2.legend(fontsize=7.5, frameon=True, framealpha=0.8)
ax2.grid(alpha=0.3, linestyle="--")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
fig2.tight_layout()
out2 = FIGURES / "fig_convergence_final.png"
fig2.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"[OK] Figure 2 saved: {out2}")
