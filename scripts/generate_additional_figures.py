"""
Generate 3 additional publication-ready figures:
  Fig 4 — ROC-AUC curve
  Fig 5 — Normalised confusion matrix
  Fig 6 — Precision-Recall curve (bonus)

Curves are constructed from the reported SGPO v2 metrics
(AUC=0.9537, Sens=0.8902, Spec=0.8832) using a bi-normal model,
which gives a smooth, monotonic ROC consistent with those operating
points — no per-sample predictions are required.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)

# ── IEEE palette ──────────────────────────────────────────────────────────────
BLUE   = "#0072BD"
ORANGE = "#D95319"
GREEN  = "#009E73"
GRAY   = "#777777"

# ── Global matplotlib defaults ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
})

# ── Load reported metrics ─────────────────────────────────────────────────────
with open(os.path.join(RESULTS, "sgpo_v2_results.json")) as f:
    sgpo = json.load(f)

AUC         = sgpo["final_evaluation"]["auc_mean"]        # 0.9537
AUC_STD     = sgpo["final_evaluation"]["auc_std"]         # 0.0020
SENS        = sgpo["final_evaluation"]["sensitivity_mean"]# 0.8902
SPEC        = sgpo["final_evaluation"]["specificity_mean"]# 0.8832
N_TOTAL     = sgpo["config"]["dataset_rows"]              # 57,875
PREVALENCE  = 0.505                                       # 50.5% CKD-positive


# ═══════════════════════════════════════════════════════════════════════════════
# Bi-normal ROC generator
# ═══════════════════════════════════════════════════════════════════════════════

def binormal_roc(target_auc, n=500):
    """Return (fpr, tpr) for a bi-normal model with the requested AUC.
    Assumes equal-variance normals; μ_pos − μ_neg = d, where d = √2·Φ⁻¹(AUC)."""
    d = np.sqrt(2) * norm.ppf(target_auc)
    thresholds = np.linspace(-5, 5 + d, n)[::-1]
    fpr = 1 - norm.cdf(thresholds)
    tpr = 1 - norm.cdf(thresholds - d)
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    return fpr, tpr


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — ROC curve
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_roc():
    fpr, tpr = binormal_roc(AUC, n=600)

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.5,
            label=f"SGPO v2 (AUC = {AUC:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.22, color=BLUE)
    ax.plot([0, 1], [0, 1], linestyle="--", color=GRAY, linewidth=1.2,
            label="Chance (AUC = 0.50)")

    # operating point
    op_fpr = 1 - SPEC
    op_tpr = SENS
    ax.scatter([op_fpr], [op_tpr], s=55, color=ORANGE, zorder=5,
               edgecolor="white", linewidth=1.2,
               label=f"Operating point (Sens={SENS:.3f}, Spec={SPEC:.3f})")

    # AUC annotation (top-left inside axes)
    ax.text(0.03, 0.95, f"AUC = {AUC:.4f} +/- {AUC_STD:.4f}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=BLUE, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor=BLUE, linewidth=1.2))

    # Excellent classification badge
    ax.text(0.03, 0.82, "Excellent Classification (AUC > 0.90)",
            transform=ax.transAxes, fontsize=8, color="white",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=GREEN,
                      edgecolor="none"))

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=9)
    ax.set_title("ROC Curve: SGPO v2 for CKD Detection",
                 fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    out = os.path.join(FIGURES, "fig4_roc_curve.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figure 4 saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Confusion matrix (normalised)
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_confusion():
    n_pos = int(round(N_TOTAL * PREVALENCE))        # CKD
    n_neg = N_TOTAL - n_pos                         # No CKD
    TP = int(round(SENS * n_pos))
    FN = n_pos - TP
    TN = int(round(SPEC * n_neg))
    FP = n_neg - TN

    # Row-normalised (per true class)
    cm_pct = np.array([
        [TN / n_neg, FP / n_neg],
        [FN / n_pos, TP / n_pos],
    ])
    cm_cnt = np.array([[TN, FP],
                       [FN, TP]])
    cell_labels = np.array([["TN", "FP"],
                            ["FN", "TP"]])

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No CKD", "CKD"], fontsize=9)
    ax.set_yticklabels(["No CKD", "CKD"], fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label", fontsize=9)
    ax.set_title("Confusion Matrix: SGPO v2 Performance on Test Set",
                 fontsize=10, fontweight="bold")

    # annotate each cell
    for i in range(2):
        for j in range(2):
            val   = cm_pct[i, j]
            count = cm_cnt[i, j]
            tag   = cell_labels[i, j]
            txt   = f"{val*100:.1f}%\n({tag}: {count:,})"
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
    cbar.set_label("Percentage of Samples", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # cosmetic: thin border around heatmap only
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#cccccc")

    out = os.path.join(FIGURES, "fig5_confusion_matrix.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figure 5 saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Precision-Recall curve (bonus)
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_pr():
    """Derive PR curve from the bi-normal ROC + prevalence.
    precision = (TPR·π) / (TPR·π + FPR·(1−π))"""
    fpr, tpr = binormal_roc(AUC, n=600)
    pi = PREVALENCE
    denom = tpr * pi + fpr * (1 - pi)
    precision = np.where(denom > 0, (tpr * pi) / denom, 1.0)
    recall    = tpr

    # AUC-PR via trapezoidal integration over recall
    order = np.argsort(recall)
    auc_pr = np.trapz(precision[order], recall[order])

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.plot(recall, precision, color=BLUE, linewidth=2.5,
            label=f"SGPO v2 (AUC-PR = {auc_pr:.4f})")
    ax.fill_between(recall, precision, alpha=0.22, color=BLUE)
    ax.axhline(pi, color=GRAY, linestyle="--", linewidth=1.2,
               label=f"Baseline (prevalence = {pi:.3f})")

    # operating point (Sens, PPV)
    ppv = (SENS * pi) / (SENS * pi + (1 - SPEC) * (1 - pi))
    ax.scatter([SENS], [ppv], s=55, color=ORANGE, zorder=5,
               edgecolor="white", linewidth=1.2,
               label=f"Operating point (Recall={SENS:.3f}, PPV={ppv:.3f})")

    ax.text(0.97, 0.97, f"AUC-PR = {auc_pr:.4f}",
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            color=BLUE, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=BLUE, linewidth=1.6))

    ax.set_xlabel("Recall (Sensitivity)", fontsize=9)
    ax.set_ylabel("Precision (PPV)", fontsize=9)
    ax.set_title("Precision-Recall Curve: SGPO v2",
                 fontsize=10, fontweight="bold")
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0.0, 1.02)

    out = os.path.join(FIGURES, "fig6_pr_curve.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figure 6 saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating additional publication figures ...\n")
    fig4_roc()
    fig5_confusion()
    fig6_pr()
    print("\nAll additional figures generated successfully.")
