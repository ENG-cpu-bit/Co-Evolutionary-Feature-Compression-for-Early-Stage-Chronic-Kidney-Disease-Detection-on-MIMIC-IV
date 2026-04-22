"""
Figure 1 - SGPO v2 Co-Evolutionary Optimization flowchart.

IEEE-ready schematic with strict orthogonal (Manhattan) routing:
  * Single vertical column, uniform box sizes.
  * Every forward arrow travels bottom-center -> top-center.
  * Feedback edge uses right-angle turns along the right margin.
  * No diagonal / curved segments anywhere.

Output: results/figures/fig1_algorithm_flowchart.png (300 DPI, white bg).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Palette (user-specified)
# ---------------------------------------------------------------------------
C_TERMINAL = "#009E73"  # Start / End       (green)
C_PROCESS  = "#0072BD"  # Process steps     (blue)
C_LOOP     = "#D95319"  # Loop / decision   (orange)
C_FGO      = "#56B4E9"  # FGO intervention  (accent)
C_EDGE     = "#222222"
C_TEXT_L   = "#FFFFFF"
C_TEXT_D   = "#1A1A1A"

FONT = "Arial"
FS_TITLE = 13
FS_BODY  = 10
FS_LEG   = 9

# ---------------------------------------------------------------------------
# Geometry - single vertical lane
# ---------------------------------------------------------------------------
CENTER_X = 5.0
BOX_W    = 5.8
BOX_H    = 1.05
V_GAP    = 0.55          # vertical whitespace between stacked boxes
MARGIN_X = 9.35          # x-coord of the right-hand feedback lane


def node(ax, cy, text, facecolor, textcolor=C_TEXT_L,
         height=BOX_H, bold=False):
    """Draw a uniformly sized rounded node centred on (CENTER_X, cy)."""
    x0 = CENTER_X - BOX_W / 2
    y0 = cy - height / 2
    patch = FancyBboxPatch(
        (x0, y0), BOX_W, height,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=1.3, edgecolor=C_EDGE, facecolor=facecolor,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        CENTER_X, cy, text, ha="center", va="center",
        color=textcolor, fontsize=FS_BODY, family=FONT,
        fontweight="bold" if bold else "normal", zorder=4,
    )
    return {"x": CENTER_X, "y": cy, "w": BOX_W, "h": height,
            "top": cy + height / 2, "bot": cy - height / 2}


def v_arrow(ax, src, dst):
    """Straight vertical arrow: bottom-center of src -> top-center of dst."""
    arr = FancyArrowPatch(
        (src["x"], src["bot"]), (dst["x"], dst["top"]),
        arrowstyle="-|>", mutation_scale=13,
        linewidth=1.4, color=C_EDGE, zorder=2,
    )
    ax.add_patch(arr)


def ortho_feedback(ax, src, dst, label):
    """
    Right-margin feedback route with right-angle turns only:
        src.right  -> MARGIN_X  (horizontal)
        MARGIN_X   -> dst.y     (vertical)
        MARGIN_X   -> dst.right (horizontal, with arrowhead)
    """
    y_src = src["y"]
    y_dst = dst["y"]
    x_src_right = src["x"] + src["w"] / 2
    x_dst_right = dst["x"] + dst["w"] / 2

    # Segment 1: horizontal out of the decision box to the margin lane.
    ax.plot([x_src_right, MARGIN_X], [y_src, y_src],
            color=C_LOOP, linewidth=1.6, solid_capstyle="butt", zorder=2)
    # Segment 2: vertical up the margin to the loop header level.
    ax.plot([MARGIN_X, MARGIN_X], [y_src, y_dst],
            color=C_LOOP, linewidth=1.6, solid_capstyle="butt", zorder=2)
    # Segment 3: horizontal back into the loop header (with arrowhead).
    arr = FancyArrowPatch(
        (MARGIN_X, y_dst), (x_dst_right, y_dst),
        arrowstyle="-|>", mutation_scale=13,
        linewidth=1.6, color=C_LOOP, zorder=2,
    )
    ax.add_patch(arr)

    # Rotated label placed along the vertical segment of the feedback lane.
    ax.text(
        MARGIN_X + 0.18, (y_src + y_dst) / 2, label,
        rotation=90, ha="center", va="center",
        fontsize=FS_LEG, style="italic", family=FONT,
        color=C_LOOP, fontweight="bold",
    )


def build(out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.4, 13.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 19)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title -----------------------------------------------------------------
    ax.text(
        CENTER_X, 18.35,
        "SGPO v2: Co-Evolutionary Feature Selection & Hyperparameter Optimization",
        ha="center", va="center",
        fontsize=FS_TITLE, fontweight="bold", family=FONT, color=C_TEXT_D,
    )

    # Stacked nodes (top -> bottom) ----------------------------------------
    y = 17.4
    n_start  = node(ax, y, "Start", C_TERMINAL, bold=True, height=0.75)

    y -= 0.75 / 2 + V_GAP + BOX_H / 2
    n_init   = node(ax, y,
                    "Init SFOA (Masks  $S_i$)  &  DOA (HP Vectors  $\\theta_j$)",
                    C_PROCESS)

    y -= BOX_H + V_GAP
    n_loop   = node(ax, y,
                    "Main Loop:  for generation  g = 1 \u2026 G",
                    C_LOOP, bold=True)

    y -= BOX_H + V_GAP
    n_eval   = node(ax, y,
                    "Evaluate Shared Fitness Matrix  F[i, j]\n"
                    "via Nested Cross-Validation",
                    C_PROCESS, height=1.25)

    y -= 1.25 / 2 + V_GAP + BOX_H / 2
    n_sfoa   = node(ax, y, "Evolve SFOA Masks (Binary Selection)", C_PROCESS)

    y -= BOX_H + V_GAP
    n_doa    = node(ax, y, "Optimize DOA Vectors (HP Tuning)", C_PROCESS)

    y -= BOX_H + V_GAP
    n_fgo    = node(ax, y,
                    "FGO Intervention: Perturb Top-10% Individuals",
                    C_FGO, textcolor=C_TEXT_D)

    y -= BOX_H + V_GAP
    n_gbest  = node(ax, y,
                    "Update Global Best  ($S^{*}$,  $\\theta^{*}$)",
                    C_PROCESS)

    y -= BOX_H + V_GAP
    n_dec    = node(ax, y, "g < G ?", C_LOOP, bold=True, height=0.9)

    y -= 0.9 / 2 + V_GAP + BOX_H / 2
    n_end    = node(ax, y,
                    "Return Optimal  $S^{*}$  and  $\\theta^{*}$",
                    C_TERMINAL, bold=True)

    # Forward edges (strictly vertical, bottom-center -> top-center) -------
    forward = [
        (n_start, n_init),
        (n_init,  n_loop),
        (n_loop,  n_eval),
        (n_eval,  n_sfoa),
        (n_sfoa,  n_doa),
        (n_doa,   n_fgo),
        (n_fgo,   n_gbest),
        (n_gbest, n_dec),
        (n_dec,   n_end),
    ]
    for src, dst in forward:
        v_arrow(ax, src, dst)

    # "No" label on decision -> return edge.
    ax.text(
        CENTER_X + 0.28, (n_dec["bot"] + n_end["top"]) / 2,
        "No", ha="left", va="center",
        fontsize=FS_LEG, style="italic", family=FONT,
        color=C_TEXT_D, fontweight="bold",
    )

    # "Yes" feedback loop: decision.right -> right margin -> loop.right.
    ortho_feedback(ax, n_dec, n_loop, label="Yes  (g += 1)")

    # Legend ----------------------------------------------------------------
    lx, ly = 0.3, 0.3
    lw, lh = 3.8, 1.25
    ax.add_patch(Rectangle(
        (lx, ly), lw, lh,
        linewidth=0.8, edgecolor="#BBBBBB", facecolor="#FAFAFA", zorder=1,
    ))
    ax.text(lx + 0.15, ly + lh - 0.22, "Legend",
            fontsize=FS_LEG, fontweight="bold", family=FONT, color=C_TEXT_D)

    entries = [
        (C_TERMINAL, "Start / End"),
        (C_PROCESS,  "Process Step"),
        (C_LOOP,     "Loop / Decision"),
        (C_FGO,      "FGO Intervention"),
    ]
    for i, (color, lbl) in enumerate(entries):
        row, col = divmod(i, 2)
        cx = lx + 0.2 + col * 1.85
        cy = ly + lh - 0.55 - row * 0.32
        ax.add_patch(Rectangle((cx, cy - 0.09), 0.26, 0.18,
                               facecolor=color, edgecolor=C_EDGE, lw=0.7))
        ax.text(cx + 0.34, cy, lbl, fontsize=FS_LEG,
                family=FONT, color=C_TEXT_D, va="center")

    # Save ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path, dpi=300, bbox_inches="tight",
        facecolor="white", edgecolor="none",
    )
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    out = repo_root / "results" / "figures" / "fig1_algorithm_flowchart.png"
    saved = build(out)
    print(f"[OK] SGPO v2 flowchart saved -> {saved}")
