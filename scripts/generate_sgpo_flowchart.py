import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
import numpy as np

# IEEE Color Palette
COLORS = {
    'process': '#0072BD',      # Steel Blue
    'decision': '#D95319',     # Muted Orange
    'start_end': '#009E73',    # Soft Green
    'fgo': '#56B4E9',          # Soft Violet
    'text': '#333333',         # Dark Gray
    'light_bg': '#F5F5F5'      # Light Gray
}

# Create figure with portrait orientation (1600x2000 at 300 DPI)
fig = plt.figure(figsize=(5.33, 6.67), dpi=300, facecolor='white')
ax = fig.add_subplot(111)
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

def draw_box(ax, x, y, width, height, text, color, fontsize=9):
    """Draw a rectangular box with text."""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.05",
                         edgecolor='black', facecolor=color, linewidth=1.5, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontname='Arial', fontweight='normal', wrap=True, zorder=3, color='white')

def draw_diamond(ax, x, y, width, height, text, color, fontsize=9):
    """Draw a diamond-shaped decision point."""
    points = np.array([[x, y + height/2], [x + width/2, y],
                       [x, y - height/2], [x - width/2, y]])
    diamond = Polygon(points, edgecolor='black', facecolor=color, linewidth=1.5, zorder=2)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize-0.5,
            fontname='Arial', fontweight='normal', wrap=True, zorder=3, color='white')

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw an arrow between two points."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=1.5, color='black', zorder=1)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=8, fontname='Arial',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def draw_text_box(ax, x, y, width, height, text, fontsize=8):
    """Draw a text box with gray background."""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.05",
                         edgecolor='#999999', facecolor=COLORS['light_bg'],
                         linewidth=1, zorder=2, linestyle='--')
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontname='Arial', fontweight='normal', wrap=True, zorder=3, color=COLORS['text'])

# Title
ax.text(5, 19.5, 'Figure 4: SGPO v2 Co-Evolutionary Optimization Framework',
        ha='center', va='top', fontsize=12, fontname='Arial', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0', edgecolor='black', linewidth=2))

# ============ PHASE 1: SHARED FITNESS EVALUATION ============
y_pos = 18.5
ax.text(0.5, y_pos, 'PHASE 1: SHARED FITNESS EVALUATION', fontsize=11, fontname='Arial',
        fontweight='bold', color=COLORS['process'], bbox=dict(boxstyle='round,pad=0.4',
        facecolor='#E8F4F8', edgecolor=COLORS['process'], linewidth=2))

# Start
y_pos -= 0.8
draw_box(ax, 5, y_pos, 1, 0.5, 'START', COLORS['start_end'], fontsize=9)
draw_arrow(ax, 5, y_pos - 0.25, 5, y_pos - 0.9)

# Input
y_pos -= 1.2
draw_box(ax, 5, y_pos, 2.5, 0.8, 'Input: Clinical Dataset X,\nLabels y', COLORS['process'], fontsize=8)
draw_arrow(ax, 5, y_pos - 0.4, 5, y_pos - 1.1)

# Initialize Populations
y_pos -= 1.4
draw_text_box(ax, 5, y_pos, 3.5, 1.2,
    'Initialize Populations\n• SFOA: {s₁...s_N}\n• DOA: {θ₁...θ_M}', fontsize=8)
draw_arrow(ax, 5, y_pos - 0.6, 5, y_pos - 1.2)

# Fitness Matrix
y_pos -= 1.5
draw_box(ax, 5, y_pos, 2.8, 0.7, 'Shared Fitness Matrix\nF ∈ ℝ^(N×M)', COLORS['process'], fontsize=8)
draw_arrow(ax, 5, y_pos - 0.35, 5, y_pos - 1.0)

# Nested Loop
y_pos -= 1.3
draw_text_box(ax, 5, y_pos, 3.8, 1.4,
    'Nested Loop: Evaluate All (sᵢ, θⱼ) Pairs\n• Apply mask sᵢ → Xₛᵤᵦ\n• Train with θⱼ\n• f_ij = ω₁·AUC + ω₂·Sens - ω₃·(||sᵢ||/D)\n• Store in F[i,j]', fontsize=7.5)
draw_arrow(ax, 5, y_pos - 0.7, 5, y_pos - 1.4)

# Marginal Fitness
y_pos -= 1.8
draw_text_box(ax, 5, y_pos, 3.8, 1.2,
    'Derive Marginal Fitness\n• Fitness_SFOA(i) = max_j F[i,:]\n• Fitness_DOA(j) = max_i F[:,j]', fontsize=8)
draw_arrow(ax, 5, y_pos - 0.6, 5, y_pos - 1.2)

# ============ PHASE 2: CO-ADAPTIVE POPULATION UPDATES ============
y_phase2 = y_pos - 1.5
ax.text(0.5, y_phase2 + 0.3, 'PHASE 2: CO-ADAPTIVE UPDATES', fontsize=11, fontname='Arial',
        fontweight='bold', color=COLORS['process'], bbox=dict(boxstyle='round,pad=0.4',
        facecolor='#E8F4F8', edgecolor=COLORS['process'], linewidth=2))

y_pos = y_phase2 - 0.7

# Update SFOA & DOA (side by side)
# Left side - SFOA
draw_box(ax, 2.5, y_pos, 2.2, 0.8, 'Update SFOA Population\n(Binary Mask Evolution)',
         COLORS['process'], fontsize=8)
# Right side - DOA
draw_box(ax, 7.5, y_pos, 2.2, 0.8, 'Update DOA Population\n(HP Vector Evolution)',
         COLORS['process'], fontsize=8)

# Arrows from fitness to updates
draw_arrow(ax, 4, y_pos + 0.5, 2.5, y_pos + 0.4)
draw_arrow(ax, 6, y_pos + 0.5, 7.5, y_pos + 0.4)

y_pos -= 1.2
# Co-adaptive feedback
draw_box(ax, 5, y_pos, 3.5, 1.0, 'Co-Adaptive Feedback Loop\nMasks ↔ HPs (Real-time Compensation)',
         COLORS['fgo'], fontsize=8)

# Arrows between updates and feedback
draw_arrow(ax, 2.5, y_pos + 1.0, 4.2, y_pos + 0.5)
draw_arrow(ax, 7.5, y_pos + 1.0, 5.8, y_pos + 0.5)

draw_arrow(ax, 5, y_pos - 0.5, 5, y_pos - 1.1)

# ============ PHASE 3: FGO STAGNATION-BREAKING ============
y_phase3 = y_pos - 1.4
ax.text(0.5, y_phase3 + 0.3, 'PHASE 3: FGO STAGNATION-BREAKING', fontsize=11, fontname='Arial',
        fontweight='bold', color=COLORS['fgo'], bbox=dict(boxstyle='round,pad=0.4',
        facecolor='#E8F4F8', edgecolor=COLORS['fgo'], linewidth=2))

y_pos = y_phase3 - 0.7

# Identify Stagnant
draw_box(ax, 5, y_pos, 2.8, 0.7, 'Identify Stagnant Individuals\n(Bottom-ρ% of Population)',
         COLORS['decision'], fontsize=8)
draw_arrow(ax, 5, y_pos - 0.35, 5, y_pos - 1.0)

# FGO Module
y_pos -= 1.3
draw_box(ax, 5, y_pos, 2.8, 1.2, 'FGO Perturbation Module 🍄\n• Mycelial Diffusion Operator\n• Gaussian Noise: δ ~ N(0, σ²I)\n• Expand Search Radius',
         COLORS['fgo'], fontsize=8)
draw_arrow(ax, 5, y_pos - 0.6, 5, y_pos - 1.2)

# Update Global Best
y_pos -= 1.5
draw_box(ax, 5, y_pos, 2.5, 0.7, 'Update Global Best\n(S*, θ*) ← argmax F[i,j]',
         COLORS['process'], fontsize=8)
draw_arrow(ax, 5, y_pos - 0.35, 5, y_pos - 1.0)

# Convergence Check (Diamond)
y_pos -= 1.3
draw_diamond(ax, 5, y_pos, 1.8, 0.8, 'g = G_max?', COLORS['decision'], fontsize=9)

# Decision branches
draw_arrow(ax, 3.8, y_pos, 2, y_pos, label='NO')
draw_arrow(ax, 6.2, y_pos, 8, y_pos, label='YES')

# Loop back arrow (NO path)
loop_y = y_pos + 4.5
draw_arrow(ax, 2, y_pos, 2, loop_y)
draw_arrow(ax, 2, loop_y, 1.5, loop_y)
draw_arrow(ax, 1.5, loop_y, 1.5, 11.5)
draw_arrow(ax, 1.5, 11.5, 4, 11.5)
ax.text(1, loop_y + 0.2, 'Next\nGeneration', fontsize=7.5, fontname='Arial',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

# Output (YES path)
y_pos -= 1.2
draw_box(ax, 8, y_pos, 3.0, 0.8, 'Output: Optimal Feature Subset S*\n& Hyperparameters θ*',
         COLORS['process'], fontsize=8)
draw_arrow(ax, 8, y_pos - 0.4, 8, y_pos - 1.0)

# End
y_pos -= 1.3
draw_box(ax, 8, y_pos, 1, 0.5, 'END', COLORS['start_end'], fontsize=9)

# ============ LEGEND AND ANNOTATIONS ============
# Legend box
legend_y = 0.8
ax.text(0.5, legend_y + 0.8, 'Legend', fontsize=10, fontname='Arial', fontweight='bold')
legend_items = [
    (COLORS['process'], 'Process/Calculation'),
    (COLORS['decision'], 'Decision Point'),
    (COLORS['start_end'], 'Start/End'),
    (COLORS['fgo'], 'FGO Module')
]

for i, (color, label) in enumerate(legend_items):
    y = legend_y - i * 0.4
    rect = Rectangle((0.5, y - 0.1), 0.3, 0.2, facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(1.0, y, label, fontsize=8, fontname='Arial', va='center')

# Key Innovation Callout
innovation_y = 0.3
draw_text_box(ax, 6, innovation_y, 3.5, 0.6,
    'Key Innovation: Shared Fitness Matrix enables\nco-adaptive evolution without sequential decoupling',
    fontsize=7.5)

plt.tight_layout()

# Save figure
output_path = 'results/figures/fig4_sgpo_v2_flowchart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Flowchart saved to {output_path}")

# Also save as PDF
pdf_path = 'results/figures/fig4_sgpo_v2_flowchart.pdf'
plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Flowchart saved to {pdf_path}")

plt.close()
print("Flowchart generation complete!")
