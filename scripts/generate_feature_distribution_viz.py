import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data
categories = ['Laboratory Tests', 'Demographics', 'Admissions History', 'Administrative']
counts = [22, 5, 5, 10]
colors = ['#0072BD', '#009E73', '#D95319', '#56B4E9']  # IEEE palette

# Create figure with specific dimensions for 300 DPI (8x5 inches = 2400x1500 pixels at 300 DPI)
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Create horizontal bar chart
bars = ax.barh(categories, counts, color=colors, edgecolor='none', height=0.6)

# Customize labels and title
ax.set_xlabel('Number of Features', fontsize=11, fontname='Arial', fontweight='normal')
ax.set_title('Distribution of 42 Initial Features by Clinical Category',
             fontsize=13, fontname='Arial', fontweight='bold', pad=20)

# Add count labels on the bars
for i, (bar, count) in enumerate(zip(bars, counts)):
    width = bar.get_width()
    ax.text(width + 0.3, bar.get_y() + bar.get_height()/2,
            f'n={count}',
            ha='left', va='center', fontsize=10, fontname='Arial', fontweight='normal')

# Customize axes
ax.set_xlim(0, 25)
ax.set_ylim(-0.5, len(categories) - 0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)
ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Set tick labels font
ax.tick_params(axis='both', which='major', labelsize=10)
for label in ax.get_xticklabels():
    label.set_fontname('Arial')
for label in ax.get_yticklabels():
    label.set_fontname('Arial')

# Add annotation
fig.text(0.5, 0.02, 'Note: SGPO v2 reduces this space to 8 critical features',
         ha='center', fontsize=9, fontname='Arial', style='italic', color='#333333')

# Adjust layout to accommodate annotation
plt.subplots_adjust(bottom=0.08)

# Save figure
output_path = 'results/figures/fig2_feature_distribution.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Figure saved to {output_path}")

# Also save as PDF for publication
pdf_path = 'results/figures/fig2_feature_distribution.pdf'
plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Figure saved to {pdf_path}")

plt.close()
print("Visualization complete!")
