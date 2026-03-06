#!/usr/bin/env python3
"""
Create a summary bar chart showing the decomposition results.
Shows Intensity and Extent anomalies for both morning and afternoon periods.
"""
import matplotlib.pyplot as plt
import numpy as np

# Verified values
data = {
    'Morning Low-Level\n(09-12 LT)': {
        '2MOM': {'Intensity': -23.4, 'Extent': -29.7},
        'GRAUPEL': {'Intensity': -37.5, 'Extent': -50.0}
    },
    'Afternoon Upper-Level\n(15-18 LT)': {
        '2MOM': {'Intensity': -9.1, 'Extent': +17.6},
        'GRAUPEL': {'Intensity': -13.6, 'Extent': +5.1}
    }
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

periods = list(data.keys())
experiments = ['2MOM', 'GRAUPEL']
components = ['Intensity', 'Extent']
colors_int = ['#2166ac', '#4393c3']  # Blues for intensity
colors_ext = ['#b2182b', '#d6604d']  # Reds for extent

x = np.arange(len(experiments))
width = 0.35

for i, period in enumerate(periods):
    ax = axes[i]
    
    int_vals = [data[period][exp]['Intensity'] for exp in experiments]
    ext_vals = [data[period][exp]['Extent'] for exp in experiments]
    
    bars1 = ax.bar(x - width/2, int_vals, width, label='Intensity', color='#2166ac', edgecolor='black')
    bars2 = ax.bar(x + width/2, ext_vals, width, label='Extent', color='#b2182b', edgecolor='black')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:+.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, -12 if height < 0 else 3),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:+.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, -12 if height < 0 else 3),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')
    
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Anomaly (%)', fontsize=11)
    ax.set_title(period, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=11)
    ax.legend(loc='lower left' if i == 0 else 'upper left', fontsize=10)
    ax.set_ylim(-60, 30)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Updraft Flux Decomposition: Anomalies vs Control', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('summary_decomposition_barplot.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: summary_decomposition_barplot.png")
plt.close()
