# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 20:09:42 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# %%
compare_name = 'batch10_price'
eval_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\factor_evaluation')
summary_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis') / compare_name
summary_dir.mkdir(exist_ok=True, parents=True)


# %%
eval_period = '160101_250301'
eval_list = [
    'agg_batch10', 
    'agg_batch10_by_mpc', 
    'agg_batch10_by_symp', 
    ]


# %%
for eval_name in eval_list:
    eval_path = eval_dir / eval_name / f'factor_eval_{eval_period}.csv'
    eval_data = pd.read_csv(eval_path)


# %%
# Column to analyze
column_to_plot = 'net_return_annualized'


# %%
# Set up the plot with a white background for better contrast
plt.figure(figsize=(12, 8), facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')

# Define bright colors for the outlines
colors = ['#FF5733', '#3366FF', '#33CC33', '#CC33FF', '#FFCC00', 
          '#FF33CC', '#33FFCC', '#9933FF', '#FF9933', '#33FF33']

# Define shared bins for consistent comparison
bin_edges = np.linspace(-1, 4, 26)  # Adjust range based on your data

# Track max frequency for y-axis scaling
max_freq = 0

# Process each evaluation dataset
for i, eval_name in enumerate(eval_list):
    # Load data
    eval_path = eval_dir / eval_name / f'factor_eval_{eval_period}.csv'
    eval_data = pd.read_csv(eval_path)
    
    # Dynamically assign color based on index
    color = colors[i % len(colors)]
    
    # Calculate histogram data
    hist, _ = np.histogram(eval_data[column_to_plot], bins=bin_edges)
    max_freq = max(max_freq, np.max(hist))
    
    # Plot histogram with ONLY outlines, no fill
    plt.hist(
        eval_data[column_to_plot],
        bins=bin_edges,
        histtype='step',  # 'step' type draws only the outlines
        linewidth=3,      # Thicker lines for better visibility
        edgecolor=color,
        label=eval_name
    )

# Add plot details
plt.xlabel(f'{column_to_plot}', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.title(f'Distribution of {column_to_plot} Across Different Evaluation Methods', fontsize=16, fontweight='bold')

# Improve legend visibility
plt.legend(fontsize=12, framealpha=1, facecolor='white', edgecolor='black')

# Add grid but keep it subtle
plt.grid(True, alpha=0.3, linestyle='--')

# Ensure y-axis has enough room
plt.ylim(0, max_freq * 1.1)

# Show plot
plt.tight_layout()

# Save plot to summary directory
save_path = summary_dir / f'{column_to_plot}_distribution_comparison.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')

print(f"Plot saved to: {save_path}")