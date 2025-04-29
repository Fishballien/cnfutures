# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:45:55 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Path to the CSV
res_path = 'D:/mnt/CNIndexFutures/timeseries/factor_test/results/analysis/momentum_or_reversal/1_2_factor_hisrtn_corr.csv'
# Directory to save results
res_dir = 'D:/mnt/CNIndexFutures/timeseries/factor_test/results/analysis/momentum_or_reversal'

# Ensure the output directory exists
os.makedirs(res_dir, exist_ok=True)

# Read the CSV
res = pd.read_csv(res_path)

# Rename the 'Unnamed: 0' column to 'factor_name'
res = res.rename(columns={'Unnamed: 0': 'factor_name'})

# Split factor_name by '-' and extract first and third elements
res['indicator'] = res['factor_name'].apply(lambda x: x.split('-')[0] if len(x.split('-')) > 0 else None)
res['ts_trans'] = res['factor_name'].apply(lambda x: x.split('-')[2] if len(x.split('-')) > 2 else None)

# Filter out rows where indicator or ts_trans is None
res = res.dropna(subset=['indicator', 'ts_trans'])

# Identify all numeric columns for heatmaps (exclude factor_name, indicator, ts_trans)
numeric_columns = res.select_dtypes(include=['number']).columns.tolist()

# Function to create and save a heatmap for a specific column
def create_heatmap(data, column):
    # Group by indicator and ts_trans, calculate mean for the specified column
    pivot_data = data.pivot_table(index='indicator', columns='ts_trans', values=column, aggfunc='mean')
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap without annotations
    sns.heatmap(pivot_data, annot=False, cmap='viridis', linewidths=.5)
    
    # Set title and labels
    plt.title(f'Heatmap of {column} Grouped by Indicator and Time Series Transformation')
    plt.xlabel('Time Series Transformation (ts_trans)')
    plt.ylabel('Indicator (indicator)')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(res_dir, f'{column}_heatmap.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved heatmap to {output_path}")

# Create heatmaps for each numeric column
for col in numeric_columns:
    # Filter out rows where the column value is NaN
    col_data = res.dropna(subset=[col])
    if not col_data.empty:
        create_heatmap(col_data, col)
        print(f"Created heatmap for {col}")
    else:
        print(f"No valid data for {col}")

print(f"All heatmaps saved to {res_dir}")