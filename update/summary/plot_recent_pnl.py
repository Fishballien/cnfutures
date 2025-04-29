# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 17:44:34 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# %%

# =============================================================================
# for tag_name, model_info in model_mapping.items():
#     model_name = model_info['model_name']
#     test_name = model_info['test_name']
#     lob_gp_path = model_dir / model_name / 'test' / test_name / 'data' / f'gpd_predict_{model_name}.pkl'
#     lob_hsr_path = model_dir / model_name / 'test' / test_name / 'data' / f'hsr_predict_{model_name}.pkl'
#     
#     with open(lob_gp_path, 'rb') as f:
#         lob_gp = pickle.load(f)
#     with open(lob_hsr_path, 'rb') as f:
#         lob_hsr = pickle.load(f)
#     
#     lob_net = lob_gp['all']['return'] - lob_hsr['all']['avg']  * fee
# =============================================================================


# %%
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_daily_returns(model_mapping, model_dir, summary_dir, start_date='2025-03-17', end_date='2025-04-18', fee=0.00024):
    """
    Plot daily returns as bar charts for different models within a specified date range.
    Annotate each bar with its return value in percentage format.
    """
    os.makedirs(summary_dir, exist_ok=True)
    
    daily_returns = {}
    
    for tag_name, model_info in model_mapping.items():
        model_name = model_info['model_name']
        test_name = model_info['test_name']

        lob_gp_path = model_dir / model_name / 'test' / test_name / 'data' / f'gpd_predict_{model_name}.pkl'
        lob_hsr_path = model_dir / model_name / 'test' / test_name / 'data' / f'hsr_predict_{model_name}.pkl'

        with open(lob_gp_path, 'rb') as f:
            lob_gp = pickle.load(f)
        with open(lob_hsr_path, 'rb') as f:
            lob_hsr = pickle.load(f)

        lob_net = lob_gp['all']['return'] - lob_hsr['all']['avg'] * fee

        daily_returns[tag_name] = lob_net
    
    df_returns = pd.DataFrame(daily_returns)
    df_filtered = df_returns.loc[start_date:end_date]
    
    fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
    
    n_models = len(model_mapping)
    dates = df_filtered.index
    n_dates = len(dates)
    
    bar_width = 0.8 / n_models

    for i, tag_name in enumerate(df_filtered.columns):
        x_positions = np.arange(n_dates) + i * bar_width - (n_models - 1) * bar_width / 2
        model_color = model_mapping[tag_name]['color']
        returns = df_filtered[tag_name]

        bars = ax.bar(x_positions, returns, width=bar_width, label=tag_name, color=model_color, alpha=0.7)

        # ===== 添加数值标注在每个bar上，按百分比格式 =====
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height):
                continue
            ax.annotate(f'{height*100:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5 if height >= 0 else -10),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=8)
    
    ax.set_xticks(np.arange(n_dates))
    ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in dates], rotation=45, ha='right')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Return')
    ax.set_title(f'Daily Returns Comparison ({start_date} to {end_date})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(summary_dir / f'daily_returns_{start_date}_to_{end_date}.png', dpi=300)
    plt.close()

    print(f"Daily returns plot saved to {summary_dir / f'daily_returns_{start_date}_to_{end_date}.png'}")


def plot_cumulative_returns(model_mapping, model_dir, summary_dir, start_date='2025-01-01', fee=0.00024):
    """
    Plot cumulative returns for different models from a specified start date.
    Uses custom colors defined in model_mapping.
    
    Parameters:
    -----------
    model_mapping : dict
        Dictionary containing model information, including 'color' for each model.
    model_dir : Path
        Path to the model directory.
    summary_dir : Path
        Path to save the output plot.
    start_date : str
        Start date for cumulative returns in 'YYYY-MM-DD' format.
    fee : float
        Trading fee per transaction.
    """
    # Create summary_dir if it doesn't exist
    os.makedirs(summary_dir, exist_ok=True)
    
    # Store daily returns for each model
    daily_returns = {}
    model_colors = {}
    
    # Load data for each model
    for tag_name, model_info in model_mapping.items():
        model_name = model_info['model_name']
        test_name = model_info['test_name']
        prod_name = model_info['prod_name']
        
        lob_gp_path = model_dir / model_name / 'test' / test_name / 'data' / f'gpd_predict_{model_name}.pkl'
        lob_hsr_path = model_dir / model_name / 'test' / test_name / 'data' / f'hsr_predict_{model_name}.pkl'
        
        with open(lob_gp_path, 'rb') as f:
            lob_gp = pickle.load(f)
        with open(lob_hsr_path, 'rb') as f:
            lob_hsr = pickle.load(f)
        
        # Calculate net return
        lob_net = lob_gp['all']['return'] - lob_hsr['all']['avg'] * fee
        
        # Create label with tag name and prod name
        label = f"{tag_name} ({prod_name})"
        
        # Store returns with combined label
        daily_returns[label] = lob_net
        
        # Store color mapping
        model_colors[label] = model_info['color']
    
    # Convert to DataFrame
    df_returns = pd.DataFrame(daily_returns)
    
    # Filter by start date
    df_filtered = df_returns.loc[start_date:]
    
    # Calculate cumulative returns (1 + r1) * (1 + r2) * ... - 1
    df_cumulative = (1 + df_filtered).cumprod() - 1
    
    # Set up the plot
    plt.figure(figsize=(12, 6))
    
    # Plot line for each model with custom colors
    for label in df_cumulative.columns:
        plt.plot(df_cumulative.index, df_cumulative[label], label=label, 
                 color=model_colors[label], linewidth=2)
    
    # Add labels and legend
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title(f'Cumulative Returns Since {start_date}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(summary_dir / f'cumulative_returns_from_{start_date}.png', dpi=300)
    plt.close()
    
    print(f"Cumulative returns plot saved to {summary_dir / f'cumulative_returns_from_{start_date}.png'}")

# Example usage:
# plot_daily_returns(model_mapping, model_dir, summary_dir, start_date='2025-03-17', end_date='2025-04-18')
# plot_cumulative_returns(model_mapping, model_dir, summary_dir, start_date='2025-01-01')

# %% 使用示例
if __name__ == "__main__":
    
    model_mapping = {
        '1_2_3_overnight': {
            'model_name': 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18',
            'test_name': 'trade_ver3_futtwap_sp1min_s240d_icim_v6',
            'prod_name': 'agg_1.2.0_3',
            'color': 'r'
            },
        '1_2_3_intraday': {
            'model_name': 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18',
            'test_name': 'trade_ver3_3_futtwap_sp1min_s240d_icim_v6',
            'prod_name': 'agg_1.2.0_4',
            'color': 'g'
            },
    }

    model_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model')
    summary_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\test_update_summary')
    summary_dir.mkdir(parents=True, exist_ok=True)
    fee = 0.00024
    
    plot_daily_returns(model_mapping, model_dir, summary_dir, start_date='2025-03-17', end_date='2025-04-18')
    plot_cumulative_returns(model_mapping, model_dir, summary_dir, start_date='2025-01-01')