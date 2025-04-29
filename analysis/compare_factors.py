# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 12:55:57 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.datautils import align_and_sort_columns
from utils.market import index_to_futures
from utils.timeutils import parse_time_string
from data_processing.ts_trans import *


# %%
def align_data(factor_data, price_data, index_to_futures):
    """
    Align factor and price data.
    """
    factor_data = factor_data.rename(columns=index_to_futures)[['IC', 'IF', 'IM']]
    factor_data, price_data = align_and_sort_columns([factor_data, price_data])
    price_data = price_data.loc[factor_data.index.min():factor_data.index.max()]
    factor_data = factor_data.reindex(price_data.index)
    return factor_data, price_data


def scale_factor(factor_data, scale_method, scale_window, sp, direction, 
                 scale_quantile=None, rtn_1p=None, pp_by_sp=None):
    """
    Scale factor data using specified method.
    """
    scale_func = globals()[scale_method]
    scale_step = int(parse_time_string(scale_window) / parse_time_string(sp))
    
    if scale_method in ['minmax_scale', 'minmax_scale_separate']:
        factor_scaled = scale_func(factor_data, window=scale_step, quantile=scale_quantile)
    elif scale_method in ['minmax_scale_adj_by_his_rtn', 'zscore_adj_by_his_rtn_and_minmax']:
        factor_scaled = scale_func(factor_data, rtn_1p, window=scale_step, 
                                   rtn_window=pp_by_sp, quantile=scale_quantile)
    elif scale_method in ['rolling_percentile']:
        factor_scaled = scale_func(factor_data, window=scale_step)
    elif scale_method in ['percentile_adj_by_his_rtn']:
        factor_scaled = scale_func(factor_data, rtn_1p, window=scale_step, rtn_window=pp_by_sp)
    
    factor_scaled = (factor_scaled - 0.5) * 2 * direction
    return factor_scaled


# %%
# compare_name = 'compare_ValueTimeDecayOrderAmount_zscore_or_not'
# factor_list = [
#     [r'D:\mnt\CNIndexFutures\timeseries\sample_factors\zscore_or_not', 
#      'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb01_dp2-org', 1],
#     [r'D:\mnt\CNIndexFutures\timeseries\sample_factors\zscore_or_not', 
#      'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb01_dp2-rollingAggMinuteMinMaxScale_w30d_q0_i5', 1],
#     [r'D:\mnt\CNIndexFutures\timeseries\sample_factors\zscore_or_not', 
#      'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb01_dp2-zscore_fxwd_p1d', 1],
#     [r'D:\mnt\CNIndexFutures\timeseries\sample_factors\zscore_or_not', 
#      'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb01_dp2-zscore_fxwd_p3d', 1],
#     [r'D:\mnt\CNIndexFutures\timeseries\sample_factors\zscore_or_not', 
#      'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb01_dp2-zscore_fxwd_p10d', 1],
#     [r'D:\mnt\CNIndexFutures\timeseries\sample_factors\zscore_or_not', 
#      'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb01_dp2-rollingAggMinuteMinMaxScale_w245d_q0_i5', 1],
#     ]
compare_name = 'compare_bid_ask_large_order_amount'
factor_list = [
    [r'D:\mnt\CNIndexFutures\timeseries\factor_test\sample_data\factors\v1.2_all_trans_3', 
     'LargeOrderAmountByValue_p1.0_v40000-avg_side_dp2_Ask-org', 1],
    [r'D:\mnt\CNIndexFutures\timeseries\factor_test\sample_data\factors\v1.2_all_trans_3', 
     'LargeOrderAmountByValue_p1.0_v40000-avg_side_dp2_Bid-org', 1],
    ]
price_name = 't1min_fq1min_dl1min'

to_scale = False
scale_method = 'minmax_scale'
scale_window = '240d'
scale_quantile = 0.02
sp = '1min'


# %%
path_config = load_path_config(project_dir)
param_dir = Path(path_config['param'])
result_dir = Path(path_config['result'])
fut_dir = Path('/mnt/data1/futuretwap')
save_dir =Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\compare_factors') / compare_name
save_dir.mkdir(parents=True, exist_ok=True)


# %%
price_data = pd.read_parquet(fut_dir / f'{price_name}.parquet')

# Load all model positions into a dictionary
factor_mapping = {}
for factor_dir, factor_name, direction in factor_list:
    factor_path = Path(factor_dir) / f'{factor_name}.parquet'
    factor_data = pd.read_parquet(factor_path)
    factor_data, price_data = align_data(factor_data, price_data, index_to_futures)
    factor_scaled = scale_factor(factor_data, scale_method, scale_window, sp, direction, 
                     scale_quantile=scale_quantile)
    factor_mapping[factor_name] = factor_scaled if to_scale else factor_data
    
    
# %%
by_week_dir = save_dir / 'by_week'
by_week_dir.mkdir(parents=True, exist_ok=True)


# %% Filter for only IC and IM columns
price_data = price_data[['IC', 'IM']]

# %% Define time range (use the last year of data or specify your own range)
start_date = pd.Timestamp('2024-09-01')
end_date = pd.Timestamp('2025-04-18')

# Filter data to the desired date range
price_data = price_data.loc[(price_data.index >= start_date) & (price_data.index <= end_date)]

# Filter model positions to the same date range
for factor_name in factor_mapping:
    factor_mapping[factor_name] = factor_mapping[factor_name].loc[
        (factor_mapping[factor_name].index >= start_date) & 
        (factor_mapping[factor_name].index <= end_date)
    ]


# %% Create a color list for the factors
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Create a color list with 20 distinct colors
# Using a combination of tab20 colormap and some additional colors
tab20_colors = plt.cm.tab20(np.linspace(0, 1, 20))
color_list = [mcolors.rgb2hex(tab20_colors[i]) for i in range(20)]

# %% Create weekly visualizations
for week_start, price_group in price_data.groupby(pd.Grouper(freq='W-MON', label='left', closed='left')):
    # Skip weeks with no data
    if price_group.empty:
        continue
    
    # Get the end of the week
    week_end = week_start + pd.Timedelta(days=7)
    
    # Create figure with 2 subplots (one for IC, one for IM)
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f"Price and Factor Data for Week: {week_start.strftime('%Y-%m-%d')} to {(week_end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}", 
                 fontsize=16)
    
    instruments = ['IC', 'IM']
    
    for i, instrument in enumerate(instruments):
        # Primary axis for price
        ax1 = axs[i]
        
        # Plot price data
        price_series = price_group[instrument].dropna()
        if not price_series.empty:
            # Use arange for x-axis instead of datetime
            x = np.arange(len(price_series))
            x_labels = price_series.index.strftime('%Y-%m-%d %H:%M')  # Convert to time labels for reference
            
            # Plot with numeric x values
            ax1.plot(x, price_series.values, color='black', linewidth=1.5, label=f'{instrument} Price')
            ax1.set_ylabel(f'{instrument} Price', color='black', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Create secondary y-axis for factor data
            ax2 = ax1.twinx()
            
            # Plot factors
            for idx, (factor_name, factor_data) in enumerate(factor_mapping.items()):
                # Get color from the color list (cycle through if more factors than colors)
                color = color_list[idx % len(color_list)]
                
                # Filter factor data for this week and instrument
                factor_week = factor_data.loc[
                    (factor_data.index >= week_start) & 
                    (factor_data.index < week_end), 
                    instrument
                ].dropna()
                
                if not factor_week.empty:
                    # Need to align factor data with price index for plotting
                    aligned_factor = factor_week.reindex(price_series.index, method='ffill')
                    # Only use non-NaN values
                    valid_indices = ~aligned_factor.isna()
                    if valid_indices.any():
                        # Get the corresponding x values for valid factors
                        valid_x = x[valid_indices.values]
                        valid_factor = aligned_factor[valid_indices].values
                        
                        # Plot factor data on secondary axis
                        ax2.plot(valid_x, valid_factor, color=color, 
                                linewidth=1.5, linestyle='-', 
                                label=f'{factor_name}')
            
            # Set the secondary y-axis label
            ax2.set_ylabel('Factor Value', fontsize=12)
            if to_scale:
                ax2.set_ylim(-1.2, 1.2)  # Assuming factor values are between -1 and 1
            ax2.tick_params(axis='y')
            
            # Add a horizontal dashed line at position 0
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            
            # Find every 9:30 AM for vertical lines (market open)
            nine_thirty_indices = [idx for idx, t in enumerate(price_series.index) if t.strftime('%H:%M') == '09:30']
            for idx in nine_thirty_indices:
                ax1.axvline(x=idx, color='gray', linestyle='--', linewidth=1, alpha=0.7)
                
            # Set title for subplot
            ax1.set_title(f"{instrument} Price and Factors", fontsize=14)
            
            # Remove individual legends from subplots
            # We'll add a combined legend outside the plots
    
    # Set x-tick locations and labels for the bottom subplot
    # Display 10 evenly spaced ticks with corresponding time labels
    if len(x) > 0:
        tick_positions = np.linspace(0, len(x)-1, num=10, dtype=int)
        axs[-1].set_xticks(tick_positions)
        axs[-1].set_xticklabels([x_labels[i] for i in tick_positions], rotation=45)
    
    # Get all handles and labels for the combined legend
    all_handles = []
    all_labels = []
    
    # Add price handle and label
    all_handles.append(plt.Line2D([0], [0], color='black', linewidth=1.5))
    all_labels.append('Price')
    
    # Add factor handles and labels
    for idx, factor_name in enumerate(factor_mapping.keys()):
        color = color_list[idx % len(color_list)]
        all_handles.append(plt.Line2D([0], [0], color=color, linewidth=1.5))
        # Use full factor names without shortening
        all_labels.append(factor_name)
    
    # Calculate appropriate bottom margin based on number of factors
    # More factors need more space for the vertical legend
    legend_height = 0.04 * (len(all_handles) + 1)  # Base height per item plus a little extra
    bottom_margin = min(0.2, legend_height)  # Cap at 40% of figure height to avoid extreme cases
    
    # Add the legend below the subplots - single column (vertical) layout
    fig.subplots_adjust(bottom=bottom_margin)  # Make room for the legend at the bottom
    fig.legend(all_handles, all_labels, loc='lower center', ncol=1, fontsize=9, 
               bbox_to_anchor=(0.5, 0), frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout with appropriate rect to leave space for legend
    plt.tight_layout(rect=[0, bottom_margin, 1, 0.95])
    
    # Save the figure
    plt.savefig(by_week_dir / f"week_{week_start.strftime('%Y-%m-%d')}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

print(f"Weekly visualizations completed and saved to {by_week_dir}")