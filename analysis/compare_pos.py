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


# %%
compare_name = 'summarize_250407'
model_list = [
    # ['merged_model', '1.2.3_fix_tfe_v4'],
    # ['merged_model', 'tf_trade_avg'],
    # # ['merged_model', 'zxt_1.2_merge_with_tf_tradelob_250321'],
    # ['model', 'trade1'],
    ['model', ('avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18', 'trade_ver3_futtwap_sp1min_s240d_icim_v6', 'hold_overnight')],
    ['model', ('avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18', 'trade_ver3_3_futtwap_sp1min_s240d_icim_v6', 'intraday_only')],
    ]
price_name = 't1min_fq1min_dl1min'


# %%
path_config = load_path_config(project_dir)
param_dir = Path(path_config['param'])
result_dir = Path(path_config['result'])
model_dir_mapping = {model_type: result_dir / model_type for model_type in ['model', 'merged_model']}
fut_dir = Path('/mnt/data1/futuretwap')
save_dir =Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\compare_pos') / compare_name
save_dir.mkdir(parents=True, exist_ok=True)


# %%
price_data = pd.read_parquet(fut_dir / f'{price_name}.parquet')

# Load all model positions into a dictionary
model_positions = {}
for model_type, model_info in model_list:
    if model_type == 'merged_model':
        model_name = model_info
        model_pos_path = model_dir_mapping[model_type] / model_name / 'pos' / f'pos_{model_name}.parquet'
        tag_name = model_name
    elif model_type == 'model':
        model_name, test_name, tag_name = model_info
        model_pos_path = model_dir_mapping[model_type] / model_name / 'test' / test_name / 'data' / f'pos_predict_{model_name}.parquet'
    model_pos = pd.read_parquet(model_pos_path)
    model_positions[tag_name] = model_pos
    
    
# %%
by_week_dir = save_dir / 'by_week'
by_week_dir.mkdir(parents=True, exist_ok=True)

# %% Set up color scheme for models (cute colors)
model_colors = {
    'hold_overnight': '#FF9AA2',  # Soft pink
    'intraday_only': '#FFDAC1',  # Peach
    'trade1': '#B5EAD7'  # Mint green
}

# %% Filter for only IC and IM columns
price_data = price_data[['IC', 'IM']]

# %% Define time range (use the last year of data or specify your own range)
start_date = pd.Timestamp('2025-03-17')
end_date = pd.Timestamp('2025-04-18')

# Filter data to the desired date range
price_data = price_data.loc[(price_data.index >= start_date) & (price_data.index <= end_date)]

# Filter model positions to the same date range
for model_name in model_positions:
    model_positions[model_name] = model_positions[model_name].loc[
        (model_positions[model_name].index >= start_date) & 
        (model_positions[model_name].index <= end_date)
    ]


# %% Create weekly visualizations
for week_start, price_group in price_data.groupby(pd.Grouper(freq='W-MON', label='left', closed='left')):
    # Skip weeks with no data
    if price_group.empty:
        continue
    
    # Get the end of the week
    week_end = week_start + pd.Timedelta(days=7)
    
    # Create figure with 2 subplots (one for IC, one for IM)
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f"Price and Position Data for Week: {week_start.strftime('%Y-%m-%d')} to {(week_end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}", 
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
            
            # Create secondary y-axis for position data
            ax2 = ax1.twinx()
            
            # Plot positions for each model
            for model_name, model_pos in model_positions.items():
                # Filter position data for this week and instrument
                pos_week = model_pos.loc[
                    (model_pos.index >= week_start) & 
                    (model_pos.index < week_end), 
                    instrument
                ].dropna()
                
                if not pos_week.empty:
                    # Need to align position data with price index for plotting
                    aligned_pos = pos_week.reindex(price_series.index, method='ffill')
                    # Only use non-NaN values
                    valid_indices = ~aligned_pos.isna()
                    if valid_indices.any():
                        # Get the corresponding x values for valid positions
                        valid_x = x[valid_indices.values]
                        valid_pos = aligned_pos[valid_indices].values
                        
                        # Plot position data on secondary axis
                        color = model_colors.get(model_name, 'blue')  # Use defined color or default to blue
                        ax2.plot(valid_x, valid_pos, color=color, 
                                linewidth=1.5, linestyle='-', 
                                label=f'{model_name}')
            
            # Set the secondary y-axis label
            ax2.set_ylabel('Position', fontsize=12)
            ax2.set_ylim(-1.2, 1.2)  # Assuming position values are between -1 and 1
            ax2.tick_params(axis='y')
            
            # Add a horizontal dashed line at position 0
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            
            # Find every 9:30 AM for vertical lines (market open)
            nine_thirty_indices = [idx for idx, t in enumerate(price_series.index) if t.strftime('%H:%M') == '09:30']
            for idx in nine_thirty_indices:
                ax1.axvline(x=idx, color='gray', linestyle='--', linewidth=1, alpha=0.7)
                
            # Set title for subplot
            ax1.set_title(f"{instrument} Price and Positions", fontsize=14)
            
            # Add legends
            # Price legend on the left plot
            ax1.legend(loc='upper left')
            
            # Position legends on the right plot
            handles, labels = ax2.get_legend_handles_labels()
            ax2.legend(handles, labels, loc='upper right')
    
    # Set x-tick locations and labels for the bottom subplot
    # Display 10 evenly spaced ticks with corresponding time labels
    if len(x) > 0:
        tick_positions = np.linspace(0, len(x)-1, num=10, dtype=int)
        axs[-1].set_xticks(tick_positions)
        axs[-1].set_xticklabels([x_labels[i] for i in tick_positions], rotation=45)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    plt.savefig(by_week_dir / f"week_{week_start.strftime('%Y-%m-%d')}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

print(f"Weekly visualizations completed and saved to {by_week_dir}")