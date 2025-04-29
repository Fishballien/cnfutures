# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 15:22:19 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[3]
sys.path.append(str(project_dir))


# %%
from test_and_eval.factor_evaluation import eval_one_factor_one_period_net_public
from test_and_eval.scores import get_general_return_metrics


# %%
compare_name = 'org_vs_sharpe_10_8y'


# %%
fee = 0.00024

pstart = '20170101'
puntil = '20250326'

factor_config = {
    'factor_name': 'predict_avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18',
    'factor_dir': r'/mnt/30.132_xt_data1/xintang/CNIndexFutures/timeseries/factor_test/results/model/avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18/predict',
    'direction': 1,
    'simplified_name': 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18'
    }

feature_name = 'realized_vol_multi_wd'


test_name_org = 'trade_ver3_futtwap_sp1min_noscale_icim_v6'

walk_forward_mapping = {
    # 'sharpe_greater_10': ['v0', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    'sharpe_greater_10': ['sharpe_10_8y', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    # 'sharpe_greater_12': ['v2', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    # 'annrtn_greater_10': ['rtn_10', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    # 'annrtn_greater_11': ['rtn_11', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    }

target_metrics = ['net_return_annualized', 'net_max_dd', 'net_sharpe_ratio', 'net_calmar_ratio',
                  'net_ulcer_index', 'hsr', 'profit_per_trade',]


# %%
analysis_dir = Path(r'/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/analysis/filter_vol_and_trend')
save_dir = analysis_dir / feature_name / factor_config['simplified_name']
test_dir = save_dir / 'test'
org_test_data_dir = test_dir / test_name_org / 'data'
wkfwd_dir = save_dir / 'walk_forward'
synthesis_dir = wkfwd_dir / 'synthesis'
compare_dir = wkfwd_dir / 'compare' / compare_name


for dir_path in [compare_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)
    
    
# %%
res_list = []
net_dict = {}

## org
res_info = {
    'pred_name': 'org',
    }
res_dict, net = eval_one_factor_one_period_net_public(
    f"{factor_config['simplified_name']}_org", res_info, org_test_data_dir, pstart, puntil, fee, return_net=True)
res_list.append(res_dict)
net_dict['org'] = net

## synthesis
for synthesis_shortcut in walk_forward_mapping:
    synthesis_name, final_test_name = walk_forward_mapping[synthesis_shortcut]
    res_info = {
        'pred_name': synthesis_shortcut,
        }
    test_data_dir = synthesis_dir / synthesis_name / 'test' / final_test_name / 'data'
    res_dict, net = eval_one_factor_one_period_net_public(
        f'predict_{synthesis_name}', res_info, test_data_dir, pstart, puntil, fee, return_net=True)
    res_list.append(res_dict)
    net_dict[synthesis_shortcut] = net
    
    
# %%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Calculate metrics for average of non-org nets
non_org_keys = [k for k in net_dict.keys() if k != 'org']
if non_org_keys:
    # Calculate average net returns for non-org strategies
    non_org_nets = pd.DataFrame({k: net_dict[k] for k in non_org_keys})
    avg_non_org = non_org_nets.mean(axis=1)
    
    # Calculate metrics for average net
    metrics = get_general_return_metrics(avg_non_org)
    renamed_metrics = {f'net_{m}': v for m, v in metrics.items()}
    
    # Create result dict for average filtered
    res_dict = {'pred_name': 'avg_filtered'}
    res_dict.update(renamed_metrics)
    
    # Add to res_list
    res_list.append(res_dict)
    
    # Calculate the average of org and avg_filtered
    avg_combined = (avg_non_org + net_dict['org']) / 2
    
    # Calculate metrics for the combined average
    metrics = get_general_return_metrics(avg_combined)
    renamed_metrics = {f'net_{m}': v for m, v in metrics.items()}
    
    # Create result dict for combined average - renamed from 'avg_org_filtered' to 'org+avg_filter'
    res_dict = {'pred_name': 'org+avg_filter'}
    res_dict.update(renamed_metrics)
    
    # Add to res_list
    res_list.append(res_dict)

# Convert res_list to a DataFrame
res_df = pd.DataFrame(res_list)
# Set pred_name as index for the results
if 'pred_name' in res_df.columns:
    res_df = res_df.set_index('pred_name')

# Calculate cumulative returns for each strategy
cum_returns = {}
for key, net in net_dict.items():
    cum_returns[key] = net.cumsum()

# Create a DataFrame from cum_returns dictionary
cum_returns_df = pd.DataFrame(cum_returns)

# Calculate average of non-org strategies
non_org_keys = [k for k in net_dict.keys() if k != 'org']
non_org_nets = pd.DataFrame({k: net_dict[k] for k in non_org_keys})
avg_non_org = non_org_nets.mean(axis=1)
cum_avg_non_org = avg_non_org.cumsum()

# Calculate difference between average non-org and org
diff_series = avg_non_org - net_dict['org']
cum_diff = diff_series.cumsum()

# Calculate average of org and avg_filtered for plotting
avg_combined = (avg_non_org + net_dict['org']) / 2
cum_avg_combined = avg_combined.cumsum()

# Create the figure and grid
fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.5])

# Plot cumulative returns
ax1 = plt.subplot(gs[0])

# Plot non-org strategies with light colors
colors = plt.cm.tab10(np.linspace(0, 1, len(non_org_keys)))
for i, key in enumerate(non_org_keys):
    ax1.plot(cum_returns[key], alpha=0.3, color=colors[i], label=key)

# Plot average of non-org strategies with a brighter blue
ax1.plot(cum_avg_non_org, linewidth=2, color='#1E88E5', label='Avg Filtered')

# Plot org strategy with a brighter red
ax1.plot(cum_returns['org'], linewidth=2, color='#D81B60', label='Original')

# Plot average of org and filtered with a brighter purple - updated label to match the table
ax1.plot(cum_avg_combined, linewidth=2, color='#8E24AA', label='org+avg_filter')

# Plot difference with a brighter green
ax1.plot(cum_diff, linewidth=2, color='#00C853', linestyle='--', label='Avg Filtered - Org')

ax1.set_title('Cumulative Returns')
ax1.legend(loc='best')
ax1.grid(True)

# Create table for the target metrics
ax2 = plt.subplot(gs[1])
ax2.axis('off')

# Filter DataFrame to include only target_metrics
target_df = res_df[target_metrics].copy() if len(res_df) > 0 else pd.DataFrame()

# Format the table data
formatted_df = target_df.applymap(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

# Create table
table = ax2.table(
    cellText=formatted_df.values,
    rowLabels=formatted_df.index,
    colLabels=formatted_df.columns,
    loc='center',
    cellLoc='center'
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Make the 'org' and 'org+avg_filter' rows bold
for row_idx, row_label in enumerate(formatted_df.index):
    if row_label == 'org' or row_label == 'org+avg_filter':
        # Bold the row label
        cell = table[row_idx+1, -1]  # +1 for the header row offset
        cell.set_text_props(fontweight='bold')
        
        # Bold all cells in the row
        for col_idx in range(len(formatted_df.columns)):
            cell = table[row_idx+1, col_idx]
            cell.set_text_props(fontweight='bold')

# Add title to the table section
ax2.text(0.5, 0.95, 'Performance Metrics', horizontalalignment='center', 
         fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(str(compare_dir / 'cumulative_returns_with_metrics.png'), dpi=300, bbox_inches='tight')
plt.show()

# Also save the metrics data to CSV for reference
formatted_df.to_csv(str(compare_dir / 'performance_metrics.csv'))

# Save the cumulative returns data for further analysis
cum_returns_df.to_csv(str(compare_dir / 'cumulative_returns.csv'))
cum_diff.to_csv(str(compare_dir / 'cumulative_diff.csv'))

# Save the full results dataframe with all metrics
res_df.to_csv(str(compare_dir / 'all_metrics.csv'))

print(f"Visualization and data files saved to {compare_dir}:"
      f"\n- cumulative_returns_with_metrics.png"
      f"\n- performance_metrics.csv"
      f"\n- cumulative_returns.csv"
      f"\n- cumulative_diff.csv"
      f"\n- all_metrics.csv")