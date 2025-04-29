# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:32:55 2025

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
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[2]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from test_and_eval.scores import get_general_return_metrics


# %%
compare_name = 'compare_v1.2_seperate_long_short'
model_list = [
    ['model', 'avg_agg_250218_3_fix_tfe_by_trade_net_v4', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    ['merged_model', '1.2_long_short_seperate_v0', 'traded_futtwap_sp1min_s240d_icim_v6_noscale'],
    ['merged_model', '1.2_long_short_seperate_v0_lb8y', 'traded_futtwap_sp1min_s240d_icim_v6_noscale'],
    ['merged_model', '1.2_long_v0_short_v1_lb8y', 'traded_futtwap_sp1min_s240d_icim_v6_noscale'],
    ]
fee = 0.00024
date_start = '20160101'
date_end = '20250101'


# %%
path_config = load_path_config(project_dir)
result_dir = Path(path_config['result'])
summary_dir = result_dir / 'analysis' / 'model_compare' / compare_name
summary_dir.mkdir(parents=True, exist_ok=True)


# %%
final_res_list = []
fig, ax = plt.subplots(figsize=(10, 6))

for model_type, model_name, test_name in model_list:
    test_data_dir = result_dir / model_type / model_name / 'test' / test_name / 'data'
    pred_name = f'predict_{model_name}' if model_type == 'model' else f'pos_{model_name}'
    
    res_dict = {'model_name': model_name, 'test_name': test_name}
    
    test_data = {}
    for data_type in ('gpd', 'hsr'):
        data_path = test_data_dir / f'{data_type}_{pred_name}.pkl'
        with open(data_path, 'rb') as f:
            test_data[data_type] = pickle.load(f)
            
    df_gp = test_data['gpd']['all']
    df_hsr = test_data['hsr']['all']
    
    df_gp = df_gp[(df_gp.index >= date_start) & (df_gp.index <= date_end)]
    df_hsr = df_hsr[(df_hsr.index >= date_start) & (df_hsr.index <= date_end)]
    
    net = (df_gp['return'] - fee * df_hsr['avg']).fillna(0)
    metrics = get_general_return_metrics(net.values)
    renamed_metrics = {f'net_{m}': v for m, v in metrics.items()}
    res_dict.update(renamed_metrics)
    
    profit_per_trade = df_gp["return"].sum() / df_hsr["avg"].sum()
    res_dict.update({'profit_per_trade': profit_per_trade * 1000})
    
    # Add net series for plotting
    ax.plot(net.index, np.cumsum(net.values), label=f"{model_name} (Sharpe: {renamed_metrics['net_sharpe_ratio']:.2f}, Profit per Trade: {profit_per_trade*1000:.2f})")
    
    final_res_list.append(res_dict)

# Convert final results to DataFrame
final_res = pd.DataFrame(final_res_list)
final_res.to_csv(summary_dir / f'compare_{compare_name}.csv', index=None)

# Plot settings
ax.set_title(f"Comparison of Models - {compare_name}")
ax.set_xlabel('Date')
ax.set_ylabel('Net Return')
ax.grid(True)
ax.legend(title='Models')

# Save the plot
plt.tight_layout()
plt.savefig(summary_dir / f'{compare_name}_comparison_plot.png')
plt.show()