# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:39:30 2025

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


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from test_and_eval.scores import get_general_return_metrics


# %%
compare_name = 'compare_v1.2_filter'
model_list = [
    ['avg_agg_250203_by_trade_net_v6', 'trade_ver0_futtwap_sp1min_s240d_icim'],
    ['avg_agg_250203_by_trade_net_v6', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    ['avg_agg_250218_2_by_trade_net_v6', 'trade_ver0_futtwap_sp1min_s240d_icim'],
    ['avg_agg_250218_2_by_trade_net_v6', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    ['avg_agg_250218_2_by_trade_net_v10', 'trade_ver0_futtwap_sp1min_s240d_icim'],
    ['avg_agg_250218_2_by_trade_net_v10', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    ['avg_agg_250218_2_by_trade_net_v11', 'trade_ver0_futtwap_sp1min_s240d_icim'],
    ['avg_agg_250218_2_by_trade_net_v11', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    ['avg_agg_250218_2_by_trade_net_v12', 'trade_ver0_futtwap_sp1min_s240d_icim'],
    ['avg_agg_250218_2_by_trade_net_v12', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    ['avg_agg_250218_2_by_trade_net_v13', 'trade_ver0_futtwap_sp1min_s240d_icim'],
    ['avg_agg_250218_2_by_trade_net_v13', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    ['avg_agg_250218_2_by_trade_net_v14', 'trade_ver0_futtwap_sp1min_s240d_icim'],
    ['avg_agg_250218_2_by_trade_net_v14', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    ['avg_agg_250218_2_by_trade_net_v15', 'trade_ver0_futtwap_sp1min_s240d_icim'],
    ['avg_agg_250218_2_by_trade_net_v15', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    ['avg_agg_250218_3_by_trade_net_v6', 'trade_ver0_futtwap_sp1min_s240d_icim'],
    ['avg_agg_250218_3_by_trade_net_v6', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    ]
fee = 0.00024
date_start = '20160101'
date_end = '20250101'


# %%
path_config = load_path_config(project_dir)
result_dir = Path(path_config['result'])
model_dir = result_dir / 'model'
summary_dir = result_dir / 'analysis' / 'model_compare' / compare_name
summary_dir.mkdir(parents=True, exist_ok=True)


# %%
final_res_list = []
for model_name, test_name in model_list:
    test_data_dir = model_dir / model_name / 'test' / test_name / 'data'
    pred_name = f'predict_{model_name}'
    
    res_dict = {
        'model_name': model_name, 
        'test_name': test_name, 
        }
    
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
    res_dict.update({'profit_per_trade': profit_per_trade*1000})
    
    # Calculate metrics year by year
    years = net.index.year.unique()
    for year in years:
        net_year = net[net.index.year == year]
        year_metrics = get_general_return_metrics(net_year)
        for m, v in year_metrics.items():
            res_dict[f'net_{m}_{year}'] = v
    
    final_res_list.append(res_dict)
final_res = pd.DataFrame(final_res_list)
final_res.to_csv(summary_dir / f'compare_{compare_name}.csv', index=None)

