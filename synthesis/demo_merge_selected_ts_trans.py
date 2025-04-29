# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 14:22:49 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datetime import datetime


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.datautils import compute_dataframe_dict_average, add_dataframe_to_dataframe_reindex
from utils.timeutils import RollingPeriods, period_shortcut


# %%
select_name = ''
date_start = '20160101'
date_end = '20250101'


# %%
path_config = load_path_config(project_dir)
result_dir = Path(path_config['result'])
test_dir = result_dir / 'test'
select_dir = result_dir / 'select_ts_trans' / select_name
merged_dir = result_dir / 'merge_selected_ts_trans'


# %%
period_name = period_shortcut(date_start, date_end)
select_period_dir = select_dir / period_name
merged_period_dir = merged_dir / period_name


final_selected_factors_path = select_period_dir / 'final_selected_factors.csv'
if not os.path.exists(final_selected_factors_path):
    pass # TODO: return
    
final_selected_factors = pd.read_csv(final_selected_factors_path)
grouped = final_selected_factors.groupby('group')
factor_dict, weight_dict = {}, {}
for group_num, group_info in tqdm(grouped, desc='load_factors_by_group'):
    group_factor_dict, group_weight_dict = {}, {}
    for idx in group_info.index:
        test_name = group_info.loc[idx, 'test_name']
        process_name = group_info.loc[idx, 'process_name']
        factor = group_info.loc[idx, 'factor']
        scaled_fac_path = test_dir / test_name / process_name / 'data' / f'scaled_{factor}.parquet'
        scaled_fac = pd.read_parquet(scaled_fac_path)
        group_factor_dict[idx] = scaled_fac
        group_weight_dict[idx] = 1
    group_avg = compute_dataframe_dict_average(group_factor_dict, group_weight_dict)
    factor_dict[group_num] = group_avg
    weight_dict[group_num] = 1
factor_avg = compute_dataframe_dict_average(factor_dict, weight_dict)
factor_avg.to_parquet(merged_period_dir / f'avg_predict_{period_name}.parquet')
