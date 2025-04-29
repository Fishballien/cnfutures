# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:12:24 2025

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
import toml


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.datautils import compute_dataframe_dict_average
from test_and_eval.factor_tester import FactorTesterByContinuous, FactorTesterByDiscrete


# %%
merge_name = '1.2.0'


# %%
path_config = load_path_config(project_dir)
param_dir = Path(path_config['param'])
result_dir = Path(path_config['result'])
model_dir = result_dir / 'model'
merged_model_dir = result_dir / 'merged_model' / merge_name
pos_dir = merged_model_dir / 'pos'
pos_dir.mkdir(parents=True, exist_ok=True)


# %%
params = toml.load(param_dir / 'merge_model' / f'{merge_name}.toml')


# %%
model_list = params['model_list']
pos_dict = {}
for model_info in model_list:
    model_name = model_info['model_name']
    test_name = model_info['test_name']
    
    test_data_dir = model_dir / model_name / 'test' / test_name / 'data'
    pos_filename = f'pos_predict_{model_name}'
    pos_path = test_data_dir / f'{pos_filename}.parquet'
    
    pos = pd.read_parquet(pos_path)
    pos_dict[(model_name, test_name)] = pos
    
pos_average = compute_dataframe_dict_average(pos_dict)
pos_average.to_csv(pos_dir / f'pos_{merge_name}.csv')
pos_average.to_parquet(pos_dir / f'pos_{merge_name}.parquet')

pos1 = pos_dict[(model_list[0]['model_name'], model_list[0]['test_name'])]
pos1 = pos_dict[(model_list[0]['model_name'], model_list[0]['test_name'])]


# %%
test_list = params['test_list']
for test_info in test_list:
    mode = test_info['mode']
    test_name = test_info['test_name']
    date_start = test_info.get('date_start')
    if mode == 'test':
        test_class = FactorTesterByContinuous
    elif mode == 'trade':
        test_class = FactorTesterByDiscrete
    else:
        NotImplementedError()

    ft = test_class(None, None, pos_dir, test_name=test_name, result_dir=merged_model_dir)
    ft.test_one_factor(f'pos_{merge_name}', date_start=date_start)