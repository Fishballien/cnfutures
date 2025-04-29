# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:07:38 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import subprocess
import toml


# %% add sys path
file_path = Path(__file__).resolve()
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from utils.dirutils import load_path_config
from filter_funcs import *


# %%
def cp_file(source_dir, target_dir, file_name):
    source_path = source_dir / file_name
    target_path = target_dir / file_name
    subprocess.run(['cp', str(source_path), str(target_path)])
    
    
def copy_factors(df, target_dir):
    for _, row in tqdm(list(df.iterrows()), desc='cping factors'):
        root_dir = Path(row['root_dir'])
        process_name = row['process_name']
        factor = row['factor']
        
        file_name = f"{factor}.parquet"
        source_dir = root_dir / process_name
        target_sub_dir = target_dir / process_name
        target_sub_dir.mkdir(parents=True, exist_ok=True)
        
        cp_file(source_dir, target_sub_dir, file_name)
        
        
def copy_test_results(df, test_dir, new_tag):
    for _, row in tqdm(list(df.iterrows()), desc='cping test results'):
        test_name = row['test_name']
        tag_name = row['tag_name']
        process_name = row['process_name']
        factor = row['factor']
        
        target_sub_dir = test_dir / test_name / new_tag / process_name
        target_data_dir = target_sub_dir / 'data'
        target_plot_dir = target_sub_dir / 'plot'
        target_data_dir.mkdir(parents=True, exist_ok=True)
        target_plot_dir.mkdir(parents=True, exist_ok=True)
        
        source_sub_dir = test_dir / test_name / tag_name / process_name
        source_data_dir = source_sub_dir / 'data'
        source_plot_dir = source_sub_dir / 'plot'
        
        for data_type in ('gp', 'hsr', 'ts_test'):
            file_name = f'{data_type}_{factor}.pkl'
            cp_file(source_data_dir, target_data_dir, file_name)
            
        plot_file_name = f'{factor}.jpg'
        cp_file(source_plot_dir, target_plot_dir, plot_file_name)


def process_and_copy_files(new_tag, eval_res, filter_func, select_dir, test_dir):
    filtered_eval_res = eval_res[filter_func(eval_res)].reset_index(drop=True)
    copy_factors(filtered_eval_res, select_dir / new_tag)
    copy_test_results(filtered_eval_res, test_dir, new_tag)

    
# %% Example usage
if __name__=='__main__':

    select_name = 'zxt_select_241211'
    
    path_config = load_path_config(project_dir)
    param_dir = Path(path_config['param']) / 'select'
    result_dir = Path(path_config['result'])
    eval_dir = result_dir / 'factor_evaluation'
    test_dir = result_dir / 'test'
    select_dir = Path(path_config['selected_fac'])
    
    param = toml.load(param_dir / f'{select_name}.toml')
    feval_name = param['feval_name']
    file_name = param['file_name']
    new_tag = param['new_tag']
    filter_func = globals()[param['filter_func']]
    
    
    feval_res_path = eval_dir / feval_name / f'{file_name}.csv'
    feval_res = pd.read_csv(feval_res_path)
    
    process_and_copy_files(new_tag, feval_res, filter_func, select_dir, test_dir)

