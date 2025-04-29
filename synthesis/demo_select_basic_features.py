# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 13:27:45 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import os
import re
import sys
import json
import shutil
from pathlib import Path
import pandas as pd
from functools import partial
from typing import List, Union


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.timeutils import period_shortcut
from synthesis.filter_methods import filter_func_dynamic
from synthesis.factor_cluster import (cluster_factors, select_best_test_name, select_topn_per_group,
                                      remove_ma_suffix_from_factors)


# %%
cluster_name = ''
date_start = '20160101'
date_end = '20250101'


# %%
feval_name = 'basis_pct_250416_org_batch_250419_batch_test_v0'


# %%
path_config = load_path_config(project_dir)
result_dir = Path(path_config['result'])
test_dir = result_dir / 'test'
feval_dir = result_dir / 'factor_evaluation' / feval_name
cluster_dir = result_dir / 'cluster'


# %% keep best test version
keep_best_test_by = 'net_sharpe_ratio_long_only'


# %% first filter param
filter_func_name = 'filter_func_dynamic'

filter_params = {
    'conditions': [
        {'target': 'net_sharpe_ratio_long_only', 'operator': 'greater', 'threshold': 0.7, 'is_multiplier': False},
    ],
    'min_count': 0,
    }


# %% cluster param
cluster_params = {
    'cluster_params': {'t': 0.5, 'criterion': 'distance'},
    'linkage_method': 'complete',
    'use_direction': 'long_only',
    }
cluster_params.update({'test_dir': test_dir,})


# %% filter after cluster
final_filter_by = 'net_sharpe_ratio_long_only'
top_n = 1


# %%
period_name = period_shortcut(date_start, date_end)
res_dir = cluster_dir / period_name
res_selected_test_dir = res_dir / 'selected'
res_selected_test_dir.mkdir(parents=True, exist_ok=True)


# %%
path = feval_dir / feval_name / f'factor_eval_{period_name}.csv'
eval_res = pd.read_csv(path)
eval_res['org_fac'] = eval_res['factor'].apply(lambda x: x.split('-', 1)[0])


# %%
org_fac = list(eval_res['org_fac'].unique())[0]
org_fac_data = eval_res[eval_res['org_fac'] == org_fac].copy()


# %% keep best test
keep_best_test_df = select_best_test_name(
    org_fac_data,
    metric=keep_best_test_by,
    )


# %% filst filter
filter_func = globals()[filter_func_name]
filter_func_with_param = partial(filter_func, **filter_params)
first_selected = keep_best_test_df[filter_func_with_param(keep_best_test_df)]


# %%
groups = cluster_factors(first_selected, date_start, date_end, **cluster_params)
first_selected['group'] = groups


# %%
top_factors = select_topn_per_group(
    first_selected, 
    metric=final_filter_by,  # Change to your desired metric
    n=top_n,                                 # Change to your desired N
    ascending=False,                      # True if lower values are better
)


# %%
final_factors = remove_ma_suffix_from_factors(top_factors)
final_factors_to_list = list(zip(final_factors['root_dir'], 
                                 final_factors['process_name'], 
                                 final_factors['factor']))
# 保存json到res_dir
with open(res_dir / 'final_factors.json', 'w', encoding='utf-8') as f:
    json.dump(final_factors_to_list, f, ensure_ascii=False, indent=4)


# %%
def find_files_with_prefix(directory_path: str, target_prefix: str) -> List[str]:
    """
    在指定目录中查找所有以目标前缀开头，后接两个由下划线分隔字段（即总共两个额外字段）的文件名。

    参数：
        directory_path (str): 要查找的文件夹路径
        target_prefix (str): 目标前缀，比如 "IC_xxx_yyy"

    返回：
        List[str]: 匹配到的完整文件名列表
    """
    matched_files = []
    # 构造正则表达式：以目标前缀开头，后接两个下划线字段
    escaped_prefix = re.escape(target_prefix)
    pattern = re.compile(rf"^{escaped_prefix}_[^_]+_[^_]+$")

    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            if pattern.match(filename):
                matched_files.append(filename)

    return matched_files

res_selected_test_info_dir = res_selected_test_dir / 'factor_info'
res_selected_test_plot_dir = res_selected_test_dir / 'plot'
res_selected_test_info_dir.mkdir(parents=True, exist_ok=True)
res_selected_test_plot_dir.mkdir(parents=True, exist_ok=True)


for idx in final_factors.index:
    test_name = final_factors.loc[idx, 'test_name']
    tag_name = final_factors.loc[idx, 'tag_name']
    process_name = final_factors.loc[idx, 'process_name']
    factor_name = final_factors.loc[idx, 'factor']
    
    target_test_dir = test_dir / test_name / tag_name / process_name
    
    factor_info_plot_dir = target_test_dir / 'factor_info'
    factor_info_plot_paths = find_files_with_prefix(factor_info_plot_dir)
    # cp到res_selected_test_info_dir
    
    factor_plot_path = target_test_dir / 'plot' / f'{factor_name}.png'
    # cp到res_selected_test_plot_dir
    
    
# %%
def find_files_with_prefix(directory_path: Union[str, Path], target_prefix: str = None) -> List[str]:
    """
    在指定目录中查找所有以目标前缀开头，后接两个由下划线分隔字段（即总共两个额外字段）的文件名。
    如果未提供前缀，则返回所有文件。

    参数：
        directory_path (str or Path): 要查找的文件夹路径
        target_prefix (str, optional): 目标前缀，比如 "IC_xxx_yyy"，不指定则返回所有文件

    返回：
        List[str]: 匹配到的完整文件名列表
    """
    directory_path = Path(directory_path)
    matched_files = []
    
    if not directory_path.exists() or not directory_path.is_dir():
        print(f"警告: 目录 {directory_path} 不存在或不是一个目录")
        return matched_files
    
    if target_prefix:
        # 构造正则表达式：以目标前缀开头，后接两个下划线字段
        escaped_prefix = re.escape(target_prefix)
        pattern = re.compile(rf"^{escaped_prefix}_[^_]+_[^_]+$")
        
        for filename in os.listdir(directory_path):
            file_path = directory_path / filename
            if file_path.is_file() and pattern.match(filename):
                matched_files.append(filename)
    else:
        # 如果没有指定前缀，则返回所有文件
        matched_files = [filename for filename in os.listdir(directory_path) 
                        if (directory_path / filename).is_file()]
    
    return matched_files


def copy_file(source_path: Union[str, Path], target_path: Union[str, Path], overwrite: bool = True) -> bool:
    """
    复制文件从源路径到目标路径

    参数:
        source_path (str or Path): 源文件路径
        target_path (str or Path): 目标文件路径
        overwrite (bool, optional): 是否覆盖已存在的文件，默认为True

    返回:
        bool: 复制成功返回True，否则返回False
    """
    source_path = Path(source_path)
    target_path = Path(target_path)
    
    # 检查源文件是否存在
    if not source_path.exists() or not source_path.is_file():
        print(f"错误: 源文件 {source_path} 不存在或不是一个文件")
        return False
    
    # 检查目标目录是否存在，不存在则创建
    target_dir = target_path.parent
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查目标文件是否已存在
    if target_path.exists() and not overwrite:
        print(f"警告: 目标文件 {target_path} 已存在且不允许覆盖")
        return False
    
    try:
        shutil.copy2(source_path, target_path)
        return True
    except Exception as e:
        print(f"复制文件时出错: {e}")
        return False


# 创建结果目录
res_selected_test_info_dir = res_selected_test_dir / 'factor_info'
res_selected_test_plot_dir = res_selected_test_dir / 'plot'
res_selected_test_info_dir.mkdir(parents=True, exist_ok=True)
res_selected_test_plot_dir.mkdir(parents=True, exist_ok=True)


# 复制选定的因子信息和图表到结果目录
for idx in final_factors.index:
    test_name = final_factors.loc[idx, 'test_name']
    tag_name = final_factors.loc[idx, 'tag_name']
    process_name = final_factors.loc[idx, 'process_name']
    factor_name = final_factors.loc[idx, 'factor']
    
    # 源目录路径
    target_test_dir = test_dir / test_name / tag_name / process_name
    
    # 复制因子信息图表
    factor_info_plot_dir = target_test_dir / 'factor_info'
    factor_info_files = find_files_with_prefix(factor_info_plot_dir, factor_name)
    
    for file_name in factor_info_files:
        source_file = factor_info_plot_dir / file_name
        target_file = res_selected_test_info_dir / file_name
        copy_result = copy_file(source_file, target_file)
        if copy_result:
            print(f"成功复制因子信息文件: {file_name}")
        else:
            print(f"复制因子信息文件失败: {file_name}")
    
    # 复制因子图表
    factor_plot_source = target_test_dir / 'plot' / f'{factor_name}.png'
    factor_plot_target = res_selected_test_plot_dir / f'{factor_name}.png'
    
    if factor_plot_source.exists():
        copy_result = copy_file(factor_plot_source, factor_plot_target)
        if copy_result:
            print(f"成功复制因子图表: {factor_name}.png")
        else:
            print(f"复制因子图表失败: {factor_name}.png")
    else:
        print(f"警告: 因子图表文件不存在: {factor_plot_source}")

# 保存筛选后的因子信息到CSV文件
final_factors.to_csv(res_dir / 'final_selected_factors.csv', index=False)
print(f"已完成因子筛选，筛选结果保存至: {res_dir}")