# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 09:58:28 2025

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
import json


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.timeutils import (get_rolling_dates, get_lb_fit_periods, get_lb_list_fit_periods, 
                             find_matching_select_period, period_shortcut)
from test_and_eval.factor_evaluation import FactorEvaluation


# %%
def generate_process_names(ind_cate, org_fac_name, selected_factors, ts_trans_list):
    """
    生成 process_name_list，根据因子路径和时间序列转换名生成路径名。

    参数:
    - ind_cate: str，行业分类
    - org_fac_name: str，原始因子名称
    - selected_factors: List[List[str]]，每个因子路径是一个列表，basic_fac_name 在索引2
    - ts_trans_list: List[str]，时间序列转换名称列表

    返回:
    - process_name_list: List[str]，生成的处理路径列表
    """
    new_ind_cate = f'{ind_cate}/{org_fac_name}'
    process_name_list = []
    for factor_paths in selected_factors:
        basic_fac_name = factor_paths[2]
        process_name_list.append(f'{new_ind_cate}/{basic_fac_name}/org')
        for ts_trans_name in ts_trans_list:
            process_name_list.append(f'{new_ind_cate}/{basic_fac_name}/org_TS_{ts_trans_name}')
    return process_name_list


def process_eval_for_period(fp, dfp, 
                            select_fit_periods, select_dir, 
                            org_fac_name, ind_cate, 
                            ts_trans_list, factor_data_dir, 
                            tag_name, test_name_list, fe):
    """
    执行单个周期的因子评估流程。
    """
    target_select_period = find_matching_select_period(fp[1], select_fit_periods)
    select_period_shortcut = period_shortcut(*target_select_period)
    
    selected_factors_path = select_dir / org_fac_name / select_period_shortcut / 'final_factors.json'
    with open(selected_factors_path, 'r') as f:
        selected_factors = json.load(f)

    process_name_list = generate_process_names(ind_cate, org_fac_name, selected_factors, ts_trans_list)

    process_info_list = [
        (factor_data_dir, tag_name, process_name, mode, test_name)
        for process_name in process_name_list
        for mode, test_name in test_name_list
    ]

    fe.eval_one_period(*fp, *dfp, process_name_list=process_info_list)


# %%
select_rolling_params = {
    "lb_list": [96],
    "fstart": "20150101",
    "rolling_params": {
        "end_by": "time",
        "rrule_kwargs": {
            "freq": "M",
            "interval": 6,
            "bymonthday": 1
        }
    }
}

eval_rolling_params = {
    "lb_list": [6, 96],
    "data_lb": 96,
    "rolling_params": {
        "end_by": "time",
        "rrule_kwargs": {
            "freq": "M",
            "interval": 3,
            "bymonthday": 1
        }
    }
}


# %%
dynamic_eval_name = ''
fstart, pstart, puntil = '20150101', '20180101', '20200101'
n_workers = 1
eval_type = 'rolling'
check_consistency = True


# %%
path_config = load_path_config(project_dir)
result_dir = Path(path_config['result'])
param_dir = Path(path_config['param'])
factor_factory_param_dir = Path(path_config['factor_factory_param'])
batch_test_param_dir = param_dir / 'batch_test'
gen_batch_config_dir = factor_factory_param_dir / 'generate_batch_config'


# %% dynamic eval params
params = toml.load(param_dir / 'dynamic_eval' / f'{dynamic_eval_name}.toml')
generate_batch_name = params['generate_batch_name']
rolling_select_name = params['rolling_select_name']
batch_test_name = params['batch_test_name']
eval_rolling_name = params['eval_rolling_name']
factor_data_dir = params['factor_data_dir']

base_eval_name = params.get('base_eval_name', 'default_v0')
tag_name = params.get('tag_name', 'zxt')


# %% rolling params
select_rolling_params = toml.load(param_dir / 'rolling_select_basic_features' / f'{rolling_select_name}.toml')
eval_rolling_params = toml.load(param_dir / 'eval_rolling' / f'{eval_rolling_name}.toml')


# %% test list
config = toml.load(batch_test_param_dir / f'{batch_test_name}.toml')
test_name_list = [(single_test_param['mode'], single_test_param['test_name'])
                  for single_test_param in config['test_list']]

# %% process list
config_path = gen_batch_config_dir / f'{generate_batch_name}.toml'
with open(config_path, 'r') as f:
    config = toml.load(f)
    
select_name = config['select_name']
ind_cate = config['ind_cate']
org_fac_name = config['org_fac_name']
ts_trans_list = config['ts_trans_list']
select_dir = result_dir / 'select_basic_features' / select_name


# %%
rolling_dates = get_rolling_dates(fstart, pstart, puntil)

select_lb = select_rolling_params['lb_list'][0]
select_rolling_pr = select_rolling_params['rolling_params']
eval_lb = eval_rolling_params['lb_list']
eval_data_lb = eval_rolling_params['data_lb']
eval_rolling_pr = eval_rolling_params['rolling_params']

select_fit_periods = get_lb_fit_periods(rolling_dates, select_rolling_pr, select_lb)
eval_fit_periods_list = get_lb_list_fit_periods(rolling_dates, eval_rolling_pr, eval_lb)
eval_data_fit_periods = get_lb_fit_periods(rolling_dates, eval_rolling_pr, eval_data_lb)


# %%
fe = FactorEvaluation(base_eval_name, n_workers=n_workers, check_consistency=check_consistency)
for fit_periods in eval_fit_periods_list:
    if eval_type == 'rolling':
        for i_p, (fp, dfp) in enumerate(list(zip(fit_periods, eval_data_fit_periods))):
            process_eval_for_period(
                fp, dfp,
                select_fit_periods, select_dir,
                org_fac_name, ind_cate,
                ts_trans_list, factor_data_dir,
                tag_name, test_name_list, fe
            )
    elif eval_type == 'update':
        fp = fit_periods[-1]
        dfp = eval_data_fit_periods[-1]
        process_eval_for_period(
            fp, dfp,
            select_fit_periods, select_dir,
            org_fac_name, ind_cate,
            ts_trans_list, factor_data_dir,
            tag_name, test_name_list, fe
        )