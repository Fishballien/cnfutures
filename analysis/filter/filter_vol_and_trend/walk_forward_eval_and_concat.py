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
import os
import sys
from pathlib import Path
import pandas as pd
import itertools
from tqdm import tqdm
from datetime import datetime
import yaml
from functools import partial


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[3]
sys.path.append(str(project_dir))


# %%
from test_and_eval.factor_evaluation import eval_one_factor_one_period_net_public
from utils.timeutils import RollingPeriods, period_shortcut
from utils.datautils import compute_dataframe_dict_average, add_dataframe_to_dataframe_reindex
from test_and_eval.factor_tester import FactorTesterByDiscrete
from synthesis.filter_methods import *


# %%
to_eval = True
eval_skip_exists = False
to_filter = True
to_synthesis = True
to_test = True


# %%
synthesis_name = 'sharpe_10_8y'


# %%
factor_config = {
    'factor_name': 'predict_avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18',
    'factor_dir': r'/mnt/30.132_xt_data1/xintang/CNIndexFutures/timeseries/factor_test/results/model/avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18/predict',
    'direction': 1,
    'simplified_name': 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18'
    }

feature_name = 'realized_vol_multi_wd'

# Parameter grid
param_grid = {
    'vol_T': [0.006, 0.007, 0.008, 0.009, 0.01, 0.0125],
    'vol_k': [600, 800, 1000],
    'trend_k': [200, 1000, 2000, 3000],
    'trend_th': [-0.02, 0, 0.0005, 0.002]
}

test_name = 'trade_ver3_futtwap_sp1min_s240d_icim_v6'
test_name_org = 'trade_ver3_futtwap_sp1min_noscale_icim_v6'

fstart = '20160101'
pstart = '20170101'
puntil = '20250326'
window_kwargs = {'months': 96}
rrule_kwargs = {'freq': 'M', 'interval': 1}
end_by = 'date'

fee = 0.00024

filter_name = 'filter_func_dynamic'
filter_params = {
    'pred_name': factor_config['factor_name'],
    'conditions': [
        {'target': 'net_sharpe_ratio', 'operator': 'greater', 'threshold': 1.1, 'is_multiplier': True},  # 大于参考值的1.2倍
    ],
    'min_count': 5,
    'sort_target': 'net_sharpe_ratio',
    'sort_ascending': False
}

final_test_name = 'trade_ver3_futtwap_sp1min_s240d_icim_v6'


# %%
analysis_dir = Path(r'/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/analysis/filter_vol_and_trend')
save_dir = analysis_dir / feature_name / factor_config['simplified_name']
filtered_fac_dir = save_dir / 'filtered_fac'
org_fac_dir = save_dir / 'org_fac'
test_dir = save_dir / 'test'
test_data_dir = test_dir / test_name / 'data'
org_test_data_dir = test_dir / test_name_org / 'data'
eval_dir = save_dir / 'eval'
wkfwd_dir = save_dir / 'walk_forward'
param_dir = wkfwd_dir / 'param'
filtered_dir = wkfwd_dir / 'filtered' / synthesis_name
synthesis_dir = wkfwd_dir / 'synthesis' / synthesis_name
final_fac_dir = synthesis_dir / 'prediction'

for dir_path in [eval_dir, wkfwd_dir, param_dir, filtered_dir, synthesis_dir, final_fac_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)
    
    
# %% save param
# Create a dictionary with all parameters
config = {
    'synthesis_name': synthesis_name,
    'factor_config': factor_config,
    'param_grid': param_grid,
    'test_name': test_name,
    'test_name_org': test_name_org,
    'fstart': fstart,
    'pstart': pstart,
    'puntil': puntil,
    'window_kwargs': window_kwargs,
    'rrule_kwargs': rrule_kwargs,
    'end_by': end_by,
    'filter_name': filter_name,
    'filter_params': filter_params,
    'final_test_name': final_test_name
}

# Save to YAML file
yaml_file_path = param_dir / f"{synthesis_name}.yaml"
with open(yaml_file_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"YAML file saved to: {yaml_file_path}")
    

# %%
rolling = RollingPeriods(
    fstart=datetime.strptime(fstart, '%Y%m%d'),
    pstart=datetime.strptime(pstart, '%Y%m%d'),
    puntil=datetime.strptime(puntil, '%Y%m%d'),
    window_kwargs=window_kwargs,
    rrule_kwargs=rrule_kwargs,
    end_by=end_by,
    )


# %% eval one period
params_list = [
    dict(zip(param_grid.keys(), values)) 
    for values in itertools.product(*param_grid.values())
]

def eval_one_period(date_start, date_end, factor_config, params_list, 
                    org_test_data_dir, test_data_dir, 
                    filtered_fac_dir,
                    fee):
    simplified_name = factor_config['simplified_name']
    factor_dir = factor_config['factor_dir']
    
    # Update the evaluation part
    res_list = []
    
    ## org
    res_info = {
        'pred_name': factor_config['factor_name'],  
        'root_dir': factor_dir,
    }
    # 评估结果
    res_dict = eval_one_factor_one_period_net_public(
        f'{simplified_name}_org', res_info, org_test_data_dir, date_start, date_end, fee)
    
    res_list.append(res_dict)
    
    ## 遍历交易规则参数
    for params in params_list:
        vol_T = params['vol_T']
        vol_k = params['vol_k']
        trend_k = params['trend_k']
        trend_th = params['trend_th']
                                            
        # 生成预测名称
        pred_name = f'{simplified_name}_vT{vol_T}_vk{vol_k}_tk{trend_k}_tth{trend_th}'
        
        # 基本结果信息
        res_info = {
            'pred_name': pred_name, 
            'root_dir': filtered_fac_dir,
            'vol_T': vol_T,
            'vol_k': vol_k,
            'trend_k': trend_k,
            'trend_th': trend_th,
        }
        
        
        # 评估结果
        res_dict = eval_one_factor_one_period_net_public(
            pred_name, res_info, test_data_dir, date_start, date_end, fee)
        
        res_list.append(res_dict)
        
    res_df = pd.DataFrame(res_list)
    return res_df


# %% rolling eval
if to_eval:
    eval_one_period_func = partial(eval_one_period, factor_config=factor_config, params_list=params_list, 
                                   org_test_data_dir=org_test_data_dir, test_data_dir=test_data_dir, 
                                   filtered_fac_dir=filtered_fac_dir, fee=fee)
    for fp in tqdm(rolling.fit_periods, 'rolling eval'):
        fit_period = period_shortcut(*fp)
        eval_path = eval_dir / f'eval_summary_{fit_period}.csv'
        if eval_skip_exists and os.path.exists(eval_path):
            continue
        eval_res = eval_one_period_func(*fp)
        eval_res.to_csv(eval_path, index=None)
    

# %% rolling filter
if to_filter:
    filter_func = globals()[filter_name]
    for fp in tqdm(rolling.fit_periods, 'rolling filter'):
        fit_period = period_shortcut(*fp)
        eval_res = pd.read_csv(eval_dir / f'eval_summary_{fit_period}.csv')
        filtered = eval_res[filter_func(eval_res, **filter_params)].reset_index()
        filtered.to_csv(filtered_dir / f'filtered_{fit_period}.csv', index=None)
    

# %% synthesis
if to_synthesis:
    predict_all = pd.DataFrame()
    for fp, pp in tqdm(list(zip(rolling.fit_periods, rolling.predict_periods)), desc='rolling predict'):
        fit_period = period_shortcut(*fp)
        filtered = pd.read_csv(filtered_dir / f'filtered_{fit_period}.csv')
        predict_dict, weight_dict = {}, {}
        for idx in filtered.index:
            pred_name, root_dir = filtered.loc[idx, ['pred_name', 'root_dir']]
            predict_path = Path(root_dir) / f'{pred_name}.parquet'
            predict = pd.read_parquet(predict_path)
            predict_dict[pred_name] = predict.loc[pp[0]:pp[1]]
            weight_dict[pred_name] = 1
        predict_avg_period = compute_dataframe_dict_average(predict_dict, weight_dict)
        predict_all = add_dataframe_to_dataframe_reindex(predict_all, predict_avg_period)
        
    predict_all.to_csv(final_fac_dir / f'predict_{synthesis_name}.csv')
    predict_all.to_parquet(final_fac_dir / f'predict_{synthesis_name}.parquet')


# %%
if to_test:
    tester = FactorTesterByDiscrete(None, None, final_fac_dir, test_name=final_test_name, 
                                    result_dir=synthesis_dir)
    tester.test_one_factor(f'predict_{synthesis_name}')
    
    