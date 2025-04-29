# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 10:43:47 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% import public
import sys
from pathlib import Path
import toml
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from utils.dirutils import load_path_config
from test_and_eval.factor_tester import *
from test_and_eval.factor_evaluation import FactorEvaluation


# %%
def batch_test(process_name, tag_name, factor_data_dir, batch_test_name):
    path_config = load_path_config(project_dir)
    param_dir = Path(path_config['param']) / 'batch_test'
    
    config = toml.load(param_dir / f'{batch_test_name}.toml')
    n_workers = config.get('n_workers')
    
    for single_test_param in config['test_list']:
        mode = single_test_param['mode']
        skip_exists = single_test_param.get('skip_exists', False)
        args_dict = {k:v for k, v in single_test_param.items() if k not in ['mode', 'skip_exists']}
        args_dict.update({
            'process_name': process_name,
            'tag_name': tag_name,
            'factor_data_dir': factor_data_dir,
            })
        if not 'n_workers' in args_dict and n_workers is not None:
            args_dict['n_workers'] = n_workers
        # main
        if mode == 'test':
            test_class = FactorTesterByContinuous
        elif mode == 'trade':
            test_class = FactorTesterByDiscrete
        else:
            NotImplementedError()
        tester = test_class(**args_dict)
        tester.test_multi_factors(skip_exists=skip_exists)
        
        
def batch_test_by_path_and_eval_one_period(ind_cate_name, final_path_name, tag_name, 
                                           factor_data_dir, batch_test_name,
                                           price_name, price_type,
                                           date_start=None, date_end=None, eval_wkr=1): #TODO: 改default的eval param
    path_config = load_path_config(project_dir)
    final_path_dir = Path(path_config['final_ts_path'])
    param_dir = Path(path_config['param'])
    batch_test_param_dir = param_dir / 'batch_test'
    
    config = toml.load(batch_test_param_dir / f'{batch_test_name}.toml')
    test_name_list = [(single_test_param['mode'], single_test_param['test_name'])
                      for single_test_param in config['test_list']]
    
    final_path_file = final_path_dir / f'{final_path_name}.json'
    with open(final_path_file, 'r') as f:
        data = json.load(f)
        final_path_list = data['final_path']
        
    ## test
    process_name_list = []
    for final_path in final_path_list:
        process_name = f'{ind_cate_name}/{final_path}'
        batch_test(process_name, tag_name, factor_data_dir, batch_test_name)
        process_name_list.append(process_name)
    
    ## eval
    process_info_list = [(factor_data_dir, tag_name, process_name, mode, test_name)
                         for process_name in process_name_list
                         for mode, test_name in test_name_list]
    eva_params = {
        'price_name': price_name,
        'price_type': price_type,
        'process_name_list': process_info_list, 
        }
    eval_name = f'{ind_cate_name}_{final_path_name}_{batch_test_name}'
    with open(param_dir / 'feval' / f'{eval_name}.toml', 'w') as toml_file:
        toml.dump(eva_params, toml_file)
        
    fe = FactorEvaluation(eval_name, n_workers=eval_wkr)
    date_start_dt = datetime.strptime(date_start, '%Y%m%d')
    date_end_dt = datetime.strptime(date_end, '%Y%m%d')
    fe.eval_one_period(date_start_dt, date_end_dt, date_start_dt, date_end_dt)
    

# =============================================================================
# # TODO:之后改成读默认eval参数
# def batch_test_by_all_path_and_eval_one_period(ind_cate_name, final_path_name, tag_name, 
#                                                    factor_data_dir, batch_test_name,
#                                                    base_eval_name,
#                                                    date_start=None, date_end=None, eval_wkr=1): #TODO: 改default的eval param
#     path_config = load_path_config(project_dir)
#     final_path_dir = Path(path_config['final_ts_path'])
#     param_dir = Path(path_config['param'])
#     batch_test_param_dir = param_dir / 'batch_test'
#     
#     config = toml.load(batch_test_param_dir / f'{batch_test_name}.toml')
#     test_name_list = [(single_test_param['mode'], single_test_param['test_name'])
#                       for single_test_param in config['test_list']]
#     
#     final_path_file = final_path_dir / f'{final_path_name}.json'
#     with open(final_path_file, 'r') as f:
#         data = json.load(f)
#         final_path_list = data['final_path']
#         
#     ## test
#     process_name_list = []
#     for final_path in final_path_list:
#         process_name = f'{ind_cate_name}/{final_path}'
#         batch_test(process_name, tag_name, factor_data_dir, batch_test_name)
#         process_name_list.append(process_name)
#     
#     ## eval
#     process_info_list = [(factor_data_dir, tag_name, process_name, mode, test_name)
#                          for process_name in process_name_list
#                          for mode, test_name in test_name_list]
#     base_eval_params = toml.load(param_dir / 'feval' / f'{base_eval_name}.toml')
#     eval_params = base_eval_params.copy()
#     eval_params.update({
#         'process_name_list': process_info_list, 
#         })
#     eval_name = f'{ind_cate_name}_{final_path_name}_{batch_test_name}_{base_eval_params}'
#     with open(param_dir / 'feval' / f'{eval_name}.toml', 'w') as toml_file:
#         toml.dump(eval_params, toml_file)
#         
#     fe = FactorEvaluation(eval_name, n_workers=eval_wkr)
#     date_start_dt = datetime.strptime(date_start, '%Y%m%d')
#     date_end_dt = datetime.strptime(date_end, '%Y%m%d')
#     fe.eval_one_period(date_start_dt, date_end_dt, date_start_dt, date_end_dt)
# =============================================================================


def batch_test_by_all_selected_and_eval_one_period(generate_batch_name,
                                                   tag_name, factor_data_dir, batch_test_name,
                                                   base_eval_name,
                                                   date_start=None, date_end=None, eval_wkr=1): #TODO: 改default的eval param
    path_config = load_path_config(project_dir)
    test_result_dir = Path(path_config['result'])
    param_dir = Path(path_config['param'])
    factor_factory_param_dir = Path(path_config['factor_factory_param'])
    select_dir = test_result_dir / 'select_basic_features'
    batch_test_param_dir = param_dir / 'batch_test'
    gen_batch_config_dir = factor_factory_param_dir / 'generate_batch_config'
    
    # test list
    config = toml.load(batch_test_param_dir / f'{batch_test_name}.toml')
    test_name_list = [(single_test_param['mode'], single_test_param['test_name'])
                      for single_test_param in config['test_list']]
    
    # process list
    config_path = gen_batch_config_dir / f'{generate_batch_name}.toml'
    with open(config_path, 'r') as f:
        config = toml.load(f)
        
    select_name = config['select_name']
    ind_cate = config['ind_cate']
    org_fac_name = config['org_fac_name']
    ts_trans_list = config['ts_trans_list']
        
    final_factors_path = select_dir / select_name / org_fac_name / 'all_final_factors.json'
    with open(final_factors_path, 'r') as f:
        final_factors = json.load(f)
        
    new_ind_cate = f'{ind_cate}/{org_fac_name}'
    process_name_list = []
    for factor_paths in final_factors:
        basic_fac_name = factor_paths[2]
        process_name_list.append(f'{new_ind_cate}/{basic_fac_name}/org')
        for ts_trans_name in ts_trans_list:
            process_name_list.append(f'{new_ind_cate}/{basic_fac_name}/org_TS_{ts_trans_name}')
        
    ## test
    for process_name in process_name_list:
        batch_test(process_name, tag_name, factor_data_dir, batch_test_name)
    
    ## eval
    process_info_list = [(factor_data_dir, tag_name, process_name, mode, test_name)
                         for process_name in process_name_list
                         for mode, test_name in test_name_list]
    base_eval_params = toml.load(param_dir / 'feval' / f'{base_eval_name}.toml')
    eval_params = base_eval_params.copy()
    eval_params.update({
        'process_name_list': process_info_list, 
        })
    eval_name = f'{generate_batch_name}_{batch_test_name}_{base_eval_name}'
    with open(param_dir / 'feval' / f'{eval_name}.toml', 'w') as toml_file:
        toml.dump(eval_params, toml_file)
        
    fe = FactorEvaluation(eval_name, n_workers=eval_wkr)
    date_start_dt = datetime.strptime(date_start, '%Y%m%d')
    date_end_dt = datetime.strptime(date_end, '%Y%m%d')
    fe.eval_one_period(date_start_dt, date_end_dt, date_start_dt, date_end_dt)