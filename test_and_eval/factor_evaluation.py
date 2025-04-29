# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:19 2024

@author: Xintang Zheng

"""
# %% imports
import os
import toml
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")


from utils.dirutils import load_path_config, DirectoryProcessor
from utils.timeutils import period_shortcut
from test_and_eval.scores import get_general_return_metrics, calc_sharpe
from utils.speedutils import timeit
from utils.datautils import check_dataframe_consistency
from utils.dateutils import get_cffex_trading_days_by_date_range


# %%
def eval_one_factor_one_period_net_public(factor_name, factor_info, test_data_dir, date_start, date_end, fee, 
                                          return_net=False):
    res_dict = {}
    res_dict.update(factor_info)
    
    try:
        test_data = {}
        for data_type in ('gpd', 'hsr'):
            data_path = test_data_dir / f'{data_type}_{factor_name}.pkl'
    
            with open(data_path, 'rb') as f:
                test_data[data_type] = pickle.load(f)
    except:
        return res_dict
    
    df_gp = test_data['gpd']['all']
    df_hsr = test_data['hsr']['all']

    
    # direction
    gp_of_data = df_gp[(df_gp.index >= date_start) & (df_gp.index <= date_end)]
    cumrtn = gp_of_data['return'].sum()
    direction = 1 if cumrtn > 0 else -1
    res_dict.update({'direction': direction})
    
    df_gp = df_gp[(df_gp.index >= date_start) & (df_gp.index <= date_end)]
    df_hsr = df_hsr[(df_hsr.index >= date_start) & (df_hsr.index <= date_end)]
    
    # net
    net = (df_gp['return']*direction - fee * df_hsr['avg']).fillna(0)
    metrics = get_general_return_metrics(net)
    renamed_metrics = {f'net_{m}': v for m, v in metrics.items()}
    res_dict.update(renamed_metrics)
    
    # hsr
    hsr = df_hsr['avg'].mean(axis=0)
    res_dict.update({'hsr': hsr})
    
    # ppt
    profit_per_trade = df_gp["return"].sum() / df_hsr["avg"].sum()
    res_dict.update({'profit_per_trade': profit_per_trade*1000})
    
    # Calculate metrics year by year
    years = net.index.year.unique()
    for year in years:
        net_year = net[net.index.year == year]
        year_metrics = get_general_return_metrics(net_year)
        for m, v in year_metrics.items():
            res_dict[f'net_{m}_{year}'] = v
            
            
    if return_net:
        return res_dict, net
    else:
        return res_dict


def eval_one_factor_one_period_by_net(net):
    res_dict = {}
    metrics = get_general_return_metrics(net)
    renamed_metrics = {f'net_{m}': v for m, v in metrics.items()}
    res_dict.update(renamed_metrics)

    # Calculate metrics year by year
    years = net.index.year.unique()
    for year in years:
        net_year = net[net.index.year == year]
        year_metrics = get_general_return_metrics(net_year)
        for m, v in year_metrics.items():
            res_dict[f'net_{m}_{year}'] = v
            
    return res_dict
    

def eval_one_factor_one_period(factor_name, *, date_start, date_end, data_date_start, data_date_end,
                               process_name, test_name, tag_name, data_dir, processed_data_dir, 
                               valid_prop_thresh, fee, price_data_path, mode='test'):
    res_dict = {
        'root_dir': processed_data_dir, 
        'test_name': test_name, 
        'tag_name': tag_name, 
        'process_name': process_name, 
        'factor': factor_name,
        }

    try:
        test_data = {}
        for data_type in ('gpd', 'hsr', 'ts_test'):
            data_path = data_dir / f'{data_type}_{factor_name}.pkl'
            if data_type == 'ts_test' and not mode == 'test':
                continue
            with open(data_path, 'rb') as f:
                test_data[data_type] = pickle.load(f)
                
        for data_type in ('pos', 'scaled'):
            data_path = data_dir / f'{data_type}_{factor_name}.parquet'
            test_data[data_type] = pd.read_parquet(data_path)
            
        price_data = pd.read_parquet(price_data_path)
        
    except:
        traceback.print_exc()
        return res_dict
    
    df_gp = test_data['gpd']['all']
    df_hsr = test_data['hsr']['all']
    df_scaled = test_data['scaled']
    df_pos = test_data['pos']
    if mode == 'test':
        ts_test_res = test_data['ts_test']
    
    # direction
    if data_date_start is None or data_date_end is None:
        data_date_start, data_date_end = date_start, date_end
    gp_of_data = df_gp[(df_gp.index >= data_date_start) & (df_gp.index <= data_date_end)]
    cumrtn = gp_of_data['return'].sum()
    direction = 1 if cumrtn > 0 else -1
    res_dict.update({'direction': direction})
    
    df_gp = df_gp[(df_gp.index >= date_start) & (df_gp.index <= date_end)]
    df_hsr = df_hsr[(df_hsr.index >= date_start) & (df_hsr.index <= date_end)]
    df_scaled = df_scaled[(df_scaled.index >= date_start) & (df_scaled.index <= date_end)]
    df_pos = df_pos[(df_pos.index >= date_start) & (df_pos.index <= date_end)]
    
    
    # check validation
    trading_days = get_cffex_trading_days_by_date_range(date_start.date(), date_end.date())
    len_trading_days = len(trading_days)
    gps = df_gp['return'].replace([0.0], np.nan)
    res_dict.update({'trading_days': len_trading_days, 'valid_prop': gps.count() / len_trading_days})
    if len(df_gp) == 0 or gps.count() == 0 or gps.count() < valid_prop_thresh * len(trading_days):
        return res_dict
    
    # long short metrics
    res_dict.update(get_general_return_metrics(df_gp['return']*direction))

    # hsr
    hsr = df_hsr['avg'].mean(axis=0)
    hsrs = df_hsr['avg'].replace([0.0], np.nan)
    hsdr = hsrs.count() / len_trading_days
    hsr_std = df_hsr['avg'].std(axis=0)
    hsr_sp = hsr / hsr_std
    res_dict.update({'hsr': hsr, 'hsdr': hsdr, 'hsr_std': hsr_std, 'hsr_sr': hsr_sp})
    
    # net
    if mode == 'trade':
        net = (df_gp['return']*direction - fee * df_hsr['avg']).fillna(0)
        metrics = get_general_return_metrics(net)
        renamed_metrics = {f'net_{m}': v for m, v in metrics.items()}
        res_dict.update(renamed_metrics)
    
    # valid
    if mode == 'test':
        adf_test_res = ts_test_res['adf']
        ratio_diff = ts_test_res['ratio_diff']
        adf_valid = all([adf_test_res[fut]["Is Stationary"] for fut in adf_test_res])
        max_pos_ratio_diff = np.max(ratio_diff)
        res_dict.update({'adf_test': adf_valid, 'max_pos_ratio_diff': max_pos_ratio_diff})
        
    direction_dict = {
        'long_only': 'pos' if direction == 1 else 'neg',
        'short_only': 'neg' if direction == 1 else 'pos',
        }
    
    for direction_type, value_sign in direction_dict.items():
        df_gp = test_data['gpd'][value_sign]
        df_hsr = test_data['hsr'][value_sign]
        
        df_gp = df_gp[(df_gp.index >= date_start) & (df_gp.index <= date_end)]
        df_hsr = df_hsr[(df_hsr.index >= date_start) & (df_hsr.index <= date_end)]
        
        net = (df_gp['return']*direction - fee * df_hsr['avg']).fillna(0)
        metrics = get_general_return_metrics(net)
        renamed_metrics = {f'net_{m}_{direction_type}': v for m, v in metrics.items()}
        res_dict.update(renamed_metrics)
        
        hsr = df_hsr['avg'].mean(axis=0)
        hsrs = df_hsr['avg'].replace([0.0], np.nan)
        hsdr = hsrs.count() / len_trading_days
        hsr_std = df_hsr['avg'].std(axis=0)
        hsr_sp = hsr / hsr_std
        res_dict.update({f'hsr_{direction_type}': hsr, f'hsdr_{direction_type}': hsdr, 
                         f'hsr_std_{direction_type}': hsr_std, f'hsr_sr_{direction_type}': hsr_sp})
        
    # 趋势反转
    price_data = price_data.reindex(index=df_scaled.index, columns=df_scaled.columns)
    for wd in (30, 60, 240, 720, 1200):
        his_rtn = price_data.pct_change(wd, fill_method=None).shift(2) #!!!
        corr_cont = his_rtn.corrwith(df_scaled * direction, drop=True).mean()
        corr_dist = his_rtn.corrwith(df_pos * direction, drop=True).mean()
        res_dict.update({f'corr_cont_wd{wd}': corr_cont, f'corr_dist_wd{wd}': corr_dist})

    return res_dict


def get_one_factor_sharpe_and_gp(factor_name, *, date_start, date_end, date_range, data_dir, valid_prop_thresh,
                                 filter_gp):
    try:
        gp_path = data_dir / f'gpd_{factor_name}.pkl'
        with open(gp_path, 'rb') as f:
            gp = pickle.load(f)
        df_gp = gp['all']
    except:
        traceback.print_exc()
        return 0.0, pd.Series(0, index=date_range)
    
    # time period
    df_gp = df_gp[(df_gp.index >= date_start) & (df_gp.index <= date_end)].reindex(date_range)
    # ic
    ic_series = df_gp[filter_gp].fillna(0)
    # ic_series = df_icd[(df_icd.index >= date_start) & (df_icd.index <= date_end)].reindex(date_range)['ic_240min'].fillna(0)
    # long short metrics
    gps = df_gp[filter_gp].fillna(0)
    if gps.count() < valid_prop_thresh * gps.size:
        return 0.0, pd.Series(0, index=date_range)
    cumrtn_lag_0 = df_gp[filter_gp].sum()
    direction = 1 if cumrtn_lag_0 > 0 else -1
    sharpe = calc_sharpe(df_gp[filter_gp]*direction)
    return sharpe, ic_series


@timeit
def filter_correlated_features(factor_name_list, data_dir, *, date_start, date_end, valid_prop_thresh, corr_thresh,
                               filter_gp):
    date_range = pd.date_range(start=date_start, end=date_end, freq='D')
    calc_func = partial(get_one_factor_sharpe_and_gp, date_start=date_start, date_end=date_end, date_range=date_range,
                        data_dir=data_dir, valid_prop_thresh=valid_prop_thresh, filter_gp=filter_gp)
    results = [calc_func(factor_name) for factor_name in factor_name_list]

    ic_list = [result[1] for result in results]
    sharpe_list = [result[0] for result in results]
    del results
    
    ic_matrix = np.array(ic_list)
    corr_matrix = np.corrcoef(ic_matrix)
    
    # 创建一个布尔数组来标记是否保留因子
    keep = np.ones(len(factor_name_list), dtype=bool)
    
    # 按照 Sharpe 比率从高到低排序因子索引
    sorted_indices = np.argsort(sharpe_list)[::-1]
    
    # 遍历排序后的因子索引，保留高 Sharpe 比率因子，并标记与其高度相关的因子
    for i in sorted_indices:
        if keep[i]:
            for j in range(len(factor_name_list)):
                if i != j and abs(corr_matrix[i, j]) > corr_thresh:
                    keep[j] = False
    
    # 筛选后的因子列表
    filtered_factors = [factor_name_list[i] for i in range(len(factor_name_list)) if keep[i]]
    return filtered_factors


class FactorEvaluation:

    valid_prop_thresh = 0.5
    
    def __init__(self, eval_name, n_workers=1, check_consistency=True):
        self.eval_name = eval_name
        self.n_workers = n_workers
        self.check_consistency = check_consistency
        
        self._load_public_paths()
        self._load_test_params()
        self._init_dirs()
        self._load_price_dir()
        self.filtered_factors = {}
        
    def _load_public_paths(self):
        file_path = Path(__file__).resolve()
        project_dir = file_path.parents[1]
        self.path_config = load_path_config(project_dir)
        
        self.result_dir = Path(self.path_config['result'])
        self.param_dir = Path(self.path_config['param'])
        self.factor_data_dir = self.path_config['factor_data']
        
    def _load_test_params(self):
        try:
            self.params = toml.load(self.param_dir / 'feval' / f'{self.eval_name}.toml')
        except:
            self.params = toml.load(self.param_dir / 'dynamic_eval' / f'{self.eval_name}.toml')
        process_name_list = self.params.get('process_name_list')
        if process_name_list is None:
            root_dir_dict = self.params.get('root_dir_dict')
            if root_dir_dict is None:
                return
            dirp = DirectoryProcessor(root_dir_dict)
            process_name_list = dirp.list_of_tuple
        self.process_name_list = process_name_list
        
    def _init_dirs(self):
        self.feval_dir = self.result_dir / 'factor_evaluation'
        self.save_dir = self.feval_dir / self.eval_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir = self.feval_dir / 'debug' / self.eval_name
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir = self.result_dir  / 'test'
        
    def _load_price_dir(self):
        twap_name = self.params['price_name']
        price_type = self.params.get('price_type', 'index')
        if price_type == 'index':
            default_dir = self.path_config['twap_price']
        elif price_type == 'future':
            default_dir = self.path_config['future_price']
        elif price_type == 'future_twap':
            default_dir = self.path_config['future_twap']
        elif price_type == 'etf_twap':
            default_dir = self.path_config['etf_twap']
        else:
            raise NotImplementedError(f'Invalid price type: {price_type}')
        twap_data_dir = Path(self.params.get('twap_dir') or default_dir)
        self.price_data_path = twap_data_dir / f'{twap_name}.parquet'
        
    def eval_one_period(self, date_start, date_end, data_date_start=None, data_date_end=None, process_name_list=None):
        test_name = self.params.get('test_name')
        corr_thresh = self.params.get('corr_thresh', None)
        filter_gp = self.params.get('filter_gp', 'return')
        check_exists = self.params.get('check_exists', False)
        name_to_autosave = self.params.get('name_to_autosave', 'test')
        fee = self.params.get('fee', 4e-4)
        mode = self.params.get('mode', 'test')
        test_dir = self.test_dir
        valid_prop_thresh = self.params.get('valid_prop_thresh', 0.2)
        if process_name_list is None or len(process_name_list) == 0:
            process_name_list = self.process_name_list
        
        res_df_list = []
        period_name = period_shortcut(date_start, date_end)
        filter_func = partial(filter_correlated_features, 
                              date_start=data_date_start, date_end=data_date_end,
                              valid_prop_thresh=valid_prop_thresh, 
                              corr_thresh=corr_thresh, filter_gp=filter_gp)
        
        # 先收集所有要处理的任务
        all_tasks_info = []
        
        for process_info in process_name_list:
            if not isinstance(process_info, str):
                if len(process_info) == 3:
                    processed_data_dir, tag_name, process_name = process_info
                elif len(process_info) == 5:
                    processed_data_dir, tag_name, process_name, mode, test_name = process_info
                processed_data_dir = processed_data_dir or self.factor_data_dir
            else:
                processed_data_dir = self.factor_data_dir
                tag_name = None
                process_name = process_info
                
            # 检查是否已经评估过
            process_eval_dir = (self.feval_dir / name_to_autosave / test_name / tag_name / process_name 
                                if tag_name is not None
                                else self.feval_dir / name_to_autosave / test_name / process_name)
            process_eval_dir.mkdir(exist_ok=True, parents=True)
            corr_thresh_suffix = '' if corr_thresh is None else f'_{str(int(corr_thresh * 100)).zfill(3)}'
            process_res_filename = f'factor_eval_{period_name}{corr_thresh_suffix}'
            process_res_path = process_eval_dir / f'{process_res_filename}.csv'
            
            # 如果文件已存在且需要检查，则读取已有结果
            if check_exists and os.path.exists(process_res_path):
                res_df = pd.read_csv(process_res_path)
                res_df_list.append(res_df)
                continue
            
            # 定位test结果
            process_dir = (test_dir / test_name / tag_name if tag_name is not None
                          else test_dir / test_name)
            data_dir = process_dir / process_name / 'data' 
            
            # 定位factors
            factor_dir = Path(processed_data_dir) / process_name
            factor_name_list = [path.stem for path in factor_dir.glob('*.parquet')]
            
            # 筛选相关性
            if corr_thresh is not None:
                filtered_factor_list = self.filtered_factors.get((data_date_start, data_date_end, process_name))
                if filtered_factor_list is None:
                    filtered_factor_list = filter_func(factor_name_list, data_dir)
                    self.filtered_factors[(data_date_start, data_date_end, process_name)] = filtered_factor_list
                factor_name_list = filtered_factor_list
            
            # 构建评估函数
            eval_func = partial(eval_one_factor_one_period, date_start=date_start, date_end=date_end,
                                data_date_start=data_date_start, data_date_end=data_date_end,
                                process_name=process_name, test_name=test_name, tag_name=tag_name, 
                                data_dir=data_dir, processed_data_dir=processed_data_dir,
                                valid_prop_thresh=valid_prop_thresh, fee=fee, 
                                price_data_path=self.price_data_path, mode=mode)
            
            # 收集此process的所有任务
            all_tasks_info.append({
                'process_name': process_name,
                'eval_func': eval_func,
                'factor_name_list': factor_name_list,
                'process_res_path': process_res_path
            })
        
        # 如果没有任务需要执行，直接返回结果
        if not all_tasks_info:
            return pd.concat(res_df_list, axis=0, ignore_index=True) if res_df_list else pd.DataFrame()
        
        # 将所有任务转换为(eval_func, factor_name)对
        all_tasks = []
        task_to_process_map = {}  # 用于跟踪每个任务属于哪个process
        
        for task_info in all_tasks_info:
            process_name = task_info['process_name']
            eval_func = task_info['eval_func']
            for factor_name in task_info['factor_name_list']:
                task_id = len(all_tasks)
                all_tasks.append((eval_func, factor_name))
                task_to_process_map[task_id] = task_info
        
        # 执行所有任务
        results_by_process = {}  # 按process分组的结果
        
        if self.n_workers is None or self.n_workers == 1:
            # 单进程执行
            for task_id, (eval_func, factor_name) in enumerate(tqdm(all_tasks, desc=f'{self.eval_name} - {period_name}')):
                res_dict = eval_func(factor_name)
                if res_dict is not None:
                    process_info = task_to_process_map[task_id]
                    process_name = process_info['process_name']
                    if process_name not in results_by_process:
                        results_by_process[process_name] = []
                    results_by_process[process_name].append(res_dict)
        else:
            # 多进程执行
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(eval_func, factor_name) for eval_func, factor_name in all_tasks]
                for task_id, future in enumerate(tqdm(as_completed(futures), total=len(futures), 
                                                     desc=f'{self.eval_name} - {period_name}')):
                    res_dict = future.result()
                    if res_dict is not None:
                        process_info = task_to_process_map[task_id]
                        process_name = process_info['process_name']
                        if process_name not in results_by_process:
                            results_by_process[process_name] = []
                        results_by_process[process_name].append(res_dict)
        
        # 将结果按process分组保存并添加到结果列表
        for task_info in all_tasks_info:
            process_name = task_info['process_name']
            process_res_path = task_info['process_res_path']
            
            if process_name in results_by_process and results_by_process[process_name]:
                res_df = pd.DataFrame(results_by_process[process_name])
                res_df_list.append(res_df)
                res_df.to_csv(process_res_path, index=None)
        
        # return pd.concat(res_df_list, axis=0, ignore_index=True) if res_df_list else pd.DataFrame()
    
        res = pd.concat(res_df_list, axis=0, ignore_index=True) if res_df_list else pd.DataFrame()
        self._save_factor_eval(res, period_name)
        self._plot_sharpe_dist(period_name, res)
        # self._plot_adf_and_sharpe(period_name, res)
        # if len(self.process_name_list) == 2:
        #     self._plot_diff(period_name, res)
  
# =============================================================================
#     def eval_one_period(self, date_start, date_end, data_date_start=None, data_date_end=None, process_name_list=None):
#         test_name = self.params.get('test_name')
#         corr_thresh = self.params.get('corr_thresh', None)
#         filter_gp = self.params.get('filter_gp', 'return')
#         check_exists = self.params.get('check_exists', False)
#         name_to_autosave = self.params.get('name_to_autosave', 'test')
#         fee = self.params.get('fee', 4e-4)
#         mode = self.params.get('mode', 'test')
#         test_dir = self.test_dir
#         valid_prop_thresh = self.params.get('valid_prop_thresh', 0.2)
#         if process_name_list is None or len(process_name_list) == 0:
#             process_name_list = self.process_name_list
#         
#         res_df_list = []
#         period_name = period_shortcut(date_start, date_end)
#         filter_func = partial(filter_correlated_features, 
#                               date_start=data_date_start, date_end=data_date_end,
#                               valid_prop_thresh=valid_prop_thresh, 
#                               corr_thresh=corr_thresh, filter_gp=filter_gp)
# 
#         for process_info in process_name_list:
#             if not isinstance(process_info, str):
#                 if len(process_info) == 3:
#                     processed_data_dir, tag_name, process_name = process_info
#                 elif len(process_info) == 5:
#                     processed_data_dir, tag_name, process_name, mode, test_name = process_info
#                 processed_data_dir = processed_data_dir or self.factor_data_dir
#             else:
#                 processed_data_dir = self.factor_data_dir
#                 tag_name = None
#                 process_name = process_info
#                 
#             # 检查是否已经评估过
#             process_eval_dir = (self.feval_dir / name_to_autosave / test_name / tag_name / process_name 
#                                 if tag_name is not None
#                                 else self.feval_dir / name_to_autosave / test_name / process_name)
#             process_eval_dir.mkdir(exist_ok=True, parents=True)
#             corr_thresh_suffix = '' if corr_thresh is None else f'_{str(int(corr_thresh * 100)).zfill(3)}'
#             process_res_filename = f'factor_eval_{period_name}{corr_thresh_suffix}'
#             process_res_path = process_eval_dir / f'{process_res_filename}.csv'
#             if check_exists and os.path.exists(process_res_path):
#                 res_df = pd.read_csv(process_res_path)
#                 res_df_list.append(res_df)
#                 continue
#             
#             # 定位test结果
#             process_dir = (test_dir / test_name / tag_name if tag_name is not None
#                           else test_dir / test_name)
#             data_dir = process_dir / process_name / 'data' 
#             
#             # 定位factors
#             factor_dir = Path(processed_data_dir) / process_name
#             factor_name_list = [path.stem for path in factor_dir.glob('*.parquet')]
#             
#             # 筛选相关性
#             if corr_thresh is not None:
#                 filtered_factor_list = self.filtered_factors.get((data_date_start, data_date_end, process_name))
#                 if filtered_factor_list is None:
#                     filtered_factor_list = filter_func(factor_name_list, data_dir)
#                     self.filtered_factors[(data_date_start, data_date_end, process_name)] = filtered_factor_list
#                 factor_name_list = filtered_factor_list
#             
#             # evaluate
#             eval_func = partial(eval_one_factor_one_period, date_start=date_start, date_end=date_end,
#                                 data_date_start=data_date_start, data_date_end=data_date_end,
#                                 process_name=process_name, test_name=test_name, tag_name=tag_name, 
#                                 data_dir=data_dir, processed_data_dir=processed_data_dir,
#                                 valid_prop_thresh=valid_prop_thresh, fee=fee, 
#                                 price_data_path=self.price_data_path, mode=mode)
#             
#             res_list = []
#             if self.n_workers is None or self.n_workers == 1:
#                 for factor_name in tqdm(factor_name_list, desc=f'{self.eval_name} - {process_name} - {period_name}'):
#                     res_dict = eval_func(factor_name)
#                     res_list.append(res_dict)
#             else:
#                 # res_list = multiprocess_with_sequenced_result(eval_func, factor_name_list, self.n_workers,
#                 #                                               desc=f'{self.eval_name} - {period_name}')
#                 with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
#                     all_tasks = [executor.submit(eval_func, factor_name)
#                                  for factor_name in factor_name_list]
#                     for task in tqdm(as_completed(all_tasks), total=len(all_tasks), 
#                                      desc=f'{self.eval_name} - {process_name} - {period_name}'):
#                         res_dict = task.result()
#                         if res_dict is not None:
#                             res_list.append(res_dict)
#             res_df = pd.DataFrame(res_list)
#             res_df_list.append(res_df)
#             res_df.to_csv(process_res_path, index=None)
#                 
#         res = pd.concat(res_df_list, axis=0, ignore_index=True)
#         
#         self._save_factor_eval(res, period_name)
#         self._plot_sharpe_dist(period_name, res)
#         # self._plot_adf_and_sharpe(period_name, res)
#         # if len(self.process_name_list) == 2:
#         #     self._plot_diff(period_name, res)
# =============================================================================
        
    def _save_factor_eval(self, res, period_name):
        """
        保存因子评估结果到CSV文件，如果debug_dir存在且已有相同文件，则先检查一致性
        
        参数:
        res (pd.DataFrame): 因子评估结果DataFrame
        period_name (str): 时间段名称
        """
        # 首先对结果按factor排序
        res = res.copy()
        res.sort_values(by='factor', inplace=True)
        
        save_path = self.save_dir / f'factor_eval_{period_name}.csv'
        
# =============================================================================
#         # 检查是否需要进行一致性检查
#         if self.check_consistency and hasattr(self, 'debug_dir') and save_path.exists():
#             try:
#                 # 读取已存在的数据
#                 existing_data = pd.read_csv(save_path)
#                 
#                 # 对已存在的数据也按factor排序，确保比较的顺序一致
#                 existing_data.sort_values(by='factor', inplace=True)
#                 
#                 # 检查一致性
#                 status, info = check_dataframe_consistency(existing_data, res)
#                 
#                 if status == "INCONSISTENT":
#                     # 保存不一致的数据到debug目录
#                     debug_path = self.debug_dir / f'factor_eval_{period_name}_inconsistent.csv'
#                     res.to_csv(debug_path, index=None)
#                     
#                     # 构造错误信息
#                     error_msg = f"DataFrame一致性检查失败! 索引: {info['index']}, 列: {info['column']}, "
#                     error_msg += f"原始值: {info['original_value']}, 新值: {info['new_value']}, "
#                     error_msg += f"不一致计数: {info['inconsistent_count']}。已保存到 {debug_path}"
#                     
#                     raise ValueError(error_msg)
#             except Exception as e:
#                 if not isinstance(e, ValueError):  # 如果不是我们自己抛出的ValueError，则记录异常但继续执行
#                     print(f"一致性检查过程中发生异常: {str(e)}")
# =============================================================================
        
        # 保存新数据
        res.to_csv(save_path, index=None)
        
    def _plot_sharpe_dist(self, period_name, res):
        
        FONTSIZE_L1 = 20
        FONTSIZE_L2 = 18
        FONTSIZE_L3 = 15
        
        title = f'{self.eval_name} Sharpe Ratio {period_name}'
        
        fig = plt.figure(figsize=(36, 27), dpi=100, layout="constrained")
        spec = fig.add_gridspec(ncols=1, nrows=1)
        
        ax0 = fig.add_subplot(spec[:, :])
        ax0.set_title(f'{title}', fontsize=FONTSIZE_L1, pad=25)
        for process_name, group_data in res.groupby('process_name'):
            ax0.hist(group_data['sharpe_ratio'], label=process_name, alpha=.5, bins=50)
        
        for ax in [ax0,]:
            ax.grid(linestyle=":")
            ax.legend(loc="upper left", borderaxespad=0.5, fontsize=FONTSIZE_L3)
            ax.tick_params(labelsize=FONTSIZE_L2, pad=15)
        
        plt.savefig(self.save_dir / f"factor_eval_{period_name}.jpg", dpi=100, bbox_inches="tight")
        plt.close()

    def _plot_diff(self, period_name, res):
         p_name_1 = self.process_name_list[0]
         p_name_2 = self.process_name_list[1]
         res_1 = res[res['process_name'] == p_name_1].set_index('factor')
         res_2 = res[res['process_name'] == p_name_2].set_index('factor')
         diff = pd.DataFrame()
         diff[f'sharpe_{p_name_1}'] = res_1['sharpe_ratio']
         diff[f'sharpe_{p_name_2}'] = res_2['sharpe_ratio']
         diff['sharpe_diff'] = res_2['sharpe_ratio'] - res_1['sharpe_ratio']
         diff.to_csv(self.save_dir / f'diff_{period_name}.csv')
                    
        
# %%
if __name__=='__main__':
    pass