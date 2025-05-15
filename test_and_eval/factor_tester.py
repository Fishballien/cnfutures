# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:41:56 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
__all__ = ["FactorTesterByContinuous", "FactorTesterByDiscrete"]


# %% imports
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import toml
import traceback
import copy
import pickle
from functools import partial
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.timeutils import parse_time_string
from data_processing.ts_trans import *
from utils.dirutils import load_path_config
from utils.datautils import align_and_sort_columns, check_dataframe_consistency, expand_factor_data
from utils.plotutils import *
from utils.trade_rules import *
from utils.market import index_to_futures, index_mapping
from test_and_eval.ts_test import check_stationarity, calculate_positive_ratio
from test_and_eval.testutils import analyze_returns_by_interval, get_factor_basic_info


# %% 
class FactorTest:
    
    def __init__(self, process_name, tag_name, factor_data_dir, test_name='', result_dir=None, params=None,
                 skip_plot=False, n_workers=1, date_start=None, date_end=None, check_consistency=True):
        self.process_name = process_name
        self.tag_name = tag_name
        self.factor_data_dir = Path(factor_data_dir)
        self.result_dir = result_dir
        self.params = params
        self.test_name = test_name
        self.skip_plot = skip_plot
        self.n_workers = n_workers
        self.date_start = date_start
        self.date_end = date_end
        self.check_consistency = check_consistency
        
        self._load_public_paths()
        self._load_test_params()
        self._init_dirs()
        self._init_plot_func()
        self._init_step()
        # self._load_twap_price()
        # self._preprocess_twap_return()
        
    def _load_public_paths(self):
        file_path = Path(__file__).resolve()
        project_dir = file_path.parents[1]
        self.path_config = load_path_config(project_dir)
        
        result_dir = self.result_dir or Path(self.path_config['result'])
        self.result_dir = result_dir / 'test' / self.test_name
        self.param_dir = Path(self.path_config['param'])
        
    def _init_dirs(self):
        self.factor_dir = (self.factor_data_dir / self.process_name if self.process_name is not None
                           else self.factor_data_dir)
        save_dir_prefix = f'{self.tag_name}/{self.process_name}' if self.tag_name is not None else f'{self.process_name}'
        save_dir = (self.result_dir / save_dir_prefix if self.process_name is not None
                    else self.result_dir)
        save_dir = save_dir
        
        self.data_dir = save_dir / 'data'
        self.factor_info_dir = save_dir / 'factor_info'
        self.plot_dir = save_dir / 'plot'
        self.debug_dir = save_dir / 'debug'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.factor_info_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_test_params(self):
        self.params = self.params or toml.load(self.param_dir / 'test' / f'{self.test_name}.toml')
        
    def _init_plot_func(self):
        plot_func_name = self.params['plot_func_name']
        plot_func = globals()[plot_func_name]
        self.plot_func = partial(plot_func, params=self.params, plot_dir=self.plot_dir)
        
    def _load_twap_price(self):
        twap_name = self.params['twap_name']
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
        
        twap_path = twap_data_dir / f'{twap_name}.parquet'
        twap_price = pd.read_parquet(twap_path)
        twap_price = twap_price.dropna(axis=1, how='all')
        
        self.twap_price = twap_price
        
    def _init_step(self):
        pp = self.params['pp']
        sp = self.params['sp']
        
        self.pp_by_sp = int(parse_time_string(pp) / parse_time_string(sp))
        
    def _preprocess_twap_return(self):
        pp_by_sp = self.pp_by_sp
        twap_price = self.twap_price
        
        twap_to_mask = twap_price.isna()
        
        rtn_1p = twap_price.pct_change(pp_by_sp, fill_method=None).shift(-pp_by_sp) / pp_by_sp
        rtn_1p = rtn_1p.replace([np.inf, -np.inf], 0)
        twap_to_mask = twap_to_mask | rtn_1p.isna()
            
        self.twap_to_mask = twap_to_mask
        self.rtn_1p = rtn_1p
        self.twap_price = twap_price

    def _load_factor(self, factor_name, date_start=None, date_end=None):
        org_factor_path = self.factor_dir / f'{factor_name}.parquet'
        if not os.path.exists(org_factor_path):
            print(self.factor_dir / f'{factor_name}.parquet')
            return None

        try:
            factor = pd.read_parquet(org_factor_path)
            factor = factor.rename(columns={'000984': '932000'})
            factor = factor.loc[date_start:date_end]
        except:
            traceback.print_exc()
            return None
        return factor
    
    def _align_his(self, factor, rtn_1p, twap_price):
        price_type = self.params.get('price_type', 'index')
        target_asset = self.params.get('target_asset')
        signal_mode = self.params.get('signal_mode', 'self-trade')
        if price_type in ['future', 'future_twap'] and signal_mode == 'self-trade':
            factor = factor.rename(columns=index_to_futures)
            if target_asset is not None:
                factor = factor[target_asset]
            factor, rtn_1p, twap_price = align_and_sort_columns([factor, rtn_1p, twap_price])

        factor = factor.reindex(rtn_1p.index) # 按twap reindex，确保等长
        return factor, rtn_1p, twap_price
    
    def _align_with_factor(self, factor, rtn_1p, twap_price):
        factor = factor.dropna(how='all')
        twap_price = twap_price.loc[factor.index.min():factor.index.max()] # 按factor头尾截取
        rtn_1p = rtn_1p.loc[factor.index.min():factor.index.max()] # 按factor头尾截取
        factor = factor.reindex(rtn_1p.index) # 按twap reindex，确保等长
        return factor, rtn_1p, twap_price
    
    def _mask(self, factor, twap_to_mask):
        twap_to_mask = twap_to_mask.reindex_like(factor)
        to_mask = factor.isna() | twap_to_mask # !!!: 不一定需要，看情况
        factor = factor.mask(to_mask)
        return factor, to_mask
    
    def _scale_factor(self, factor, rtn_1p):
        scale_window = self.params.get('scale_window')
        if scale_window is None:
            if factor.min().min() >= 0:
                factor = (factor - 0.5) * 2
            return factor
        scale_quantile = self.params['scale_quantile']
        sp = self.params['sp']
        scale_method = self.params.get('scale_method', 'minmax_scale')
        pp_by_sp = self.pp_by_sp
        
        
        scale_func = globals()[scale_method]
        scale_step = int(parse_time_string(scale_window) / parse_time_string(sp))
        # factor_scaled = ts_quantile_scale(factor, window=scale_step, quantile=scale_quantile)
        if scale_method in ['minmax_scale', 'minmax_scale_separate']:
            factor_scaled = scale_func(factor, window=scale_step, quantile=scale_quantile)
        elif scale_method in ['minmax_scale_adj_by_his_rtn', 'zscore_adj_by_his_rtn_and_minmax']:
            factor_scaled = scale_func(factor, rtn_1p, window=scale_step, rtn_window=pp_by_sp, quantile=scale_quantile)
        elif scale_method in ['rolling_percentile']:
            factor_scaled = scale_func(factor, window=scale_step)
        elif scale_method in ['percentile_adj_by_his_rtn']:
            factor_scaled = scale_func(factor, rtn_1p, window=scale_step, rtn_window=pp_by_sp)

        factor_scaled = (factor_scaled - 0.5) * 2
        return factor_scaled
    
    def _get_clipped(self, factor_scaled):
        direction_choices = self.params['direction_choices']
        
        factor_scaled_dict = {}
        for direction in direction_choices:
            if direction == 'all':
                factor_scaled_direction = factor_scaled.copy()
            elif direction == 'pos':
                factor_scaled_direction = factor_scaled.clip(lower=0)
            elif direction == 'neg':
                factor_scaled_direction = factor_scaled.clip(upper=0)

            factor_scaled_dict[direction] = factor_scaled_direction
        return factor_scaled_dict
        
# =============================================================================
#     def _save_pickle(self, obj_to_save, name): # 因为结果不是array，所以不方便用h5，就用pickle
#         with open(self.data_dir / f'{name}_{self.factor_name}.pkl', 'wb') as file:
#             pickle.dump(obj_to_save, file)
# =============================================================================

    def _save_pickle(self, obj_to_save, name):
        """
        保存对象到pickle文件，如果debug_dir存在且已有相同文件，则先检查一致性
        
        参数:
        obj_to_save (dict of DataFrames): 要保存的对象，假设是字典，其值为DataFrame
        name (str): 文件名前缀
        """
        save_path = self.data_dir / f'{name}_{self.factor_name}.pkl'
        
        # 检查是否需要进行一致性检查
        if self.check_consistency and hasattr(self, 'debug_dir') and save_path.exists():
            try:
                # 读取已存在的数据
                with open(save_path, 'rb') as file:
                    existing_data = pickle.load(file)
                
                # 检查obj_to_save和existing_data中的每个DataFrame是否一致
                if isinstance(obj_to_save, dict) and isinstance(existing_data, dict):
                    for key in obj_to_save.keys():
                        if key in existing_data and isinstance(obj_to_save[key], pd.DataFrame) and isinstance(existing_data[key], pd.DataFrame):
                            status, info = check_dataframe_consistency(existing_data[key], obj_to_save[key])
                            
                            if status == "INCONSISTENT":
                                # 保存不一致的数据到debug目录
                                debug_path = self.debug_dir / f'{name}_{self.factor_name}_inconsistent.pkl'
                                with open(debug_path, 'wb') as debug_file:
                                    pickle.dump(obj_to_save, debug_file)
                                
                                # 构造错误信息
                                error_msg = f"DataFrame一致性检查失败! 键: {key}, 索引: {info['index']}, 列: {info['column']}, "
                                error_msg += f"原始值: {info['original_value']}, 新值: {info['new_value']}, "
                                error_msg += f"不一致计数: {info['inconsistent_count']}。已保存到 {debug_path}"
                                
                                raise ValueError(error_msg)
            except Exception as e:
                if not isinstance(e, ValueError):  # 如果不是我们自己抛出的ValueError，则记录异常但继续执行
                    print(f"一致性检查过程中发生异常: {str(e)}")
        
        # 保存新数据
        with open(save_path, 'wb') as file:
            pickle.dump(obj_to_save, file)
            
    def _check_validation(self, factor, factor_scaled, rtn_1p):
        adf_test_res = {fut: check_stationarity(factor[fut].resample('1d').mean().dropna()) 
                        for fut in factor.columns}
        pos_ratio = calculate_positive_ratio(factor_scaled)
        rtn_ratio = calculate_positive_ratio(rtn_1p)
        if pos_ratio.mean() > 0.5:
            pos_ratio = 1 - pos_ratio
        ratio_diff = rtn_ratio - pos_ratio
        test_res = {
            'adf': adf_test_res, 
            'pos_ratio': pos_ratio,
            'ratio_diff': ratio_diff,
            }
        self._save_pickle(test_res, 'ts_test')
        
    def _calc_and_save_gp(self, rtn_1p, factor_scaled_dict):
        direction_choices = self.params['direction_choices']
        signal_mode = self.params.get('signal_mode', 'self-trade')
        target_asset = self.params.get('target_asset')
        interday_rtn_to_pre = self.params.get('interday_rtn_to_pre', False)
        pp_by_sp = self.pp_by_sp
        
        gp_dict = {}
        gpd_dict = {}
        for direction in direction_choices:
            factor_scaled_direction = factor_scaled_dict[direction]
            if signal_mode == 'self-trade':
                gp = (factor_scaled_direction * rtn_1p) #.fillna(0)
                # breakpoint()
            elif signal_mode == 'cross':
                # breakpoint()
                gp = pd.DataFrame()
                for signal_asset_combo in target_asset:
                    signal = signal_asset_combo['signal']
                    target = signal_asset_combo['target']
                    gp[f'{signal} To {target}'] = (factor_scaled_direction[index_mapping.get(signal, signal)] 
                                                 * rtn_1p[target])
            gp['return'] = gp.mean(axis=1)
            gpd = (gp.shift(pp_by_sp).resample('D').sum(min_count=1).dropna(how='all')
                   if not interday_rtn_to_pre else gp.resample('D').sum(min_count=1).dropna(how='all'))
            gpd['return'] = gpd.mean(axis=1).fillna(0)
            gp_dict[direction] = gp
            gpd_dict[direction] = gpd
            
        self._save_pickle(gp_dict, 'gp')
        self._save_pickle(gpd_dict, 'gpd')
        
        return gp_dict, gpd_dict
        
    def _calc_and_save_hsr(self, factor_scaled_dict, cluster_method='mean'):
        direction_choices = self.params['direction_choices']
        pp_by_sp = self.pp_by_sp
        
        hsr_dict = {}
        for direction in direction_choices:
            factor_scaled_direction = factor_scaled_dict[direction]
            hsr = ((factor_scaled_direction - factor_scaled_direction.shift(pp_by_sp)) / 2).abs().replace(
                [np.inf, -np.inf, np.nan], np.nan)
            if cluster_method == 'mean':
                hsrd = hsr.resample('1d').mean().dropna(how='all')
            elif cluster_method == 'sum':
                hsrd = hsr.resample('1d').sum(min_count=1).dropna(how='all')
            else:
                raise NotImplementedError()
            hsrd['avg'] = hsrd.mean(axis=1).fillna(0)
            hsr_dict[direction] = hsrd
            
        self._save_pickle(hsr_dict, 'hsr')
        
        return hsr_dict
    
    def test_multi_factors(self, skip_exists=False):
        factor_name_list = [path.stem for path in self.factor_dir.glob('*.parquet')]
        factor_name_list = [fac for fac in factor_name_list
                            if not os.path.exists(self.data_dir / f'hsr_{fac}.pkl') or not skip_exists]
        if self.n_workers is None or self.n_workers == 1:
            for factor_name in tqdm(factor_name_list, desc=f'{self.process_name} - {self.test_name} [single]'):
                self.test_one_factor(factor_name)
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                all_tasks = [executor.submit(self.test_one_factor, factor_name)
                             for factor_name in factor_name_list]
                num_of_success = 0
                for task in tqdm(as_completed(all_tasks), total=len(all_tasks), desc=f'{self.process_name} - {self.test_name} [multi]'):
                    res = task.result()
                    if res == 0:
                        num_of_success += 1
            print(f'num_of_success: {num_of_success}, num_of_failed: {len(factor_name_list)-num_of_success}')
            
            
class FactorTesterByContinuous(FactorTest):
    
    def test_one_factor(self, factor_name):
        self.factor_name = factor_name
        
        self._load_twap_price()
        self._preprocess_twap_return()
        
        twap_to_mask = self.twap_to_mask
        rtn_1p = self.rtn_1p
        twap_price = self.twap_price
        
        # load factor
        factor = self._load_factor(factor_name)
        if factor is None:
            return 1
        # align index & columns
        factor, rtn_1p, twap_price = self._align_his(factor, rtn_1p, twap_price)
        # scale
        factor = self._scale_factor(factor, rtn_1p) # TODO: 之后需要考虑，是否预先保存这些变换后的因子
        # align with factor 
        factor, rtn_1p, twap_price = self._align_with_factor(factor, rtn_1p, twap_price)
        # rough mask
        factor, _ = self._mask(factor, twap_to_mask)
        # clipped
        factor_scaled_dict = self._get_clipped(factor)
        # check validation
        self._check_validation(factor, factor, rtn_1p)
        # test and save results
        gp_dict, gpd_dict = self._calc_and_save_gp(rtn_1p, factor_scaled_dict)
        hsr_dict = self._calc_and_save_hsr(factor_scaled_dict)
        # plot
        if not self.skip_plot:
            self.plot_func(factor_name, gp_dict, gpd_dict, hsr_dict, twap_price)
        return 0


class FactorTesterByDiscrete(FactorTest):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_trade_func()
    
    def _init_trade_func(self):
        trade_rule_name = self.params.get('trade_rule_name')
        if trade_rule_name is None:
            self.trade_rule_func = None
            return
        trade_rule_param = self.params['trade_rule_param']
        self.trade_rule_func = partial(globals()[trade_rule_name], **trade_rule_param)
        
    def _to_discrete_pos(self, factor_scaled, to_mask):
        if self.trade_rule_func is None:
            return factor_scaled
        trade_rule_input = self.params.get('trade_rule_input', 'array')
        if trade_rule_input == 'array':
            actual_pos = factor_scaled.apply(lambda col: self.trade_rule_func(col.values), axis=0).mask(to_mask)
        elif  trade_rule_input == 'series':
            actual_pos = factor_scaled.apply(lambda col: self.trade_rule_func(col), axis=0).mask(to_mask)
        else:
            raise NotImplementedError()
        return actual_pos
    
    def _save_scaled_and_pos(self, scaled_factor, factor_pos):
        scaled_factor_filename = f'scaled_{self.factor_name}'
        pos_filename = f'pos_{self.factor_name}'
        self._save_factor_data(scaled_factor, factor_pos, scaled_factor_filename, pos_filename)
        
    def _save_factor_data(self, scaled_factor, factor_pos, scaled_factor_filename, pos_filename):
        """
        保存因子数据和因子仓位数据到parquet和csv文件，对parquet文件进行一致性检查
        
        参数:
        scaled_factor (pd.DataFrame): 缩放后的因子DataFrame
        factor_pos (pd.DataFrame): 因子仓位DataFrame
        scaled_factor_filename (str): 缩放因子文件名前缀
        pos_filename (str): 仓位文件名前缀
        """
        # 保存缩放因子数据
        scaled_factor_path = self.data_dir / f'{scaled_factor_filename}.parquet'
        
        # 检查缩放因子数据一致性
        if self.check_consistency and hasattr(self, 'debug_dir') and scaled_factor_path.exists():
            try:
                # 读取已存在的数据
                existing_scaled_factor = pd.read_parquet(scaled_factor_path)
                
                # 检查一致性
                status, info = check_dataframe_consistency(existing_scaled_factor, scaled_factor)
                
                if status == "INCONSISTENT":
                    # 保存不一致的数据到debug目录
                    debug_path = self.debug_dir / f'{scaled_factor_filename}_inconsistent.parquet'
                    scaled_factor.to_parquet(debug_path)
                    
                    # 构造错误信息
                    error_msg = f"缩放因子数据一致性检查失败! 索引: {info['index']}, 列: {info['column']}, "
                    error_msg += f"原始值: {info['original_value']}, 新值: {info['new_value']}, "
                    error_msg += f"不一致计数: {info['inconsistent_count']}。已保存到 {debug_path}"
                    
                    raise ValueError(error_msg)
            except Exception as e:
                if not isinstance(e, ValueError):  # 如果不是我们自己抛出的ValueError，则记录异常但继续执行
                    print(f"缩放因子数据一致性检查过程中发生异常: {str(e)}")
        
        # 保存因子仓位数据
        factor_pos_path = self.data_dir / f'{pos_filename}.parquet'
        
        # 检查因子仓位数据一致性
        if hasattr(self, 'debug_dir') and factor_pos_path.exists():
            try:
                # 读取已存在的数据
                existing_factor_pos = pd.read_parquet(factor_pos_path)
                
                # 检查一致性
                status, info = check_dataframe_consistency(existing_factor_pos, factor_pos)
                
                if status == "INCONSISTENT":
                    # 保存不一致的数据到debug目录
                    debug_path = self.debug_dir / f'{pos_filename}_inconsistent.parquet'
                    factor_pos.to_parquet(debug_path)
                    
                    # 构造错误信息
                    error_msg = f"因子仓位数据一致性检查失败! 索引: {info['index']}, 列: {info['column']}, "
                    error_msg += f"原始值: {info['original_value']}, 新值: {info['new_value']}, "
                    error_msg += f"不一致计数: {info['inconsistent_count']}。已保存到 {debug_path}"
                    
                    raise ValueError(error_msg)
            except Exception as e:
                if not isinstance(e, ValueError):  # 如果不是我们自己抛出的ValueError，则记录异常但继续执行
                    print(f"因子仓位数据一致性检查过程中发生异常: {str(e)}")
        
        # 确保目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存文件
        scaled_factor.to_parquet(scaled_factor_path)
        scaled_factor.to_csv(self.data_dir / f'{scaled_factor_filename}.csv')
        factor_pos.to_parquet(factor_pos_path)
        factor_pos.to_csv(self.data_dir / f'{pos_filename}.csv')
    
    def test_one_factor(self, factor_name, date_start=None, date_end=None):
        signal_mode = self.params.get('signal_mode', 'self-trade')
        basic_info_params = self.params.get('basic_info_params', {})
        group_pp_list = basic_info_params.get('group_pp', ['30min', '2h', '1d', '3d'])
        sp = self.params['sp']
        group_pp_by_sp = [(int(parse_time_string(group_pp) / parse_time_string(sp)), group_pp)
                          for group_pp in group_pp_list]
        
        self.factor_name = factor_name
        
        self._load_twap_price()
        self._preprocess_twap_return()
        
        twap_to_mask = self.twap_to_mask
        rtn_1p = self.rtn_1p
        twap_price = self.twap_price
        
        
        # load factor
        factor = self._load_factor(factor_name, date_start, date_end)
        if factor is None:
            return 1
        try:
            # expand if necessary
            factor = expand_factor_data(factor, twap_price.columns)
            # align index & columns
            factor, rtn_1p, twap_price = self._align_his(factor, rtn_1p, twap_price)
            # get factor basic info
            get_factor_basic_info(factor_name, factor, twap_price, group_pp_by_sp, self.data_dir, self.factor_info_dir)
            # scale
            factor = self._scale_factor(factor, rtn_1p)
            # align with factor 
            factor, rtn_1p, twap_price = self._align_with_factor(factor, rtn_1p, twap_price)
            # rough mask
            factor, to_mask = self._mask(factor, twap_to_mask)
            # to pos
            factor_pos = self._to_discrete_pos(factor, to_mask)
            # save scaled & pos
            self._save_scaled_and_pos(factor, factor_pos)
            # clipped
            factor_scaled_dict = self._get_clipped(factor_pos)
            # test and save results
            gp_dict, gpd_dict = self._calc_and_save_gp(rtn_1p, factor_scaled_dict)
            hsr_dict = self._calc_and_save_hsr(factor_scaled_dict, cluster_method='sum')
            # plot
            if not self.skip_plot:
                return_category_by_interval = (analyze_returns_by_interval(factor_pos, gp_dict['all']) 
                                               if signal_mode == 'self-trade' else None)
                self.plot_func(factor_name, self.test_name, gp_dict, gpd_dict, hsr_dict, twap_price, 
                               factor, factor_pos, return_category_by_interval)
        except:
            print(factor_name)
            traceback.print_exc()
            return 1
        return 0
        

# %%
if __name__=='__main__':
    tag_name = 'test'
    
    # process_name = 'batch10'
    # factor_data_dir = r'D:\CNIndexFutures\timeseries\factor_test\sample_data\factors'
    
    process_name = 'oi_510500'
    factor_data_dir = r'D:/mnt/idx_opt_processed'
    
    
    # test_name = 'regular_sp1min_pp60min_s40d'
    test_name = 'trade_ver3_futtwap_sp1min_s240d_icim_v6'
    
    tester = FactorTesterByDiscrete(process_name, tag_name, factor_data_dir, test_name=test_name)
    tester.test_multi_factors()