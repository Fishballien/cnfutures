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
import numpy as np
from tqdm import tqdm
from typing import Union, Dict, Any
import yaml
from functools import partial
import concurrent.futures


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.datautils import compute_dataframe_dict_average
from utils.timeutils import period_shortcut
from test_and_eval.factor_tester import FactorTesterByContinuous, FactorTesterByDiscrete
from data_processing.ts_trans import ts_normalize
from test_and_eval.factor_evaluation import eval_one_factor_one_period


# %%
def process_group(args):
    """
    处理单个组的函数，适用于并行执行
    
    Args:
        args: 包含所需参数的元组 (group_num, group_info, test_dir, normalization_func)
        
    Returns:
        tuple: (group_num, group_scaled) - 组号和组的标准化后的平均因子
    """
    group_num, group_info, test_dir, normalization_func, price_path, fstart = args
    
    price_data = pd.read_parquet(price_path)
    price_index = price_data.loc[fstart:].index
    group_factor_dict, group_weight_dict = {}, {}
    
    # 处理组内每个因子
    for idx in group_info.index:
        tag_name = group_info.loc[idx, 'tag_name']
        test_name = group_info.loc[idx, 'test_name']
        process_name = group_info.loc[idx, 'process_name']
        factor = group_info.loc[idx, 'factor']
        direction = group_info.loc[idx, 'direction']
        
        # 加载缩放因子
        scaled_fac_path = test_dir / test_name / tag_name / process_name / 'data' / f'scaled_{factor}.parquet'
        scaled_fac = pd.read_parquet(scaled_fac_path)
        
        # 存储因子及其权重
        group_factor_dict[idx] = (direction * scaled_fac).reindex(index=price_index).replace([-np.inf, np.inf], np.nan).fillna(0)
        # if scaled_fac.count() / len(scaled_fac) < 0.5:
        #     print(test_name, process_name, factor)
        group_weight_dict[idx] = 1
    
    # 计算组平均值并标准化
    group_avg = compute_dataframe_dict_average(group_factor_dict, group_weight_dict)
    group_scaled = normalization_func(group_avg).replace([-np.inf, np.inf], np.nan).fillna(0)
    return group_num, group_scaled


# %%
class FactorMerger:
    
    def __init__(self, merge_name, max_workers=None):
        """
        初始化FactorMerger，使用项目目录和选择名称。
        
        Args:
            project_dir: 项目根目录
            select_name: 选择名称
        """
        self.merge_name = merge_name
        self.max_workers = max_workers
        
        # 加载路径配置
        self.path_config = load_path_config(project_dir)
        self.result_dir = Path(self.path_config['result'])
        self.param_dir = Path(self.path_config['param']) / 'merge_selected_basic_features'
        self.test_dir = self.result_dir / 'test'
        self.merged_dir = self.result_dir / 'merge_selected_basic_features' / merge_name
        
        # 加载配置文件
        config_path = self.param_dir / f'{merge_name}.yaml'
        self.config = self._load_config(config_path)
        
        self.select_dir = self.result_dir / 'select_basic_features' / self.config['select_name']
        
        # 确保输出目录存在
        self.merged_dir.mkdir(parents=True, exist_ok=True)
        
        self._init_utils()
        
    def _load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载配置文件
        
        参数:
            config_path (str or Path): 配置文件路径
            
        返回:
            Dict[str, Any]: 配置字典
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _init_utils(self):
        preprocess_params = self.config['preprocess_params']
        self.normalization_func = partial(ts_normalize, param=preprocess_params)
        
    def run_one_period(self, date_start, date_end):
        # 生成期间名称
        period_name = period_shortcut(date_start, date_end)
        
        # 设置此期间的目录
        self.select_period_dir = self.select_dir / period_name
        self.merged_period_dir = self.merged_dir / period_name
        self.merged_period_dir.mkdir(parents=True, exist_ok=True)
        
        self._merge_one_period(period_name)
        self._test_predicted(period_name)
        self._eval_predicted(date_start, date_end, period_name)
    
    def _merge_one_period(self, period_name):
        """
        合并指定日期范围内的因子。
        
        Args:
            date_start: 开始日期
            date_end: 结束日期
            
        Returns:
            pd.DataFrame: 该期间的合并平均因子
        """
        select_period_dir = self.select_period_dir
        merged_period_dir = self.merged_period_dir
        
        # 检查是否已经存在
        output_path = merged_period_dir / f'avg_predict_{period_name}.parquet'
        # if os.path.exists(output_path):
        #     return
        
        # 最终选定因子的路径
        final_selected_factors_path = select_period_dir / 'top_factors.csv'
        
        # 检查最终选定的因子是否存在
        if not os.path.exists(final_selected_factors_path):
            print(f"未找到期间 {period_name} 的最终选定因子")
            return None
        
        # 加载最终选定的因子
        final_selected_factors = pd.read_csv(final_selected_factors_path)
        
        # 按组分组因子
        grouped = final_selected_factors.groupby('group')
        factor_dict, weight_dict = self._process_groups_parallel(grouped, period_name, max_workers=self.max_workers)
        
        # 计算跨组的总体平均值
        factor_avg = compute_dataframe_dict_average(factor_dict, weight_dict)
        factor_scaled = self.normalization_func(factor_avg).replace([-np.inf, np.inf], np.nan).fillna(0)
        
        # 保存结果
        factor_scaled.to_parquet(output_path)
        
        print(f"合并因子已保存至 {output_path}")
        
    def _process_groups_parallel(self, grouped, period_name, max_workers=None):
        """
        并行处理所有组
        
        Args:
            grouped: 分组后的数据
            period_name: 周期名称
            max_workers: 最大工作进程数，None表示使用默认值(CPU核心数)
            
        Returns:
            tuple: (factor_dict, weight_dict) - 因子字典和权重字典
        """
        price_path = self.config['price_path']
        fstart = self.config['fstart']
        factor_dict, weight_dict = {}, {}
        
        # 准备并行处理的参数
        group_args = [(group_num, group_info, self.test_dir, self.normalization_func, price_path, fstart) for group_num, group_info in grouped]
        total_groups = len(group_args)
        
        if max_workers == 1 or max_workers is None:
            # 单进程顺序处理
            with tqdm(total=total_groups, desc=f'处理 {period_name} 的Groups') as pbar:
                for args in group_args:
                    group_num = args[0]
                    try:
                        group_num, group_avg = process_group(args)
                        factor_dict[group_num] = group_avg
                        weight_dict[group_num] = 1
                    except Exception as exc:
                        print(f'处理组 {group_num} 时发生错误: {exc}')
                    finally:
                        pbar.update(1)
        else:
            # 多进程处理
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_group = {executor.submit(process_group, args): args[0] for args in group_args}
        
                with tqdm(total=total_groups, desc=f'处理 {period_name} 的Groups') as pbar:
                    for future in concurrent.futures.as_completed(future_to_group):
                        group_num = future_to_group[future]
                        try:
                            group_num, group_avg = future.result()
                            factor_dict[group_num] = group_avg
                            weight_dict[group_num] = 1
                        except Exception as exc:
                            print(f'处理组 {group_num} 时发生错误: {exc}')
                        finally:
                            pbar.update(1)

        return factor_dict, weight_dict
    
    def _test_predicted(self, period_name):
        merged_period_dir = self.merged_period_dir
        
        process_name = None
        factor_data_dir = merged_period_dir
        result_dir = merged_period_dir
        params = self.config
        
        test_list = params['test_list']
        for test_info in test_list:
            mode = test_info['mode']
            test_name = test_info['test_name']
            if mode == 'test':
                test_class = FactorTesterByContinuous
            elif mode == 'trade':
                test_class = FactorTesterByDiscrete
            else:
                NotImplementedError()
        
            ft = test_class(process_name, None, factor_data_dir, test_name=test_name, result_dir=result_dir)
            ft.test_one_factor(f'avg_predict_{period_name}')
            
    def _eval_predicted(self, date_start, date_end, period_name):
        merged_period_dir = self.merged_period_dir
        max_workers = self.max_workers
        params = self.config
        test_list = params['test_list']
        eval_param = params['eval']
        price_path = self.config['price_path']
        factor_name = f'avg_predict_{period_name}'
        
        # 准备输入参数列表
        input_params = []
        for test_info in test_list:
            input_params.append((
                test_info, 
                factor_name, 
                date_start, 
                date_end, 
                merged_period_dir, 
                eval_param, 
                price_path
            ))
        
        total_tasks = len(input_params)
        res_list = []
        
        # 根据max_workers决定是否使用多进程
        if max_workers == 1 or max_workers is None:
            # 单进程顺序执行，但显示进度条
            for params in tqdm(input_params, desc="Processing tests", total=total_tasks):
                res_dict = process_test_info(*params)
                res_list.append(res_dict)
        else:
            # 多进程并行执行，使用as_completed捕获进度
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_params = {executor.submit(process_test_info, *params): params for params in input_params}
                
                # 使用as_completed获取完成的任务并显示进度
                for future in tqdm(concurrent.futures.as_completed(future_to_params), total=total_tasks, desc="Processing tests"):
                    try:
                        res_dict = future.result()
                        res_list.append(res_dict)
                    except Exception as exc:
                        test_info = future_to_params[future][0]
                        print(f'Test {test_info["test_name"]} generated an exception: {exc}')
        
        # 转换为DataFrame
        res_df = pd.DataFrame(res_list)
        res_df.to_csv(merged_period_dir / 'evaluation.csv', index=None)
        

def process_test_info(test_info, factor_name, date_start, date_end, merged_period_dir, eval_param, price_path):
    mode = test_info['mode']
    test_name = test_info['test_name']
    eval_inputs = {
        "factor_name": factor_name,
        "date_start": date_start,
        "date_end": date_end,
        "data_date_start": date_start,
        "data_date_end": date_end,
        "process_name": '',
        "test_name": test_name,
        "tag_name": '',
        "data_dir": merged_period_dir / 'test' / test_name / 'data',
        "processed_data_dir": merged_period_dir,
        "valid_prop_thresh": eval_param['valid_prop_thresh'],
        "fee": eval_param['fee'],
        "price_data_path": price_path,
        "mode": mode, 
    }
    
    return eval_one_factor_one_period(**eval_inputs)
