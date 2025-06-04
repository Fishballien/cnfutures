# -*- coding: utf-8 -*-
"""
Created on Wed May 29 2025

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
import json
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
def process_group_applied_filters(args):
    """
    处理应用过滤器选定因子的单个组的函数，适用于并行执行
    
    Args:
        args: 包含所需参数的元组 (group_num, group_info, group_normalization_func, 
              factor_normalization_func, price_path, fstart)
        
    Returns:
        tuple: (group_num, group_scaled) - 组号和组的标准化后的平均因子
    """
    (group_num, group_info, group_normalization_func, factor_normalization_func, 
     price_path, fstart) = args
    
    price_data = pd.read_parquet(price_path)
    price_index = price_data.loc[fstart:].index
    group_factor_dict, group_weight_dict = {}, {}
    
    # 处理组内每个因子
    for idx in group_info.index:
        root_dir = group_info.loc[idx, 'root_dir']
        factor_name = group_info.loc[idx, 'factor']
        direction = group_info.loc[idx, 'direction']
        
        # 直接从root_dir读取因子文件
        fac_path = Path(root_dir) / f'{factor_name}.parquet'
        
        if not fac_path.exists():
            print(f"警告: 未找到因子文件: {fac_path}")
            continue
            
        fac = pd.read_parquet(fac_path)
        
        # 使用单因子标准化函数
        scaled_fac = factor_normalization_func(fac)
        
        # 存储因子及其权重
        group_factor_dict[idx] = (direction * scaled_fac).reindex(index=price_index).replace([-np.inf, np.inf], np.nan).fillna(0)
        group_weight_dict[idx] = 1
    
    # 如果组内没有有效因子，返回零因子
    if not group_factor_dict:
        zero_factor = pd.DataFrame(0, index=price_index, columns=price_data.columns)
        return group_num, zero_factor
    
    # 计算组平均值并使用组标准化函数标准化
    group_avg = compute_dataframe_dict_average(group_factor_dict, group_weight_dict)
    group_scaled = group_normalization_func(group_avg).replace([-np.inf, np.inf], np.nan).fillna(0)
    return group_num, group_scaled


class AppliedFiltersMerger:
    
    def __init__(self, fac_merge_name, test_eval_filtered_alpha_name, select_name, filter_merge_name, max_workers=None):
        """
        初始化AppliedFiltersMerger，用于合并应用过滤器选定的因子。
        
        Args:
            fac_merge_name: 因子合并配置名称（决定从哪里读filtered factors）
            test_eval_filtered_alpha_name: 测试评估过滤alpha配置名称
            select_name: 选择配置名称  
            filter_merge_name: 过滤器合并配置名称（决定合并方法）
            max_workers: 最大并行工作进程数
        """
        self.fac_merge_name = fac_merge_name
        self.test_eval_filtered_alpha_name = test_eval_filtered_alpha_name
        self.select_name = select_name
        self.filter_merge_name = filter_merge_name
        self.max_workers = max_workers
        
        # 加载路径配置
        self.path_config = load_path_config(project_dir)
        self.result_dir = Path(self.path_config['result'])
        self.param_dir = Path(self.path_config['param']) / 'merge_selected_applied_filters'
        
        # 设置目录路径
        self.filtered_base_dir = self.result_dir / 'apply_filters_on_merged' / f'{fac_merge_name}'
        self.select_dir = self.result_dir / 'select_applied_filters' / f'{fac_merge_name}_{test_eval_filtered_alpha_name}_{select_name}'
        self.merged_dir = self.result_dir / 'merge_selected_applied_filters' / f'{fac_merge_name}_{test_eval_filtered_alpha_name}_{select_name}_{filter_merge_name}'
        
        # 加载配置文件
        config_path = self.param_dir / f'{filter_merge_name}.yaml'
        self.config = self._load_config(config_path)
        
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
        # 从配置中读取两个不同的预处理参数
        group_preprocess_params = self.config['group_preprocess_params']
        factor_preprocess_params = self.config['factor_preprocess_params']
        
        # 初始化两个不同的标准化函数
        self.group_normalization_func = partial(ts_normalize, param=group_preprocess_params)
        self.factor_normalization_func = partial(ts_normalize, param=factor_preprocess_params)
        
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
        合并指定期间的应用过滤器选定因子。
        
        Args:
            period_name: 期间名称
            
        Returns:
            pd.DataFrame: 该期间的合并平均因子
        """
        select_period_dir = self.select_period_dir
        merged_period_dir = self.merged_period_dir
        
        # 检查是否已经存在
        output_path = merged_period_dir / f'avg_predict_{period_name}.parquet'
        
        # 最终选定因子的路径
        final_selected_factors_path = select_period_dir / 'final_selected_factors.csv'
        
        # 检查最终选定的因子是否存在
        if not os.path.exists(final_selected_factors_path):
            print(f"未找到期间 {period_name} 的最终选定因子")
            return None
        
        # 加载最终选定的因子
        final_selected_factors = pd.read_csv(final_selected_factors_path)
        
        print(f"加载了 {len(final_selected_factors)} 个选定的应用过滤器因子")
        
        # 不需要过滤原始alpha，因为可以统一处理
        if final_selected_factors.empty:
            print(f"期间 {period_name} 没有选定的因子")
            # 创建零因子作为占位符
            price_path = self.config['price_path']
            fstart = self.config['fstart']
            price_data = pd.read_parquet(price_path)
            price_index = price_data.loc[fstart:].index
            zero_factor = pd.DataFrame(0, index=price_index, columns=price_data.columns)
            zero_factor.to_parquet(output_path)
            return None
        
        print(f"选定的因子数量: {len(final_selected_factors)}")
        
        # 按组分组因子
        grouped = final_selected_factors.groupby('group')
        factor_dict, weight_dict = self._process_groups_parallel(grouped, period_name, max_workers=self.max_workers)
        
        # 计算跨组的总体平均值
        factor_avg = compute_dataframe_dict_average(factor_dict, weight_dict)
        # 使用组标准化函数来标准化最终结果
        factor_scaled = self.group_normalization_func(factor_avg).replace([-np.inf, np.inf], np.nan).fillna(0)
        
        # 保存结果
        factor_scaled.to_parquet(output_path)
        
        print(f"合并的应用过滤器因子已保存至 {output_path}")
        
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
        group_args = [(group_num, group_info, self.group_normalization_func, 
                       self.factor_normalization_func, price_path, fstart) 
                      for group_num, group_info in grouped]
        total_groups = len(group_args)
        
        if max_workers == 1 or max_workers is None:
            # 单进程顺序处理
            with tqdm(total=total_groups, desc=f'处理 {period_name} 的应用过滤器Groups [Single]') as pbar:
                for args in group_args:
                    group_num = args[0]
                    try:
                        group_num, group_avg = process_group_applied_filters(args)
                        factor_dict[group_num] = group_avg
                        weight_dict[group_num] = 1
                    except Exception as exc:
                        print(f'处理应用过滤器组 {group_num} 时发生错误: {exc}')
                    finally:
                        pbar.update(1)
        else:
            # 多进程处理
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_group = {executor.submit(process_group_applied_filters, args): args[0] for args in group_args}
        
                with tqdm(total=total_groups, desc=f'处理 {period_name} 的应用过滤器Groups [Multi]') as pbar:
                    for future in concurrent.futures.as_completed(future_to_group):
                        group_num = future_to_group[future]
                        try:
                            group_num, group_avg = future.result()
                            factor_dict[group_num] = group_avg
                            weight_dict[group_num] = 1
                        except Exception as exc:
                            print(f'处理应用过滤器组 {group_num} 时发生错误: {exc}')
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
                res_dict = process_test_info_applied_filters(*params)
                res_list.append(res_dict)
        else:
            # 多进程并行执行，使用as_completed捕获进度
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_params = {executor.submit(process_test_info_applied_filters, *params): params for params in input_params}
                
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
        

def process_test_info_applied_filters(test_info, factor_name, date_start, date_end, merged_period_dir, eval_param, price_path):
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


def example_usage():
    """
    使用示例
    """
    # 初始化AppliedFiltersMerger
    afm = AppliedFiltersMerger(
        fac_merge_name='batch_till20_newma_batch_test_v3_icim_nsr22_m0',
        test_eval_filtered_alpha_name='corr_and_diffusion_v1',
        select_name='gt_nsr_ppt',
        filter_merge_name='m0',
        max_workers=4
    )
    
    # 对单个期间进行合并
    from datetime import datetime
    afm.run_one_period(datetime(2015, 1, 1), datetime(2016, 1, 1))
    
    print("应用过滤器因子合并完成")


if __name__ == "__main__":
    example_usage()