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


def process_group_applied_filters(args):
    """
    处理单个组的函数，适用于并行执行（过滤后因子版本）
    
    Args:
        args: 包含所需参数的元组
        
    Returns:
        tuple: (group_num, group_scaled) - 组号和组的标准化后的平均因子
    """
    group_num, group_info, group_normalization_func, factor_normalization_func, price_path, fstart = args
    
    price_data = pd.read_parquet(price_path)
    price_index = price_data.loc[fstart:].index
    group_factor_dict, group_weight_dict = {}, {}
    
    # 处理组内每个因子
    for idx in group_info.index:
        root_dir = group_info.loc[idx, 'root_dir']
        factor_name = group_info.loc[idx, 'factor_name']
        direction = group_info.loc[idx, 'direction']
        
        # 加载因子文件
        fac_path = Path(root_dir) / f'{factor_name}.parquet'
        if not fac_path.exists():
            print(f"警告: 因子文件不存在: {fac_path}")
            continue
            
        fac = pd.read_parquet(fac_path)
        
        # 使用单因子标准化函数
        scaled_fac = factor_normalization_func(fac)
        
        # 存储因子及其权重
        group_factor_dict[idx] = (direction * scaled_fac).reindex(index=price_index).replace([-np.inf, np.inf], np.nan).fillna(0)
        group_weight_dict[idx] = 1
    
    # 如果组内没有有效因子，返回空DataFrame
    if not group_factor_dict:
        return group_num, pd.DataFrame(index=price_index)
    
    # 计算组平均值并使用组标准化函数标准化
    group_avg = compute_dataframe_dict_average(group_factor_dict, group_weight_dict)
    group_scaled = group_normalization_func(group_avg).replace([-np.inf, np.inf], np.nan).fillna(0)
    return group_num, group_scaled


class AppliedFilterMerger:
    
    def __init__(self, merge_select_applied_filters_name: str, select_name: str, 
                 merge_name: str, test_eval_filtered_alpha_name: str, max_workers=None):
        """
        初始化过滤后因子合并器
        
        Args:
            merge_select_applied_filters_name: 合并配置名称
            select_name: 选择名称
            merge_name: 原始合并名称
            test_eval_filtered_alpha_name: 测试评估配置名称
            max_workers: 最大并行工作进程数
        """
        self.merge_select_applied_filters_name = merge_select_applied_filters_name
        self.select_name = select_name
        self.merge_name = merge_name
        self.test_eval_filtered_alpha_name = test_eval_filtered_alpha_name
        self.max_workers = max_workers
        
        # 加载路径配置
        self.path_config = load_path_config(project_dir)
        self.result_dir = Path(self.path_config['result'])
        self.param_dir = Path(self.path_config['param']) / 'merge_selected_applied_filters'
        
        # 设置目录路径
        self.select_dir = self.result_dir / 'select_applied_filters' / f'{merge_name}_{test_eval_filtered_alpha_name}_{select_name}'
        self.merged_dir = self.result_dir / 'merge_selected_applied_filters' / f'{merge_name}_{test_eval_filtered_alpha_name}_{select_name}_{merge_select_applied_filters_name}'
        
        # 加载配置文件
        config_path = self.param_dir / f'{merge_select_applied_filters_name}.yaml'
        self.config = self._load_config(config_path)
        
        # 确保输出目录存在
        self.merged_dir.mkdir(parents=True, exist_ok=True)
        
        self._init_utils()
        
    def _load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载配置文件
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
        
    def run_one_period(self, date_start, date_end, eval_date_start=None, eval_date_end=None):
        """
        运行单个期间的合并操作
        
        Args:
            date_start: 开始日期
            date_end: 结束日期
            eval_date_start: 评估开始日期，如果为None则使用date_start
            eval_date_end: 评估结束日期，如果为None则使用date_end
        """
        # 生成期间名称
        period_name = period_shortcut(date_start, date_end)
        
        if eval_date_start is None:
            eval_date_start = date_start
        if eval_date_end is None:
            eval_date_end = date_end
        
        # 设置此期间的目录
        self.select_period_dir = self.select_dir / period_name
        self.merged_period_dir = self.merged_dir / period_name
        self.merged_period_dir.mkdir(parents=True, exist_ok=True)
        
        self._merge_one_period(period_name)
        self._test_predicted(period_name)
        self._eval_predicted(eval_date_start, eval_date_end, period_name)
    
    def _merge_one_period(self, period_name):
        """
        合并指定期间内的过滤后因子
        """
        select_period_dir = self.select_period_dir
        merged_period_dir = self.merged_period_dir
        
        # 检查是否已经存在
        output_path = merged_period_dir / f'avg_predict_{period_name}.parquet'
        
        # 最终选定因子的路径
        final_selected_factors_path = select_period_dir / 'final_selected_factors.csv'
        
        # 检查最终选定的因子是否存在
        if not final_selected_factors_path.exists():
            print(f"未找到期间 {period_name} 的最终选定过滤后因子")
            return None
        
        # 加载最终选定的因子
        final_selected_factors = pd.read_csv(final_selected_factors_path)
        
        if len(final_selected_factors) == 0:
            print(f"期间 {period_name} 没有选定的过滤后因子")
            return None
        
        # 按组分组因子
        grouped = final_selected_factors.groupby('group')
        factor_dict, weight_dict = self._process_groups_parallel(grouped, period_name, max_workers=self.max_workers)
        
        if not factor_dict:
            print(f"期间 {period_name} 没有有效的因子组")
            return None
        
        # 计算跨组的总体平均值
        factor_avg = compute_dataframe_dict_average(factor_dict, weight_dict)
        # 使用组标准化函数来标准化最终结果
        factor_scaled = self.group_normalization_func(factor_avg).replace([-np.inf, np.inf], np.nan).fillna(0)
        
        # 保存结果
        factor_scaled.to_parquet(output_path)
        
        print(f"合并过滤后因子已保存至 {output_path}")
        
    def _process_groups_parallel(self, grouped, period_name, max_workers=None):
        """
        并行处理所有组
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
            with tqdm(total=total_groups, desc=f'处理 {period_name} 的过滤后因子Groups [Single]') as pbar:
                for args in group_args:
                    group_num = args[0]
                    try:
                        group_num, group_avg = process_group_applied_filters(args)
                        if not group_avg.empty:
                            factor_dict[group_num] = group_avg
                            weight_dict[group_num] = 1
                    except Exception as exc:
                        print(f'处理过滤后因子组 {group_num} 时发生错误: {exc}')
                    finally:
                        pbar.update(1)
        else:
            # 多进程处理
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
    """
    处理测试信息的独立函数（过滤后因子版本）
    """
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
    # 初始化AppliedFilterMerger
    afm = AppliedFilterMerger(
        merge_select_applied_filters_name='basic_merge',
        select_name='basic_select',
        merge_name='merge_v1',
        test_eval_filtered_alpha_name='basic_test_eval',
        max_workers=4
    )
    
    # 对单个期间进行合并
    afm.run_one_period('20240101', '20240331')
    
    print("过滤后因子合并完成")


if __name__ == "__main__":
    example_usage()