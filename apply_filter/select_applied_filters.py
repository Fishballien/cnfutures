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
import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from functools import partial
from datetime import datetime
from typing import Union, Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.timeutils import period_shortcut
from utils.fsutils import find_files_with_prefix, copy_file
from utils.datautils import deduplicate_nested_list
from synthesis.factor_cluster import cluster_factors


def extend_metrics(eval_res):
    """
    Extend evaluation metrics with additional calculated values.
    
    Parameters:
    -----------
    eval_res : DataFrame
        DataFrame containing the base evaluation results with metrics like
        'net_return_annualized', 'hsr', and correlation metrics.
        
    Returns:
    -----------
    eval_res : DataFrame
        The same DataFrame with additional calculated metrics.
    """
    # Calculate net_ppt for different directions
    for direction_suffix in ('', '_long_only', '_short_only'):
        eval_res[f'net_ppt{direction_suffix}'] = (eval_res[f'net_return_annualized{direction_suffix}'] 
                                                  / eval_res[f'hsr{direction_suffix}'] / 245)
    
    # Calculate average correlations for different time windows
    for corr_type in ('cont', 'dist'):
        lt720_cols = [f'corr_{corr_type}_wd30', f'corr_{corr_type}_wd60', 
                      f'corr_{corr_type}_wd240', f'corr_{corr_type}_wd720']
        eval_res[f'corr_{corr_type}_lt720_avg'] = eval_res[lt720_cols].mean(axis=1)
    
    return eval_res


def filter_func_dynamic_for_applied_filters(eval_res, conditions, min_count=1, 
                                           sort_target=None, sort_ascending=False,
                                           target_tag_name=None, target_process_name=None, 
                                           target_factor_name=None, target_test_name=None,
                                           reference_process_name=None, reference_factor_name=None,
                                           reference_test_name=None, reference_tag_name=None):
    """
    针对applied filters的动态筛选函数，保持与原始filter_func_dynamic的兼容性
    
    参数:
        eval_res (pd.DataFrame): 评估结果数据框
        conditions (list): 筛选条件列表，每个条件是包含metric, threshold, direction等的字典
        min_count (int): 最少保留的因子数量
        sort_target (str): 排序目标指标
        sort_ascending (bool): 是否升序排序
        target_tag_name (str): 目标tag_name，替代原来的pred_name功能
        target_process_name (str): 目标process_name
        target_factor_name (str): 目标factor_name  
        target_test_name (str): 目标test_name
        reference_process_name (str): 参考process_name，如果为None则自动设置
        reference_factor_name (str): 参考factor_name，如果为None则自动推断
        reference_test_name (str): 参考test_name，如果为None则使用第一个可用的
        reference_tag_name (str): 参考tag_name，如果为None则自动设置为org_alpha
        
    返回:
        pd.Series: 布尔索引，表示是否通过筛选
    """
    # 如果指定了目标条件，先进行预筛选
    if any([target_tag_name, target_process_name, target_factor_name, target_test_name]):
        target_mask = pd.Series([True] * len(eval_res), index=eval_res.index)
        
        if target_tag_name is not None:
            target_mask &= (eval_res['tag_name'] == target_tag_name)
        if target_process_name is not None:
            target_mask &= (eval_res['process_name'] == target_process_name)
        if target_factor_name is not None:
            target_mask &= (eval_res['factor_name'] == target_factor_name)
        if target_test_name is not None:
            target_mask &= (eval_res['test_name'] == target_test_name)
            
        eval_res_filtered = eval_res[target_mask]
    else:
        eval_res_filtered = eval_res.copy()
        target_mask = pd.Series([True] * len(eval_res), index=eval_res.index)
    
    if len(eval_res_filtered) == 0:
        print("警告: 目标筛选后没有数据")
        return pd.Series([False] * len(eval_res), index=eval_res.index)
    
    # 应用筛选条件
    selected_mask = pd.Series([True] * len(eval_res_filtered), index=eval_res_filtered.index)
    
    for condition in conditions:
        metric = condition['metric']
        threshold = condition.get('threshold')
        direction = condition.get('direction', 'ge')
        use_reference = condition.get('use_reference', False)
        
        if use_reference:
            # 自动设置参考基准
            ref_tag_name = reference_tag_name or 'org_alpha'
            ref_process_name = reference_process_name or ''
            ref_test_name = reference_test_name or eval_res['test_name'].iloc[0]
            
            if reference_factor_name is None:
                # 从org_alpha中推断factor_name
                org_alpha_rows = eval_res[
                    (eval_res['tag_name'] == ref_tag_name) & 
                    (eval_res['process_name'] == ref_process_name) &
                    (eval_res['test_name'] == ref_test_name)
                ]
                if len(org_alpha_rows) > 0:
                    ref_factor_name = org_alpha_rows['factor_name'].iloc[0]
                else:
                    raise ValueError(f"无法找到参考基准: tag_name={ref_tag_name}, "
                                   f"process_name={ref_process_name}, test_name={ref_test_name}")
            else:
                ref_factor_name = reference_factor_name
            
            # 找到参考行
            ref_condition = (
                (eval_res['process_name'] == ref_process_name) & 
                (eval_res['factor_name'] == ref_factor_name) &
                (eval_res['test_name'] == ref_test_name) &
                (eval_res['tag_name'] == ref_tag_name)
            )
            
            ref_rows = eval_res[ref_condition]
            if len(ref_rows) == 0:
                raise ValueError(f"无法找到参考基准行: process_name={ref_process_name}, "
                               f"factor_name={ref_factor_name}, test_name={ref_test_name}, "
                               f"tag_name={ref_tag_name}")
            
            ref_value = ref_rows[metric].iloc[0]
            print(f"使用参考基准值: {metric} = {ref_value}")
            
            # 相对于参考基准进行筛选
            if direction == 'ge':
                condition_mask = eval_res_filtered[metric] >= ref_value
            elif direction == 'gt':
                condition_mask = eval_res_filtered[metric] > ref_value
            elif direction == 'le':
                condition_mask = eval_res_filtered[metric] <= ref_value
            elif direction == 'lt':
                condition_mask = eval_res_filtered[metric] < ref_value
            else:
                raise ValueError(f"不支持的筛选方向: {direction}")
        else:
            # 绝对阈值筛选
            if threshold is None:
                raise ValueError(f"使用绝对阈值筛选时，threshold不能为None")
                
            if direction == 'ge':
                condition_mask = eval_res_filtered[metric] >= threshold
            elif direction == 'gt':
                condition_mask = eval_res_filtered[metric] > threshold
            elif direction == 'le':
                condition_mask = eval_res_filtered[metric] <= threshold
            elif direction == 'lt':
                condition_mask = eval_res_filtered[metric] < threshold
            else:
                raise ValueError(f"不支持的筛选方向: {direction}")
        
        selected_mask &= condition_mask
        print(f"应用条件 {metric} {direction} {threshold if not use_reference else 'reference'}: "
              f"剩余 {selected_mask.sum()} 个因子")
    
    # 检查最小数量要求
    if selected_mask.sum() < min_count and sort_target is not None:
        print(f"筛选结果少于最小数量 {min_count}，使用排序方式保留前 {min_count} 个")
        sorted_indices = eval_res_filtered[sort_target].sort_values(ascending=sort_ascending).index[:min_count]
        selected_mask = eval_res_filtered.index.isin(sorted_indices)
    
    # 将结果映射回原始数据框
    result_mask = pd.Series([False] * len(eval_res), index=eval_res.index)
    result_mask.loc[eval_res_filtered.index] = selected_mask
    result_mask &= target_mask  # 确保只返回目标范围内的结果
    
    return result_mask


class AppliedFiltersSelector:
    """
    应用过滤器结果选择器类：用于筛选、聚类并保存最佳过滤后的因子
    """
    
    def __init__(self, select_name: str, merge_name: str, test_eval_filtered_alpha_name: str):
        """
        初始化应用过滤器结果选择器
        
        参数:
            select_name (str): 选择配置名称
            merge_name (str): 合并名称
            test_eval_filtered_alpha_name (str): 测试评估过滤alpha名称
        """
        self.select_name = select_name
        self.merge_name = merge_name
        self.test_eval_filtered_alpha_name = test_eval_filtered_alpha_name
        
        # 加载项目路径配置
        self.project_dir = project_dir
        self.path_config = load_path_config(self.project_dir)
        self.result_dir = Path(self.path_config['result'])
        self.param_dir = Path(self.path_config['param']) / 'select_applied_filters'
        
        # 设置相关目录
        self.test_eval_dir = self.result_dir / 'test_eval_filtered_alpha' / f'{merge_name}_{test_eval_filtered_alpha_name}'
        self.select_dir = self.result_dir / 'select_applied_filters' / f'{merge_name}_{test_eval_filtered_alpha_name}_{select_name}'
        
        # 加载配置文件
        config_path = self.param_dir / f'{select_name}.yaml'
        self.config = self._load_config(config_path)
        
        # 设置筛选参数
        self.cluster_params = self.config['cluster']
        self.cluster_params.update({'test_dir': None})  # applied filters不需要test_dir
        
        # 创建筛选函数，支持原始filter_func_dynamic的参数结构
        filter_config = self.config['filter']
        self.filter_func = partial(
            filter_func_dynamic_for_applied_filters,
            conditions=filter_config['conditions'],
            min_count=filter_config.get('min_count', 1),
            sort_target=filter_config.get('sort_target'),
            sort_ascending=filter_config.get('sort_ascending', False),
            target_tag_name=filter_config.get('target_tag_name'),
            target_process_name=filter_config.get('target_process_name'),
            target_factor_name=filter_config.get('target_factor_name'),
            target_test_name=filter_config.get('target_test_name'),
            reference_process_name=filter_config.get('reference_process_name'),
            reference_factor_name=filter_config.get('reference_factor_name'),
            reference_test_name=filter_config.get('reference_test_name'),
            reference_tag_name=filter_config.get('reference_tag_name')
        )

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
    
    def run_one_period(self, date_start: str, date_end: str):
        """
        对指定期间运行筛选
        
        Args:
            date_start: 开始日期
            date_end: 结束日期
        """
        # 生成期间名称
        period_name = period_shortcut(date_start, date_end)
        
        # 设置结果目录
        res_dir = self.select_dir / period_name
        res_selected_dir = res_dir / 'selected'
        res_selected_info_dir = res_selected_dir / 'factor_info'
        res_selected_plot_dir = res_selected_dir / 'plot'
        
        # 创建目录
        res_selected_info_dir.mkdir(parents=True, exist_ok=True)
        res_selected_plot_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"开始对期间 {period_name} 进行应用过滤器结果筛选")
        
        # 读取评估结果
        # 直接在test_eval_dir下寻找对应期间的子目录
        period_test_eval_dir = self.test_eval_dir / period_name / 'eval'
        
        if not period_test_eval_dir.exists():
            print(f"警告: 未找到期间 {period_name} 的评估目录: {period_test_eval_dir}")
            return
            
        # 寻找评估文件
        eval_files = list(period_test_eval_dir.glob('eval_res_*.csv'))
        if not eval_files:
            print(f"警告: 未找到期间 {period_name} 的评估结果文件")
            return
            
        eval_file = eval_files[0]  # 取第一个找到的文件
            
        print(f"读取评估结果文件: {eval_file}")
        eval_res = pd.read_csv(eval_file)
        
        # 扩展评估指标
        eval_res = extend_metrics(eval_res)
        
        print(f"评估结果包含 {len(eval_res)} 个因子")
        
        # 应用筛选函数
        print(f"开始筛选，初始因子数量: {len(eval_res)}")
        
        selected_mask = self.filter_func(eval_res)
        selected_eval_res = eval_res[selected_mask].copy()
        
        print(f"筛选后剩余 {len(selected_eval_res)} 个因子")
        
        # 如果筛选后没有因子，则跳过后续步骤
        if selected_eval_res.empty:
            print(f"期间 {period_name} 筛选后没有因子")
            return
        
        # 聚类
        if len(selected_eval_res) < 2:
            selected_eval_res['group'] = 1
            print("因子数量少于2个，不进行聚类")
        else:
            print(f"对 {len(selected_eval_res)} 个因子进行聚类")
            
            # 为聚类准备参数，去掉test_dir
            cluster_params_copy = self.cluster_params.copy()
            cluster_params_copy.pop('test_dir', None)
            
            try:
                groups = cluster_factors(
                    selected_eval_res, 
                    datetime.strptime(date_start, '%Y%m%d'), 
                    datetime.strptime(date_end, '%Y%m%d'), 
                    **cluster_params_copy
                )
                selected_eval_res['group'] = groups
                print(f"聚类完成，共 {selected_eval_res['group'].nunique()} 个组")
            except Exception as e:
                print(f"聚类失败: {str(e)}，设置所有因子为同一组")
                selected_eval_res['group'] = 1
        
        # 保存筛选结果
        self._save_results(selected_eval_res, res_dir, res_selected_dir)
        
        print(f"期间 {period_name} 应用过滤器结果筛选完成")
      
    def _save_results(self, final_factors: pd.DataFrame, res_dir: Path, res_selected_dir: Path) -> None:
        """
        保存筛选结果和复制相关文件
        
        参数:
            final_factors (pd.DataFrame): 筛选后的因子数据
            res_dir (Path): 结果目录
            res_selected_dir (Path): 选定结果目录
        """
        print(f"保存 {len(final_factors)} 个筛选后的因子")
        
        # 为applied filters创建因子列表格式
        # 这里使用tag_name和factor_name来标识因子
        final_factors_to_list = []
        for idx in final_factors.index:
            tag_name = final_factors.loc[idx, 'tag_name']
            factor_name = final_factors.loc[idx, 'factor_name']
            test_name = final_factors.loc[idx, 'test_name']
            final_factors_to_list.append([tag_name, factor_name, test_name])
        
        final_factors_to_list = deduplicate_nested_list(final_factors_to_list)
        
        # 保存到结果目录
        with open(res_dir / 'final_factors.json', 'w', encoding='utf-8') as f:
            json.dump(final_factors_to_list, f, ensure_ascii=False, indent=4)
            
        print(f"因子列表已保存到: {res_dir / 'final_factors.json'}")
        
        # 保存筛选后的因子信息到CSV文件
        final_factors.to_csv(res_dir / 'final_selected_factors.csv', index=False)
        print(f"筛选结果已保存到: {res_dir / 'final_selected_factors.csv'}")
        
        # 可选：复制相关文件（如果配置中启用）
        if self.config.get('copy_selected', False):
            print("注意: 应用过滤器结果通常不复制文件，因为测试文件结构不同")
            # 这里可以根据需要实现文件复制逻辑
            # 但由于applied filters的文件结构与原始factors不同，可能需要特殊处理
        
        print(f"已完成应用过滤器结果筛选，结果保存至: {res_dir}")


def example_usage():
    """
    使用示例
    """
    # 初始化AppliedFiltersSelector
    afs = AppliedFiltersSelector(
        select_name='basic_select',
        merge_name='merge_v1',
        test_eval_filtered_alpha_name='basic_test_eval'
    )
    
    # 对单个期间进行筛选
    afs.run_one_period('20240101', '20240331')
    
    print("应用过滤器结果筛选完成")


if __name__ == "__main__":
    example_usage()