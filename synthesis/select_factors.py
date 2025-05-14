# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:15:45 2025

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
import yaml
from pathlib import Path
import pandas as pd
from functools import partial
from datetime import datetime
from typing import Union, Dict, Any, Optional
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
from synthesis.filter_methods import filter_func_dynamic
from synthesis.factor_cluster import (cluster_factors, select_best_test_name)
from utils.datautils import align_to_primary_by_col_list


# %%
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


def select_by_multi_period_multi_filters(eval_res_list, filter_func_list, align_key_list=[]):
    for i, (eval_res, filter_func) in enumerate(list(zip(eval_res_list, filter_func_list))):
        if i == 0:
            selected_idx = filter_func(eval_res)
            first_eval_res = eval_res.copy()
        else:
            eval_res = align_to_primary_by_col_list(first_eval_res, eval_res, align_key_list)
            selected_idx_add = filter_func(eval_res)
            selected_idx &=  selected_idx_add
    return first_eval_res[selected_idx]


# %%
class FactorSelector:
    """
    因子选择器类：用于筛选、聚类并保存最佳因子
    """
    
    def __init__(self, select_name: str, eval_name: str = None):
        """
        初始化因子选择器
        
        参数:
            select_name (str): 选择配置名称
            eval_name (str, optional): 评估名称，如果为None则从配置文件中读取
        """
        self.select_name = select_name
        
        # 加载项目路径配置
        self.project_dir = project_dir
        self.path_config = load_path_config(self.project_dir)
        self.result_dir = Path(self.path_config['result'])
        self.param_dir = Path(self.path_config['param']) / 'select_factors'
        self.test_dir = self.result_dir / 'test'
        self.select_dir = self.result_dir / 'select_factors' / f'{eval_name}_{select_name}'
        
        # 加载配置文件
        config_path = self.param_dir / f'{select_name}.yaml'
        self.config = self._load_config(config_path)
        
        # 设置基本参数
        # 如果传入了eval_name，则使用传入的值，否则从配置文件中读取
        if eval_name is None:
            self.eval_name = self.config['basic'].get('eval_name')
            if self.eval_name is None:
                raise ValueError("eval_name不能为空，请在初始化时提供eval_name参数或在配置文件中设置")
        else:
            self.eval_name = eval_name
            
        self.eval_dir = self.result_dir / 'factor_evaluation' / self.eval_name
        
        # 筛选参数
        self.keep_best_test_by = self.config['best_test']['metric']
        
        self.cluster_params = self.config['cluster']
        self.cluster_params.update({'test_dir': self.test_dir})
        
        filter_param_list = self.config['filter_param_list']
        self.filter_func_list = [partial(globals()[filter_param['func_name']], **filter_param['params'])
                                 for filter_param in filter_param_list]

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
    
    def run_one_period(self, period_pairs):
        
        period_name = period_shortcut(*period_pairs[0])
        res_dir = self.select_dir / period_name
        res_selected_test_dir = res_dir / 'selected'
        res_selected_test_dir.mkdir(parents=True, exist_ok=True)
        
        filter_func_list = self.filter_func_list
        cluster_params = self.cluster_params
        
        eval_res_list = []
        for (date_start, date_end) in period_pairs:
            # 如果输入是字符串格式，转换为datetime对象
            if isinstance(date_start, str):
                date_start = datetime.strptime(date_start, '%Y%m%d')
            if isinstance(date_end, str):
                date_end = datetime.strptime(date_end, '%Y%m%d')
                
            # 设置基础时间段
            period_name = period_shortcut(date_start, date_end)
            
            # 读取因子评估结果
            path = self.eval_dir / f'factor_eval_{period_name}.csv'
            if not os.path.exists(path):
                print(f'Period: {period_name}未读取到时序衍生的评估结果（当期没有基础因子入选）')
                return
            eval_res = pd.read_csv(path)
            eval_res = extend_metrics(eval_res)
            eval_res_list.append(eval_res)
            
        # 多期评估
        selected_eval_res = select_by_multi_period_multi_filters(eval_res_list, filter_func_list, 
                                                                 align_key_list=['process_name', 'factor', 'test_name'])
        
        # 如果筛选后没有因子，则跳过后续步骤
        if selected_eval_res.empty:
            return
        
        # 保留最佳测试版本
        selected_eval_res = select_best_test_name(
            selected_eval_res,
            metric=self.keep_best_test_by,
        )
        
        # 聚类
        if len(selected_eval_res) < 2:
            selected_eval_res['group'] = 1
        else:
            groups = cluster_factors(selected_eval_res, date_start, date_end, **cluster_params)
            selected_eval_res['group'] = groups
            
        # 保存筛选结果
        self._save_results(selected_eval_res, res_dir, res_selected_test_dir)
      
    def _save_results(self, final_factors: pd.DataFrame, res_dir, res_selected_test_dir) -> None:
        """
        保存筛选结果和复制相关文件
        
        参数:
            final_factors (pd.DataFrame): 筛选后的因子数据
            top_factors (pd.DataFrame): 顶部因子数据（聚类前）
            res_dir (Path): 结果目录
            res_selected_test_dir (Path): 选定测试的结果目录
            save_to_all (bool): 是否将结果保存到类变量中，默认为False
        """

        # 保存因子列表为JSON
        final_factors_to_list = list(zip(final_factors['root_dir'], 
                                     final_factors['process_name'], 
                                     final_factors['factor']))
        final_factors_to_list = deduplicate_nested_list(final_factors_to_list)
        
        # 保存到结果目录
        with open(res_dir / 'final_factors.json', 'w', encoding='utf-8') as f:
            json.dump(final_factors_to_list, f, ensure_ascii=False, indent=4)
            
        if self.config.get('copy_selected', True):
            # 创建结果目录
            res_selected_test_info_dir = res_selected_test_dir / 'factor_info'
            res_selected_test_plot_dir = res_selected_test_dir / 'plot'
            res_selected_test_info_dir.mkdir(parents=True, exist_ok=True)
            res_selected_test_plot_dir.mkdir(parents=True, exist_ok=True)
            
            # 收集所有需要复制的文件对
            file_pairs = []
            
            # 遍历所有需要复制的因子
            for idx in final_factors.index:
                test_name = final_factors.loc[idx, 'test_name']
                tag_name = final_factors.loc[idx, 'tag_name']
                process_name = final_factors.loc[idx, 'process_name']
                factor_name = final_factors.loc[idx, 'factor']
                
                # 源目录路径
                target_test_dir = self.test_dir / test_name / tag_name / process_name
                
                # 收集因子信息图表文件
                factor_info_plot_dir = target_test_dir / 'factor_info'
                factor_info_files = find_files_with_prefix(factor_info_plot_dir, factor_name)
                
                for file_name in factor_info_files:
                    source_file = factor_info_plot_dir / file_name
                    target_file = res_selected_test_info_dir / file_name
                    file_pairs.append((source_file, target_file))
                
                # 收集因子图表文件
                factor_plot_source = target_test_dir / 'plot' / f'{factor_name}.jpg'
                factor_plot_target = res_selected_test_plot_dir / f'{factor_name}.jpg'
                
                if factor_plot_source.exists():
                    file_pairs.append((factor_plot_source, factor_plot_target))
                else:
                    print(f"警告: 因子图表文件不存在: {factor_plot_source}")
            
            # 使用多线程并行复制所有文件
            if file_pairs:
                print(f"开始并行复制 {len(file_pairs)} 个文件...")
                start_time = time.time()
                
                # 使用ThreadPoolExecutor进行并行复制
                with ThreadPoolExecutor(max_workers=min(len(file_pairs), 8)) as executor:
                    # 为每个文件对提交复制任务
                    future_to_file = {
                        executor.submit(copy_file, src, dst): (src, dst) 
                        for src, dst in file_pairs
                    }
                    
                    # 收集结果
                    success_count = 0
                    for future in as_completed(future_to_file):
                        src, dst = future_to_file[future]
                        try:
                            result = future.result()
                            if result:
                                success_count += 1
                                print(f"成功复制: {src.name}")
                            else:
                                print(f"复制失败: {src.name}")
                        except Exception as e:
                            print(f"复制 {src.name} 时发生异常: {e}")
                
                end_time = time.time()
                duration = end_time - start_time
                print(f"复制完成! 成功: {success_count}/{len(file_pairs)}, 用时: {duration:.2f} 秒")
            else:
                print("没有找到需要复制的文件。")
        
        # 保存筛选后的因子信息到CSV文件
        final_factors.to_csv(res_dir / 'final_selected_factors.csv', index=False)
        print(f"已完成因子筛选，筛选结果保存至: {res_dir}")