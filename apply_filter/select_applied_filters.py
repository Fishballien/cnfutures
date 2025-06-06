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
from synthesis.filter_methods import filter_func_dynamic


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
        self.eval_dir = self.result_dir / 'eval_filtered_alpha' / f'{merge_name}_{test_eval_filtered_alpha_name}'
        self.select_dir = self.result_dir / 'select_applied_filters' / f'{merge_name}_{test_eval_filtered_alpha_name}_{select_name}'
        
        # 加载配置文件
        config_path = self.param_dir / f'{select_name}.yaml'
        self.config = self._load_config(config_path)
        
        # 设置筛选参数
        self.cluster_params = self.config['cluster']
        self.cluster_params.update({'test_dir': None})  # applied filters不需要test_dir
        
        # 创建筛选函数，直接使用配置中的参数
        filter_config = self.config['filter']
        
        self.filter_func = partial(
            filter_func_dynamic,
            **filter_config
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
    
    def _apply_target_filter(self, eval_res):
        """
        应用目标筛选条件（如果配置中有target_filter参数）
        
        参数:
            eval_res (pd.DataFrame): 评估结果数据框
            
        返回:
            pd.DataFrame: 筛选后的数据框
        """
        target_filter = self.config['filter'].get('target_filter')
        if target_filter is None:
            return eval_res
        
        target_mask = pd.Series([True] * len(eval_res), index=eval_res.index)
        
        for col, value in target_filter.items():
            if col in eval_res.columns:
                target_mask &= (eval_res[col] == value)
            else:
                print(f"警告: 目标筛选列 '{col}' 不存在于数据中")
        
        filtered_data = eval_res[target_mask]
        print(f"目标筛选: {len(eval_res)} -> {len(filtered_data)} 个因子")
        
        return filtered_data
    
    def _get_factor_data_dir(self, factor_row):
        """
        根据因子信息获取数据目录路径，参考cluster_factors中的逻辑
        
        参数:
            factor_row: 包含因子信息的行数据
            
        返回:
            Path: 数据目录路径
        """
        # 检查是否存在test_data_dir列
        has_test_data_dir = 'test_data_dir' in factor_row.index
        
        if has_test_data_dir:
            test_data_dir = factor_row.get('test_data_dir')
            # 检查test_data_dir是否存在且不为空
            if pd.notna(test_data_dir) and str(test_data_dir).strip():
                # 直接使用test_data_dir作为数据目录的父目录
                return Path(test_data_dir).parent
        
        # 如果没有test_data_dir或者为空，使用传统的路径构建方式
        # 对于applied filters，我们需要从eval结果中推断路径结构
        test_name = factor_row.get('test_name', '')
        tag_name = factor_row.get('tag_name', '')
        process_name = factor_row.get('process_name', '')
        
        # 构建传统路径：test_dir / test_name / tag_name / process_name
        # 注意：这里我们没有test_dir，所以需要根据实际情况调整
        # 可能需要从配置或其他地方获取基础路径
        base_test_dir = self.result_dir / 'test'  # 假设基础测试目录
        
        if isinstance(tag_name, str) and tag_name.strip():
            target_dir = base_test_dir / test_name / tag_name / process_name
        else:
            target_dir = base_test_dir / test_name / process_name
            
        return target_dir
    
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
        # 直接在test_dir下寻找对应期间的子目录
        period_eval_dir = self.eval_dir / period_name / 'eval'
        
        if not period_eval_dir.exists():
            print(f"警告: 未找到期间 {period_name} 的评估目录: {period_eval_dir}")
            return
            
        # 寻找评估文件
        eval_files = list(period_eval_dir.glob('eval_res_*.csv'))
        if not eval_files:
            print(f"警告: 未找到期间 {period_name} 的评估结果文件")
            return
            
        eval_file = eval_files[0]  # 取第一个找到的文件
            
        print(f"读取评估结果文件: {eval_file}")
        eval_res = pd.read_csv(eval_file)
        
        # 扩展评估指标
        eval_res = extend_metrics(eval_res)
        
        print(f"评估结果包含 {len(eval_res)} 个因子")
        
        # 应用目标筛选
        eval_res_filtered = self._apply_target_filter(eval_res)
        
        if eval_res_filtered.empty:
            print(f"期间 {period_name} 目标筛选后没有因子")
            return
        
        # 应用筛选函数
        print(f"开始筛选，筛选前因子数量: {len(eval_res_filtered)}")
        
        selected_mask = self.filter_func(eval_res_filtered)
        selected_eval_res = eval_res_filtered[selected_mask].copy()
        
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
            
            # try:
            groups = cluster_factors(
                selected_eval_res, 
                date_start, 
                date_end, 
                **cluster_params_copy
            )
            selected_eval_res['group'] = groups
            print(f"聚类完成，共 {selected_eval_res['group'].nunique()} 个组")
            # except Exception as e:
            #     print(f"聚类失败: {str(e)}，设置所有因子为同一组")
            #     selected_eval_res['group'] = 1
        
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
            factor_name = final_factors.loc[idx, 'factor']
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
        
        # 复制相关文件（如果配置中启用）
        if self.config.get('copy_selected', False):
            print("开始复制筛选后的因子相关文件...")
            
            # 创建结果目录
            res_selected_info_dir = res_selected_dir / 'factor_info'
            res_selected_plot_dir = res_selected_dir / 'plot'
            res_selected_info_dir.mkdir(parents=True, exist_ok=True)
            res_selected_plot_dir.mkdir(parents=True, exist_ok=True)
            
            # 收集所有需要复制的文件对
            file_pairs = []
            
            # 遍历所有需要复制的因子
            for idx in final_factors.index:
                factor_name = final_factors.loc[idx, 'factor']
                
                try:
                    # 获取因子数据目录
                    target_dir = self._get_factor_data_dir(final_factors.loc[idx])
                    
                    # 收集因子信息图表文件
                    factor_info_plot_dir = target_dir / 'factor_info'
                    if factor_info_plot_dir.exists():
                        factor_info_files = find_files_with_prefix(factor_info_plot_dir, factor_name)
                        
                        for file_name in factor_info_files:
                            source_file = factor_info_plot_dir / file_name
                            target_file = res_selected_info_dir / file_name
                            if source_file.exists():
                                file_pairs.append((source_file, target_file))
                    else:
                        print(f"警告: 因子信息目录不存在: {factor_info_plot_dir}")
                    
                    # 收集因子图表文件
                    factor_plot_source = target_dir / 'plot' / f'{factor_name}.jpg'
                    factor_plot_target = res_selected_plot_dir / f'{factor_name}.jpg'
                    
                    if factor_plot_source.exists():
                        file_pairs.append((factor_plot_source, factor_plot_target))
                    else:
                        print(f"警告: 因子图表文件不存在: {factor_plot_source}")
                        
                except Exception as e:
                    print(f"警告: 处理因子 {factor_name} 时发生错误: {e}")
                    continue
            
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
        else:
            print("配置中未启用文件复制功能 (copy_selected=False)")
        
        print(f"已完成应用过滤器结果筛选，结果保存至: {res_dir}")


def example_usage():
    """
    使用示例
    """
    # 初始化AppliedFiltersSelector
    afs = AppliedFiltersSelector(
        select_name='gt_nsr_ppt',
        merge_name='batch_till20_newma_batch_test_v3_icim_nsr22_m0',
        test_eval_filtered_alpha_name='corr_and_diffusion_v1'
    )
    
    # 对单个期间进行筛选
    from datetime import datetime
    afs.run_one_period(datetime(2015, 1, 1), datetime(2016, 1, 1))
    
    print("应用过滤器结果筛选完成")


if __name__ == "__main__":
    example_usage()