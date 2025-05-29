# -*- coding: utf-8 -*-
"""
Created on Wed May 28 2025

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
from typing import Union, Dict, Any, Optional, List
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
from utils.timeutils import period_shortcut
from apply_filter.apply_filters_to_target import process_signal_filters
from apply_filter.apply_methods import mul_filter, conditional_mul_filter


class FilterApplier:
    
    def __init__(self, apply_filters_name: str, merge_name: str, max_workers: Optional[int] = None):
        """
        初始化FilterApplier，用于对合并后的因子应用过滤器
        
        Args:
            apply_filters_name: 过滤器应用配置名称
            select_name: 选择名称
            merge_name: 合并名称
            max_workers: 最大并行工作进程数
        """
        self.apply_filters_name = apply_filters_name
        self.merge_name = merge_name
        self.max_workers = max_workers
        
        # 加载路径配置
        self.path_config = load_path_config(project_dir)
        self.result_dir = Path(self.path_config['result'])
        self.param_dir = Path(self.path_config['param']) / 'apply_filters_on_merged'
        
        # 设置目录路径
        self.merged_dir = self.result_dir / 'merge_selected_factors' / f'{merge_name}'
        self.filtered_dir = self.result_dir / 'apply_filters_on_merged' / f'{merge_name}' / self.apply_filters_name
        
        # 加载配置文件
        config_path = self.param_dir / f'{apply_filters_name}.yaml'
        self.config = self._load_config(config_path)
        
        # 确保输出目录存在
        self.filtered_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备apply_filter模块的globals，包含过滤函数
        self.apply_filter_globals = {
            'mul_filter': mul_filter,
            'conditional_mul_filter': conditional_mul_filter
        }
        
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
        对指定期间运行过滤器应用
        
        Args:
            date_start: 开始日期
            date_end: 结束日期
        """
        # 生成期间名称
        period_name = period_shortcut(date_start, date_end)
        
        # 设置此期间的目录
        self.merged_period_dir = self.merged_dir / period_name
        self.filtered_period_dir = self.filtered_dir / period_name
        self.filtered_period_dir.mkdir(parents=True, exist_ok=True)
        
        self._apply_filters_one_period(period_name)
    
    def _apply_filters_one_period(self, period_name: str):
        """
        对指定期间应用过滤器
        
        Args:
            period_name: 期间名称
        """
        print(f"开始对期间 {period_name} 应用过滤器")
        
        # 检查合并后的因子文件是否存在
        alpha_path = self.merged_period_dir / f'avg_predict_{period_name}.parquet'
        if not alpha_path.exists():
            print(f"警告: 未找到期间 {period_name} 的合并因子文件: {alpha_path}")
            return
        
        # 从配置中获取过滤器配置
        filter_configs = self.config['filter_configs']
        
        # 处理过滤器配置，确保路径正确
        processed_configs = []
        for config in filter_configs:
            processed_config = config.copy()
            # 如果filter_path是相对路径，转换为绝对路径
            filter_path = Path(processed_config['filter_path'])
            if not filter_path.is_absolute():
                filter_path = self.result_dir / filter_path
            processed_config['filter_path'] = str(filter_path)
            processed_configs.append(processed_config)
        
        # 使用process_signal_filters函数应用过滤器
        try:
            result = process_signal_filters(
                alpha_path=str(alpha_path),
                filter_configs=processed_configs,
                save_dir=str(self.filtered_period_dir),
                apply_filter_module_globals=self.apply_filter_globals,
                max_workers=self.max_workers
            )
            
            print(f"期间 {period_name} 过滤器应用完成: {result['successful_tasks']}/{result['total_tasks']} 任务成功")
            
            if result['failed_tasks']:
                print(f"警告: {len(result['failed_tasks'])} 个任务失败")
                
        except Exception as e:
            print(f"错误: 期间 {period_name} 过滤器应用失败: {str(e)}")


def example_usage():
    """
    使用示例
    """
    # 初始化FilterApplier
    fa = FilterApplier(
        apply_filters_name='basic_filters',
        # select_name='selected_v1',
        merge_name='merge_v1',
        max_workers=4
    )
    
    # 对单个期间应用过滤器
    fa.run_one_period('20240101', '20240331')
    
    print("过滤器应用完成")


if __name__ == "__main__":
    example_usage()