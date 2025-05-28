# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:51:37 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
from functools import partial
from tqdm import tqdm


def read_parquet_files(folder_path: str, file_patterns: List[str] = None, 
                      suffix_list: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    读取指定文件夹下的parquet文件
    
    Args:
        folder_path: 文件夹路径
        file_patterns: 文件名模式列表，如果为None则读取所有.parquet文件
        suffix_list: 后缀列表，用于匹配特定后缀的文件
    
    Returns:
        文件名到DataFrame的映射字典
    """
    folder_path = Path(folder_path)
    files_dict = {}
    
    if file_patterns is None:
        # 读取所有parquet文件
        parquet_files = list(folder_path.glob('*.parquet'))
        for file_path in parquet_files:
            file_name = file_path.stem  # 去掉.parquet后缀
            files_dict[file_name] = pd.read_parquet(file_path)
    else:
        # 根据指定模式读取文件
        for pattern in file_patterns:
            if isinstance(pattern, dict) and 'pos' in pattern and 'neg' in pattern:
                # 处理pos/neg格式
                pos_name = pattern['pos']
                neg_name = pattern['neg']
                
                if suffix_list:
                    for suffix in suffix_list:
                        pos_files = list(folder_path.glob(f'{pos_name}*{suffix}.parquet'))
                        neg_files = list(folder_path.glob(f'{neg_name}*{suffix}.parquet'))
                        
                        if pos_files and neg_files:
                            pos_df = pd.read_parquet(pos_files[0])
                            neg_df = pd.read_parquet(neg_files[0])
                            key = f"{pos_name}_{neg_name}_{suffix}"
                            files_dict[key] = {
                                'pos_filter': pos_df,
                                'neg_filter': neg_df
                            }
                else:
                    # 找到所有匹配的文件并按后缀配对
                    pos_files = list(folder_path.glob(f'{pos_name}*.parquet'))
                    neg_files = list(folder_path.glob(f'{neg_name}*.parquet'))
                    
                    # 提取后缀并配对
                    pos_suffixes = {f.stem.replace(pos_name, '').lstrip('_'): f for f in pos_files}
                    neg_suffixes = {f.stem.replace(neg_name, '').lstrip('_'): f for f in neg_files}
                    
                    common_suffixes = set(pos_suffixes.keys()) & set(neg_suffixes.keys())
                    
                    for suffix in common_suffixes:
                        pos_df = pd.read_parquet(pos_suffixes[suffix])
                        neg_df = pd.read_parquet(neg_suffixes[suffix])
                        key = f"{pos_name}_{neg_name}_{suffix}" if suffix else f"{pos_name}_{neg_name}"
                        files_dict[key] = {
                            'pos_filter': pos_df,
                            'neg_filter': neg_df
                        }
            else:
                # 处理普通字符串模式
                pattern_files = list(folder_path.glob(f'{pattern}*.parquet'))
                for file_path in pattern_files:
                    file_name = file_path.stem
                    files_dict[file_name] = pd.read_parquet(file_path)
    
    return files_dict


def create_filter_task(alpha_df: pd.DataFrame, filter_data: Union[pd.DataFrame, Dict], 
                      filter_func_name: str, filter_name: str, save_path: str) -> Dict:
    """
    创建单个过滤任务
    
    Args:
        alpha_df: 原始信号DataFrame
        filter_data: 过滤因子数据
        filter_func_name: 过滤函数名称
        filter_name: 过滤器名称
        save_path: 保存路径
    
    Returns:
        任务字典
    """
    return {
        'alpha_df': alpha_df,
        'filter_data': filter_data,
        'filter_func_name': filter_func_name,
        'filter_name': filter_name,
        'save_path': save_path
    }


def execute_filter_task(task: Dict, apply_filter_module_globals: Dict) -> bool:
    """
    执行单个过滤任务
    
    Args:
        task: 任务字典
        apply_filter_module_globals: 包含apply_filter函数的globals字典
    
    Returns:
        执行成功返回True，否则返回False
    """
    try:
        alpha_df = task['alpha_df']
        filter_data = task['filter_data']
        filter_func_name = task['filter_func_name']
        filter_name = task['filter_name']
        save_path = task['save_path']
        
        # 获取过滤函数
        filter_func = apply_filter_module_globals.get(filter_func_name)
        if filter_func is None:
            print(f"Error: Function {filter_func_name} not found in globals")
            return False
        
        # 执行过滤
        filtered_result = filter_func(alpha_df, filter_data)
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存结果
        filtered_result.to_parquet(save_path)
        print(f"Successfully saved filtered result: {filter_name} -> {save_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing task {task.get('filter_name', 'unknown')}: {str(e)}")
        return False


def process_signal_filters(alpha_path: str, 
                          filter_configs: List[Dict], 
                          save_dir: str,
                          apply_filter_module_globals: Dict,
                          max_workers: Optional[int] = None) -> None:
    """
    批量处理信号过滤任务
    
    Args:
        alpha_path: 原始信号文件路径
        filter_configs: 过滤配置列表，每个元素包含：
            - 'filter_path': 过滤因子文件夹路径
            - 'filter_names': 选取因子名列表（可选）
            - 'suffix_list': 后缀列表（可选，用于pos/neg模式）
            - 'apply_filter_func': apply_filter函数名称
            - 'save_name': 该过滤的命名
        save_dir: 最终保存目录
        apply_filter_module_globals: 包含apply_filter函数的模块globals
        max_workers: 最大工作进程数，默认为CPU核心数
    
    Example:
        filter_configs = [
            {
                'filter_path': '/path/to/filter1',
                'filter_names': ['factor1', 'factor2'],  # 可选
                'apply_filter_func': 'basic_filter',
                'save_name': 'basic_filtering'
            },
            {
                'filter_path': '/path/to/filter2',
                'filter_names': [{'pos': 'pos_factor', 'neg': 'neg_factor'}],
                'suffix_list': ['1d', '5d'],  # 可选
                'apply_filter_func': 'conditional_mul_filter',
                'save_name': 'conditional_filtering'
            }
        ]
    """
    # 读取原始信号
    print(f"Loading alpha data from: {alpha_path}")
    alpha_df = pd.read_parquet(alpha_path)
    
    # 初始化所有任务
    all_tasks = []
    task_names = []  # 用于进度条显示
    
    for config in filter_configs:
        filter_path = config['filter_path']
        filter_names = config.get('filter_names', None)
        suffix_list = config.get('suffix_list', None)
        apply_filter_func = config['apply_filter_func']
        save_name = config['save_name']
        
        print(f"Processing filter config: {save_name}")
        
        # 读取过滤因子文件
        filter_files = read_parquet_files(filter_path, filter_names, suffix_list)
        
        # 为每个过滤因子创建任务
        for filter_name, filter_data in filter_files.items():
            save_path = os.path.join(save_dir, save_name, f"{filter_name}.parquet")
            
            task = create_filter_task(
                alpha_df=alpha_df,
                filter_data=filter_data,
                filter_func_name=apply_filter_func,
                filter_name=filter_name,
                save_path=save_path
            )
            all_tasks.append(task)
            task_names.append(f"{save_name}_{filter_name}")
    
    print(f"Created {len(all_tasks)} filtering tasks")
    
    # 设置工作进程数
    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, len(all_tasks))
    
    print(f"Starting concurrent processing with {max_workers} workers")
    
    # 创建部分函数，包含apply_filter_module_globals
    execute_task_with_globals = partial(execute_filter_task, 
                                       apply_filter_module_globals=apply_filter_module_globals)
    
    # 使用concurrent.futures执行任务并监控进度
    results = []
    success_count = 0
    failed_tasks = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(execute_task_with_globals, task): (task, task_name) 
            for task, task_name in zip(all_tasks, task_names)
        }
        
        # 使用tqdm监控进度
        with tqdm(total=len(all_tasks), desc="Processing filters", 
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
            
            for future in as_completed(future_to_task):
                task, task_name = future_to_task[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result:
                        success_count += 1
                        pbar.set_postfix({'Success': success_count, 'Failed': len(failed_tasks)})
                    else:
                        failed_tasks.append(task_name)
                        pbar.set_postfix({'Success': success_count, 'Failed': len(failed_tasks)})
                        
                except Exception as e:
                    failed_tasks.append(task_name)
                    results.append(False)
                    print(f"\nError in task {task_name}: {str(e)}")
                    pbar.set_postfix({'Success': success_count, 'Failed': len(failed_tasks)})
                
                pbar.update(1)
    
    # 统计结果
    total_count = len(results)
    
    print(f"\nProcessing completed: {success_count}/{total_count} tasks successful")
    
    if failed_tasks:
        print(f"Warning: {len(failed_tasks)} tasks failed:")
        for failed_task in failed_tasks[:10]:  # 只显示前10个失败的任务
            print(f"  - {failed_task}")
        if len(failed_tasks) > 10:
            print(f"  ... and {len(failed_tasks) - 10} more failed tasks")
    
    return {
        'total_tasks': total_count,
        'successful_tasks': success_count,
        'failed_tasks': failed_tasks
    }


# 使用示例函数
def example_usage():
    """
    使用示例
    """
    # 假设你已经从apply_filter模块导入了所有函数
    # from your_apply_filter_module import *
    
    # 配置过滤参数
    filter_configs = [
        {
            'filter_path': '/path/to/basic_filters',
            'filter_names': ['momentum', 'reversal'],  # 可选，如果不提供则读取所有parquet文件
            'apply_filter_func': 'basic_filter',
            'save_name': 'basic_filtering'
        },
        {
            'filter_path': '/path/to/conditional_filters',
            'filter_names': [
                {'pos': 'bull_market', 'neg': 'bear_market'},
                {'pos': 'high_vol', 'neg': 'low_vol'}
            ],
            'suffix_list': ['1d', '5d', '20d'],
            'apply_filter_func': 'conditional_mul_filter',
            'save_name': 'conditional_filtering'
        }
    ]
    
    # 执行过滤处理
    result = process_signal_filters(
        alpha_path='/path/to/original_alpha.parquet',
        filter_configs=filter_configs,
        save_dir='/path/to/filtered_results',
        apply_filter_module_globals=globals(),  # 传入当前模块的globals
        max_workers=4
    )
    
    print(f"Processing summary: {result}")
    
    # 可以根据返回结果进行后续处理
    if result['failed_tasks']:
        print("Some tasks failed, consider rerunning or checking the logs")


if __name__ == "__main__":
    example_usage()