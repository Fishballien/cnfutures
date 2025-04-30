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
from synthesis.factor_cluster import (cluster_factors, select_best_test_name, select_topn_per_group,
                                     remove_ma_suffix_from_factors)


# %%
class FactorSelector:
    """
    因子选择器类：用于筛选、聚类并保存最佳因子
    """
    
    def __init__(self, select_name: str):
        """
        初始化因子选择器
        
        参数:
            cluster_name (str): 聚类名称
        """
        self.select_name = select_name
        
        # 加载项目路径配置
        self.project_dir = project_dir
        self.path_config = load_path_config(self.project_dir)
        self.result_dir = Path(self.path_config['result'])
        self.param_dir = Path(self.path_config['param']) / 'select_basic_features'
        self.test_dir = self.result_dir / 'test'
        self.select_dir = self.result_dir / 'select_basic_features' / select_name
        
        # 加载配置文件
        config_path = self.param_dir / f'{select_name}.yaml'
        self.config = self._load_config(config_path)
        
        # 设置基本参数
        self.feval_name = self.config['basic']['feval_name']
        self.feval_dir = self.result_dir / 'factor_evaluation' / self.feval_name
        
        # 筛选参数
        self.keep_best_test_by = self.config['best_test']['metric']
        self.test_name_range = self.config.get('test_name_range')
        self.filter_func_name = self.config['first_filter']['func_name']
        self.filter_params = self.config['first_filter']['params']
        self.cluster_params = self.config['cluster']
        self.cluster_params.update({'test_dir': self.test_dir})
        self.final_filter_by = self.config['final_filter']['metric']
        self.top_n = self.config['final_filter']['top_n']

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
    
    def run_one_period(self, date_start: Union[str, datetime], date_end: Union[str, datetime], 
                       save_to_all: bool = False) -> pd.DataFrame:
        """
        运行单个时间段的因子筛选流程，遍历所有原始因子
        
        参数:
            date_start (str or datetime): 开始日期，str格式为YYYYMMDD或datetime对象
            date_end (str or datetime): 结束日期，str格式为YYYYMMDD或datetime对象
            save_to_all (bool): 是否将结果保存到类变量中，默认为False
            
        返回:
            pd.DataFrame: 所有原始因子筛选后的结果合并
        """
        # 如果输入是字符串格式，转换为datetime对象
        if isinstance(date_start, str):
            date_start = datetime.strptime(date_start, '%Y%m%d')
        if isinstance(date_end, str):
            date_end = datetime.strptime(date_end, '%Y%m%d')
            
        # 设置基础时间段
        period_name = period_shortcut(date_start, date_end)
        
        # 读取因子评估结果
        path = self.feval_dir / f'factor_eval_{period_name}.csv'
        eval_res = pd.read_csv(path)
        eval_res['org_fac'] = eval_res['factor'].apply(lambda x: x.split('-', 1)[0])
        for direction_suffix in ('', '_long_only', '_short_only'):
            eval_res[f'net_ppt{direction_suffix}'] = (eval_res[f'net_return_annualized{direction_suffix}'] 
                                                      / eval_res[f'hsr{direction_suffix}'] / 245)
        for corr_type in ('cont', 'dist'):
            lt720_cols = [f'corr_{corr_type}_wd30', f'corr_{corr_type}_wd60', f'corr_{corr_type}_wd240', f'corr_{corr_type}_wd720']
            eval_res[f'corr_{corr_type}_lt720_avg'] = eval_res[lt720_cols].mean(axis=1)
        
        # 获取所有唯一的原始因子
        org_facs = eval_res['org_fac'].unique()
        
        # 存储所有筛选结果
        all_final_factors = []
        
        # 为每个原始因子进行筛选
        for org_fac in org_facs:
            # 为当前原始因子设置结果目录
            org_fac_dir = self.select_dir / org_fac
            res_dir = org_fac_dir / period_name
            res_selected_test_dir = res_dir / 'selected'
            res_selected_test_dir.mkdir(parents=True, exist_ok=True)
            
            # 筛选当前原始因子的数据
            org_fac_data = eval_res[eval_res['org_fac'] == org_fac].copy()
            
            # 筛选测试方法
            if self.test_name_range is not None:
                org_fac_data = org_fac_data[org_fac_data['test_name'].apply(lambda x: x in self.test_name_range)]
            
            # 保留最佳测试版本
            keep_best_test_df = select_best_test_name(
                org_fac_data,
                metric=self.keep_best_test_by,
            )
            
            # 第一次筛选
            filter_func = globals()[self.filter_func_name]
            filter_func_with_param = partial(filter_func, **self.filter_params)
            first_selected = keep_best_test_df[filter_func_with_param(keep_best_test_df)]
            
            # 如果筛选后没有因子，则跳过后续步骤
            if first_selected.empty:
                continue
            
            # 聚类
            if len(first_selected) < 2:
                first_selected['group'] = 1
            else:
                groups = cluster_factors(first_selected, date_start, date_end, **self.cluster_params)
                first_selected['group'] = groups
            
            # 每组选择顶部因子
            top_factors = select_topn_per_group(
                first_selected, 
                metric=self.final_filter_by,
                n=self.top_n,
                ascending=False,
            )
            
            # 移除MA后缀
            final_factors = remove_ma_suffix_from_factors(top_factors)
            
            # 保存筛选结果
            self._save_results(final_factors, top_factors, res_dir, res_selected_test_dir, save_to_all)
            
            # 将当前原始因子的结果添加到总结果中
            all_final_factors.append(final_factors)
        
        # 合并所有原始因子的结果
        if all_final_factors:
            return pd.concat(all_final_factors, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _save_results(self, final_factors: pd.DataFrame, top_factors: pd.DataFrame, res_dir: Path, 
                      res_selected_test_dir: Path, save_to_all: bool = False) -> None:
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
            
        # 保存top_factors到CSV
        top_factors.to_csv(res_dir / 'top_factors.csv', index=False)

        # 如果需要保存到all_final_factors
        if save_to_all:
            # 获取原始因子名称
            org_fac_dir = res_dir.parent
            
            # 尝试读取原始因子目录中已有的因子列表
            all_factors_path = org_fac_dir / 'all_final_factors.json'
            existing_factors = []
            if all_factors_path.exists():
                try:
                    with open(all_factors_path, 'r', encoding='utf-8') as f:
                        existing_factors = json.load(f)
                except:
                    existing_factors = []
            
            # 合并并去重
            combined_factors = existing_factors + final_factors_to_list
            unique_factors = deduplicate_nested_list(combined_factors)
            
            # 保存到原始因子目录
            with open(all_factors_path, 'w', encoding='utf-8') as f:
                json.dump(unique_factors, f, ensure_ascii=False, indent=4)
        
        # 创建结果目录
        res_selected_test_info_dir = res_selected_test_dir / 'factor_info'
        res_selected_test_plot_dir = res_selected_test_dir / 'plot'
        res_selected_test_info_dir.mkdir(parents=True, exist_ok=True)
        res_selected_test_plot_dir.mkdir(parents=True, exist_ok=True)
        
        # 收集所有需要复制的文件对
        file_pairs = []
        
        # 遍历所有需要复制的因子
        for idx in top_factors.index:
            test_name = top_factors.loc[idx, 'test_name']
            tag_name = top_factors.loc[idx, 'tag_name']
            process_name = top_factors.loc[idx, 'process_name']
            factor_name = top_factors.loc[idx, 'factor']
            
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