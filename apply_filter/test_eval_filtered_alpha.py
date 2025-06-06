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
# %% imports
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Union, Dict, Any, Optional, List
import yaml
import concurrent.futures
import multiprocessing

# 设置多进程启动方法为spawn，避免OpenMP fork问题
multiprocessing.set_start_method('spawn', force=True)


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% 独立的评估任务执行函数（放在类外，避免多进程调用类方法的问题）
def execute_eval_task(task: Dict, eval_date_start: str, eval_date_end: str,
                      eval_config: Dict, price_path: str):
    """
    执行单个评估任务的独立函数
    
    Args:
        task: 评估任务字典
        eval_date_start: 评估开始日期
        eval_date_end: 评估结束日期
        eval_config: 评估配置
        price_path: 价格数据路径
        
    Returns:
        Dict: 评估结果字典，失败时返回None
    """
    try:
        eval_inputs = {
            "factor_name": task['factor_name'],
            "date_start": eval_date_start,
            "date_end": eval_date_end,
            "data_date_start": eval_date_start,
            "data_date_end": eval_date_end,
            "process_name": task['process_name'],
            "test_name": task['test_name'],
            "tag_name": task['tag_name'],
            "data_dir": task['data_dir'],
            "processed_data_dir": task['processed_data_dir'],
            "valid_prop_thresh": eval_config['valid_prop_thresh'],
            "fee": eval_config['fee'],
            "price_data_path": price_path,
            "mode": task['mode'],
        }
        
        result = eval_one_factor_one_period(**eval_inputs)
        
        # 添加评估类型标识
        result['eval_type'] = task['eval_type']
        
        return result
        
    except Exception as e:
        print(f"执行评估任务失败 {task['factor_name']}: {str(e)}")
        return None


# %%
from utils.dirutils import load_path_config
from utils.timeutils import period_shortcut
from test_and_eval.factor_tester import FactorTesterByContinuous, FactorTesterByDiscrete
from test_and_eval.factor_evaluation import eval_one_factor_one_period


class TestEvalFilteredAlpha:
    
    def __init__(self, test_eval_filtered_alpha_name: str, merge_name: str, 
                 max_workers: Optional[int] = None):
        """
        初始化TestEvalFilteredAlpha，用于测试评估过滤后的alpha
        
        Args:
            test_eval_filtered_alpha_name: 测试评估配置名称
            merge_name: 合并名称
            max_workers: 最大并行工作进程数
        """
        self.test_eval_filtered_alpha_name = test_eval_filtered_alpha_name
        self.merge_name = merge_name
        self.max_workers = max_workers
        
        # 加载路径配置
        self.path_config = load_path_config(project_dir)
        self.result_dir = Path(self.path_config['result'])
        self.param_dir = Path(self.path_config['param']) / 'test_eval_filtered_alpha'
        
        # 加载配置文件
        config_path = self.param_dir / f'{test_eval_filtered_alpha_name}.yaml'
        self.config = self._load_config(config_path)
        
        # 设置目录路径
        self.filtered_base_dir = self.result_dir / 'apply_filters_on_merged' / f'{merge_name}'
        self.merged_dir = self.result_dir / 'merge_selected_factors' / f'{merge_name}'
        self.test_dir = self.result_dir / 'test_filtered_alpha' / f'{merge_name}'
        self.eval_dir = self.result_dir / 'eval_filtered_alpha' / f'{merge_name}_{test_eval_filtered_alpha_name}'
        
        # 确保输出目录存在
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def run_one_period(self, date_start: str, date_end: str, eval_date_start: str, eval_date_end: str):
        """
        对指定期间运行测试评估
        
        Args:
            date_start: 开始日期
            date_end: 结束日期
            eval_date_start: 评估开始日期
            eval_date_end: 评估结束日期
        """
        # 生成期间名称
        period_name = period_shortcut(date_start, date_end)
        eval_period_name = period_shortcut(eval_date_start, eval_date_end)
        
        # 设置此期间的目录
        self.merged_period_dir = self.merged_dir / period_name
        self.test_period_dir = self.test_dir / period_name
        self.eval_period_dir = self.eval_dir / eval_period_name
        self.test_period_dir.mkdir(parents=True, exist_ok=True)
        self.eval_period_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"开始对期间 {period_name} 进行测试评估")
        
        # 1. 测试过滤后的alpha
        self._test_filtered_alphas(period_name)
        
        # 2. 测试原始alpha
        self._test_original_alpha(period_name)
        
        # 3. 评估所有测试结果（合并到一个文件）
        self._evaluate_all_results(date_start, date_end, eval_date_start, eval_date_end, 
                                 period_name, eval_period_name)
        
        print(f"期间 {period_name} 测试评估完成")
    
    def _test_filtered_alphas(self, period_name: str):
        """
        测试过滤后的alpha文件
        
        Args:
            period_name: 期间名称
        """
        print(f"开始测试过滤后的alpha - 期间 {period_name}")
        
        # 获取root_dir_mapping
        root_dir_mapping = self.config['root_dir_mapping']
        test_list = self.config['test_list']
        
        # 遍历每个apply_filters_name（root_dir_mapping的第一层键）
        for apply_filters_name, sub_dirs in root_dir_mapping.items():
            print(f"处理过滤器配置: {apply_filters_name}")
            
            # 构建该apply_filters_name对应的过滤后目录
            filtered_apply_dir = self.filtered_base_dir / apply_filters_name / period_name
            
            if not filtered_apply_dir.exists():
                print(f"警告: 未找到过滤器目录: {filtered_apply_dir}")
                continue
            
            # 遍历每个子目录
            for sub_dir in sub_dirs:
                filtered_sub_dir = filtered_apply_dir / sub_dir
                if not filtered_sub_dir.exists():
                    print(f"警告: 未找到子目录: {filtered_sub_dir}")
                    continue
                
                # 获取所有parquet文件
                parquet_files = list(filtered_sub_dir.glob('*.parquet'))
                if not parquet_files:
                    print(f"警告: 在目录 {filtered_sub_dir} 中未找到parquet文件")
                    continue
                
                # 准备因子列表
                factor_names = [pf.stem for pf in parquet_files]
                
                # 对每个test_name进行测试
                for test_info in test_list:
                    mode = test_info['mode']
                    test_name = test_info['test_name']
                    skip_exists = test_info.get('skip_exists', False)
                    
                    print(f"  测试配置: {test_name} (模式: {mode}) - 过滤器: {apply_filters_name}/{sub_dir}")
                    print(f"  将测试 {len(factor_names)} 个因子")
                    
                    # 创建测试器
                    if mode == 'test':
                        test_class = FactorTesterByContinuous
                    elif mode == 'trade':
                        test_class = FactorTesterByDiscrete
                    else:
                        raise NotImplementedError(f"不支持的模式: {mode}")
                    
                    # 设置输出目录 - 调整为test_period_dir
                    test_result_dir = (self.test_period_dir / 'test' / test_name / 
                                      apply_filters_name / sub_dir)
                    test_result_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 创建测试器实例
                    tester = test_class(
                        process_name=sub_dir,  # 使用sub_dir作为process_name
                        tag_name=None,
                        factor_data_dir=filtered_apply_dir,
                        test_name=test_name,
                        save_dir=test_result_dir,
                        n_workers=self.max_workers,
                        skip_plot=test_info.get('skip_plot', True)
                    )
                    
                    # 使用test_multi批量测试所有因子
                    try:
                        tester.test_multi_factors(skip_exists=skip_exists)
                        print(f"    成功测试 {len(factor_names)} 个因子")
                    except Exception as e:
                        print(f"    错误: 批量测试因子失败: {str(e)}")
    
    def _test_original_alpha(self, period_name: str):
        """
        测试原始alpha文件
        
        Args:
            period_name: 期间名称
        """
        print(f"开始测试原始alpha - 期间 {period_name}")
        
        # 检查原始alpha文件是否存在
        alpha_path = self.merged_period_dir / f'avg_predict_{period_name}.parquet'
        if not alpha_path.exists():
            print(f"警告: 未找到原始alpha文件: {alpha_path}")
            return
        
        test_list = self.config['test_list']
        factor_name = f'avg_predict_{period_name}'
        
        # 对每个test_name进行测试
        for test_info in test_list:
            mode = test_info['mode']
            test_name = test_info['test_name']
            
            print(f"  测试原始alpha配置: {test_name} (模式: {mode})")
            
            # 创建测试器
            if mode == 'test':
                test_class = FactorTesterByContinuous
            elif mode == 'trade':
                test_class = FactorTesterByDiscrete
            else:
                raise NotImplementedError(f"不支持的模式: {mode}")
            
            # 设置输出目录
            test_result_dir = self.test_period_dir / 'test' / test_name / 'org_alpha'
            test_result_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建测试器实例
            tester = test_class(
                process_name=None,
                tag_name=None,
                factor_data_dir=self.merged_period_dir,
                test_name=test_name,
                save_dir=test_result_dir,
                n_workers=self.max_workers,
                skip_plot=test_info.get('skip_plot', True)
            )
            
            # 测试原始alpha因子
            try:
                tester.test_one_factor(factor_name)
                print(f"    成功测试原始alpha因子: {factor_name}")
            except Exception as e:
                print(f"    错误: 测试原始alpha {factor_name} 失败: {str(e)}")
    
    def _evaluate_all_results(self, date_start: str, date_end: str, eval_date_start: str, 
                            eval_date_end: str, period_name: str, eval_period_name: str):
        """
        评估所有测试结果并合并到一个文件
        
        Args:
            date_start: 开始日期
            date_end: 结束日期
            eval_date_start: 评估开始日期
            eval_date_end: 评估结束日期
            period_name: 期间名称
            eval_period_name: 评估期间名称
        """
        print(f"开始评估所有测试结果 - 期间 {period_name}")
        
        eval_config = self.config['eval']
        test_list = self.config['test_list']
        root_dir_mapping = self.config['root_dir_mapping']
        price_path = eval_config['price_path']
        
        # 设置评估输出目录
        eval_result_dir = self.eval_period_dir / 'eval'
        eval_result_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备所有评估任务
        eval_tasks = []
        
        # 1. 评估原始alpha
        for test_info in test_list:
            test_name = test_info['test_name']
            mode = test_info['mode']
            
            data_dir = (self.test_period_dir / 'test' / test_name / 'org_alpha' / 
                       'data')
            
            if data_dir.exists():
                factor_name = f'avg_predict_{period_name}'
                
                eval_tasks.append({
                    'factor_name': factor_name,
                    'test_name': test_name,
                    'mode': mode,
                    'tag_name': 'org_alpha',
                    'process_name': '',
                    'data_dir': data_dir,
                    'processed_data_dir': self.merged_period_dir,
                    'eval_type': 'original'
                })
        
        # 2. 评估过滤后的alpha
        for apply_filters_name, sub_dirs in root_dir_mapping.items():
            for sub_dir in sub_dirs:
                for test_info in test_list:
                    test_name = test_info['test_name']
                    mode = test_info['mode']
                    
                    data_dir = (self.test_period_dir / 'test' / test_name / 
                               apply_filters_name / sub_dir / 'data')
                    
                    if data_dir.exists():
                        # 获取所有测试过的因子
                        pkl_files = list(data_dir.glob('gpd_*.pkl'))
                        for pkl_file in pkl_files:
                            factor_name = pkl_file.name.replace('gpd_', '').replace('.pkl', '')
                            
                            # 构建对应的processed_data_dir路径
                            processed_data_dir = self.filtered_base_dir / apply_filters_name / period_name / sub_dir
                            
                            eval_tasks.append({
                                'factor_name': factor_name,
                                'test_name': test_name,
                                'mode': mode,
                                'tag_name': f'{apply_filters_name}_{sub_dir}',
                                'process_name': '',
                                'data_dir': data_dir,
                                'processed_data_dir': processed_data_dir,
                                'eval_type': 'filtered'
                            })
        
        # 执行评估任务
        print(f"准备执行 {len(eval_tasks)} 个评估任务")
        
        all_results = []
        
        if self.max_workers == 1 or self.max_workers is None:
            # 单进程执行
            for task in tqdm(eval_tasks, desc="评估进度"):
                result = execute_eval_task(task, eval_date_start, eval_date_end, 
                                          eval_config, price_path)
                if result is not None:
                    all_results.append(result)
        else:
            # 多进程执行
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        execute_eval_task, task, eval_date_start, eval_date_end,
                        eval_config, price_path
                    ) for task in eval_tasks
                ]
                
                for future in tqdm(concurrent.futures.as_completed(futures), 
                                 total=len(futures), desc="评估进度"):
                    try:
                        result = future.result()
                        if result is not None:
                            all_results.append(result)
                    except Exception as e:
                        print(f"评估任务失败: {str(e)}")
        
        # 合并所有结果并保存到一个文件
        if all_results:
            result_df = pd.DataFrame(all_results)
            result_path = eval_result_dir / f'eval_res_{eval_period_name}.csv'
            result_df.to_csv(result_path, index=False)
            print(f"所有评估结果已保存到: {result_path}")
            print(f"共评估了 {len(all_results)} 个因子")
        else:
            print("警告: 没有成功的评估结果")


def example_usage():
    """
    使用示例
    """
    # 初始化TestEvalFilteredAlpha
    tefa = TestEvalFilteredAlpha(
        test_eval_filtered_alpha_name='basic_test_eval',
        merge_name='merge_v1',
        max_workers=4
    )
    
    # 对单个期间进行测试评估
    tefa.run_one_period(
        date_start='20240101', 
        date_end='20240331',
        eval_date_start='20240101',
        eval_date_end='20240331'
    )
    
    print("测试评估完成")


if __name__ == "__main__":
    example_usage()