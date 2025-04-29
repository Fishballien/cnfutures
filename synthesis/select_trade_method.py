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
import yaml
from pathlib import Path
import pandas as pd
from functools import partial
from datetime import datetime
from typing import Union, Dict, Any


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.timeutils import period_shortcut
from utils.datautils import compute_dataframe_dict_average, check_dataframe_consistency
from test_and_eval.evalutils import extend_metrics
from utils.datautils import add_dataframe_to_dataframe_reindex
from test_and_eval.factor_tester import FactorTesterByContinuous, FactorTesterByDiscrete
from synthesis.filter_methods import *


# %%
class TradeSelector:
    """
    因子选择器类：用于筛选、聚类并保存最佳交易方法
    
    该类负责：
    1. 加载配置文件指定的参数
    2. 根据评估结果筛选最佳交易方法
    3. 合并预测结果并生成持仓数据
    4. 进行回测验证
    """
    
    def __init__(self, select_name: str):
        """
        初始化因子选择器
        
        参数:
            select_name (str): 选择器名称，用于加载对应的配置文件和创建结果目录
        """
        self.select_name = select_name
        
        # 加载项目路径配置
        self.project_dir = project_dir  # 项目根目录，应该从外部导入
        self.path_config = load_path_config(self.project_dir)  # 加载路径配置
        self.result_dir = Path(self.path_config['result'])  # 结果根目录
        self.param_dir = Path(self.path_config['param']) / 'select_trade_method'  # 参数目录
        
        # 创建结果存储目录结构
        self.select_dir = self.result_dir / 'select_trade_method' / select_name  # 当前选择器的结果目录
        self.selected_dir = self.select_dir / 'selected'  # 存储筛选结果的目录
        self.pos_dir = self.select_dir / 'pos'  # 存储持仓数据的目录
        self.selected_dir.mkdir(parents=True, exist_ok=True)  # 创建目录，如果已存在则不报错
        self.pos_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置文件
        config_path = self.param_dir / f'{select_name}.yaml'
        self.config = self._load_config(config_path)
        
        # 设置基本参数
        self.eval_name = self.config['basic']['merge_name']  # 从配置中获取评估名称
        self.eval_type = self.config['basic']['merge_type']
        self.eval_dir = self.result_dir / self.eval_type / self.eval_name  # 评估结果目录
        
        # 设置筛选函数
        filter_param = self.config['filter_param']
        # 使用partial函数创建筛选函数，动态加载指定的函数并传入参数
        self.filter_func = partial(globals()[filter_param['func_name']], **filter_param['params'])
        
        # 设置一致性检查标志（默认为False，可在配置中开启）
        self.check_consistency = self.config.get('check_consistency', False)

    def _load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        参数:
            config_path (str or Path): 配置文件路径
            
        返回:
            Dict[str, Any]: 配置字典
        
        异常:
            FileNotFoundError: 当配置文件不存在时抛出
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def run_one_period(self, fit_date_start, fit_date_end, predict_date_start, predict_date_end):
        """
        运行一个时间段的因子选择和预测
        
        参数:
            fit_date_start: 拟合开始日期，可以是字符串或datetime对象
            fit_date_end: 拟合结束日期，可以是字符串或datetime对象
            predict_date_start: 预测开始日期
            predict_date_end: 预测结束日期
        """
        # 将字符串日期转换为datetime对象
        if isinstance(fit_date_start, str):
            fit_date_start = datetime.strptime(fit_date_start, '%Y%m%d')
        if isinstance(fit_date_end, str):
            fit_date_end = datetime.strptime(fit_date_end, '%Y%m%d')
            
        # 设置基础时间段名称
        period_name = period_shortcut(fit_date_start, fit_date_end)  # 拟合期间的简称
        predict_period = period_shortcut(predict_date_start, predict_date_end)  # 预测期间的简称
        
        # 选择最佳交易方法
        selected_eval_res = self._select_one_period(period_name)
        if selected_eval_res is not None:
            # 合并预测持仓数据
            self._merge_predict_pos(selected_eval_res, predict_period, predict_date_start, predict_date_end)
        else:
            print(f"警告: 期间 {period_name} 没有选出任何交易方法，跳过预测阶段")
        
    def _select_one_period(self, period_name):
        """
        选择一个时间段内的最佳交易方法
        
        参数:
            period_name: 时间段名称
            
        返回:
            pd.DataFrame: 筛选后的评估结果，如果没有符合条件的结果则返回None
        """
        # 创建结果目录
        res_dir = self.selected_dir / period_name
        res_dir.mkdir(parents=True, exist_ok=True)
            
        # 读取交易方式评估结果
        path = self.eval_dir / period_name / 'evaluation.csv'
        if not os.path.exists(path):
            print(f'Period: {period_name}未读取到交易方式的评估结果（当期没有基础因子入选）')
            return None
            
        # 加载评估结果并扩展指标
        eval_res = pd.read_csv(path)
        print(eval_res)
        if len(eval_res) == 0:
            print(f'Period: {period_name}交易方式的评估结果为空（当期没有基础因子入选）')
            return None
        eval_res = extend_metrics(eval_res)  # 计算扩展指标
            
        # 应用筛选函数选择最佳交易方法
        selected_eval_res = eval_res[self.filter_func(eval_res)]
        
        # 如果筛选后没有交易方法，则返回None
        if selected_eval_res.empty:
            print(f"Period: {period_name} 筛选后没有交易方法入选")
            return None

        # 保存筛选结果
        self._save_results(selected_eval_res, res_dir)
        
        return selected_eval_res
      
    def _save_results(self, final_factors: pd.DataFrame, res_dir) -> None:
        """
        保存筛选结果
        
        参数:
            final_factors (pd.DataFrame): 筛选后的交易方法数据
            res_dir (Path): 结果保存目录
        """
        # 保存筛选后的交易方法信息到CSV文件
        final_factors.to_csv(res_dir / 'final_selected_trades.csv', index=False)
        print(f"已完成交易方法筛选，筛选结果保存至: {res_dir}")
        
    def _merge_predict_pos(self, selected_eval_res, predict_period, predict_date_start, predict_date_end):
        """
        合并选定交易方法的预测持仓数据
        
        参数:
            selected_eval_res (pd.DataFrame): 选定的交易方法评估结果
            predict_period (str): 预测期间的简称
            predict_date_start: 预测开始日期
            predict_date_end: 预测结束日期
        """
        # 初始化持仓字典和权重字典
        pos_dict = {}
        weight_dict = {}
        
        # 遍历每个选中的交易方法
        for idx in selected_eval_res.index:
            # 获取交易方法的路径和名称信息
            root_dir = selected_eval_res.loc[idx, 'root_dir']
            test_name = selected_eval_res.loc[idx, 'test_name']
            factor = selected_eval_res.loc[idx, 'factor']
            
            # 读取该交易方法的持仓数据
            pos_path = Path(root_dir) / 'test' / test_name / 'data' / f'pos_{factor}.parquet'
            pos_of_trade = pd.read_parquet(pos_path)
            
            # 添加到字典中，后续用于计算加权平均
            pos_dict[test_name] = pos_of_trade
            weight_dict[test_name] = 1  # 默认每个交易方法权重相同
        
        # 计算加权平均持仓
        pos_avg = compute_dataframe_dict_average(pos_dict, weight_dict)
        
        # 提取预测期间的持仓数据
        pos_predict_period = pos_avg.loc[predict_date_start:predict_date_end]
        
        # 保存预测期间的持仓数据
        pos_predict_period.to_parquet(self.pos_dir / f'pos_{predict_period}.parquet')
        pos_predict_period.to_csv(self.pos_dir / f'pos_{predict_period}.csv')
        
        # 更新汇总预测文件
        self._save_predictions(pos_predict_period)
        
    def _save_predictions(self, y_pred):
        """
        将新的预测结果添加到已有的汇总预测文件中
        
        如果启用了一致性检查且存在debug_dir，则先检查数据一致性
        
        参数:
            y_pred (pd.DataFrame): 新的预测结果DataFrame
        """
        # 汇总预测文件路径
        pred_all_path = self.pos_dir / f'pos_{self.select_name}.parquet'
        
        # 检查是否有已存在的汇总文件
        if os.path.exists(pred_all_path):
            # 读取已存在的数据
            pred_all = pd.read_parquet(pred_all_path)
            # 过滤掉全为0或全为NaN的行，优化数据质量
            pred_all = pred_all[(~(pred_all == 0).all(axis=1)) & (~pred_all.isna().all(axis=1))]
            
            # 检查是否需要进行一致性检查
            if self.check_consistency and hasattr(self, 'debug_dir'):
                try:
                    # 使用check_dataframe_consistency函数检查数据一致性
                    status, info = check_dataframe_consistency(pred_all, y_pred)
                    
                    if status == "INCONSISTENT":
                        # 如果不一致，保存不一致的数据到debug目录
                        debug_path = self.debug_dir / f'predict_{self.select_name}_inconsistent.parquet'
                        y_pred.to_parquet(debug_path)
                        
                        # 构造详细的错误信息
                        error_msg = f"DataFrame一致性检查失败! 索引: {info['index']}, 列: {info['column']}, "
                        error_msg += f"原始值: {info['original_value']}, 新值: {info['new_value']}, "
                        error_msg += f"不一致计数: {info['inconsistent_count']}。已保存到 {debug_path}"
                        
                        raise ValueError(error_msg)
                except Exception as e:
                    if not isinstance(e, ValueError):  # 如果不是自己抛出的ValueError，则记录异常但继续执行
                        print(f"一致性检查过程中发生异常: {str(e)}")
                        
            # 进行拼接操作，合并新旧数据
            pred_all = add_dataframe_to_dataframe_reindex(pred_all, y_pred)
            # 再次过滤无效行
            pred_all = pred_all[(~(pred_all == 0).all(axis=1)) & (~pred_all.isna().all(axis=1))]
        else:
            # 如果文件不存在，则直接使用y_pred创建新的DataFrame
            pred_all = y_pred.copy()
            # 过滤无效行
            pred_all = pred_all[(~(pred_all == 0).all(axis=1)) & (~pred_all.isna().all(axis=1))]
        
        # 保存更新后的汇总数据
        pred_all.to_parquet(pred_all_path)
        print(f"已更新汇总预测数据: {pred_all_path}")
        
    def test_predicted(self):
        """
        对预测结果进行回测验证
        
        根据配置文件中指定的测试列表，对生成的交易信号进行回测
        """
        process_name = None  # 回测过程名称，可以为空
        factor_data_dir = self.pos_dir  # 持仓数据目录
        result_dir = self.select_dir  # 结果保存目录
        params = self.config  # 配置参数
        
        # 获取测试列表
        test_list = params['test_list']
        for test_info in test_list:
            # 解析测试配置
            mode = test_info['mode']  # 测试模式：test或trade
            test_name = test_info['test_name']  # 测试名称
            
            # 根据模式选择测试类
            if mode == 'test':
                # 连续信号测试类
                test_class = FactorTesterByContinuous
            elif mode == 'trade':
                # 离散信号测试类
                test_class = FactorTesterByDiscrete
            else:
                raise NotImplementedError(f"不支持的测试模式: {mode}")
        
            # 初始化测试器并执行测试
            ft = test_class(process_name, None, factor_data_dir, test_name=test_name, result_dir=result_dir)
            ft.test_one_factor(f'pos_{self.select_name}')
            print(f"已完成 {test_name} 测试，模式: {mode}")
            