# -*- coding: utf-8 -*-
"""
workflow_manager.py - 因子评估和交易策略工作流管理脚本

这个脚本管理完整的因子评估和交易策略选择流程:
1. 测试和评估因子 (agg_test_and_eval 或 rolling_eval)
2. 选择最佳因子 (select_factors)
3. 合并选定的因子 (merge_selected_factors)
4. 选择交易方法 (select_trade_method)

作者: [您的名字]
"""

import os
import sys
import argparse
import yaml
import toml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union, Tuple

# 添加项目根目录到系统路径
file_path = Path(__file__).resolve()
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))

# 导入各个模块
from test_and_eval.agg_batch_test_and_eval import AggTestEval
from test_and_eval.rolling_eval import RollingEval
from synthesis.select_factors import FactorSelector
from synthesis.rolling_select_factors import run_rolling_factor_select
from synthesis.merge_selected_factors import FactorMerger
from synthesis.rolling_merge_selected_factors import run_rolling_merge_selected_factors
from synthesis.select_trade_method_02 import TradeSelector
from synthesis.rolling_select_trade_method_02 import run_rolling_trade_select
from utils.dirutils import load_path_config
from utils.logutils import FishStyleLogger


class WorkflowManager:
    """
    因子评估和交易策略工作流管理器
    
    负责协调和执行完整的因子评估、选择、合并和交易策略选择流程。
    支持选择性执行流程中的特定步骤，以及处理多组参数配置。
    """
    
    def __init__(self, workflow_name: str):
        """
        初始化工作流管理器
        
        Args:
            workflow_name: 工作流配置文件名称（不含扩展名），将从param/workflow目录中读取
        """
        self.workflow_name = workflow_name
        self.logger = FishStyleLogger()
        
        # 加载路径配置
        self.path_config = load_path_config(project_dir)
        self.param_dir = Path(self.path_config['param'])
        self.result_dir = Path(self.path_config['result'])
        
        # 设置工作流配置目录
        self.workflow_dir = self.param_dir / 'workflow'
        if not self.workflow_dir.exists():
            self.workflow_dir.mkdir(parents=True, exist_ok=True)
            self.logger.warning(f"创建工作流配置目录: {self.workflow_dir}")
        
        # 加载配置
        self._load_config()

    def _load_config(self):
        """从param/workflow目录加载工作流配置文件"""
        # 尝试不同的扩展名
        config_extensions = ['.yaml', '.yml', '.toml']
        config_path = None
        
        for ext in config_extensions:
            test_path = self.workflow_dir / f"{self.workflow_name}{ext}"
            if test_path.exists():
                config_path = test_path
                break
        
        if config_path is None:
            raise FileNotFoundError(f"在 {self.workflow_dir} 目录中找不到 {self.workflow_name} 的配置文件")
        
        self.logger.info(f"加载工作流配置: {config_path}")
        
        # 根据文件扩展名加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                self.config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.toml':
                self.config = toml.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        # 提取触发器和时间范围设置
        self.triggers = self.config.get('triggers', {})
        self.time_range = self.config.get('time_range', {})
        
        # 提取共同参数
        self.eval_name = self.config.get('eval_name')
        if not self.eval_name:
            self.logger.warning("配置中未指定eval_name，这可能会导致一些步骤失败")
        
        # 提取各步骤的配置
        self.agg_test_eval_config = self.config.get('agg_test_eval', {})
        self.rolling_eval_config = self.config.get('rolling_eval', {})
        self.select_factors_config = self.config.get('select_factors', {})
        self.merge_factors_config = self.config.get('merge_factors', {})
        self.select_trade_config = self.config.get('select_trade', {})
    
    def run(self):
        """执行工作流"""
        # 设置时间范围
        pstart = self.time_range.get('pstart', '20230701')
        puntil = self.time_range.get('puntil')
        mode = self.time_range.get('mode', 'rolling')  # 'rolling' 或 'update'
        
        # 将日期字符串标准化为YYYYMMDD格式，如果puntil为None则使用当前日期
        if pstart:
            if isinstance(pstart, datetime):
                pstart = pstart.strftime('%Y%m%d')
            else:
                # 确保pstart是YYYYMMDD格式
                pstart = datetime.strptime(pstart, '%Y%m%d').strftime('%Y%m%d')
        
        if puntil:
            if isinstance(puntil, datetime):
                puntil = puntil.strftime('%Y%m%d')
            else:
                # 确保puntil是YYYYMMDD格式
                puntil = datetime.strptime(puntil, '%Y%m%d').strftime('%Y%m%d')
        else:
            # 如果没有指定puntil，使用当前日期
            puntil = datetime.now().strftime('%Y%m%d')
        
        self.logger.info(f"工作流开始执行 - 时间范围: {pstart} 到 {puntil}, 模式: {mode}")
        
        # 步骤1: 测试和评估因子
        if self.triggers.get('test', False):
            self._run_agg_test_eval()
        elif self.triggers.get('eval', False):
            self._run_rolling_eval(pstart, puntil, mode)
        
        # 步骤2-4: 选择因子、合并因子、选择交易方法
        # 如果启用了factor_select，则处理所有后续步骤
        
        # 获取所有需要处理的select_names
        select_names = self.select_factors_config.get('select_names', [])
        
        if not select_names:
            self.logger.warning("没有找到要处理的select_names，跳过因子选择和后续步骤")
            return
        
        # 处理每个select_name
        for select_name in select_names:
            self.logger.info(f"开始处理select_name: {select_name}")
            
            # 步骤2: 选择因子
            if self.triggers.get('factor_select', False):
                self._run_select_factors(select_name, pstart, puntil, mode)
            
            # 获取所有需要处理的merge_names
            merge_names = self.merge_factors_config.get('merge_names', [])
            
            if not merge_names:
                self.logger.warning(f"没有找到要处理的merge_names，跳过因子合并和后续步骤")
                continue
            
            # 处理每个merge_name
            for merge_name in merge_names:
                self.logger.info(f"开始处理merge_name: {merge_name}")
                
                # 构造完整的select_name (eval_name_select_name)
                full_select_name = f"{self.eval_name}_{select_name}"
                
                # 步骤3: 合并选定的因子
                # 如果启用了merge_factors，则处理合并步骤
                if self.triggers.get('merge_factors', False):
                    self._run_merge_factors(merge_name, full_select_name, pstart, puntil, mode)
                
                # 如果启用了trade_select，则处理交易选择步骤
                if self.triggers.get('trade_select', False):
                    # 获取所有需要处理的trade_select_names
                    trade_select_names = self.select_trade_config.get('select_names', [])
                    
                    if not trade_select_names:
                        self.logger.warning(f"没有找到要处理的trade_select_names，跳过交易方法选择步骤")
                        continue
                    
                    # 处理每个trade_select_name
                    for trade_select_name in trade_select_names:
                        self.logger.info(f"开始处理trade_select_name: {trade_select_name}")
                        
                        # 构造交易选择的merge_name (eval_name_select_name_merge_name)
                        trade_merge_name = f"{self.eval_name}_{select_name}_{merge_name}"
                        
                        # 获取rolling_select_name
                        rolling_select_name = self.select_trade_config.get('rolling_select_name')
                        
                        # 步骤4: 选择交易方法
                        self._run_select_trade(trade_select_name, rolling_select_name, 
                                               trade_merge_name, pstart, puntil, mode)
        
        self.logger.success(f"工作流执行完成!")

    def _run_agg_test_eval(self):
        """运行聚合测试和评估"""
        # 从配置中获取参数
        agg_eval_name = self.agg_test_eval_config.get('agg_eval_name', self.eval_name)
        test_wkr = self.agg_test_eval_config.get('test_wkr', 1)
        eval_wkr = self.agg_test_eval_config.get('eval_wkr', 1)
        skip_exists = self.agg_test_eval_config.get('skip_exists', True)
        
        if not agg_eval_name:
            self.logger.warning("没有指定agg_eval_name或eval_name，跳过聚合测试和评估")
            return
        
        self.logger.info(f"开始运行聚合测试和评估: {agg_eval_name}")
        
        try:
            # 初始化和运行AggTestEval
            agg_test_eval = AggTestEval(agg_eval_name, test_wkr, eval_wkr, skip_exists)
            agg_test_eval.run()
            self.logger.success(f"聚合测试和评估完成: {agg_eval_name}")
        except Exception as e:
            self.logger.error(f"聚合测试和评估失败: {str(e)}")
            raise

    def _run_rolling_eval(self, pstart, puntil, mode):
        """运行滚动评估"""
        # 从配置中获取参数
        eval_name = self.rolling_eval_config.get('eval_name', self.eval_name)
        eval_rolling_name = self.rolling_eval_config.get('eval_rolling_name')
        n_workers = self.rolling_eval_config.get('n_workers', 1)
        check_consistency = self.rolling_eval_config.get('check_consistency', True)
        
        if not eval_name:
            self.logger.warning("没有指定eval_name，跳过滚动评估")
            return
            
        if not eval_rolling_name:
            self.logger.warning("没有指定eval_rolling_name，跳过滚动评估")
            return
        
        self.logger.info(f"开始运行滚动评估: {eval_name}, rolling配置: {eval_rolling_name}")
        
        try:
            # 初始化和运行RollingEval
            rolling_eval = RollingEval(
                eval_name=eval_name,
                eval_rolling_name=eval_rolling_name,
                pstart=pstart,
                puntil=puntil,
                eval_type=mode,
                n_workers=n_workers,
                check_consistency=check_consistency
            )
            rolling_eval.run()
            self.logger.success(f"滚动评估完成: {eval_name}")
        except Exception as e:
            self.logger.error(f"滚动评估失败: {str(e)}")
            raise

    def _run_select_factors(self, select_name, pstart, puntil, mode):
        """运行因子选择"""
        # 从配置中获取参数
        rolling_select_name = self.select_factors_config.get('rolling_select_name')
        eval_name = self.select_factors_config.get('eval_name', self.eval_name)
        
        if not rolling_select_name:
            self.logger.warning("没有指定rolling_select_name，跳过因子选择")
            return
            
        if not eval_name:
            self.logger.warning("没有指定eval_name，跳过因子选择")
            return
        
        self.logger.info(f"开始运行因子选择: {select_name}, eval_name: {eval_name}")
        
        try:
            # 运行滚动因子选择
            run_rolling_factor_select(
                select_name=select_name,
                rolling_select_name=rolling_select_name,
                pstart=pstart,
                puntil=puntil,
                mode=mode,
                eval_name=eval_name
            )
            self.logger.success(f"因子选择完成: {select_name}")
        except Exception as e:
            self.logger.error(f"因子选择失败: {str(e)}")
            raise

    def _run_merge_factors(self, merge_name, select_name, pstart, puntil, mode):
        """运行因子合并"""
        # 从配置中获取参数
        rolling_merge_name = self.merge_factors_config.get('rolling_merge_name')
        n_workers = self.merge_factors_config.get('n_workers', 1)
        max_workers = self.merge_factors_config.get('max_workers', None)
        
        if not rolling_merge_name:
            self.logger.warning("没有指定rolling_merge_name，跳过因子合并")
            return
        
        self.logger.info(f"开始运行因子合并: merge_name: {merge_name}, select_name: {select_name}")
        
        try:
            # 运行滚动因子合并
            run_rolling_merge_selected_factors(
                merge_name=merge_name,
                select_name=select_name,
                rolling_merge_name=rolling_merge_name,
                pstart=pstart,
                puntil=puntil,
                mode=mode,
                max_workers=max_workers,
                n_workers=n_workers,
            )
            self.logger.success(f"因子合并完成: {merge_name}")
        except Exception as e:
            self.logger.error(f"因子合并失败: {str(e)}")
            raise

    def _run_select_trade(self, select_name, rolling_select_name, merge_name, pstart, puntil, mode):
        """运行交易方法选择"""
        if not rolling_select_name:
            self.logger.warning("没有指定rolling_select_name，跳过交易方法选择")
            return
        
        # 从配置中获取其他可选参数
        merge_type = self.select_trade_config.get('merge_type', 'merge_selected_factors')
        
        self.logger.info(f"开始运行交易方法选择: {select_name}, merge_name: {merge_name}")
        
        try:
            # 运行滚动交易选择
            run_rolling_trade_select(
                select_name=select_name,
                rolling_select_name=rolling_select_name,
                pstart=pstart,
                puntil=puntil,
                mode=mode,
                merge_type=merge_type,
                merge_name=merge_name,
            )
            self.logger.success(f"交易方法选择完成: {select_name}")
        except Exception as e:
            self.logger.error(f"交易方法选择失败: {str(e)}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="因子评估和交易策略工作流管理器")
    parser.add_argument('-w', '--workflow', type=str, required=True,
                        help='工作流配置名称（不含扩展名），将从param/workflow目录中读取')
    
    args = parser.parse_args()
    
    # 初始化并运行工作流管理器
    workflow = WorkflowManager(args.workflow)
    workflow.run()


if __name__ == "__main__":
    main()