# -*- coding: utf-8 -*-
"""
apply_filter_workflow.py - 应用过滤器工作流管理脚本

这个脚本管理完整的应用过滤器工作流程:
1. 测试和评估过滤后的alpha (test_eval_filtered_alpha)
2. 选择最佳应用过滤器 (select_applied_filters)
3. 合并选定的应用过滤器 (merge_selected_applied_filters)
4. 选择交易方法 (select_trade_method with merge_type=merge_selected_applied_filters)

作者: Xintang Zheng
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
from apply_filter.rolling_test_eval_filtered_alpha import run_rolling_test_eval_filtered_alpha
from apply_filter.rolling_select_applied_filters import run_rolling_select_applied_filters
from apply_filter.rolling_merge_selected_applied_filters import run_rolling_merge_selected_applied_filters
from synthesis.rolling_select_trade_method_02 import run_rolling_trade_select
from utils.dirutils import load_path_config
from utils.logutils import FishStyleLogger


class AppliedFilterWorkflowManager:
    """
    应用过滤器工作流管理器
    
    负责协调和执行完整的应用过滤器测试评估、选择、合并和交易策略选择流程。
    支持选择性执行流程中的特定步骤，以及处理多组参数配置。
    """
    
    def __init__(self, workflow_name: str):
        """
        初始化工作流管理器
        
        Args:
            workflow_name: 工作流配置文件名称（不含扩展名），将从param/apply_filter_workflow目录中读取
        """
        self.workflow_name = workflow_name
        self.logger = FishStyleLogger()
        
        # 加载路径配置
        self.path_config = load_path_config(project_dir)
        self.param_dir = Path(self.path_config['param'])
        self.result_dir = Path(self.path_config['result'])
        
        # 设置工作流配置目录
        self.workflow_dir = self.param_dir / 'apply_filter_workflow'
        if not self.workflow_dir.exists():
            self.workflow_dir.mkdir(parents=True, exist_ok=True)
            self.logger.warning(f"创建应用过滤器工作流配置目录: {self.workflow_dir}")
        
        # 加载配置
        self._load_config()

    def _load_config(self):
        """从param/apply_filter_workflow目录加载工作流配置文件"""
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
        
        self.logger.info(f"加载应用过滤器工作流配置: {config_path}")
        
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
        self.fac_merge_name = self.config.get('fac_merge_name')
        if not self.fac_merge_name:
            self.logger.warning("配置中未指定fac_merge_name，这可能会导致一些步骤失败")
        
        # 提取各步骤的配置
        self.test_eval_config = self.config.get('test_eval_filtered_alpha', {})
        self.select_applied_filters_config = self.config.get('select_applied_filters', {})
        self.merge_applied_filters_config = self.config.get('merge_applied_filters', {})
        self.select_trade_config = self.config.get('select_trade', {})
    
    def run(self):
        """执行应用过滤器工作流"""
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
        
        self.logger.info(f"应用过滤器工作流开始执行 - 时间范围: {pstart} 到 {puntil}, 模式: {mode}")
        
        # 步骤1: 测试和评估过滤后的alpha
        test_eval_names = []
        if self.triggers.get('test_eval', False):
            test_eval_names = self._run_test_eval_filtered_alpha(pstart, puntil, mode)
        else:
            # 如果不运行测试评估，从配置中获取existing的test_eval_names
            test_eval_names = self.test_eval_config.get('test_eval_filtered_alpha_names', [])
        
        if not test_eval_names:
            self.logger.warning("没有找到要处理的test_eval_filtered_alpha_names，跳过后续步骤")
            return
        
        # 处理每个test_eval_filtered_alpha_name
        for test_eval_filtered_alpha_name in test_eval_names:
            self.logger.info(f"开始处理test_eval_filtered_alpha_name: {test_eval_filtered_alpha_name}")
            
            # 步骤2: 选择应用过滤器
            select_names = []
            if self.triggers.get('select_applied_filters', False):
                select_names = self._run_select_applied_filters(test_eval_filtered_alpha_name, pstart, puntil, mode)
            else:
                # 如果不运行选择，从配置中获取existing的select_names
                select_names = self.select_applied_filters_config.get('select_names', [])
            
            if not select_names:
                self.logger.warning(f"没有找到要处理的select_names，跳过合并和后续步骤")
                continue
            
            # 处理每个select_name
            for select_name in select_names:
                self.logger.info(f"开始处理select_name: {select_name}")
                
                # 步骤3: 合并选定的应用过滤器
                filter_merge_names = []
                if self.triggers.get('merge_applied_filters', False):
                    filter_merge_names = self._run_merge_applied_filters(
                        test_eval_filtered_alpha_name, select_name, pstart, puntil, mode)
                else:
                    # 如果不运行合并，从配置中获取existing的filter_merge_names
                    filter_merge_names = self.merge_applied_filters_config.get('filter_merge_names', [])
                
                if not filter_merge_names:
                    self.logger.warning(f"没有找到要处理的filter_merge_names，跳过交易方法选择步骤")
                    continue
                
                # 步骤4: 选择交易方法
                if self.triggers.get('trade_select', False):
                    # 处理每个filter_merge_name
                    for filter_merge_name in filter_merge_names:
                        self.logger.info(f"开始处理filter_merge_name: {filter_merge_name}")
                        
                        # 获取所有需要处理的trade_select_names
                        trade_select_names = self.select_trade_config.get('select_names', [])
                        
                        if not trade_select_names:
                            self.logger.warning(f"没有找到要处理的trade_select_names，跳过交易方法选择步骤")
                            continue
                        
                        # 处理每个trade_select_name
                        for trade_select_name in trade_select_names:
                            self.logger.info(f"开始处理trade_select_name: {trade_select_name}")
                            
                            # 构造交易选择的merge_name (fac_merge_name_test_eval_filtered_alpha_name_select_name_filter_merge_name)
                            trade_merge_name = f"{self.fac_merge_name}_{test_eval_filtered_alpha_name}_{select_name}_{filter_merge_name}"
                            
                            # 获取rolling_select_name
                            rolling_select_name = self.select_trade_config.get('rolling_select_name')
                            
                            # 执行交易方法选择
                            self._run_select_trade(trade_select_name, rolling_select_name, 
                                                   trade_merge_name, pstart, puntil, mode)
        
        self.logger.success(f"应用过滤器工作流执行完成!")

    def _run_test_eval_filtered_alpha(self, pstart, puntil, mode):
        """运行测试和评估过滤后的alpha"""
        # 从配置中获取参数
        test_eval_filtered_alpha_names = self.test_eval_config.get('test_eval_filtered_alpha_names', [])
        rolling_name = self.test_eval_config.get('rolling_name')
        eval_rolling_name = self.test_eval_config.get('eval_rolling_name')
        max_workers = self.test_eval_config.get('max_workers', None)
        n_workers = self.test_eval_config.get('n_workers', 1)
        
        if not test_eval_filtered_alpha_names:
            self.logger.warning("没有指定test_eval_filtered_alpha_names，跳过测试评估")
            return []
            
        if not rolling_name:
            self.logger.warning("没有指定rolling_name，跳过测试评估")
            return []
            
        if not eval_rolling_name:
            self.logger.warning("没有指定eval_rolling_name，跳过测试评估")
            return []
        
        processed_names = []
        
        for test_eval_filtered_alpha_name in test_eval_filtered_alpha_names:
            self.logger.info(f"开始运行测试评估过滤后的alpha: {test_eval_filtered_alpha_name}")
            
            try:
                # 运行滚动测试评估
                run_rolling_test_eval_filtered_alpha(
                    test_eval_filtered_alpha_name=test_eval_filtered_alpha_name,
                    merge_name=self.fac_merge_name,
                    rolling_name=rolling_name,
                    eval_rolling_name=eval_rolling_name,
                    pstart=pstart,
                    puntil=puntil,
                    mode=mode,
                    max_workers=max_workers,
                    n_workers=n_workers
                )
                self.logger.success(f"测试评估过滤后的alpha完成: {test_eval_filtered_alpha_name}")
                processed_names.append(test_eval_filtered_alpha_name)
            except Exception as e:
                self.logger.error(f"测试评估过滤后的alpha失败 {test_eval_filtered_alpha_name}: {str(e)}")
                # 继续处理其他的test_eval_filtered_alpha_name
                continue
        
        return processed_names

    def _run_select_applied_filters(self, test_eval_filtered_alpha_name, pstart, puntil, mode):
        """运行选择应用过滤器"""
        # 从配置中获取参数
        select_names = self.select_applied_filters_config.get('select_names', [])
        rolling_name = self.select_applied_filters_config.get('rolling_name')
        n_workers = self.select_applied_filters_config.get('n_workers', 1)
        
        if not select_names:
            self.logger.warning("没有指定select_names，跳过应用过滤器选择")
            return []
            
        if not rolling_name:
            self.logger.warning("没有指定rolling_name，跳过应用过滤器选择")
            return []
        
        processed_names = []
        
        for select_name in select_names:
            self.logger.info(f"开始运行应用过滤器选择: {select_name}")
            
            try:
                # 运行滚动应用过滤器选择
                run_rolling_select_applied_filters(
                    select_name=select_name,
                    merge_name=self.fac_merge_name,
                    test_eval_filtered_alpha_name=test_eval_filtered_alpha_name,
                    rolling_name=rolling_name,
                    pstart=pstart,
                    puntil=puntil,
                    mode=mode,
                    n_workers=n_workers
                )
                self.logger.success(f"应用过滤器选择完成: {select_name}")
                processed_names.append(select_name)
            except Exception as e:
                self.logger.error(f"应用过滤器选择失败 {select_name}: {str(e)}")
                # 继续处理其他的select_name
                continue
        
        return processed_names

    def _run_merge_applied_filters(self, test_eval_filtered_alpha_name, select_name, pstart, puntil, mode):
        """运行合并选定的应用过滤器"""
        # 从配置中获取参数
        filter_merge_names = self.merge_applied_filters_config.get('filter_merge_names', [])
        rolling_merge_name = self.merge_applied_filters_config.get('rolling_merge_name')
        max_workers = self.merge_applied_filters_config.get('max_workers', None)
        n_workers = self.merge_applied_filters_config.get('n_workers', 1)
        
        if not filter_merge_names:
            self.logger.warning("没有指定filter_merge_names，跳过应用过滤器合并")
            return []
            
        if not rolling_merge_name:
            self.logger.warning("没有指定rolling_merge_name，跳过应用过滤器合并")
            return []
        
        processed_names = []
        
        for filter_merge_name in filter_merge_names:
            self.logger.info(f"开始运行应用过滤器合并: filter_merge_name: {filter_merge_name}")
            
            try:
                # 运行滚动应用过滤器合并
                run_rolling_merge_selected_applied_filters(
                    fac_merge_name=self.fac_merge_name,
                    test_eval_filtered_alpha_name=test_eval_filtered_alpha_name,
                    select_name=select_name,
                    filter_merge_name=filter_merge_name,
                    rolling_merge_name=rolling_merge_name,
                    pstart=pstart,
                    puntil=puntil,
                    mode=mode,
                    max_workers=max_workers,
                    n_workers=n_workers
                )
                self.logger.success(f"应用过滤器合并完成: {filter_merge_name}")
                processed_names.append(filter_merge_name)
            except Exception as e:
                self.logger.error(f"应用过滤器合并失败 {filter_merge_name}: {str(e)}")
                # 继续处理其他的filter_merge_name
                continue
        
        return processed_names

    def _run_select_trade(self, select_name, rolling_select_name, merge_name, pstart, puntil, mode):
        """运行交易方法选择"""
        if not rolling_select_name:
            self.logger.warning("没有指定rolling_select_name，跳过交易方法选择")
            return
        
        # 设置merge_type为merge_selected_applied_filters
        merge_type = 'merge_selected_applied_filters'
        
        self.logger.info(f"开始运行交易方法选择: {select_name}, merge_name: {merge_name}")
        
        try:
            # 运行滚动交易选择，复用synthesis下的select_trade_method_02
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
    parser = argparse.ArgumentParser(description="应用过滤器工作流管理器")
    parser.add_argument('-w', '--workflow', type=str, required=True,
                        help='工作流配置名称（不含扩展名），将从param/apply_filter_workflow目录中读取')
    
    args = parser.parse_args()
    
    # 初始化并运行工作流管理器
    workflow = AppliedFilterWorkflowManager(args.workflow)
    workflow.run()


if __name__ == "__main__":
    main()