# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 11:25:30 2025

@author: Xintang Zheng

Factor Selector Main Script
用于从命令行运行因子选择流程
"""
# %% imports
import sys
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))

# %% import self-defined
from synthesis.select_basic_features import FactorSelector

# %% main
def main():
    '''解析命令行参数并运行因子选择'''
    parser = argparse.ArgumentParser(description='运行因子选择流程')
    parser.add_argument('-s', '--select_name', type=str, required=True, help='选择器配置名称')
    parser.add_argument('-st', '--start', type=str, default='20150101', help='开始日期(YYYYMMDD格式)')
    parser.add_argument('-ed', '--end', type=str, required=True, help='结束日期(YYYYMMDD格式)')
    args = parser.parse_args()
    
    # 解析参数
    select_name = args.select_name
    date_start = args.start
    date_end = args.end
    
    # 打印运行信息
    print("⭐ 开始因子选择 ⭐")
    print(f"📅 时间段: {date_start} 至 {date_end}")
    print(f"🔢 选择器配置: {select_name}")
    
    try:
        # 初始化因子选择器
        selector = FactorSelector(select_name)
        
        # 运行因子选择
        print("🚀 开始运行因子选择流程...")
        final_factors = selector.run_one_period(date_start, date_end)
        
        # 打印结果摘要
        print("\n✅ 因子选择完成!")
        print(f"📊 共选出 {len(final_factors)} 个因子:")
        for idx, row in final_factors.iterrows():
            print(f"  - {row['factor']} (组别: {row['group']})")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

# %% run main
if __name__ == '__main__':
    sys.exit(main())