# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 2025

@author: Claude

Run Trade Method Analysis Tool
运行交易方法分析工具的主脚本
"""

# %% imports
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))

# %% import self-defined
from synthesis.trade_method_analysis_tool import TradeMethodAnalyzer

# %% main
def main():
    """主函数：解析命令行参数并运行交易方法分析"""
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='Run comprehensive analysis on select trade method results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 运行完整分析
  python run_trade_method_analysis.py -sf method1_select_v1 -tn trade_ver3_futtwap_sp1min_s240d_icim_v6
  
  # 指定配置文件和方向
  python run_trade_method_analysis.py -sf method1_select_v1 -tn trade_ver3_futtwap_sp1min_s240d_icim_v6 -c custom_config -d pos
  
  # 只运行特定分析
  python run_trade_method_analysis.py -sf method1_select_v1 -tn trade_ver3_futtwap_sp1min_s240d_icim_v6 --monthly-only
        """
    )
    
    # 必需参数
    parser.add_argument('-sf', '--select_folder', type=str, required=True,
                        help='Select trade method folder name')
    parser.add_argument('-tn', '--test_name', type=str, required=True,
                        help='Test name corresponding to the select folder')
    
    # 可选参数
    parser.add_argument('-c', '--config', type=str, default='default',
                        help='Configuration file name (without .yaml extension, default: default)')
    parser.add_argument('-d', '--direction', type=str, default='all', 
                        choices=['all', 'pos', 'neg'],
                        help='Analysis direction (default: all)')
    
    # 单独分析开关
    parser.add_argument('--monthly-only', action='store_true',
                        help='Run only monthly heatmap analysis')
    parser.add_argument('--weekly-only', action='store_true', 
                        help='Run only weekly position analysis')
    parser.add_argument('--sharpe-only', action='store_true',
                        help='Run only rolling Sharpe ratio analysis')
    
    # 调试和输出控制
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be analyzed without actually running')
    
    args = parser.parse_args()
    
    # 检查互斥参数
    analysis_flags = [args.monthly_only, args.weekly_only, args.sharpe_only]
    if sum(analysis_flags) > 1:
        parser.error("Only one of --monthly-only, --weekly-only, --sharpe-only can be specified")
    
    # 显示分析计划（干运行模式）
    if args.dry_run:
        print("🔍 Dry run mode - showing analysis plan:")
        print(f"📂 Select folder: {args.select_folder}")
        print(f"🧪 Test name: {args.test_name}")
        print(f"📋 Config: {args.config}")
        print(f"➡️  Direction: {args.direction}")
        
        if args.monthly_only:
            print("📊 Analysis: Monthly heatmap only")
        elif args.weekly_only:
            print("📊 Analysis: Weekly positions only")
        elif args.sharpe_only:
            print("📊 Analysis: Rolling Sharpe only")
        else:
            print("📊 Analysis: Full analysis (all components)")
        
        print("\nTo run the actual analysis, remove --dry-run flag")
        return
    
    try:
        # 创建分析器
        if args.verbose:
            print("🚀 Initializing Trade Method Analyzer...")
            print(f"📂 Select folder: {args.select_folder}")
            print(f"🧪 Test name: {args.test_name}")
            print(f"📋 Config: {args.config}")
            print(f"➡️  Direction: {args.direction}")
            print()
        
        analyzer = TradeMethodAnalyzer(
            select_folder_name=args.select_folder,
            test_name=args.test_name,
            config_name=args.config
        )
        
        # 根据参数运行相应分析
        if args.monthly_only:
            if args.verbose:
                print("📊 Running monthly heatmap analysis only...")
            analyzer.analyze_monthly_heatmap(direction=args.direction)
            
        elif args.weekly_only:
            if args.verbose:
                print("📊 Running weekly position analysis only...")
            analyzer.analyze_weekly_positions()
            
        elif args.sharpe_only:
            if args.verbose:
                print("📊 Running rolling Sharpe ratio analysis only...")
            analyzer.analyze_rolling_sharpe(direction=args.direction)
            
        else:
            # 运行完整分析
            if args.verbose:
                print("📊 Running full analysis...")
            analyzer.run_full_analysis(direction=args.direction)
        
        if args.verbose:
            print("\n✅ Analysis completed successfully!")
            print(f"📁 Results saved to: {analyzer.analysis_dir}")
    
    except FileNotFoundError as e:
        print(f"❌ File not found error: {e}")
        print("\nPlease check:")
        print("1. Select folder name and test name are correct")
        print("2. Required data files exist (gp_*.pkl, hsr_*.pkl, pos_*.parquet)")
        print("3. Path configuration is correct")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_batch_analysis():
    """批量分析函数（可作为模块调用）"""
    
    # 定义批量分析配置
    batch_configs = [
        {
            'select_folder': 'method1_select_v1',
            'test_name': 'trade_ver3_futtwap_sp1min_s240d_icim_v6',
            'config': 'default',
            'direction': 'all'
        },
        {
            'select_folder': 'method2_select_v1',
            'test_name': 'trade_ver3_futtwap_sp1min_s240d_icim_v6', 
            'config': 'default',
            'direction': 'all'
        },
        # 添加更多配置...
    ]
    
    print(f"🔄 Starting batch analysis for {len(batch_configs)} configurations...")
    
    success_count = 0
    failed_configs = []
    
    for i, config in enumerate(batch_configs, 1):
        try:
            print(f"\n📊 [{i}/{len(batch_configs)}] Analyzing: {config['select_folder']}")
            
            analyzer = TradeMethodAnalyzer(
                select_folder_name=config['select_folder'],
                test_name=config['test_name'],
                config_name=config['config']
            )
            
            analyzer.run_full_analysis(direction=config['direction'])
            
            print(f"✅ Completed: {config['select_folder']}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Failed: {config['select_folder']} - {e}")
            failed_configs.append(config['select_folder'])
            continue
    
    # 输出批量分析结果
    print(f"\n🎯 Batch analysis completed!")
    print(f"✅ Successful: {success_count}/{len(batch_configs)}")
    if failed_configs:
        print(f"❌ Failed: {len(failed_configs)} configurations")
        print(f"   Failed configs: {', '.join(failed_configs)}")


# %% main
if __name__ == '__main__':
    main()
