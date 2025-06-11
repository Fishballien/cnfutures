# -*- coding: utf-8 -*-
"""
交易阈值对比分析工具
比较两种不同threshold_combinations的交易表现

@author: Based on Xintang Zheng's code
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import logging
from datetime import datetime
import seaborn as sns

# 使用英文字体设置
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_threshold_comparison_analysis(config, threshold_set1, threshold_set2, 
                                    threshold_name1="Strategy 1", threshold_name2="Strategy 2"):
    """
    Run comparison analysis for two threshold combinations
    
    Parameters:
    -----------
    config : dict
        Basic configuration parameters
    threshold_set1 : list
        First threshold combination, e.g. [[0.8, 0.0]]
    threshold_set2 : list
        Second threshold combination, e.g. [[0.6, 0.2]]
    threshold_name1 : str
        Name for first strategy
    threshold_name2 : str
        Name for second strategy
    """
    
    logger.info(f"Starting comparison analysis: {threshold_name1} vs {threshold_name2}")
    
    # 导入必要的函数（假设已经导入）
    from generate_and_analysis_signals import (
        load_data, scale_factor, apply_trade_rules, 
        generate_trade_df, process_trade_dfs
    )
    
    # 加载数据
    factor_data, price_data = load_data(
        config['factor_dir'], 
        config['factor_name'], 
        config['fut_dir'], 
        config['price_name']
    )
    
    # 缩放因子
    factor_scaled = scale_factor(
        factor_data, 
        config['scale_method'], 
        config['scale_window'], 
        config['sp'], 
        config['scale_quantile'], 
        config['direction']
    )
    
    # 创建两个不同的交易规则配置
    trade_config1 = config['trade_rule_param'].copy()
    trade_config1['threshold_combinations'] = threshold_set1
    
    trade_config2 = config['trade_rule_param'].copy()
    trade_config2['threshold_combinations'] = threshold_set2
    
    # Execute trading rules for two strategies
    logger.info(f"Executing Strategy 1: {threshold_name1} - Thresholds: {threshold_set1}")
    actual_pos1 = apply_trade_rules(
        factor_scaled, 
        config['trade_rule_name'], 
        trade_config1,
        config.get('trade_rule_input', 'series')
    )
    
    logger.info(f"Executing Strategy 2: {threshold_name2} - Thresholds: {threshold_set2}")
    actual_pos2 = apply_trade_rules(
        factor_scaled, 
        config['trade_rule_name'], 
        trade_config2,
        config.get('trade_rule_input', 'series')
    )
    
    # Generate trade records
    trade_dfs1 = {}
    trade_dfs2 = {}
    
    for col in actual_pos1.columns:
        # Trade records for Strategy 1
        trade_df1 = generate_trade_df(actual_pos1[col], price_data[col])
        trade_dfs1[col] = trade_df1
        
        # Trade records for Strategy 2
        trade_df2 = generate_trade_df(actual_pos2[col], price_data[col])
        trade_dfs2[col] = trade_df2
    
    # Process trade records
    trade_dfs1 = process_trade_dfs(trade_dfs1, config['fee'])
    trade_dfs2 = process_trade_dfs(trade_dfs2, config['fee'])
    
    # Analyze returns by minute
    minute_analysis1 = analyze_returns_by_minute(trade_dfs1)
    minute_analysis2 = analyze_returns_by_minute(trade_dfs2)
    
    # 创建输出目录
    comparison_dir = config['analysis_dir'] / f"threshold_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成对比图表
    plot_side_by_side_comparison(
        minute_analysis1, minute_analysis2, 
        threshold_name1, threshold_name2, 
        comparison_dir, config['factor_name']
    )
    
    plot_difference_analysis(
        minute_analysis1, minute_analysis2,
        threshold_name1, threshold_name2,
        comparison_dir, config['factor_name']
    )
    
    # 保存详细分析结果
    save_comparison_results(
        trade_dfs1, trade_dfs2, minute_analysis1, minute_analysis2,
        threshold_name1, threshold_name2, comparison_dir
    )
    
    return {
        'strategy1': {
            'name': threshold_name1,
            'thresholds': threshold_set1,
            'trade_dfs': trade_dfs1,
            'minute_analysis': minute_analysis1
        },
        'strategy2': {
            'name': threshold_name2,
            'thresholds': threshold_set2,
            'trade_dfs': trade_dfs2,
            'minute_analysis': minute_analysis2
        },
        'comparison_dir': comparison_dir
    }

def analyze_returns_by_minute(trade_dfs):
    """
    Analyze returns by minute
    
    Returns:
    --------
    dict: Contains minute-level return analysis for each instrument and overall
    """
    minute_analysis = {}
    
    # Combine all instruments' trade records
    all_trades = pd.concat([
        df.assign(instrument=col) 
        for col, df in trade_dfs.items() 
        if not df.empty
    ], ignore_index=True)
    
    if all_trades.empty:
        logger.warning("No trade records available for analysis")
        return {}
    
    # Extract minute from open time
    all_trades['open_minute'] = all_trades['open_time'].dt.strftime('%H:%M')
    
    # Aggregate returns by minute
    minute_returns = all_trades.groupby('open_minute')['net_return'].sum().sort_index()
    
    # Statistics by instrument
    for instrument in all_trades['instrument'].unique():
        instrument_trades = all_trades[all_trades['instrument'] == instrument]
        minute_returns_instrument = instrument_trades.groupby('open_minute')['net_return'].sum().sort_index()
        minute_analysis[instrument] = minute_returns_instrument
    
    # Overall statistics
    minute_analysis['total'] = minute_returns
    
    # Trade count statistics
    minute_counts = all_trades.groupby('open_minute').size().sort_index()
    minute_analysis['trade_counts'] = minute_counts
    
    return minute_analysis

def plot_side_by_side_comparison(minute_analysis1, minute_analysis2, 
                                name1, name2, save_dir, factor_name):
    """
    Plot side-by-side comparison bar chart
    """
    logger.info("Generating side-by-side comparison chart")
    
    # Get overall return data
    returns1 = minute_analysis1.get('total', pd.Series())
    returns2 = minute_analysis2.get('total', pd.Series())
    
    if returns1.empty and returns2.empty:
        logger.warning("Both strategies have no trading data")
        return
    
    # Combine data with consistent time index
    combined_data = pd.DataFrame({
        name1: returns1,
        name2: returns2
    }).fillna(0)
    
    # Create chart
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Set bar chart parameters
    x = np.arange(len(combined_data.index))
    width = 0.35
    
    # Plot side-by-side bars
    bars1 = ax.bar(x - width/2, combined_data[name1], width, 
                   label=name1, alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width/2, combined_data[name2], width, 
                   label=name2, alpha=0.8, color='lightcoral')
    
    # Set chart properties
    ax.set_xlabel('Trading Time (Minutes)')
    ax.set_ylabel('Total Net Returns (Log Scale)')
    ax.set_title(f'Trading Strategy Comparison Analysis - {factor_name}\n{name1} vs {name2}')
    ax.set_xticks(x)
    ax.set_xticklabels(combined_data.index, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to log scale if there are positive values
    min_val1 = combined_data[name1].min()
    min_val2 = combined_data[name2].min()
    max_val1 = combined_data[name1].max()
    max_val2 = combined_data[name2].max()
    
    # Use log scale only if all values are positive, otherwise use symlog
    if min_val1 > 0 and min_val2 > 0:
        ax.set_yscale('log')
    elif max_val1 > 0 or max_val2 > 0:
        # Use symlog for data that includes negative values
        ax.set_yscale('symlog', linthresh=1e-6)
        ax.set_ylabel('Total Net Returns (Symlog Scale)')
    
    # Show only partial x-axis labels to avoid overlap
    n_labels = len(combined_data.index)
    if n_labels > 30:
        step = max(1, n_labels // 30)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(combined_data.index[::step], rotation=45)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Add statistical information
    stats1 = f"{name1}:\\nTotal Return: {returns1.sum():.6f}\\nTrade Count: {len(returns1[returns1 != 0])}\\nAvg Return: {returns1.mean():.6f}"
    stats2 = f"{name2}:\\nTotal Return: {returns2.sum():.6f}\\nTrade Count: {len(returns2[returns2 != 0])}\\nAvg Return: {returns2.mean():.6f}"
    
    ax.text(0.02, 0.98, stats1, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax.text(0.02, 0.82, stats2, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.7))
    
    plt.tight_layout()
    
    # Save chart
    save_path = save_dir / f'{factor_name}_side_by_side_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Side-by-side comparison chart saved: {save_path}")

def plot_difference_analysis(minute_analysis1, minute_analysis2, 
                           name1, name2, save_dir, factor_name):
    """
    Plot return difference analysis chart
    """
    logger.info("Generating return difference analysis chart")
    
    # Get overall return data
    returns1 = minute_analysis1.get('total', pd.Series())
    returns2 = minute_analysis2.get('total', pd.Series())
    
    if returns1.empty and returns2.empty:
        logger.warning("Both strategies have no trading data")
        return
    
    # Combine data and calculate differences
    combined_data = pd.DataFrame({
        name1: returns1,
        name2: returns2
    }).fillna(0)
    
    # Calculate difference (Strategy 1 - Strategy 2)
    combined_data['difference'] = combined_data[name1] - combined_data[name2]
    
    # Create chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Top chart: Original return comparison (line chart)
    ax1.plot(combined_data.index, combined_data[name1], 
             label=name1, linewidth=2, marker='o', markersize=3, alpha=0.7)
    ax1.plot(combined_data.index, combined_data[name2], 
             label=name2, linewidth=2, marker='s', markersize=3, alpha=0.7)
    
    ax1.set_ylabel('Total Net Returns (Log Scale)')
    ax1.set_title(f'Trading Strategy Return Comparison - {factor_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Set y-axis to log scale for the top chart
    min_val1 = combined_data[name1].min()
    min_val2 = combined_data[name2].min()
    max_val1 = combined_data[name1].max()
    max_val2 = combined_data[name2].max()
    
    # Use log scale only if all values are positive, otherwise use symlog
    if min_val1 > 0 and min_val2 > 0:
        ax1.set_yscale('log')
    elif max_val1 > 0 or max_val2 > 0:
        # Use symlog for data that includes negative values
        ax1.set_yscale('symlog', linthresh=1e-6)
        ax1.set_ylabel('Total Net Returns (Symlog Scale)')
    
    # Set x-axis labels
    n_labels = len(combined_data.index)
    if n_labels > 20:
        step = max(1, n_labels // 20)
        x_ticks = range(0, n_labels, step)
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels([combined_data.index[i] for i in x_ticks], rotation=45)
    else:
        ax1.set_xticklabels(combined_data.index, rotation=45)
    
    # Bottom chart: Difference analysis (bar chart)
    colors = ['green' if x > 0 else 'red' for x in combined_data['difference']]
    bars = ax2.bar(range(len(combined_data)), combined_data['difference'], 
                   color=colors, alpha=0.7)
    
    ax2.set_xlabel('Trading Time (Minutes)')
    ax2.set_ylabel(f'Return Difference ({name1} - {name2}) (Log Scale)')
    ax2.set_title(f'Strategy Return Difference Analysis\\nPositive values indicate {name1} performs better, negative values indicate {name2} performs better')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Set y-axis to log scale for the difference chart
    min_diff = combined_data['difference'].min()
    max_diff = combined_data['difference'].max()
    
    # Use symlog for difference data (which often includes both positive and negative values)
    if max_diff > 0 or min_diff < 0:
        ax2.set_yscale('symlog', linthresh=1e-6)
        ax2.set_ylabel(f'Return Difference ({name1} - {name2}) (Symlog Scale)')
    
    # Set x-axis labels
    if n_labels > 20:
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels([combined_data.index[i] for i in x_ticks], rotation=45)
    else:
        ax2.set_xticks(range(len(combined_data)))
        ax2.set_xticklabels(combined_data.index, rotation=45)
    
    # Add difference statistics
    total_diff = combined_data['difference'].sum()
    positive_diff = combined_data['difference'][combined_data['difference'] > 0].sum()
    negative_diff = combined_data['difference'][combined_data['difference'] < 0].sum()
    avg_diff = combined_data['difference'].mean()
    
    diff_stats = f"Total Difference: {total_diff:.6f}\\nPositive Diff Sum: {positive_diff:.6f}\\nNegative Diff Sum: {negative_diff:.6f}\\nAverage Difference: {avg_diff:.6f}"
    
    ax2.text(0.02, 0.98, diff_stats, transform=ax2.transAxes, 
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save chart
    save_path = save_dir / f'{factor_name}_difference_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Difference analysis chart saved: {save_path}")

def save_comparison_results(trade_dfs1, trade_dfs2, minute_analysis1, minute_analysis2,
                          name1, name2, save_dir):
    """
    Save comparison analysis results
    """
    logger.info("Saving comparison analysis results")
    
    # Save trade records
    for strategy_name, trade_dfs in [(name1, trade_dfs1), (name2, trade_dfs2)]:
        strategy_dir = save_dir / strategy_name.replace(' ', '_')
        strategy_dir.mkdir(exist_ok=True)
        
        for instrument, df in trade_dfs.items():
            if not df.empty:
                save_path = strategy_dir / f"{instrument}_trades.parquet"
                df.to_parquet(save_path)
        
        # Save combined trade records
        all_trades = pd.concat([
            df.assign(instrument=col) 
            for col, df in trade_dfs.items() 
            if not df.empty
        ], ignore_index=True)
        
        if not all_trades.empty:
            all_trades.to_parquet(strategy_dir / "all_trades.parquet")
    
    # Save minute-level analysis results
    minute_comparison = pd.DataFrame({
        f'{name1}_returns': minute_analysis1.get('total', pd.Series()),
        f'{name2}_returns': minute_analysis2.get('total', pd.Series())
    }).fillna(0)
    
    minute_comparison['difference'] = minute_comparison.iloc[:, 0] - minute_comparison.iloc[:, 1]
    minute_comparison.to_csv(save_dir / 'minute_comparison.csv')
    
    # Generate summary report
    generate_summary_report(trade_dfs1, trade_dfs2, name1, name2, save_dir)
    # 保存分钟级分析结果
    minute_comparison = pd.DataFrame({
        f'{name1}_returns': minute_analysis1.get('total', pd.Series()),
        f'{name2}_returns': minute_analysis2.get('total', pd.Series())
    }).fillna(0)
    
    minute_comparison['difference'] = minute_comparison.iloc[:, 0] - minute_comparison.iloc[:, 1]
    minute_comparison.to_csv(save_dir / 'minute_comparison.csv')
    
    # 生成汇总报告
    generate_summary_report(trade_dfs1, trade_dfs2, name1, name2, save_dir)

def generate_summary_report(trade_dfs1, trade_dfs2, name1, name2, save_dir):
    """
    Generate summary report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("Trading Strategy Comparison Analysis Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Strategy 1: {name1}")
    report_lines.append(f"Strategy 2: {name2}")
    report_lines.append("")
    
    # Analyze each strategy
    for strategy_name, trade_dfs in [(name1, trade_dfs1), (name2, trade_dfs2)]:
        report_lines.append(f"=== {strategy_name} Analysis Results ===")
        
        all_trades = pd.concat([
            df for df in trade_dfs.values() if not df.empty
        ], ignore_index=True)
        
        if not all_trades.empty:
            total_return = all_trades['net_return'].sum()
            avg_return = all_trades['net_return'].mean()
            win_rate = (all_trades['net_return'] > 0).mean()
            total_trades = len(all_trades)
            
            report_lines.append(f"Total Trades: {total_trades}")
            report_lines.append(f"Total Net Return: {total_return:.6f}")
            report_lines.append(f"Average Return: {avg_return:.6f}")
            report_lines.append(f"Win Rate: {win_rate:.4f}")
            
            # Statistics by instrument
            for instrument in ['IC', 'IF', 'IM']:
                if instrument in trade_dfs and not trade_dfs[instrument].empty:
                    inst_df = trade_dfs[instrument]
                    inst_return = inst_df['net_return'].sum()
                    inst_trades = len(inst_df)
                    inst_win_rate = (inst_df['net_return'] > 0).mean()
                    
                    report_lines.append(f"  {instrument}: {inst_trades} trades, return {inst_return:.6f}, win rate {inst_win_rate:.4f}")
        else:
            report_lines.append(f"No trade records")
        
        report_lines.append("")
    
    # Save report
    with open(save_dir / 'comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write('\\n'.join(report_lines))
    
    logger.info("Summary report saved")

# Usage Example
if __name__ == "__main__":
    # Basic configuration
    base_config = {
        # Data paths
        'fut_dir': Path('/mnt/nfs/30.132_xt_data1/futuretwap'),
        'factor_dir': Path('/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/merge_selected_factors/batch_till20_newma_batch_test_v3_icim_nsr22_m2/210401_250401'),
        'factor_name': 'avg_predict_210401_250401',
        'price_name': 't1min_fq1min_dl1min',
        
        # Scaling parameters
        'scale_method': 'minmax_scale',
        'scale_window': '240d',
        'scale_quantile': 0.02,
        'sp': '1min',
        'direction': 1,
        
        # Trading parameters
        'trade_rule_name': 'trade_rule_by_trigger_v3_4',
        'trade_rule_input': 'series',
        'trade_rule_param': {
            'time_threshold_minutes': 240,
            'close_long': True,
            'close_short': True
        },
        'fee': 0.00024,
        
        # Output paths
        'analysis_dir': Path('/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/analysis/threshold_comparison')
    }
    
    # Define two threshold sets for comparison
    threshold_set1 = [[0.8, 0.0]]  # High open threshold, no close threshold
    threshold_set2 = [[0.5, 0.0]]  # Medium open threshold, with close threshold
    
    # Execute comparison analysis
    results = run_threshold_comparison_analysis(
        config=base_config,
        threshold_set1=threshold_set1,
        threshold_set2=threshold_set2,
        threshold_name1="High Threshold Strategy (0.8,0.0)",
        threshold_name2="Medium Threshold Strategy (0.5,0.0)"
    )
    
    print("Comparison analysis completed!")
    print(f"Results saved in: {results['comparison_dir']}")