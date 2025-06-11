# -*- coding: utf-8 -*-
"""
Factor GP Analysis Script
Analyze factor returns and create visualizations based on Sharpe ratios
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def calc_sharpe(returns):
    """计算年化夏普比率"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252)  # 假设252个交易日

def load_factor_gp(data_dir, factor_name):
    """加载单个因子的gp数据"""
    try:
        gp_path = data_dir / f'gpd_{factor_name}.pkl'
        with open(gp_path, 'rb') as f:
            gp_data = pickle.load(f)
        return gp_data['all']  # 返回'all'方向的数据
    except:
        return None

def analyze_factors(data_dir):
    """分析所有因子数据"""
    data_dir = Path(data_dir)
    
    # 获取所有gpd文件
    gpd_files = list(data_dir.glob('gpd_*.pkl'))
    factor_names = [f.stem.replace('gpd_', '') for f in gpd_files]
    
    print(f"Found {len(factor_names)} factors to analyze")
    
    # 存储结果
    factor_returns = {}
    factor_cumrets = {}
    sharpe_results = []
    
    # 处理每个因子
    for factor_name in tqdm(factor_names, desc="Loading factor data"):
        gp_data = load_factor_gp(data_dir, factor_name)
        if gp_data is not None and 'return' in gp_data.columns:
            returns = gp_data['return'].dropna()
            if len(returns) > 0:
                # 检查数据时间跨度
                start_date = returns.index.min()
                end_date = returns.index.max()
                time_span_years = (end_date - start_date).days / 365.25
                
                # 计算累积收益
                cumret = returns.cumsum()
                factor_returns[factor_name] = returns
                factor_cumrets[factor_name] = cumret
                
                # 计算Sharpe比率
                sharpe = calc_sharpe(returns)
                sharpe_results.append({
                    'factor': factor_name,
                    'sharpe_ratio': sharpe,
                    'total_return': cumret.iloc[-1] if len(cumret) > 0 else 0,
                    'num_days': len(returns),
                    'start_date': start_date,
                    'end_date': end_date,
                    'time_span_years': time_span_years
                })
    
    return factor_returns, factor_cumrets, sharpe_results

def create_visualization(factor_cumrets, sharpe_df, save_dir):
    """创建可视化图表"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置图形参数
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (20, 12)
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # 筛选时间跨度大于5年的因子，然后取Top20
    long_term_factors = sharpe_df[sharpe_df['time_span_years'] >= 5.0]
    if len(long_term_factors) < 20:
        print(f"Warning: Only {len(long_term_factors)} factors have data span >= 5 years")
        top20_factors = long_term_factors['factor'].tolist()
    else:
        top20_factors = long_term_factors.head(20)['factor'].tolist()
    
    print(f"Selected {len(top20_factors)} factors with data span >= 5 years for Top20 visualization")
    
    # 绘制所有因子的累积收益（浅色）
    for factor_name, cumret in factor_cumrets.items():
        if factor_name not in top20_factors:
            ax.plot(cumret.index, cumret.values, 
                   color='lightgray', alpha=0.3, linewidth=0.5)
    
    # 绘制Top20因子的累积收益（突出显示）
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(top20_factors))))
    for i, factor_name in enumerate(top20_factors):
        if factor_name in factor_cumrets:
            cumret = factor_cumrets[factor_name]
            factor_info = sharpe_df[sharpe_df["factor"]==factor_name].iloc[0]
            ax.plot(cumret.index, cumret.values, 
                   color=colors[i], linewidth=2, 
                   label=f'{factor_name} (Sharpe: {factor_info["sharpe_ratio"]:.3f}, {factor_info["time_span_years"]:.1f}y)')
    
    # 设置图形属性
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Cumulative Return', fontsize=14)
    ax.set_title('Factor Cumulative Returns (Top 20 by Sharpe Ratio, Data Span >= 5 Years)', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    
    # 设置图例
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_dir / 'factor_cumulative_returns_top20_5y.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(save_dir / 'factor_cumulative_returns_top20_5y.jpg', 
                dpi=150, bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
    print(f"Visualization saved to {save_dir}")
    
    return top20_factors, long_term_factors

def main():
    # 指定数据目录
    data_dir = "/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/test/icim_intraday_scale_around_op05cl0/zxt/Batch18_250425/stocks/data"
    
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"Error: Data directory does not exist: {data_dir}")
        return
    
    print(f"Analyzing factors in: {data_dir}")
    
    # 分析因子数据
    factor_returns, factor_cumrets, sharpe_results = analyze_factors(data_dir)
    
    if not sharpe_results:
        print("No valid factor data found!")
        return
    
    # 创建Sharpe比率排名DataFrame
    sharpe_df = pd.DataFrame(sharpe_results)
    sharpe_df = sharpe_df.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)
    sharpe_df['rank'] = range(1, len(sharpe_df) + 1)
    
    # 保存排名结果到CSV
    save_dir = Path(data_dir).parent / 'analysis_results'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = save_dir / 'factor_sharpe_ranking.csv'
    sharpe_df.to_csv(csv_path, index=False)
    print(f"Sharpe ranking saved to: {csv_path}")
    
    # 创建可视化
    print(f"\nCreating visualization with {len(factor_cumrets)} factors...")
    top20_factors, long_term_factors = create_visualization(factor_cumrets, sharpe_df, save_dir)
    
    # 显示Top20（满足5年条件的）
    print(f"\nTop 20 Factors by Sharpe Ratio (Data Span >= 5 Years):")
    print("="*80)
    if len(long_term_factors) > 0:
        top20_filtered = long_term_factors.head(20)
        for i, row in top20_filtered.iterrows():
            print(f"{row['rank']:2d}. {row['factor']:<30} Sharpe: {row['sharpe_ratio']:8.4f} "
                  f"Total Return: {row['total_return']:8.4f} Span: {row['time_span_years']:5.1f}y")
    else:
        print("No factors found with data span >= 5 years")
    
    # 输出统计信息
    print(f"\nSummary Statistics:")
    print(f"Total factors analyzed: {len(sharpe_df)}")
    print(f"Factors with data span >= 5 years: {len(long_term_factors)}")
    print(f"Average Sharpe ratio (all): {sharpe_df['sharpe_ratio'].mean():.4f}")
    print(f"Average Sharpe ratio (5y+): {long_term_factors['sharpe_ratio'].mean():.4f}" if len(long_term_factors) > 0 else "N/A")
    print(f"Median Sharpe ratio (all): {sharpe_df['sharpe_ratio'].median():.4f}")
    print(f"Best Sharpe ratio (all): {sharpe_df['sharpe_ratio'].max():.4f}")
    print(f"Best Sharpe ratio (5y+): {long_term_factors['sharpe_ratio'].max():.4f}" if len(long_term_factors) > 0 else "N/A")
    
    # 保存筛选后的排名
    if len(long_term_factors) > 0:
        long_term_csv_path = save_dir / 'factor_sharpe_ranking_5y_plus.csv'
        long_term_factors.to_csv(long_term_csv_path, index=False)
        print(f"5-year+ factors ranking saved to: {long_term_csv_path}")
    
    # 保存详细统计信息
    stats_path = save_dir / 'analysis_summary.txt'
    with open(stats_path, 'w') as f:
        f.write(f"Factor Analysis Summary\n")
        f.write(f"="*50 + "\n")
        f.write(f"Data Directory: {data_dir}\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total factors analyzed: {len(sharpe_df)}\n")
        f.write(f"Factors with data span >= 5 years: {len(long_term_factors)}\n")
        f.write(f"Average Sharpe ratio (all): {sharpe_df['sharpe_ratio'].mean():.4f}\n")
        if len(long_term_factors) > 0:
            f.write(f"Average Sharpe ratio (5y+): {long_term_factors['sharpe_ratio'].mean():.4f}\n")
            f.write(f"Best Sharpe ratio (5y+): {long_term_factors['sharpe_ratio'].max():.4f}\n")
        f.write(f"Median Sharpe ratio (all): {sharpe_df['sharpe_ratio'].median():.4f}\n")
        f.write(f"Best Sharpe ratio (all): {sharpe_df['sharpe_ratio'].max():.4f}\n")
        f.write(f"Worst Sharpe ratio (all): {sharpe_df['sharpe_ratio'].min():.4f}\n\n")
        
        if len(long_term_factors) > 0:
            f.write("Top 20 Factors (Data Span >= 5 Years):\n")
            f.write("-" * 80 + "\n")
            top20_filtered = long_term_factors.head(20)
            for i, row in top20_filtered.iterrows():
                f.write(f"{row['rank']:2d}. {row['factor']:<30} Sharpe: {row['sharpe_ratio']:8.4f} "
                       f"Span: {row['time_span_years']:5.1f}y\n")
        else:
            f.write("No factors found with data span >= 5 years\n")
    
    print(f"Analysis summary saved to: {stats_path}")

if __name__ == "__main__":
    main()