# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 2025

因子选择趋势分析脚本
分析每期选中的因子按大类的分布趋势

@author: Analysis Script
"""

# %% imports
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
import re
import warnings
warnings.filterwarnings("ignore")

# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))

# %%
from utils.dirutils import load_path_config


# %%
def analyze_factor_selection_trends(select_dir, save_plots=True):
    """
    分析因子选择趋势
    
    参数:
        select_dir: 选择结果的根目录
        save_plots: 是否保存图片，默认True
    
    返回:
        all_data: 所有期的原始数据
        stats_df: 统计数据
        factor_count_pivot: 因子数量透视表
        group_count_pivot: 分组数量透视表
    """
    
    print("🔍 开始分析因子选择趋势...")
    
    # 收集所有期的数据
    all_periods_data = []
    
    # 遍历所有期的结果目录
    period_dirs = [d for d in select_dir.iterdir() if d.is_dir()]
    period_dirs = sorted(period_dirs, key=lambda x: x.name)
    
    print(f"📅 找到 {len(period_dirs)} 个期数目录")
    
    for period_dir in period_dirs:
        result_file = period_dir / 'final_selected_factors.csv'
        if result_file.exists():
            try:
                # 读取该期的结果
                df = pd.read_csv(result_file)
                if not df.empty:
                    # 添加期数信息
                    df['period'] = period_dir.name
                    all_periods_data.append(df)
                    print(f"✅ 成功读取期数 {period_dir.name}: {len(df)} 个因子")
                else:
                    print(f"⚠️ 期数 {period_dir.name} 的结果文件为空")
            except Exception as e:
                print(f"❌ 读取 {result_file} 时出错: {e}")
        else:
            print(f"⚠️ 期数 {period_dir.name} 缺少结果文件")
    
    if not all_periods_data:
        print("❌ 未找到任何有效的选择结果文件")
        return None, None, None, None
    
    # 合并所有期的数据
    all_data = pd.concat(all_periods_data, ignore_index=True)
    print(f"📊 合并完成，共 {len(all_data)} 条记录")
    
    # 提取因子大类名（factor列中第一个"_"前的字符）
    all_data['factor_category'] = all_data['factor'].str.split('_').str[0]
    
    # 从period中提取日期信息（period格式：xxxx_YYMMDD）
    all_data['period_date'] = all_data['period'].str.split('_').str[-1]
    
    # 将YYMMDD转换为日期格式（假设YY是20xx年）
    def convert_period_date(date_str):
        if len(date_str) == 6:
            year = '20' + date_str[:2]
            month = date_str[2:4]
            day = date_str[4:6]
            return f'{year}-{month}-{day}'
        return date_str
    
    all_data['date_formatted'] = all_data['period_date'].apply(convert_period_date)
    all_data['date_formatted'] = pd.to_datetime(all_data['date_formatted'], errors='coerce')
    
    # 按日期排序
    all_data = all_data.sort_values('date_formatted')
    
    # 显示提取的因子大类
    categories = all_data['factor_category'].unique()
    print(f"🏷️ 识别到的因子大类: {list(categories)}")
    
    # 按期和因子大类统计
    period_category_stats = []
    
    for period in sorted(all_data['period'].unique()):
        period_data = all_data[all_data['period'] == period]
        period_date = period_data['date_formatted'].iloc[0]
        
        # 统计每个因子大类的因子数量和分组数
        category_stats = period_data.groupby('factor_category').agg({
            'factor': 'count',  # 因子数量
            'group': 'nunique'  # 分组数
        }).rename(columns={'factor': 'factor_count', 'group': 'group_count'})
        
        category_stats['period'] = period
        category_stats['period_date'] = period_date
        category_stats['factor_category'] = category_stats.index
        period_category_stats.append(category_stats.reset_index(drop=True))
    
    # 合并统计结果
    stats_df = pd.concat(period_category_stats, ignore_index=True)
    
    # 创建透视表用于画图，使用日期作为索引
    factor_count_pivot = stats_df.pivot(index='period_date', columns='factor_category', values='factor_count').fillna(0)
    group_count_pivot = stats_df.pivot(index='period_date', columns='factor_category', values='group_count').fillna(0)
    
    # 按日期排序
    factor_count_pivot = factor_count_pivot.sort_index()
    group_count_pivot = group_count_pivot.sort_index()
    
    # 绘制趋势图
    _plot_trends(factor_count_pivot, group_count_pivot, select_dir, save_plots)
    
    # 打印统计摘要
    _print_summary(all_data, factor_count_pivot, group_count_pivot)
    
    return all_data, stats_df, factor_count_pivot, group_count_pivot


def _plot_trends(factor_count_pivot, group_count_pivot, select_dir, save_plots):
    """绘制趋势图"""
    
    # 设置字体和图形样式
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8')
    
    # 设置颜色调色板
    colors = plt.cm.Set3(np.linspace(0, 1, len(factor_count_pivot.columns)))
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # 图1：因子数量趋势
    for i, category in enumerate(factor_count_pivot.columns):
        ax1.plot(factor_count_pivot.index, factor_count_pivot[category], 
                marker='o', linewidth=2.5, markersize=6, label=category, 
                color=colors[i], alpha=0.8)
    
    ax1.set_title('Time Series Trend of Selected Factor Count by Category', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Number of Factors (Log Scale)', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(title='Factor Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 设置x轴日期格式
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for i, category in enumerate(factor_count_pivot.columns):
        if factor_count_pivot[category].max() > 0:  # 只为有数据的类别添加标签
            for j, (date, value) in enumerate(factor_count_pivot[category].items()):
                if value > 0:
                    ax1.annotate(f'{int(value)}', 
                               (date, value), 
                               textcoords="offset points", 
                               xytext=(0,8), 
                               ha='center', fontsize=8, alpha=0.7)
    
    # 图2：分组数趋势
    for i, category in enumerate(group_count_pivot.columns):
        ax2.plot(group_count_pivot.index, group_count_pivot[category], 
                marker='s', linewidth=2.5, markersize=6, label=category, 
                color=colors[i], alpha=0.8)
    
    ax2.set_title('Time Series Trend of Group Count by Factor Category', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Number of Groups (Log Scale)', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(title='Factor Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 设置x轴日期格式
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for i, category in enumerate(group_count_pivot.columns):
        if group_count_pivot[category].max() > 0:
            for j, (date, value) in enumerate(group_count_pivot[category].items()):
                if value > 0:
                    ax2.annotate(f'{int(value)}', 
                               (date, value), 
                               textcoords="offset points", 
                               xytext=(0,8), 
                               ha='center', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图片
    if save_plots:
        plot_path = select_dir / 'factor_trend_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 趋势图已保存至: {plot_path}")
    
    plt.show()


def _print_summary(all_data, factor_count_pivot, group_count_pivot):
    """打印统计摘要"""
    
    print("\n" + "="*80)
    print("📊 因子选择趋势分析摘要")
    print("="*80)
    
    print(f"\n📅 总期数: {len(factor_count_pivot)}")
    print(f"🏷️ 因子大类数: {len(factor_count_pivot.columns)}")
    print(f"📝 因子大类: {list(factor_count_pivot.columns)}")
    
    print(f"\n📋 数据概览:")
    print(f"   总因子记录数: {len(all_data)}")
    print(f"   日期范围: {factor_count_pivot.index[0].strftime('%Y-%m-%d')} ~ {factor_count_pivot.index[-1].strftime('%Y-%m-%d')}")
    
    print(f"\n📈 各期因子总数统计:")
    total_factors_per_period = factor_count_pivot.sum(axis=1)
    print(f"   平均每期因子数: {total_factors_per_period.mean():.1f}")
    print(f"   最多一期因子数: {total_factors_per_period.max()}")
    print(f"   最少一期因子数: {total_factors_per_period.min()}")
    print(f"   标准差: {total_factors_per_period.std():.1f}")
    
    print(f"\n🔢 各期分组总数统计:")
    total_groups_per_period = group_count_pivot.sum(axis=1)
    print(f"   平均每期分组数: {total_groups_per_period.mean():.1f}")
    print(f"   最多一期分组数: {total_groups_per_period.max()}")
    print(f"   最少一期分组数: {total_groups_per_period.min()}")
    print(f"   标准差: {total_groups_per_period.std():.1f}")
    
    print(f"\n🏆 各因子大类平均选中情况 (按因子数量排序):")
    avg_factors = factor_count_pivot.mean().sort_values(ascending=False)
    avg_groups = group_count_pivot.mean().sort_values(ascending=False)
    
    for i, category in enumerate(avg_factors.index, 1):
        factor_avg = avg_factors[category]
        group_avg = avg_groups[category]
        factor_total = factor_count_pivot[category].sum()
        group_total = group_count_pivot[category].sum()
        
        print(f"   {i:2d}. {category:20s}: "
              f"平均因子数 {factor_avg:5.1f} (总计 {factor_total:3.0f}), "
              f"平均分组数 {group_avg:5.1f} (总计 {group_total:3.0f})")
    
    # 活跃度分析
    print(f"\n🔥 因子大类活跃度分析:")
    for category in factor_count_pivot.columns:
        active_periods = (factor_count_pivot[category] > 0).sum()
        total_periods = len(factor_count_pivot)
        activity_rate = active_periods / total_periods * 100
        
        print(f"   {category:20s}: {active_periods:2d}/{total_periods:2d} 期有选中 "
              f"(活跃度: {activity_rate:5.1f}%)")


def run_trend_analysis(eval_name, select_name, save_plots=True):
    """
    运行趋势分析的主函数
    
    参数:
        eval_name: 评估名称
        select_name: 选择配置名称
        save_plots: 是否保存图片
    """
    
    # 加载路径配置
    path_config = load_path_config(project_dir)
    result_dir = Path(path_config['result'])
    
    # 构建选择结果目录路径
    select_dir = result_dir / 'select_factors' / f'{eval_name}_{select_name}'
    
    if not select_dir.exists():
        print(f"❌ 选择结果目录不存在: {select_dir}")
        return None
    
    print(f"📁 分析目录: {select_dir}")
    
    # 执行分析
    results = analyze_factor_selection_trends(select_dir, save_plots)
    
    if results[0] is not None:
        # 保存统计结果
        all_data, stats_df, factor_count_pivot, group_count_pivot = results
        
        # 保存详细统计数据
        stats_file = select_dir / 'trend_analysis_stats.csv'
        stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
        
        # 保存透视表
        pivot_file = select_dir / 'factor_count_pivot.csv'
        factor_count_pivot.to_csv(pivot_file, encoding='utf-8-sig')
        
        group_pivot_file = select_dir / 'group_count_pivot.csv' 
        group_count_pivot.to_csv(group_pivot_file, encoding='utf-8-sig')
        
        print(f"💾 统计数据已保存至: {stats_file}")
        print(f"💾 因子数量透视表已保存至: {pivot_file}")
        print(f"💾 分组数量透视表已保存至: {group_pivot_file}")
        
    return results


# %%
if __name__ == "__main__":
    # 使用示例
    # 需要根据实际情况修改这些参数
    
    # 示例参数 - 请根据实际情况修改
    eval_name = "your_eval_name"      # 替换为实际的评估名称
    select_name = "your_select_name"  # 替换为实际的选择配置名称
    
    print("🚀 启动因子选择趋势分析")
    print(f"📊 评估名称: {eval_name}")
    print(f"⚙️ 选择配置: {select_name}")
    
    # 运行分析
    results = run_trend_analysis(eval_name, select_name, save_plots=True)
    
    if results and results[0] is not None:
        all_data, stats_df, factor_count_pivot, group_count_pivot = results
        print("\n✅ 分析完成！")
        
        # 可以进一步分析或导出结果
        # 例如：生成更详细的报告、导出到Excel等
        
    else:
        print("\n❌ 分析失败，请检查路径和数据文件")