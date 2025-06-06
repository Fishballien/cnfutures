# -*- coding: utf-8 -*-
"""
Created on Wed Jun 04 2025

@author: Based on Xintang Zheng's analysis framework

因子分析：按原始因子和合并类型分组分析
提取factor字符串中的raw_fac和merge_type，统计net_sharpe_ratio和hsr的分布

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pathlib import Path


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[2]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config


# %%
def load_eval_data(eval_name, period):
    """加载评估数据"""
    path_config = load_path_config(project_dir)
    result_dir = Path(path_config['result'])
    eval_dir = result_dir / 'factor_evaluation'
    
    path = eval_dir / eval_name / f'factor_eval_{period}.csv'
    
    if not path.exists():
        raise FileNotFoundError(f"评估文件不存在: {path}")
    
    print(f"📂 读取评估数据: {path}")
    eval_res = pd.read_csv(path)
    
    return eval_res


def setup_analysis_directories(eval_name, period):
    """设置分析输出目录"""
    path_config = load_path_config(project_dir)
    result_dir = Path(path_config['result'])
    analysis_dir = result_dir / 'analysis/eval_analysis'
    
    # 创建主分析目录
    main_analysis_dir = analysis_dir / f'{eval_name}/factor_eval_{period}'
    main_analysis_dir.mkdir(exist_ok=True, parents=True)
    
    # 创建因子分析子目录
    factor_analysis_dir = main_analysis_dir / 'factor_analysis'
    factor_analysis_dir.mkdir(exist_ok=True)
    
    print(f"📁 分析结果将保存到: {factor_analysis_dir}")
    
    return factor_analysis_dir


def extract_factor_components(factor_str):
    """
    从factor字符串中提取raw_fac和merge_type
    例如: "factor1-merge-type-other" -> raw_fac="factor1", merge_type="merge"
    """
    parts = factor_str.split('-')
    if len(parts) >= 2:
        raw_fac = parts[0]
        merge_type = parts[1] if len(parts) > 1 else ''
    else:
        raw_fac = factor_str
        merge_type = ''
    
    return raw_fac, merge_type


def prepare_factor_data(eval_data):
    """
    准备因子分析数据，提取raw_fac和merge_type
    """
    # 复制数据避免修改原始数据
    data = eval_data.copy()
    
    # 提取因子组成部分
    factor_components = data['factor'].apply(extract_factor_components)
    data['raw_fac'] = [comp[0] for comp in factor_components]
    data['merge_type'] = [comp[1] for comp in factor_components]
    
    print(f"📋 因子数据预处理完成:")
    print(f"  原始数据记录数: {len(eval_data)}")
    print(f"  处理后记录数: {len(data)}")
    print(f"  唯一原始因子: {data['raw_fac'].nunique()}")
    print(f"  唯一合并类型: {data['merge_type'].nunique()}")
    
    return data


# 设置全局变量
METRICS_DICT = {
    'net_sharpe_ratio': 'Net Sharpe Ratio',
    'hsr': 'HSR (Hit Success Rate)'
}

COLOR_SCHEME = {
    'heat_main': 'RdYlBu_r',
    'heat_secondary': 'viridis', 
    'box': 'Set3',
    'line': 'tab10'
}

def setup_output_directory(summary_dir):
    """设置输出目录"""
    base_dir = Path(summary_dir)
    base_dir.mkdir(exist_ok=True, parents=True)
    
    factor_analysis_dir = base_dir / 'factor_analysis'
    factor_analysis_dir.mkdir(exist_ok=True)
    
    return factor_analysis_dir


def setup_output_directory(summary_dir):
    """设置输出目录（保持兼容性）"""
    # 这个函数保持不变，但在主函数中会被setup_analysis_directories替代
    base_dir = Path(summary_dir)
    base_dir.mkdir(exist_ok=True, parents=True)
    
    factor_analysis_dir = base_dir / 'factor_analysis'
    factor_analysis_dir.mkdir(exist_ok=True)
    
    return factor_analysis_dir


def run_factor_analysis_main(eval_name, period, metrics=None, percentiles=None):
    """
    主分析函数 - 只需要eval_name和period两个参数
    """
    if metrics is None:
        metrics = ['net_sharpe_ratio', 'hsr']
    
    if percentiles is None:
        percentiles = [90, 75]
    
    print("🚀 开始因子分析...")
    print(f"📊 评估名称: {eval_name}")
    print(f"📅 分析期间: {period}")
    print(f"📈 分析指标: {metrics}")
    print(f"📊 分位数: {percentiles}")
    
    # 加载数据
    eval_data = load_eval_data(eval_name, period)
    
    # 设置输出目录
    output_dir = setup_analysis_directories(eval_name, period)
    
    # 准备数据
    data = prepare_factor_data(eval_data)
    
    # 检查指标是否存在
    available_metrics = []
    for metric in metrics:
        if metric in data.columns:
            available_metrics.append(metric)
            print(f"✅ 指标 {metric} 可用")
        else:
            print(f"⚠️  指标 {metric} 在数据中不存在，跳过分析")
    
    if not available_metrics:
        print("❌ 没有可用的分析指标，退出分析")
        return
    
    # 为每个指标生成分析
    for metric in available_metrics:
        print(f"\n🔍 分析指标: {METRICS_DICT.get(metric, metric)}")
        
        for percentile in percentiles:
            analyze_single_metric(data, metric, percentile, output_dir)
    
    # 生成综合比较分析
    generate_comprehensive_comparison(data, available_metrics, output_dir)
    
    print(f"\n✅ 分析完成！结果保存在: {output_dir}")
    return output_dir


def analyze_single_metric(data, metric, percentile, output_dir):
    """分析单个指标"""
    print(f"  📊 分析 {metric}, 分位数: {percentile}")
    
    # 获取唯一值
    raw_facs = sorted(data['raw_fac'].unique())
    merge_types = sorted(data['merge_type'].unique())
    
    # 1. 生成热力图
    generate_heatmaps(data, metric, percentile, raw_facs, merge_types, output_dir)
    
    # 2. 生成分布图
    generate_distribution_plots(data, metric, raw_facs, merge_types, output_dir)
    
    # 3. 生成汇总统计表
    generate_summary_table(data, metric, percentile, output_dir)


def generate_heatmaps(data, metric, percentile, raw_facs, merge_types, output_dir):
    """生成热力图：均值、分位数、最大值、最小值"""
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, figure=fig)
    
    # 计算统计量
    pivot_mean = data.pivot_table(values=metric, index='raw_fac', columns='merge_type', aggfunc='mean')
    pivot_percentile_high = data.pivot_table(values=metric, index='raw_fac', columns='merge_type', 
                                           aggfunc=lambda x: np.percentile(x, percentile))
    pivot_percentile_low = data.pivot_table(values=metric, index='raw_fac', columns='merge_type', 
                                          aggfunc=lambda x: np.percentile(x, 100-percentile))
    pivot_max = data.pivot_table(values=metric, index='raw_fac', columns='merge_type', aggfunc='max')
    pivot_min = data.pivot_table(values=metric, index='raw_fac', columns='merge_type', aggfunc='min')
    
    # 确保一致的顺序
    pivot_mean = pivot_mean.reindex(index=raw_facs, columns=merge_types)
    pivot_percentile_high = pivot_percentile_high.reindex(index=raw_facs, columns=merge_types)
    pivot_percentile_low = pivot_percentile_low.reindex(index=raw_facs, columns=merge_types)
    pivot_max = pivot_max.reindex(index=raw_facs, columns=merge_types)
    pivot_min = pivot_min.reindex(index=raw_facs, columns=merge_types)
    
    # 创建热力图
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(pivot_mean, annot=True, fmt='.3f', cmap=COLOR_SCHEME['heat_main'], ax=ax1, 
                cbar_kws={'label': 'Mean'})
    ax1.set_title(f'Mean {METRICS_DICT.get(metric, metric)}')
    ax1.set_xlabel('Merge Type')
    ax1.set_ylabel('Raw Factor')
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(pivot_percentile_high, annot=True, fmt='.3f', cmap=COLOR_SCHEME['heat_main'], ax=ax2, 
                cbar_kws={'label': f'{percentile}th Percentile'})
    ax2.set_title(f'{percentile}th Percentile {METRICS_DICT.get(metric, metric)}')
    ax2.set_xlabel('Merge Type')
    ax2.set_ylabel('Raw Factor')
    
    ax3 = fig.add_subplot(gs[1, 0])
    sns.heatmap(pivot_max, annot=True, fmt='.3f', cmap=COLOR_SCHEME['heat_main'], ax=ax3, 
                cbar_kws={'label': 'Maximum'})
    ax3.set_title(f'Maximum {METRICS_DICT.get(metric, metric)}')
    ax3.set_xlabel('Merge Type')
    ax3.set_ylabel('Raw Factor')
    
    ax4 = fig.add_subplot(gs[1, 1])
    sns.heatmap(pivot_min, annot=True, fmt='.3f', cmap=COLOR_SCHEME['heat_main'], ax=ax4, 
                cbar_kws={'label': 'Minimum'})
    ax4.set_title(f'Minimum {METRICS_DICT.get(metric, metric)}')
    ax4.set_xlabel('Merge Type')
    ax4.set_ylabel('Raw Factor')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'heatmaps_{metric}_pct{percentile}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存数据到CSV
    pivot_mean.to_csv(output_dir / f'heatmap_mean_{metric}.csv')
    pivot_percentile_high.to_csv(output_dir / f'heatmap_percentile_{percentile}_{metric}.csv')
    pivot_percentile_low.to_csv(output_dir / f'heatmap_percentile_{100-percentile}_{metric}.csv')
    pivot_max.to_csv(output_dir / f'heatmap_max_{metric}.csv')
    pivot_min.to_csv(output_dir / f'heatmap_min_{metric}.csv')


def generate_distribution_plots(data, metric, raw_facs, merge_types, output_dir):
    """生成分布图"""
    # 1. 按raw_fac分组的分布
    plt.figure(figsize=(12, 8))
    ax = sns.violinplot(x='raw_fac', y=metric, data=data, inner='box', 
                       palette=sns.color_palette(COLOR_SCHEME['box'], len(raw_facs)))
    ax.set_title(f'Distribution of {METRICS_DICT.get(metric, metric)} by Raw Factor')
    ax.set_xlabel('Raw Factor')
    ax.set_ylabel(METRICS_DICT.get(metric, metric))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / f'dist_by_raw_fac_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 按merge_type分组的分布
    plt.figure(figsize=(10, 8))
    ax = sns.violinplot(x='merge_type', y=metric, data=data, inner='box', 
                      palette=sns.color_palette(COLOR_SCHEME['box'], len(merge_types)))
    ax.set_title(f'Distribution of {METRICS_DICT.get(metric, metric)} by Merge Type')
    ax.set_xlabel('Merge Type')
    ax.set_ylabel(METRICS_DICT.get(metric, metric))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / f'dist_by_merge_type_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 组合分布图
    # 创建组合名称
    data['combo'] = data['raw_fac'] + ' | ' + data['merge_type']
    unique_combos = data['combo'].unique()
    
    if len(unique_combos) <= 20:
        # 如果组合数量不多，显示所有组合
        plt.figure(figsize=(14, 8))
        ax = sns.boxplot(x='combo', y=metric, data=data, 
                       palette=sns.color_palette(COLOR_SCHEME['box'], len(unique_combos)))
        ax.set_title(f'Distribution of {METRICS_DICT.get(metric, metric)} by Factor Combination')
        ax.set_xlabel('Raw Factor | Merge Type')
        ax.set_ylabel(METRICS_DICT.get(metric, metric))
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(output_dir / f'dist_by_combo_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # 如果组合太多，显示top10和bottom10
        combo_stats = data.groupby(['raw_fac', 'merge_type'])[metric].mean().reset_index()
        combo_stats['combo'] = combo_stats['raw_fac'] + ' | ' + combo_stats['merge_type']
        
        top10 = combo_stats.nlargest(10, metric)
        bottom10 = combo_stats.nsmallest(10, metric)
        extremes = pd.concat([top10, bottom10])
        
        extreme_combos = list(extremes['combo'].unique())
        filtered_data = data[data['combo'].isin(extreme_combos)]
        
        plt.figure(figsize=(16, 8))
        ax = sns.boxplot(x='combo', y=metric, data=filtered_data, 
                       palette=sns.color_palette(COLOR_SCHEME['box'], len(extreme_combos)))
        ax.set_title(f'Distribution of {METRICS_DICT.get(metric, metric)} - Top 10 & Bottom 10 Combinations')
        ax.set_xlabel('Raw Factor | Merge Type')
        ax.set_ylabel(METRICS_DICT.get(metric, metric))
        plt.xticks(rotation=90)
        
        # 为top和bottom组合着色
        for i, combo in enumerate(ax.get_xticklabels()):
            combo_text = combo.get_text()
            if combo_text in top10['combo'].values:
                ax.get_xticklabels()[i].set_color('green')
                ax.get_xticklabels()[i].set_weight('bold')
            else:
                ax.get_xticklabels()[i].set_color('red')
                ax.get_xticklabels()[i].set_weight('bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'dist_extremes_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 二维热力图分布（如果数据量合适）
    if len(raw_facs) <= 15 and len(merge_types) <= 15:
        plt.figure(figsize=(12, 8))
        
        # 计算每个组合的数据点数量
        count_pivot = data.pivot_table(values=metric, index='raw_fac', columns='merge_type', 
                                     aggfunc='count', fill_value=0)
        count_pivot = count_pivot.reindex(index=raw_facs, columns=merge_types, fill_value=0)
        
        ax = sns.heatmap(count_pivot, annot=True, fmt='d', cmap='Blues', 
                        cbar_kws={'label': 'Sample Count'})
        ax.set_title(f'Sample Count Distribution for {METRICS_DICT.get(metric, metric)}')
        ax.set_xlabel('Merge Type')
        ax.set_ylabel('Raw Factor')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'sample_count_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 保存分布统计信息
    dist_stats = data.groupby(['raw_fac', 'merge_type'])[metric].describe()
    dist_stats.to_csv(output_dir / f'distribution_stats_{metric}.csv')


def generate_summary_table(data, metric, percentile, output_dir):
    """生成汇总统计表"""
    summary_stats = []
    
    # 按raw_fac和merge_type分组计算统计量
    for (raw_fac, merge_type), group in data.groupby(['raw_fac', 'merge_type']):
        if not group[metric].empty and not group[metric].isna().all():
            stats = {
                'raw_fac': raw_fac,
                'merge_type': merge_type,
                f'{metric}_mean': group[metric].mean(),
                f'{metric}_median': group[metric].median(),
                f'{metric}_std': group[metric].std(),
                f'{metric}_percentile_{percentile}': np.percentile(group[metric].dropna(), percentile),
                f'{metric}_percentile_{100-percentile}': np.percentile(group[metric].dropna(), 100-percentile),
                f'{metric}_max': group[metric].max(),
                f'{metric}_min': group[metric].min(),
                'count': len(group),
                'valid_count': group[metric].notna().sum()
            }
            summary_stats.append(stats)
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(output_dir / f'summary_statistics_{metric}.csv', index=False)
        
        # 按均值排序
        sorted_df = summary_df.sort_values(by=f'{metric}_mean', ascending=False)
        
        # 创建可视化摘要表格
        top_bottom = pd.concat([sorted_df.head(15), sorted_df.tail(15)])
        
        display_cols = ['raw_fac', 'merge_type', f'{metric}_mean', f'{metric}_median', 
                       f'{metric}_percentile_{percentile}', f'{metric}_max', f'{metric}_min', 'count']
        
        fig, ax = plt.subplots(figsize=(14, max(10, len(top_bottom) * 0.4)))
        ax.axis('tight')
        ax.axis('off')
        
        # 准备表格数据
        headers = [col.replace(f'{metric}_', '').replace('_', ' ').title() for col in display_cols]
        table_data = []
        
        for _, row in top_bottom.reset_index(drop=True).iterrows():
            formatted_row = []
            for col in display_cols:
                val = row[col]
                if col == 'count':
                    formatted_row.append(str(int(val)))
                elif isinstance(val, (int, float)) and not pd.isna(val):
                    formatted_row.append(f'{val:.4f}')
                else:
                    formatted_row.append(str(val))
            table_data.append(formatted_row)
        
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # 设置颜色
        n_top = min(15, len(sorted_df))
        for i in range(1, n_top + 1):
            for j in range(len(display_cols)):
                table[(i, j)].set_facecolor('#d8f3dc')  # 浅绿色
        
        for i in range(n_top + 1, len(top_bottom) + 1):
            for j in range(len(display_cols)):
                table[(i, j)].set_facecolor('#ffdde1')  # 浅红色
        
        plt.title(f'Top 15 and Bottom 15 Factor Combinations by {METRICS_DICT.get(metric, metric)}', 
                 pad=20, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'summary_top_bottom_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 按raw_fac汇总
        raw_fac_summary = data.groupby('raw_fac')[metric].agg(['mean', 'std', 'median', 'min', 'max', 'count']).reset_index()
        raw_fac_summary.to_csv(output_dir / f'summary_by_raw_fac_{metric}.csv', index=False)
        
        # 按merge_type汇总
        merge_type_summary = data.groupby('merge_type')[metric].agg(['mean', 'std', 'median', 'min', 'max', 'count']).reset_index()
        merge_type_summary.to_csv(output_dir / f'summary_by_merge_type_{metric}.csv', index=False)


def generate_comprehensive_comparison(data, metrics, output_dir):
    """生成综合比较分析"""
    print("  📋 生成综合比较分析...")
    
    # 1. 指标间相关性分析
    if len(metrics) > 1:
        correlation_data = data[metrics].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0,
                   square=True, cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Matrix between Metrics')
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        correlation_data.to_csv(output_dir / 'metrics_correlation.csv')
    
    # 2. 综合排名分析
    ranking_data = []
    
    for metric in metrics:
        if metric in data.columns:
            # 计算每个组合的均值
            combo_means = data.groupby(['raw_fac', 'merge_type'])[metric].mean().reset_index()
            
            # 计算排名
            combo_means[f'{metric}_rank'] = combo_means[metric].rank(ascending=False, method='min')
            combo_means[f'{metric}_percentile_rank'] = combo_means[metric].rank(pct=True, ascending=False)
            
            ranking_data.append(combo_means[['raw_fac', 'merge_type', metric, f'{metric}_rank', f'{metric}_percentile_rank']])
    
    if ranking_data:
        # 合并所有排名数据
        combined_ranking = ranking_data[0]
        for df in ranking_data[1:]:
            combined_ranking = combined_ranking.merge(df, on=['raw_fac', 'merge_type'], how='outer')
        
        # 计算综合排名（如果有多个指标）
        if len(metrics) > 1:
            rank_cols = [f'{metric}_percentile_rank' for metric in metrics if f'{metric}_percentile_rank' in combined_ranking.columns]
            combined_ranking['combined_rank'] = combined_ranking[rank_cols].mean(axis=1)
            combined_ranking = combined_ranking.sort_values('combined_rank', ascending=False)
        
        combined_ranking.to_csv(output_dir / 'comprehensive_ranking.csv', index=False)
        
        # 可视化综合排名
        if len(combined_ranking) <= 20:
            plt.figure(figsize=(12, 8))
            combined_ranking['combo'] = combined_ranking['raw_fac'] + ' | ' + combined_ranking['merge_type']
            
            if len(metrics) > 1 and 'combined_rank' in combined_ranking.columns:
                ax = sns.barplot(x='combo', y='combined_rank', data=combined_ranking.head(15),
                               palette='viridis')
                ax.set_title('Top 15 Factor Combinations - Combined Ranking')
                ax.set_ylabel('Combined Percentile Rank')
            else:
                first_metric = metrics[0]
                ax = sns.barplot(x='combo', y=f'{first_metric}_percentile_rank', data=combined_ranking.head(15),
                               palette='viridis')
                ax.set_title(f'Top 15 Factor Combinations - {METRICS_DICT.get(first_metric, first_metric)} Ranking')
                ax.set_ylabel('Percentile Rank')
            
            ax.set_xlabel('Raw Factor | Merge Type')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(output_dir / 'comprehensive_ranking_chart.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 3. 保存因子组成部分的基本统计
    factor_info = {
        'total_records': len(data),
        'unique_raw_factors': data['raw_fac'].nunique(),
        'unique_merge_types': data['merge_type'].nunique(),
        'total_combinations': data.groupby(['raw_fac', 'merge_type']).ngroups,
        'raw_factors_list': sorted(data['raw_fac'].unique().tolist()),
        'merge_types_list': sorted(data['merge_type'].unique().tolist())
    }
    
    # 保存为文本文件
    with open(output_dir / 'factor_analysis_info.txt', 'w', encoding='utf-8') as f:
        f.write("🔍 因子分析基本信息\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"📊 总记录数: {factor_info['total_records']}\n")
        f.write(f"🧩 唯一原始因子数: {factor_info['unique_raw_factors']}\n")
        f.write(f"🔗 唯一合并类型数: {factor_info['unique_merge_types']}\n")
        f.write(f"🎯 总组合数: {factor_info['total_combinations']}\n\n")
        
        f.write("📋 原始因子列表:\n")
        for i, fac in enumerate(factor_info['raw_factors_list'], 1):
            f.write(f"  {i:2d}. {fac}\n")
        
        f.write(f"\n🔗 合并类型列表:\n")
        for i, merge_type in enumerate(factor_info['merge_types_list'], 1):
            f.write(f"  {i:2d}. {merge_type}\n")


# 主执行部分
if __name__ == "__main__":
    # 设置分析参数
    eval_name = 'batch_18_v1_batch_test_v2_icim'
    period = '160101_250101'
    
    # 运行分析
    run_factor_analysis_main(
        eval_name=eval_name,
        period=period,
        metrics=['net_sharpe_ratio', 'hsr'],
        percentiles=[90, 75, 50]
    )