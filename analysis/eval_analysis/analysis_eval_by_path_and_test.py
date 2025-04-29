# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 13:27:45 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
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
eval_name = 'basis_pct_250416_org_batch_250419_batch_test_v1'
period = '160101_250101'


# %%
path_config = load_path_config(project_dir)
result_dir = Path(path_config['result'])
eval_dir = result_dir / 'factor_evaluation'
analysis_dir = result_dir / 'analysis/eval_analysis'
summary_dir = analysis_dir / f'{eval_name}/factor_eval_{period}'
 

# %%
path = eval_dir / eval_name / f'factor_eval_{period}.csv'
eval_res = pd.read_csv(path)
eval_res['final_path'] = eval_res['process_name'].apply(lambda x: x.split('/')[1])
eval_res['org_fac'] = eval_res['factor'].apply(lambda x: x.split('-', 1)[0])
eval_res['trans_fac'] = eval_res['factor'].apply(lambda x: x.split('-', 1)[1])


# %%
# 设置全局变量
METRICS_DICT = {
    'net_sharpe_ratio': 'Net Sharpe Ratio',
    'net_calmar_ratio': 'Net Calmar Ratio',
    'net_sortino_ratio': 'Net Sortino Ratio',
    'net_return_annualized': 'Net Return (Annualized)',
    'net_max_dd': 'Net Maximum Drawdown',
    'net_sharpe_ratio_long_only': 'Net Sharpe Ratio (Long Only)',
    'net_sharpe_ratio_short_only': 'Net Sharpe Ratio (Short Only)'
}

COLOR_SCHEME = {
    'heat_main': 'RdYlBu_r',
    'heat_secondary': 'viridis',
    'box': 'Set3',
    'line': 'tab10'
}

def setup_output_directory(summary_dir, org_fac):
    """设置输出目录"""
    base_dir = Path(summary_dir)
    base_dir.mkdir(exist_ok=True, parents=True)
    
    org_fac_dir = base_dir / org_fac
    org_fac_dir.mkdir(exist_ok=True)
    
    return org_fac_dir

def run_analysis(df, summary_dir, metrics=None, percentiles=None):
    """运行完整的分析流程"""
    if metrics is None:
        metrics = ['net_sharpe_ratio']
    
    if percentiles is None:
        percentiles = [90]
    
    # 按org_fac分组
    org_facs = df['org_fac'].unique()
    
    for org_fac in org_facs:
        print(f"分析 org_fac: {org_fac}")
        
        # 筛选当前org_fac的数据
        org_fac_data = df[df['org_fac'] == org_fac].copy()
        
        # 创建输出目录
        output_dir = setup_output_directory(summary_dir, org_fac)
        
        # 为每个指标生成分析
        for metric in metrics:
            for percentile in percentiles:
                analyze_single_metric(org_fac_data, metric, percentile, output_dir)
        
        # 比较策略（只需运行一次）
        compare_strategies(org_fac_data, output_dir)

def analyze_single_metric(data, metric, percentile, output_dir):
    """分析单个指标"""
    print(f"  分析指标: {metric}, 分位数: {percentile}")
    
    # 提取唯一路径和测试名称
    final_paths = sorted(data['final_path'].unique())
    test_names = sorted(data['test_name'].unique())
    
    # 1. 生成热力图
    generate_heatmaps(data, metric, percentile, final_paths, test_names, output_dir)
    
    # 2. 生成分布箱线图
    generate_distribution_plots(data, metric, final_paths, test_names, output_dir)
    
    # 3. 生成汇总统计表
    generate_summary_table(data, metric, percentile, output_dir)

def generate_heatmaps(data, metric, percentile, final_paths, test_names, output_dir):
    """生成热力图：均值、分位数、最大值、最小值"""
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, figure=fig)
    
    # 计算统计量
    pivot_mean = data.pivot_table(values=metric, index='final_path', columns='test_name', aggfunc='mean')
    pivot_percentile_high = data.pivot_table(values=metric, index='final_path', columns='test_name', 
                                           aggfunc=lambda x: np.percentile(x, percentile))
    pivot_percentile_low = data.pivot_table(values=metric, index='final_path', columns='test_name', 
                                          aggfunc=lambda x: np.percentile(x, 100-percentile))
    pivot_max = data.pivot_table(values=metric, index='final_path', columns='test_name', aggfunc='max')
    pivot_min = data.pivot_table(values=metric, index='final_path', columns='test_name', aggfunc='min')
    
    # 确保一致的顺序
    pivot_mean = pivot_mean.reindex(index=final_paths, columns=test_names)
    pivot_percentile_high = pivot_percentile_high.reindex(index=final_paths, columns=test_names)
    pivot_percentile_low = pivot_percentile_low.reindex(index=final_paths, columns=test_names)
    pivot_max = pivot_max.reindex(index=final_paths, columns=test_names)
    pivot_min = pivot_min.reindex(index=final_paths, columns=test_names)
    
    # 创建热力图
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(pivot_mean, annot=True, fmt='.1f', cmap=COLOR_SCHEME['heat_main'], ax=ax1, cbar_kws={'label': 'Mean'})
    ax1.set_title(f'Mean {METRICS_DICT.get(metric, metric)}')
    ax1.set_xlabel('Test Name')
    ax1.set_ylabel('Final Path')
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(pivot_percentile_high, annot=True, fmt='.1f', cmap=COLOR_SCHEME['heat_main'], ax=ax2, cbar_kws={'label': f'{percentile}th Percentile'})
    ax2.set_title(f'{percentile}th Percentile {METRICS_DICT.get(metric, metric)}')
    ax2.set_xlabel('Test Name')
    ax2.set_ylabel('Final Path')
    
    ax3 = fig.add_subplot(gs[1, 0])
    sns.heatmap(pivot_max, annot=True, fmt='.1f', cmap=COLOR_SCHEME['heat_main'], ax=ax3, cbar_kws={'label': 'Maximum'})
    ax3.set_title(f'Maximum {METRICS_DICT.get(metric, metric)}')
    ax3.set_xlabel('Test Name')
    ax3.set_ylabel('Final Path')
    
    ax4 = fig.add_subplot(gs[1, 1])
    sns.heatmap(pivot_min, annot=True, fmt='.1f', cmap=COLOR_SCHEME['heat_main'], ax=ax4, cbar_kws={'label': 'Minimum'})
    ax4.set_title(f'Minimum {METRICS_DICT.get(metric, metric)}')
    ax4.set_xlabel('Test Name')
    ax4.set_ylabel('Final Path')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'heatmaps_{metric}_pct{percentile}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存数据到CSV
    pivot_mean.to_csv(output_dir / f'heatmap_mean_{metric}.csv')
    pivot_percentile_high.to_csv(output_dir / f'heatmap_percentile_{percentile}_{metric}.csv')
    pivot_percentile_low.to_csv(output_dir / f'heatmap_percentile_{100-percentile}_{metric}.csv')
    pivot_max.to_csv(output_dir / f'heatmap_max_{metric}.csv')
    pivot_min.to_csv(output_dir / f'heatmap_min_{metric}.csv')

def generate_distribution_plots(data, metric, final_paths, test_names, output_dir):
    """生成分布图，使用更紧凑的方式展示"""
    # 1. 每个final_path的所有test_name的分布
    plt.figure(figsize=(12, 8))
    
    # 按final_path分组绘制小提琴图
    ax = sns.violinplot(x='final_path', y=metric, data=data, inner='box', 
                       palette=sns.color_palette(COLOR_SCHEME['box'], len(final_paths)))
    
    ax.set_title(f'Distribution of {METRICS_DICT.get(metric, metric)} by Final Path')
    ax.set_xlabel('Final Path')
    ax.set_ylabel(METRICS_DICT.get(metric, metric))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'dist_by_final_path_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 每个test_name的所有final_path的分布
    plt.figure(figsize=(12, 8))
    
    ax = sns.violinplot(x='test_name', y=metric, data=data, inner='box', 
                      palette=sns.color_palette(COLOR_SCHEME['box'], len(test_names)))
    
    ax.set_title(f'Distribution of {METRICS_DICT.get(metric, metric)} by Test Name')
    ax.set_xlabel('Test Name')
    ax.set_ylabel(METRICS_DICT.get(metric, metric))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'dist_by_test_name_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 如果组合数量不太多，则画出每个组合的箱线图
    if len(final_paths) * len(test_names) <= 20:
        # 创建组合名称
        data['combo'] = data['final_path'] + ' | ' + data['test_name']
        
        plt.figure(figsize=(14, 8))
        
        ax = sns.boxplot(x='combo', y=metric, data=data, 
                       palette=sns.color_palette(COLOR_SCHEME['box'], len(data['combo'].unique())))
        
        ax.set_title(f'Distribution of {METRICS_DICT.get(metric, metric)} by Combination')
        ax.set_xlabel('Final Path | Test Name')
        ax.set_ylabel(METRICS_DICT.get(metric, metric))
        plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'dist_by_combo_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # 如果组合太多，就画出top5和bottom5的组合
        combo_stats = data.groupby(['final_path', 'test_name'])[metric].mean().reset_index()
        combo_stats['combo'] = combo_stats['final_path'] + ' | ' + combo_stats['test_name']
        
        # 获取top5和bottom5
        top5 = combo_stats.nlargest(5, metric)
        bottom5 = combo_stats.nsmallest(5, metric)
        extremes = pd.concat([top5, bottom5])
        
        # 筛选数据
        extreme_combos = list(extremes['combo'].unique())
        data['combo'] = data['final_path'] + ' | ' + data['test_name']
        filtered_data = data[data['combo'].isin(extreme_combos)]
        
        plt.figure(figsize=(14, 8))
        
        ax = sns.boxplot(x='combo', y=metric, data=filtered_data, 
                       palette=sns.color_palette(COLOR_SCHEME['box'], 10))
        
        ax.set_title(f'Distribution of {METRICS_DICT.get(metric, metric)} - Top 5 & Bottom 5 Combinations')
        ax.set_xlabel('Final Path | Test Name')
        ax.set_ylabel(METRICS_DICT.get(metric, metric))
        plt.xticks(rotation=90)
        
        # 添加标记表示top和bottom
        for i, combo in enumerate(ax.get_xticklabels()):
            combo_text = combo.get_text()
            if combo_text in top5['combo'].values:
                ax.get_xticklabels()[i].set_color('green')
            else:
                ax.get_xticklabels()[i].set_color('red')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'dist_extremes_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 保存分布统计信息到CSV
    dist_stats = data.groupby(['final_path', 'test_name'])[metric].describe()
    dist_stats.to_csv(output_dir / f'distribution_stats_{metric}.csv')

def compare_strategies(data, output_dir):
    """比较long-only, short-only和combined策略表现"""
    # 定义策略类型和对应的指标
    strategy_metrics = {
        'Combined': 'net_sharpe_ratio',
        'Long Only': 'net_sharpe_ratio_long_only',
        'Short Only': 'net_sharpe_ratio_short_only'
    }
    
    # 准备比较数据
    stats_data = []
    for strategy, metric in strategy_metrics.items():
        if metric in data.columns:
            # 计算每个final_path的平均值
            means_by_path = data.groupby(['final_path'])[metric].mean().reset_index()
            means_by_path['Strategy'] = strategy
            means_by_path['Metric'] = metric
            means_by_path = means_by_path.rename(columns={metric: 'Value'})
            stats_data.append(means_by_path)
            
            # 计算每个test_name的平均值
            means_by_test = data.groupby(['test_name'])[metric].mean().reset_index()
            means_by_test['Strategy'] = strategy
            means_by_test['Metric'] = metric
            means_by_test = means_by_test.rename(columns={metric: 'Value'})
            stats_data.append(means_by_test)
    
    # 合并所有数据
    if stats_data:
        all_stats = pd.concat(stats_data)
        
        # 1. 按final_path绘制策略比较
        plt.figure(figsize=(14, 8))
        
        # 按final_path筛选
        path_stats = all_stats[all_stats['final_path'].notna()]
        
        # 使用seaborn的barplot更清晰地展示分组数据
        ax = sns.barplot(x='final_path', y='Value', hue='Strategy', data=path_stats,
                       palette=sns.color_palette(COLOR_SCHEME['line'], 3))
        
        ax.set_title('Strategy Comparison by Final Path')
        ax.set_xlabel('Final Path')
        ax.set_ylabel('Mean Sharpe Ratio')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Strategy Type')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'strategy_comparison_by_path.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 按test_name绘制策略比较
        plt.figure(figsize=(14, 8))
        
        # 按test_name筛选
        test_stats = all_stats[all_stats['test_name'].notna()]
        
        ax = sns.barplot(x='test_name', y='Value', hue='Strategy', data=test_stats,
                       palette=sns.color_palette(COLOR_SCHEME['line'], 3))
        
        ax.set_title('Strategy Comparison by Test Name')
        ax.set_xlabel('Test Name')
        ax.set_ylabel('Mean Sharpe Ratio')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Strategy Type')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'strategy_comparison_by_test.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 策略之间的整体比较（不区分路径和测试）
        plt.figure(figsize=(10, 6))
        
        overall_stats = all_stats.groupby('Strategy')['Value'].mean().reset_index()
        overall_std = all_stats.groupby('Strategy')['Value'].std().reset_index()
        overall_stats['Std'] = overall_std['Value']
        
        ax = sns.barplot(x='Strategy', y='Value', data=overall_stats, 
                       palette=sns.color_palette(COLOR_SCHEME['line'], 3))
        
        # 添加误差线
        for i, row in overall_stats.iterrows():
            ax.errorbar(i, row['Value'], yerr=row['Std'], color='black', capsize=5)
        
        ax.set_title('Overall Strategy Comparison')
        ax.set_xlabel('Strategy Type')
        ax.set_ylabel('Mean Sharpe Ratio')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'strategy_comparison_overall.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 策略表现的分布图比较
        plt.figure(figsize=(12, 8))
        
        long_data = data['net_sharpe_ratio_long_only'].dropna()
        short_data = data['net_sharpe_ratio_short_only'].dropna()
        combined_data = data['net_sharpe_ratio'].dropna()
        
        all_values = pd.DataFrame({
            'Value': pd.concat([long_data, short_data, combined_data]),
            'Strategy': ['Long Only'] * len(long_data) + ['Short Only'] * len(short_data) + ['Combined'] * len(combined_data)
        })
        
        ax = sns.violinplot(x='Strategy', y='Value', data=all_values, 
                         palette=sns.color_palette(COLOR_SCHEME['line'], 3), inner='box')
        
        ax.set_title('Strategy Performance Distribution')
        ax.set_xlabel('Strategy Type')
        ax.set_ylabel('Sharpe Ratio')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'strategy_comparison_violin.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存比较数据
        all_stats.to_csv(output_dir / 'strategy_comparison_stats.csv', index=False)
        overall_stats.to_csv(output_dir / 'strategy_comparison_overall.csv', index=False)

def generate_summary_table(data, metric, percentile, output_dir):
    """生成汇总统计表"""
    summary_stats = []
    
    # 按final_path和test_name分组计算统计量
    for (final_path, test_name), group in data.groupby(['final_path', 'test_name']):
        if not group[metric].empty:
            stats = {
                'final_path': final_path,
                'test_name': test_name,
                f'{metric}_mean': group[metric].mean(),
                f'{metric}_median': group[metric].median(),
                f'{metric}_std': group[metric].std(),
                f'{metric}_percentile_{percentile}': np.percentile(group[metric], percentile),
                f'{metric}_percentile_{100-percentile}': np.percentile(group[metric], 100-percentile),
                f'{metric}_max': group[metric].max(),
                f'{metric}_min': group[metric].min(),
                'count': len(group)
            }
            
            # 添加策略比较指标（如果可用）
            if 'net_sharpe_ratio_long_only' in group.columns:
                stats['sharpe_long_only_mean'] = group['net_sharpe_ratio_long_only'].mean()
            if 'net_sharpe_ratio_short_only' in group.columns:
                stats['sharpe_short_only_mean'] = group['net_sharpe_ratio_short_only'].mean()
                
            summary_stats.append(stats)
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(output_dir / f'summary_statistics_{metric}.csv', index=False)
        
        # 按均值排序
        sorted_df = summary_df.sort_values(by=f'{metric}_mean', ascending=False)
        
        # 取top 10和bottom 10
        top_bottom = pd.concat([sorted_df.head(10), sorted_df.tail(10)])
        
        # 创建一个视觉摘要表格（仅显示主要列）
        display_cols = ['final_path', 'test_name', f'{metric}_mean', f'{metric}_median', 
                         f'{metric}_percentile_{percentile}', f'{metric}_max', f'{metric}_min']
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(top_bottom) * 0.4)))
        ax.axis('tight')
        ax.axis('off')
        
        # 将数据框转换为表格数据
        table_data = []
        headers = [col.replace(f'{metric}_', '') for col in display_cols]
        
        for _, row in top_bottom.reset_index(drop=True).iterrows():
            row_data = [row[col] for col in display_cols]
            formatted_row = []
            for v in row_data:
                if isinstance(v, float):
                    formatted_row.append(f'{v:.3f}')
                else:
                    formatted_row.append(str(v))
            table_data.append(formatted_row)
        
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # 为top 10和bottom 10设置不同的颜色
        for i in range(1, 11):
            for j in range(len(display_cols)):
                cell = table[(i, j)]
                cell.set_facecolor('#d8f3dc')  # 浅绿色
        
        for i in range(11, len(top_bottom) + 1):
            for j in range(len(display_cols)):
                cell = table[(i, j)]
                cell.set_facecolor('#ffdde1')  # 浅红色
        
        plt.title(f'Top 10 and Bottom 10 Combinations by {METRICS_DICT.get(metric, metric)}', pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / f'summary_top_bottom_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 汇总统计表 - 按final_path
        path_summary = data.groupby('final_path')[metric].agg(['mean', 'std', 'median', 'min', 'max']).reset_index()
        path_summary.to_csv(output_dir / f'summary_by_path_{metric}.csv', index=False)
        
        # 汇总统计表 - 按test_name
        test_summary = data.groupby('test_name')[metric].agg(['mean', 'std', 'median', 'min', 'max']).reset_index()
        test_summary.to_csv(output_dir / f'summary_by_test_{metric}.csv', index=False)

# 示例用法:
run_analysis(
    df=eval_res,
    summary_dir=summary_dir,
    metrics=['net_sharpe_ratio', 'net_calmar_ratio', 'net_sharpe_ratio_long_only'],
    percentiles=[90, 75]
)