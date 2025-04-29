# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 12:31:21 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
from pathlib import Path
import pandas as pd
from functools import partial
import numpy as np


from utils.datautils import align_and_sort_columns
from utils.market import index_to_futures
from utils.timeutils import parse_time_string
from utils.trade_rules import *
from data_processing.ts_trans import *


# %%
# factor_name = 'LargeOrderAmountByValue_p1.0_v40000-avg_imb01_dp2-org'
# direction = 1

factor_name = 'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb04_dp2-org'
direction = -1

factor_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\sample_data\filters\1_2_org')


vol_threshold = 0.012
slope_threshold = 2e-5
pen = 30000


price_name = 't1min_fq1min_dl1min'

scale_method = 'minmax_scale'
scale_window = '240d'
scale_quantile = 0.02
sp = '1min'
pp_by_sp = 1

trade_rule_name = 'trade_rule_by_trigger_v0'
trade_rule_param = {
    'openthres': 0.8,
    'closethres': 0,
    }

fee = 0.00024


# %%
label_name = f'rv{vol_threshold}_slp{slope_threshold}_pen{pen}'
label_dir = Path('/mnt/data1/labels')


# %%
labels_df = pd.read_parquet(label_dir / f'{label_name}.parquet')


# %%
fut_dir = Path('/mnt/data1/future_twap')
analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\label_return')
summary_dir = analysis_dir / factor_name
summary_dir.mkdir(parents=True, exist_ok=True)


# %%
factor_data = pd.read_parquet(factor_dir / f'{factor_name}.parquet')
twap_price = pd.read_parquet(fut_dir / f'{price_name}.parquet')


# %%
rtn_1p = twap_price.pct_change(pp_by_sp, fill_method=None).shift(-pp_by_sp) / pp_by_sp
rtn_1p = rtn_1p.replace([np.inf, -np.inf], 0)


# %% align
factor_data = factor_data.rename(columns=index_to_futures)[['IC', 'IF', 'IM']]
factor_data, rtn_1p, twap_price = align_and_sort_columns([factor_data, rtn_1p, twap_price])

twap_price = twap_price.loc[factor_data.index.min():factor_data.index.max()] # 按factor头尾截取
rtn_1p = rtn_1p.loc[factor_data.index.min():factor_data.index.max()] # 按factor头尾截取
factor_data = factor_data.reindex(rtn_1p.index) # 按twap reindex，确保等长


# %% scale
scale_func = globals()[scale_method]
scale_step = int(parse_time_string(scale_window) / parse_time_string(sp))
# factor_scaled = ts_quantile_scale(factor, window=scale_step, quantile=scale_quantile)
if scale_method in ['minmax_scale', 'minmax_scale_separate']:
    factor_scaled = scale_func(factor_data, window=scale_step, quantile=scale_quantile)
# elif scale_method in ['minmax_scale_adj_by_his_rtn', 'zscore_adj_by_his_rtn_and_minmax']:
#     factor_scaled = scale_func(factor, rtn_1p, window=scale_step, rtn_window=pp_by_sp, quantile=scale_quantile)
# elif scale_method in ['rolling_percentile']:
#     factor_scaled = scale_func(factor, window=scale_step)
# elif scale_method in ['percentile_adj_by_his_rtn']:
#     factor_scaled = scale_func(factor, rtn_1p, window=scale_step, rtn_window=pp_by_sp)

factor_scaled_direction = (factor_scaled - 0.5) * 2 * direction


# %% to pos
trade_rule_func = partial(globals()[trade_rule_name], **trade_rule_param)
actual_pos = factor_scaled_direction.apply(lambda col: trade_rule_func(col.values), axis=0)


# %% test
gp = (factor_scaled_direction * rtn_1p)
hsr = ((factor_scaled_direction - factor_scaled_direction.shift(pp_by_sp)) / 2).abs().replace(
    [np.inf, -np.inf, np.nan], np.nan)


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

# 忽略 pandas 的警告
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# %% 辅助函数：计算连续仓位的收益
def calculate_continuous_position_returns(positions, returns, labels):
    """
    计算连续相同仓位的聚合收益
    
    Parameters:
    ----------
    positions : pd.Series
        仓位数据，值为1(做多),-1(做空),0(不持仓)
    returns : pd.Series
        对应的分钟收益率
    labels : pd.Series
        对应的标签
        
    Returns:
    -------
    DataFrame包含连续仓位的聚合收益和对应的标签
    """
    # 调试信息
    print(f"Calculating continuous position returns")
    print(f"Positions shape: {positions.shape}")
    print(f"Returns shape: {returns.shape}")
    print(f"Labels shape: {labels.shape}")
    # 初始化结果
    continuous_returns = []
    
    # 确保输入数据没有NaN
    positions = positions.fillna(0)
    returns = returns.fillna(0)
    
    # 检测仓位变化点
    position_changes = positions.ne(positions.shift()).fillna(True)
    
    # 调试输出
    print(f"Number of position changes: {position_changes.sum()}")
    
    # 用于存储当前连续仓位段的开始索引
    start_idx = None
    current_pos = 0
    
    for idx in positions.index:
        try:
            if position_changes.loc[idx]:
                # 如果有一段连续仓位结束，计算其总收益
                if start_idx is not None and current_pos != 0:
                    # 获取这段连续仓位的索引
                    segment_indices = positions.loc[start_idx:idx].index[:-1]
                    if len(segment_indices) > 0:
                        try:
                                                    # 计算总收益
                            total_return = returns.loc[segment_indices].sum()
                            
                            # 为了处理没有标签的情况，默认使用"Unknown"
                            main_label = "Unknown"
                            
                            # 尝试获取主要标签（众数）
                            try:
                                # 先确保我们可以获取标签数据
                                valid_indices = [idx for idx in segment_indices if idx in labels.index]
                                if valid_indices:
                                    segment_labels = labels.loc[valid_indices].dropna()
                                    if len(segment_labels) > 0:
                                        mode_result = segment_labels.mode()
                                        if len(mode_result) > 0:
                                            main_label = mode_result[0]
                            except Exception as e:
                                print(f"Error determining label mode: {e}")
                                
                            # 添加到结果
                            continuous_returns.append({
                                'Position': 'Positive' if current_pos > 0 else 'Negative',
                                'Return': total_return,
                                'Label': main_label,
                                'IsProfit': total_return > 0,
                                'Duration': len(segment_indices)
                            })

                        except Exception as e:
                            print(f"Error calculating return for segment: {e}")
                
                # 更新新段的开始索引和当前仓位
                start_idx = idx
                current_pos = positions.loc[idx]
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
    
    # 处理最后一段连续仓位
    try:
        if start_idx is not None and current_pos != 0:
            segment_indices = positions.loc[start_idx:].index
            if len(segment_indices) > 0:
                try:
                    total_return = returns.loc[segment_indices].sum()
                    
                    # 为了处理没有标签的情况，默认使用"Unknown"
                    main_label = "Unknown"
                    
                    # 尝试获取主要标签
                    try:
                        valid_indices = [idx for idx in segment_indices if idx in labels.index]
                        if valid_indices:
                            segment_labels = labels.loc[valid_indices].dropna()
                            if len(segment_labels) > 0:
                                mode_result = segment_labels.mode()
                                if len(mode_result) > 0:
                                    main_label = mode_result[0]
                    except Exception as e:
                        print(f"Error determining label mode for last segment: {e}")
                        
                    # 添加到结果
                    continuous_returns.append({
                        'Position': 'Positive' if current_pos > 0 else 'Negative',
                        'Return': total_return,
                        'Label': main_label,
                        'IsProfit': total_return > 0,
                        'Duration': len(segment_indices)
                    })

                except Exception as e:
                    print(f"Error calculating return for last segment: {e}")
    except Exception as e:
        print(f"Error processing last segment: {e}")
    
    return pd.DataFrame(continuous_returns)

# %% 对齐数据索引
common_index = labels_df.index.intersection(actual_pos.index)
aligned_labels = labels_df.loc[common_index]
aligned_pos = actual_pos.loc[common_index]
aligned_net_returns = pd.DataFrame()

# 为每个期货品种计算净收益
for fut in aligned_pos.columns:
    if fut in gp.columns:
        gp_fut = gp[fut].loc[common_index].fillna(0)
        hsr_fut = hsr[fut].loc[common_index].fillna(0)
        net_fut = gp_fut - hsr_fut * fee
        aligned_net_returns[fut] = net_fut

# %% 计算连续仓位收益
continuous_results = {}

for fut in aligned_pos.columns:
    print(f"\nProcessing {fut}...")
    if fut not in aligned_labels.columns or fut not in aligned_net_returns.columns:
        print(f"Skipping {fut} - missing labels or returns")
        continue
    
    # 检查数据是否对齐
    print(f"Position range: {aligned_pos[fut].index.min()} to {aligned_pos[fut].index.max()}")
    print(f"Returns range: {aligned_net_returns[fut].index.min()} to {aligned_net_returns[fut].index.max()}")
    print(f"Labels range: {aligned_labels[fut].index.min()} to {aligned_labels[fut].index.max()}")
    
    # 计算连续仓位收益
    continuous_data = calculate_continuous_position_returns(
        aligned_pos[fut], 
        aligned_net_returns[fut], 
        aligned_labels[fut]
    )
    
    if continuous_data.empty:
        print(f"No continuous positions found for {fut}")
        continue
    
    print(f"Found {len(continuous_data)} continuous position segments for {fut}")
    
    # 统计结果
    all_labels = sorted([label for label in continuous_data['Label'].unique() if label != "Unknown"])
    print(f"Valid labels found: {all_labels}")
    
    # 创建结果DataFrame
    summary = []
    
    for label in all_labels:
        # 筛选特定标签的数据
        label_data = continuous_data[continuous_data['Label'] == label]
        
        # 正仓位数据
        pos_data = label_data[label_data['Position'] == 'Positive']
        pos_profit = pos_data[pos_data['IsProfit']]['Return'].sum()
        pos_loss = pos_data[~pos_data['IsProfit']]['Return'].sum()
        pos_count = len(pos_data)
        pos_minutes = pos_data['Duration'].sum()
        
        # 负仓位数据
        neg_data = label_data[label_data['Position'] == 'Negative']
        neg_profit = neg_data[neg_data['IsProfit']]['Return'].sum()
        neg_loss = neg_data[~neg_data['IsProfit']]['Return'].sum()
        neg_count = len(neg_data)
        neg_minutes = neg_data['Duration'].sum()
        
        # 添加到结果中
        summary.append({
            'Label': label,
            'Position': 'Positive',
            'ProfitReturn': pos_profit,
            'LossReturn': pos_loss,
            'TotalReturn': pos_profit + pos_loss,
            'TradeCount': pos_count,
            'MinuteCount': pos_minutes
        })
        
        summary.append({
            'Label': label,
            'Position': 'Negative',
            'ProfitReturn': neg_profit,
            'LossReturn': neg_loss,
            'TotalReturn': neg_profit + neg_loss,
            'TradeCount': neg_count,
            'MinuteCount': neg_minutes
        })
    
    continuous_results[fut] = pd.DataFrame(summary)
    
    # 计算百分比
    total_minutes = (aligned_pos[fut] != 0).sum()
    continuous_results[fut]['Percentage'] = continuous_results[fut]['MinuteCount'] / total_minutes * 100 if total_minutes > 0 else 0

# %% 创建图表
for fut, df in continuous_results.items():
    # 使用matplotlib创建堆叠柱状图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 处理数据 - 过滤掉Unknown标签
    df_filtered = df[df['Label'] != "Unknown"]
    if df_filtered.empty:
        print(f"No valid labels for {fut} after filtering out 'Unknown'")
        continue
        
    labels = sorted(df_filtered['Label'].unique())
    if not labels:
        print(f"No valid labels for {fut}")
        continue
        
    x = np.arange(len(labels))
    width = 0.35
    
    # 分别获取正仓位和负仓位的数据
    pos_profit = []
    pos_loss = []
    neg_profit = []
    neg_loss = []
    pos_percentage = []
    neg_percentage = []
    pos_trades = []
    neg_trades = []
    
    for label in labels:
        # 正仓位
        pos_row = df_filtered[(df_filtered['Label'] == label) & (df_filtered['Position'] == 'Positive')].iloc[0] if not df_filtered[(df_filtered['Label'] == label) & (df_filtered['Position'] == 'Positive')].empty else None
        if pos_row is not None:
            pos_profit.append(pos_row['ProfitReturn'])
            pos_loss.append(pos_row['LossReturn'])  # 已经是负值
            pos_percentage.append(pos_row['Percentage'])
            pos_trades.append(pos_row['TradeCount'])
        else:
            pos_profit.append(0)
            pos_loss.append(0)
            pos_percentage.append(0)
            pos_trades.append(0)
            
        # 负仓位
        neg_row = df_filtered[(df_filtered['Label'] == label) & (df_filtered['Position'] == 'Negative')].iloc[0] if not df_filtered[(df_filtered['Label'] == label) & (df_filtered['Position'] == 'Negative')].empty else None
        if neg_row is not None:
            neg_profit.append(neg_row['ProfitReturn'])
            neg_loss.append(neg_row['LossReturn'])  # 已经是负值
            neg_percentage.append(neg_row['Percentage'])
            neg_trades.append(neg_row['TradeCount'])
        else:
            neg_profit.append(0)
            neg_loss.append(0)
            neg_percentage.append(0)
            neg_trades.append(0)
    
    # 确保有数据可视化
    if len(x) == 0:
        print(f"No data to visualize for {fut}")
        continue
    
    # 绘制正仓位的盈利和亏损
    if len(x) > 0:
        pos_profit_bars = ax.bar(x - width/2, pos_profit, width, color='firebrick', alpha=0.7, label='Positive Profit')
        pos_loss_bars = ax.bar(x - width/2, pos_loss, width, color='firebrick', alpha=0.4, bottom=[0] * len(labels), label='Positive Loss')
        
        # 绘制负仓位的盈利和亏损
        neg_profit_bars = ax.bar(x + width/2, neg_profit, width, color='forestgreen', alpha=0.7, label='Negative Profit')
        neg_loss_bars = ax.bar(x + width/2, neg_loss, width, color='forestgreen', alpha=0.4, bottom=[0] * len(labels), label='Negative Loss')
    
    # 添加标题和标签
    plt.title(f'Continuous Position Returns by Label for {fut}\nFactor: {factor_name}\nLabel: {label_name}', 
              fontsize=16, 
              fontweight='bold',
              pad=20)
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Net Return', fontsize=14)
    plt.xticks(x, labels, rotation=0)
    plt.legend()
    
    # 添加百分比标签和交易次数
    for i, label in enumerate(labels):
        # 获取交易次数
        pos_row = df[(df['Label'] == label) & (df['Position'] == 'Positive')].iloc[0] if not df[(df['Label'] == label) & (df['Position'] == 'Positive')].empty else None
        neg_row = df[(df['Label'] == label) & (df['Position'] == 'Negative')].iloc[0] if not df[(df['Label'] == label) & (df['Position'] == 'Negative')].empty else None
        
        pos_trades = pos_row['TradeCount'] if pos_row is not None else 0
        neg_trades = neg_row['TradeCount'] if neg_row is not None else 0
        
        # 正仓位标签
        if pos_profit[i] > 0:
            height = pos_profit[i] / 2
            ax.text(i - width/2, height, f'{pos_profit[i]:.1f}%\n({pos_trades})', 
                    ha='center', va='center', color='white', fontweight='bold')
        
        if pos_loss[i] < 0:
            height = pos_loss[i] / 2
            ax.text(i - width/2, height, f'{pos_loss[i]:.1f}%\n({pos_trades})', 
                    ha='center', va='center', color='white', fontweight='bold')
        
        # 负仓位标签
        if neg_profit[i] > 0:
            height = neg_profit[i] / 2
            ax.text(i + width/2, height, f'{neg_profit[i]:.1f}%\n({neg_trades})', 
                    ha='center', va='center', color='white', fontweight='bold')
        
        if neg_loss[i] < 0:
            height = neg_loss[i] / 2
            ax.text(i + width/2, height, f'{neg_loss[i]:.1f}%\n({neg_trades})', 
                    ha='center', va='center', color='white', fontweight='bold')
            
            
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(summary_dir / f'{fut}_continuous_returns_stacked.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()

# %% 创建汇总分析表格并保存
for fut, df in continuous_results.items():
    # 过滤掉Unknown标签
    df_filtered = df[df['Label'] != "Unknown"]
    if df_filtered.empty:
        print(f"No valid labels for {fut} summary table")
        continue
        
    # 保存结果为CSV
    summary_table = df_filtered.pivot_table(
        index='Label',
        columns='Position',
        values=['TotalReturn', 'ProfitReturn', 'LossReturn', 'TradeCount', 'MinuteCount', 'Percentage'],
        aggfunc='sum'
    )
    
    # 保存为CSV
    summary_table.to_csv(summary_dir / f'{fut}_continuous_returns_summary.csv')
    
    # 打印结果
    print(f"\n{fut} Continuous Position Summary:")
    print(summary_table)

# %% 创建所有期货品种的汇总图表
if len(continuous_results) > 0:
    # 过滤有效的结果（有非Unknown标签的结果）
    valid_results = {}
    for fut, df in continuous_results.items():
        df_filtered = df[df['Label'] != "Unknown"]
        if not df_filtered.empty and len(df_filtered['Label'].unique()) > 0:
            valid_results[fut] = df_filtered
    
    if not valid_results:
        print("No valid results with non-Unknown labels found for any futures")
    else:
        fig, axes = plt.subplots(1, len(valid_results), figsize=(18, 8), sharey=True)
        
        # 确保axes是数组，即使只有一个结果
        if len(valid_results) == 1:
            axes = [axes]
        
        # 为每个期货品种创建子图
        for i, (fut, df) in enumerate(valid_results.items()):
            ax = axes[i]
            
            # 处理数据
            labels = sorted(df['Label'].unique())
            x = np.arange(len(labels))
            width = 0.35
        
        # 分别获取正仓位和负仓位的数据
        pos_profit = []
        pos_loss = []
        neg_profit = []
        neg_loss = []
        pos_trades = []
        neg_trades = []
        
        for label in labels:
            # 正仓位
            pos_row = df[(df['Label'] == label) & (df['Position'] == 'Positive')].iloc[0] if not df[(df['Label'] == label) & (df['Position'] == 'Positive')].empty else None
            if pos_row is not None:
                pos_trades.append(pos_row['TradeCount'])
            else:
                pos_trades.append(0)
                
            # 负仓位
            neg_row = df[(df['Label'] == label) & (df['Position'] == 'Negative')].iloc[0] if not df[(df['Label'] == label) & (df['Position'] == 'Negative')].empty else None
            if neg_row is not None:
                neg_trades.append(neg_row['TradeCount'])
            else:
                neg_trades.append(0)
        
        # 绘制正仓位的盈利和亏损
        pos_profit_bars = ax.bar(x - width/2, pos_profit, width, color='firebrick', alpha=0.7, label='Positive Profit')
        pos_loss_bars = ax.bar(x - width/2, pos_loss, width, color='firebrick', alpha=0.4, bottom=[0] * len(labels), label='Positive Loss')
        
        # 绘制负仓位的盈利和亏损
        neg_profit_bars = ax.bar(x + width/2, neg_profit, width, color='forestgreen', alpha=0.7, label='Negative Profit')
        neg_loss_bars = ax.bar(x + width/2, neg_loss, width, color='forestgreen', alpha=0.4, bottom=[0] * len(labels), label='Negative Loss')
        
        # 设置子图标题和标签
        ax.set_title(f'{fut}', fontsize=14)
        ax.set_xlabel('Label', fontsize=12)
        if i == 0:
            ax.set_ylabel('Net Return', fontsize=12)
        
        # 设置x轴标签
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        
        # 添加交易次数标签
        font_size = 8  # 较小的字体以适应并排图
        for j, label in enumerate(labels):
            # 正仓位标签
            if pos_profit[j] > 0:
                height = pos_profit[j] / 2
                ax.text(j - width/2, height, f'n={pos_trades[j]}', 
                        ha='center', va='center', color='white', fontsize=font_size, fontweight='bold')
            
            if pos_loss[j] < 0:
                height = pos_loss[j] / 2
                ax.text(j - width/2, height, f'n={pos_trades[j]}', 
                        ha='center', va='center', color='white', fontsize=font_size, fontweight='bold')
            
            # 负仓位标签
            if neg_profit[j] > 0:
                height = neg_profit[j] / 2
                ax.text(j + width/2, height, f'n={neg_trades[j]}', 
                        ha='center', va='center', color='white', fontsize=font_size, fontweight='bold')
            
            if neg_loss[j] < 0:
                height = neg_loss[j] / 2
                ax.text(j + width/2, height, f'n={neg_trades[j]}', 
                        ha='center', va='center', color='white', fontsize=font_size, fontweight='bold')
        
        # 添加网格线
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 设置总标题
    fig.suptitle(f'Continuous Position Returns by Label for All Futures\nFactor: {factor_name}\nLabel: {label_name}', 
                 fontsize=16, 
                 fontweight='bold',
                 y=1.05)
    
    # 为第一个子图添加图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    # 保存图表
    plt.savefig(summary_dir / 'all_futures_continuous_returns_stacked.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()