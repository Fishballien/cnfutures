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
# pos_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\portfolio\mulfac_pos')

# pos_name = 'pos_tradeAddLob_250307'


# %%
pos_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model\avg_agg_250218_3_fix_tfe_by_trade_net_v4\test\trade_ver3_futtwap_sp1min_s240d_icim_v6\data')

pos_name = 'pos_predict_avg_agg_250218_3_fix_tfe_by_trade_net_v4'



# %%
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

direction = 1


# %%
label_name = f'rv{vol_threshold}_slp{slope_threshold}_pen{pen}'
label_dir = Path('/mnt/data1/labels')


# %%
labels_df = pd.read_parquet(label_dir / f'{label_name}.parquet')


# %%
fut_dir = Path('/mnt/data1/future_twap')
analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\label_return')
summary_dir = analysis_dir / pos_name
summary_dir.mkdir(parents=True, exist_ok=True)


# %%
pos_data = pd.read_parquet(pos_dir / f'{pos_name}.parquet')
twap_price = pd.read_parquet(fut_dir / f'{price_name}.parquet')


# %%
rtn_1p = twap_price.pct_change(pp_by_sp, fill_method=None).shift(-pp_by_sp) / pp_by_sp
rtn_1p = rtn_1p.replace([np.inf, -np.inf], 0)


# %% align
pos_data = pos_data.rename(columns=index_to_futures)[['IC', 'IM']]
pos_data, rtn_1p, twap_price = align_and_sort_columns([pos_data, rtn_1p, twap_price])

twap_price = twap_price.loc[pos_data.index.min():pos_data.index.max()] # 按factor头尾截取
rtn_1p = rtn_1p.loc[pos_data.index.min():pos_data.index.max()] # 按factor头尾截取
pos_data = pos_data.reindex(rtn_1p.index) # 按twap reindex，确保等长


actual_pos = pos_data


# %% test
gp = (actual_pos * rtn_1p)
hsr = ((actual_pos - actual_pos.shift(pp_by_sp)) / 2).abs().replace(
    [np.inf, -np.inf, np.nan], np.nan)


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# %% 对齐数据索引
# 确保labels_df和实际的交易数据有相同的索引
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
        # breakpoint()

# %% 统计不同label和仓位下的收益
# 获取所有非空label类型
all_labels = []
for col in aligned_labels.columns:
    unique_labels = aligned_labels[col].dropna().unique()
    all_labels.extend([label for label in unique_labels if label != 'None'])
all_labels = sorted(list(set(all_labels)))

# 创建结果DataFrame来存储统计结果
results = {}

# 计算每个期货品种的总有效时间（实际持仓不为0的时间）
total_active_minutes = {}

for fut in aligned_pos.columns:
    if fut not in aligned_labels.columns or fut not in aligned_net_returns.columns:
        continue
    
    # 计算总有效交易时间（持仓不为0）
    total_active_minutes[fut] = (aligned_pos[fut] != 0).sum()
        
    fut_results = {'Label': [], 'Position': [], 'TotalReturn': [], 'MinuteCount': [], 'Percentage': []}
    
    # 处理每个标签
    for label in all_labels:
        # 获取该标签的所有时间点
        label_mask = aligned_labels[fut] == label
        
        # 正仓位
        pos_mask = (aligned_pos[fut] > 0) & label_mask
        pos_returns = aligned_net_returns[fut][pos_mask].sum()
        pos_minutes = pos_mask.sum()
        pos_percentage = 100 * pos_minutes / total_active_minutes[fut] if total_active_minutes[fut] > 0 else 0
        
        fut_results['Label'].append(label)
        fut_results['Position'].append('Positive')
        fut_results['TotalReturn'].append(pos_returns)
        fut_results['MinuteCount'].append(pos_minutes)
        fut_results['Percentage'].append(pos_percentage)
        
        # 负仓位
        neg_mask = (aligned_pos[fut] < 0) & label_mask
        neg_returns = aligned_net_returns[fut][neg_mask].sum()
        neg_minutes = neg_mask.sum()
        neg_percentage = 100 * neg_minutes / total_active_minutes[fut] if total_active_minutes[fut] > 0 else 0
        
        fut_results['Label'].append(label)
        fut_results['Position'].append('Negative')
        fut_results['TotalReturn'].append(neg_returns)
        fut_results['MinuteCount'].append(neg_minutes)
        fut_results['Percentage'].append(neg_percentage)
    
    # 转换为DataFrame
    results[fut] = pd.DataFrame(fut_results)

# %% 绘图
# 为每个期货品种创建一个图表
for fut, df in results.items():
    plt.figure(figsize=(14, 8))
    
    # 使用seaborn创建分组柱状图
    sns.set_style("whitegrid")
    ax = sns.barplot(
        x="Label", 
        y="TotalReturn", 
        hue="Position", 
        data=df,
        palette={"Positive": "firebrick", "Negative": "forestgreen"}
    )
    
    # 添加标题和标签
    plt.title(f'Total Net Returns by Label and Position for {fut}\nPos: {pos_name}\nLabel: {label_name}', 
              fontsize=16, 
              fontweight='bold',
              pad=20)  # 增加标题与图表之间的距离
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Total Net Return', fontsize=14)
    # 不旋转x轴标签
    plt.xticks(rotation=0)
    
    # 在每个柱状图上显示持仓时间占比
    # 使用字典创建标签位置的映射
    label_pos_map = {}
    for i, row in enumerate(df.itertuples()):
        key = (row.Label, row.Position)
        label_pos_map[key] = row.Percentage
    
    # 为每个柱状图添加标签
    for i, p in enumerate(ax.patches):
        # 确定当前补丁对应的分类和位置
        num_labels = len(df['Label'].unique())
        label_idx = i // 2
        pos_idx = i % 2
        
        if label_idx < len(df['Label'].unique()):
            label = df['Label'].unique()[label_idx]
            position = ['Positive', 'Negative'][pos_idx]
            
            key = (label, position)
            if key in label_pos_map:
                percentage = label_pos_map[key]
                
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    p.get_height() + (0.0001 if p.get_height() > 0 else -0.0012),
                    f'{percentage:.2f}%',
                    ha='center', 
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7)
                )
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(summary_dir / f'{fut}_label_returns.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()

# %% 创建汇总分析表格并保存
for fut, df in results.items():
    # 按标签和仓位分组计算统计数据
    summary_table = df.pivot_table(
        index='Label',
        columns='Position',
        values=['TotalReturn', 'MinuteCount', 'Percentage'],
        aggfunc='sum'
    )
    
    # 保存为CSV
    summary_table.to_csv(summary_dir / f'{fut}_label_returns_summary.csv')
    
    # 打印结果
    print(f"\n{fut} Summary:")
    print(summary_table)

# 创建所有期货品种的汇总图表
plt.figure(figsize=(16, 10))

# 准备数据
all_data = pd.concat([df.assign(Future=fut) for fut, df in results.items()])

# 使用seaborn创建分组柱状图
g = sns.catplot(
    x="Label", 
    y="TotalReturn", 
    hue="Position",
    col="Future",
    data=all_data,
    kind="bar",
    height=6,
    aspect=1.2,
    palette={"Positive": "firebrick", "Negative": "forestgreen"},
    sharey=True
)

# 添加百分比标签到每个子图
for i, ax in enumerate(g.axes.flat):
    fut = list(results.keys())[i]
    df = results[fut]
    
    # 使用字典创建标签位置的映射
    label_pos_map = {}
    for i, row in enumerate(df.itertuples()):
        key = (row.Label, row.Position)
        label_pos_map[key] = row.Percentage
    
    # 为每个柱状图添加标签
    for i, p in enumerate(ax.patches):
        # 确定当前补丁对应的分类和位置
        num_labels = len(df['Label'].unique())
        label_idx = i // 2
        pos_idx = i % 2
        
        if label_idx < len(df['Label'].unique()):
            label = df['Label'].unique()[label_idx]
            position = ['Positive', 'Negative'][pos_idx]
            
            key = (label, position)
            if key in label_pos_map:
                percentage = label_pos_map[key]
                
                ax.text(
                    p.get_x() + p.get_width() / 2,
                    p.get_height() + (0.0001 if p.get_height() > 0 else -0.0012),
                    f'{percentage:.2f}%',
                    ha='center', 
                    fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7)
                )
    
    # 不旋转x轴标签
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# 调整标题和标签
g.set_xlabels('Label', fontsize=12)
g.set_ylabels('Total Net Return', fontsize=12)
g.set_titles("{col_name}", fontsize=14)
g.fig.suptitle(f'Total Net Returns by Label and Position for All Futures\nPos: {pos_name}\nLabel: {label_name}', 
                fontsize=16, 
                fontweight='bold',
                y=1.05)  # 更高的标题位置
g.fig.subplots_adjust(top=0.80)  # 给标题留出更多空间

# 保存图表
plt.savefig(summary_dir / 'all_futures_label_returns.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()