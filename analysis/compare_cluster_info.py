# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:25:03 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


from utils.dirutils import get_file_names_without_extension


# %%
cluster_name1 = 'agg_250218_3_by_trade_net_double3m_v6'
cluster_name2 = 'agg_250218_3_fix_trade_by_trade_net_double3m_v4'
# cluster_name2 = 'agg_250218_3_fix_trade_by_trade_net_double3m_v6'
# cluster_name = 'agg_250203_by_trade_net_double3m_v6'
cluster_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\cluster')


# %%
this_cluster_dir1 = cluster_dir / cluster_name1
this_cluster_dir2 = cluster_dir / cluster_name2

# 获取文件名
filenames1 = get_file_names_without_extension(this_cluster_dir1)
filenames2 = get_file_names_without_extension(this_cluster_dir2)

df1 = pd.DataFrame()
df2 = pd.DataFrame()

# 遍历 cluster1 文件，整理数据
for filename in filenames1:
    path = this_cluster_dir1 / f'{filename}.csv'
    cluster_info = pd.read_csv(path)
    date_cut = filename.split('_')[-1]
    if len(date_cut) != 6:
        continue
    date = datetime.strptime(date_cut, '%y%m%d')
    df1.loc[date, 'facNum'] = len(cluster_info)
    df1.loc[date, 'facGroup'] = np.max(cluster_info['group']) + 1

# 遍历 cluster2 文件，整理数据
for filename in filenames2:
    path = this_cluster_dir2 / f'{filename}.csv'
    cluster_info = pd.read_csv(path)
    date_cut = filename.split('_')[-1]
    if len(date_cut) != 6:
        continue
    date = datetime.strptime(date_cut, '%y%m%d')
    df2.loc[date, 'facNum'] = len(cluster_info)
    df2.loc[date, 'facGroup'] = np.max(cluster_info['group']) + 1

# 统一两个 DataFrame 的日期索引
df1 = df1.sort_index().loc[:'20241201']
df2 = df2.sort_index().loc[:'20241201']

# 获取两个 DataFrame 共同的日期
common_dates = df1.index.intersection(df2.index)

# 对比绘制图形
x = np.arange(len(common_dates))  # 时间戳对应的索引
width = 0.4  # 条形宽度

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))  # 创建两个子图

# 绘制 facNum 对比
ax1.bar(x - width/2, df1.loc[common_dates, 'facNum'], width, label=f'{cluster_name1} facNum', alpha=0.7)
ax1.bar(x + width/2, df2.loc[common_dates, 'facNum'], width, label=f'{cluster_name2} facNum', alpha=0.7)
ax1.set_xticks(x)
ax1.set_xticklabels(common_dates.strftime('%Y-%m-%d'), rotation=45, ha='right')
ax1.set_xlabel('Date')
ax1.set_ylabel('facNum')
ax1.set_title('facNum Comparison')
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 绘制 facGroup 对比
ax2.bar(x - width/2, df1.loc[common_dates, 'facGroup'], width, label=f'{cluster_name1} facGroup', alpha=0.7)
ax2.bar(x + width/2, df2.loc[common_dates, 'facGroup'], width, label=f'{cluster_name2} facGroup', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(common_dates.strftime('%Y-%m-%d'), rotation=45, ha='right')
ax2.set_xlabel('Date')
ax2.set_ylabel('facGroup')
ax2.set_title('facGroup Comparison')
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

# 保存图形
save_path = this_cluster_dir1 / f'{cluster_name1}_vs_{cluster_name2}.jpg'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()