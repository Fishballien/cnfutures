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
cluster_name = 'agg_250203_by_trade_net_double3m_v6'
cluster_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\results\cluster')


# %%
this_cluster_dir = cluster_dir / cluster_name


# %%
filenames = get_file_names_without_extension(this_cluster_dir)

df = pd.DataFrame()

for filename in filenames:
    path = this_cluster_dir / f'{filename}.csv'
    cluster_info = pd.read_csv(path)
    date_cut = filename.split('_')[-1]
    if len(date_cut) != 6:
        continue
    date = datetime.strptime(date_cut, '%y%m%d')
    df.loc[date, 'facNum'] = len(cluster_info)
    df.loc[date, 'facGroup'] = np.max(cluster_info['group']) + 1
    
    
x = np.arange(len(df))  # 时间戳对应的索引
width = 0.4  # 条形宽度

fig, ax = plt.subplots(figsize=(14, 8))  # 增大图片尺寸

# 绘制 facNum 和 facGroup 的条形图
ax.bar(x - width/2, df['facNum'], width, label='facNum', alpha=0.7)
ax.bar(x + width/2, df['facGroup'], width, label='facGroup', alpha=0.7)

# 设置 x 轴标签为日期
ax.set_xticks(x)
ax.set_xticklabels(df.index.strftime('%Y-%m-%d'), rotation=45, ha='right')

# 添加网格线
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 添加标签和图例
ax.set_xlabel('Date')
ax.set_ylabel('Values')
ax.set_title(cluster_name, fontsize=14, pad=15)
ax.legend()

plt.tight_layout()
save_path = this_cluster_dir / f'{cluster_name}.jpg'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()