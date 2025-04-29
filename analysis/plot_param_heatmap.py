# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 20:09:42 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns


# %%
eval_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\results\factor_evaluation')


# %%
eval_1 = 'agg_batch10_250102_downscale'
eval_2 = 'agg_batch10_241231_wrong_wgt'
eval_name = 'factor_eval_151201_240901'


# %%
eval_1_path = eval_dir / eval_1 / f'{eval_name}.csv'
eval_2_path = eval_dir / eval_2 / f'{eval_name}.csv'


# %%
eval_1_data = pd.read_csv(eval_1_path)
eval_2_data = pd.read_csv(eval_2_path)

eval_data = pd.concat([eval_1_data, eval_2_data]).reset_index(drop=True)
eval_data['dp'] = eval_data['factor'].apply(lambda x: re.search(r"dp\d+|dp[a-zA-Z]+", x.split('-')[1]).group() 
                                            if re.search(r"dp\d+|dp[a-zA-Z]+", x.split('-')[1]) is not None else 'wrong')
eval_data['agg'] = eval_data['factor'].apply(lambda x: re.sub(r"_dp\d+|dp[a-zA-Z]+", "", x.split('-')[1]))


# %%
# Get unique 'dp' values
unique_dps = eval_data['dp'].unique()

# Set up the plot
plt.figure(figsize=(10, 6))

# Plot histograms for each unique 'dp'
for dp in unique_dps:
    if not dp in ['dp2', 'dpall', 'wrong']:
        continue
    subset = eval_data[eval_data['dp'] == dp]
    plt.hist(subset['net_sharpe_ratio'], bins=50, histtype='step',  # Outline only
            linewidth=1.5, label=f"dp: {dp}")

# Customize the plot
plt.title("Histogram of Net Sharpe by dp")
plt.xlabel("Net Sharpe")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()


# %%
# 按 'dp' 和 'agg' 分组，计算平均 'net_sharpe'
heatmap_data = eval_data.groupby(['dp', 'agg'])['net_sharpe_ratio'].mean().unstack()

# 创建热力图
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar_kws={'label': 'Avg Net Sharpe'})

# 设置标题和轴标签
plt.title("Heatmap of Avg Net Sharpe by dp and agg", fontsize=14)
plt.xlabel("agg", fontsize=12)
plt.ylabel("dp", fontsize=12)

# 显示图表
plt.tight_layout()
plt.show()