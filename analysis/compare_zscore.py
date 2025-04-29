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
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns


# %%
eval_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\results\factor_evaluation')


# %%
eval_name = 'test_zscore_fxwd'
file_name = 'factor_eval_150101_250201'


# %%
eval_path = eval_dir / eval_name / f'{file_name}.csv'


# %%
eval_data = pd.read_csv(eval_path)


# %%
# 筛选包含 "zscore" 但不包含 "zscore_fxwd" 的数据
df_zscore = eval_data[eval_data['factor'].str.contains("zscore") & ~eval_data['factor'].str.contains("zscore_fxwd")]

# 筛选仅包含 "zscore_fxwd" 的数据
df_zscore_fxwd = eval_data[eval_data['factor'].str.contains("zscore_fxwd")]

# =============================================================================
# # 计算每个因子类型的平均 Sharpe Ratio 和 90% 分位数
# zscore_mean_sharpe = df_zscore["net_sharpe_ratio"].mean()
# zscore_75th_percentile = np.percentile(df_zscore["net_sharpe_ratio"], 75)
# 
# zscore_fxwd_mean_sharpe = df_zscore_fxwd["net_sharpe_ratio"].mean()
# zscore_fxwd_75th_percentile = np.percentile(df_zscore_fxwd["net_sharpe_ratio"], 75)
# 
# # 设定边缘颜色为深粉色和深蓝色
# zscore_edge_color = '#0000B3'  # 深蓝色
# zscore_fxwd_edge_color = '#FF007F'  # 深粉色
# 
# # 画 Sharpe Ratio 的直方图
# plt.figure(figsize=(10, 5))
# plt.hist(df_zscore["net_sharpe_ratio"], bins=50, edgecolor=zscore_edge_color, linewidth=1.5, histtype='step', label=f'zscore (mean={zscore_mean_sharpe:.2f}, 75%={zscore_75th_percentile:.2f})')
# plt.hist(df_zscore_fxwd["net_sharpe_ratio"], bins=50, edgecolor=zscore_fxwd_edge_color, linewidth=1.5, histtype='step', label=f'zscore_fxwd (mean={zscore_fxwd_mean_sharpe:.2f}, 75%={zscore_fxwd_75th_percentile:.2f})')
# 
# plt.xlabel("Sharpe Ratio")
# plt.ylabel("Frequency")
# plt.title("Distribution of Sharpe Ratio for Different Factor Types")
# plt.legend()
# plt.grid(True)
# 
# # 显示图表
# plt.show()
# =============================================================================


# %%
# =============================================================================
# # 按因子名称排序两个 DataFrame，使它们的因子列一致
# df_zscore = df_zscore.sort_values('factor').reset_index()
# df_zscore_fxwd = df_zscore_fxwd.sort_values('factor').reset_index()
# 
# # 计算 profit_per_trade: net_return / hsr
# df_zscore['profit_per_trade'] = df_zscore['net_return_annualized'] / df_zscore['hsr'] / 245
# df_zscore_fxwd['profit_per_trade'] = df_zscore_fxwd['net_return_annualized'] / df_zscore_fxwd['hsr'] / 245
# 
# # 计算 Sharpe Ratio 和 profit_per_trade 之间的差值
# df_zscore_fxwd['profit_diff'] = df_zscore_fxwd['profit_per_trade'] - df_zscore['profit_per_trade']
# df_zscore_fxwd['maxdd_diff'] = df_zscore_fxwd['net_max_dd'] - df_zscore['net_max_dd']
# 
# # 绘制散点图，x 轴为 df_zscore 的 Sharpe Ratio，y 轴为 profit_diff
# plt.figure(figsize=(10, 6))
# # plt.scatter(df_zscore['net_sharpe_ratio'], df_zscore_fxwd['profit_diff'], color='#1f77b4', label='Profit Difference (zscore_fxwd - zscore)', alpha=0.7)
# plt.scatter(df_zscore['net_sharpe_ratio'], df_zscore_fxwd['maxdd_diff'], color='#1f77b4', label='MaxDD Difference (zscore_fxwd - zscore)', alpha=0.7)
# 
# 
# # 添加 y=0 的虚线
# plt.axhline(0, color='black', linestyle='--', linewidth=1)
# 
# # 添加图表标签和标题
# plt.xlabel("Sharpe Ratio (zscore)")
# # plt.ylabel("Profit Difference (zscore_fxwd - zscore)")
# # plt.title("Scatter Plot of Sharpe Ratio vs. Profit Difference Between zscore and zscore_fxwd")
# plt.ylabel("MaxDD Difference (zscore_fxwd - zscore)")
# plt.title("Scatter Plot of Sharpe Ratio vs. MaxDD Difference Between zscore and zscore_fxwd")
# plt.grid(True)
# plt.legend()
# 
# # 显示图表
# plt.show()
# =============================================================================


# %%
# 计算 profit_per_trade: net_return / hsr
df_zscore['profit_per_trade'] = df_zscore['net_return_annualized'] / df_zscore['hsr'] / 245 * 1000
df_zscore_fxwd['profit_per_trade'] = df_zscore_fxwd['net_return_annualized'] / df_zscore_fxwd['hsr'] / 245 * 1000

# 使用 apply 和 str.split 来从 'factor' 列中提取第一个"-"前后的部分
df_zscore[['factor_part1', 'factor_part2']] = df_zscore['factor'].apply(lambda x: pd.Series(x.split('-', 1)))
df_zscore_fxwd[['factor_part1', 'factor_part2']] = df_zscore_fxwd['factor'].apply(lambda x: pd.Series(x.split('-', 1)))

# 对 'factor_part2' 再进行分割，取最后一个 "-" 后的部分
df_zscore['factor_part2'] = df_zscore['factor_part2'].apply(lambda x: x.rsplit('-', 1)[-1])
df_zscore_fxwd['factor_part2'] = df_zscore_fxwd['factor_part2'].apply(lambda x: x.rsplit('-', 1)[-1])

# 准备目标值列
target_columns = ['net_sharpe_ratio', 'net_max_dd', 'profit_per_trade']

# 创建一个绘图函数，用于生成热图
def plot_heatmap(df, target_column, title):
    # 计算透视表（pivot table），行列分别是 factor_part1 和 factor_part2，数值是目标值
    pivot_table = df.pivot_table(index='factor_part1', columns='factor_part2', values=target_column, aggfunc=np.mean)

    # 绘制热图
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt=".2f", cbar_kws={'label': target_column})
    plt.title(f'{title} - {target_column}')
    plt.xlabel('Factor Part 2')
    plt.ylabel('Factor Part 1')
    # plt.grid(True)
    plt.show()

# 分别绘制每个目标值的热图
for target_column in target_columns:
    plot_heatmap(df_zscore, target_column, 'df_zscore')
    plot_heatmap(df_zscore_fxwd, target_column, 'df_zscore_fxwd')
