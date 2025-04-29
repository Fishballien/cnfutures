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


# %%
eval_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\results\factor_evaluation')


# %%
eval_name = 'test_agg_minute'
file_name = 'factor_eval_160101_250101'


# %%
eval_path = eval_dir / eval_name / f'{file_name}.csv'


# %%
eval_data = pd.read_csv(eval_path)


# %%
df_minute_min_max = eval_data[eval_data['factor'].str.contains("rollingMinuteQuantileScale")]
df_agg_minute_min_max = eval_data[eval_data['factor'].str.contains("rollingAggregatedMinuteQuantileScale")]
df_minute_pct = eval_data[eval_data['factor'].str.contains("rollingAggMinutePercentile") & eval_data['factor'].str.contains("i1")]
df_agg_minute_pct = eval_data[eval_data['factor'].str.contains("rollingAggMinutePercentile") & ~eval_data['factor'].str.contains("i1")]


# %%
# 计算每个因子类型的平均 Sharpe Ratio 和 90% 分位数
df_list = [df_minute_min_max, df_agg_minute_min_max, df_minute_pct, df_agg_minute_pct]
factor_labels = ['minute_min_max', 'agg_minute_min_max', 'minute_pct', 'agg_minute_pct']

# 初始化存储 Sharpe Ratio 平均值和 90% 分位数的字典
mean_sharpe = {}
percentile_90th = {}

# 计算每个 DataFrame 的平均 Sharpe Ratio 和 90% 分位数
for df, label in zip(df_list, factor_labels):
    mean_sharpe[label] = df["net_sharpe_ratio"].mean()
    percentile_90th[label] = np.percentile(df["net_sharpe_ratio"], 90)

# 设定边缘颜色
edge_colors = ['#0000B3', '#FF007F', '#00B300', '#FF6600']  # 深蓝色、深粉色、绿色、橙色

# 画 Sharpe Ratio 的直方图
plt.figure(figsize=(10, 5))

# 遍历所有 df 进行绘制
for df, label, edge_color in zip(df_list, factor_labels, edge_colors):
    plt.hist(df["net_sharpe_ratio"], bins=50, edgecolor=edge_color, linewidth=1.5, histtype='step', 
             label=f'{label} (mean={mean_sharpe[label]:.2f}, 90th={percentile_90th[label]:.2f})')

plt.xlabel("Sharpe Ratio")
plt.ylabel("Frequency")
plt.title("Distribution of Sharpe Ratio for Different Factor Types")
plt.legend()
plt.grid(True)

# 显示图表
plt.show()


# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 提取每个df的分类（例如 'w30d'）
def extract_wd_category(df):
    # 从factor列提取类似 'w30d' 这样的分类信息
    df['wd_category'] = df['factor'].str.extract(r'(w\d+d)')
    return df

# 提取分类并计算每个分类的 net_sharpe_ratio 均值
def calculate_sharpe_by_wd(df_list, factor_labels):
    # 用于存储结果的字典
    sharpe_means = {}
    
    # 遍历所有df
    for df, label in zip(df_list, factor_labels):
        # 提取分类
        df = extract_wd_category(df)
        # 计算每个分类的 net_sharpe_ratio 均值
        category_sharpe = df.groupby('wd_category')['net_sharpe_ratio'].mean()
        sharpe_means[label] = category_sharpe
    
    return sharpe_means

# 定义要处理的DataFrame和标签
df_list = [df_minute_min_max, df_agg_minute_min_max, df_minute_pct, df_agg_minute_pct]
factor_labels = ['minute_min_max', 'agg_minute_min_max', 'minute_pct', 'agg_minute_pct']

# 计算每个df中每个分类的平均net_sharpe_ratio
sharpe_means = calculate_sharpe_by_wd(df_list, factor_labels)

# 将结果转化为DataFrame，便于画热力图
sharpe_matrix = pd.DataFrame(sharpe_means)

# 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(sharpe_matrix.T, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'Mean Sharpe Ratio'})
plt.title("Mean Sharpe Ratio by Factor Type and w*d Category")
plt.xlabel("w*d Category")
plt.ylabel("Factor Type")
plt.show()


# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 提取 'i*' 分类参数，确保可以区分 i1, i10 等
def extract_i_category(df):
    # 使用正则表达式提取 'i' 开头的分类，例如 'i1', 'i10' 等
    df['i_category'] = df['factor'].str.extract(r'(i\d+)')
    return df

# 计算每个 i 类别的 net_sharpe_ratio 均值
def calculate_sharpe_by_i(df_list, factor_labels):
    sharpe_means = {}
    
    for df, label in zip(df_list, factor_labels):
        # 如果是 df_agg_minute_pct，先过滤掉 'i1' 类别
        if label == 'agg_minute_pct':
            df = df[df['factor'].str.endswith('i1') == False]
        
        # 提取 i 类别
        df = extract_i_category(df)
        # 计算每个 i 类别的 net_sharpe_ratio 均值
        category_sharpe = df.groupby('i_category')['net_sharpe_ratio'].mean()
        sharpe_means[label] = category_sharpe
    
    return sharpe_means

# 定义要处理的DataFrame和标签
df_list = [df_agg_minute_min_max, df_agg_minute_pct]
factor_labels = ['agg_minute_min_max', 'agg_minute_pct']

# 计算每个df中每个i类别的平均net_sharpe_ratio
sharpe_means = calculate_sharpe_by_i(df_list, factor_labels)

# 将结果转化为DataFrame，便于画热力图
sharpe_matrix = pd.DataFrame(sharpe_means)

# 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(sharpe_matrix.T, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'Mean Sharpe Ratio'})
plt.title("Mean Sharpe Ratio by Factor Type and i* Category")
plt.xlabel("i* Category")
plt.ylabel("Factor Type")
plt.show()


# %%
import seaborn as sns

# 筛选并提取 factor_part1 和 factor_part2
def extract_factor_parts(df):
    df[['factor_part1', 'factor_part2']] = df['factor'].apply(lambda x: pd.Series(x.split('-', 1)))
    df['factor_part2'] = df['factor_part2'].apply(lambda x: x.rsplit('-', 1)[-1])
    df['factor_part2'] = df['factor_part2'].str.extract(r'^(.*w\d+d)', expand=False)
    return df

# 处理每个 DataFrame，提取 factor_part1 和 factor_part2
df_minute_min_max = extract_factor_parts(df_minute_min_max)
df_agg_minute_min_max = extract_factor_parts(df_agg_minute_min_max)
df_minute_pct = extract_factor_parts(df_minute_pct)
df_agg_minute_pct = extract_factor_parts(df_agg_minute_pct)

# 计算 profit_per_trade
def calculate_profit_per_trade(df):
    df['profit_per_trade'] = df['net_return_annualized'] / df['hsr'] / 245 * 1000
    return df

# 对每个 DataFrame 计算 profit_per_trade
df_minute_min_max = calculate_profit_per_trade(df_minute_min_max)
df_agg_minute_min_max = calculate_profit_per_trade(df_agg_minute_min_max)
df_minute_pct = calculate_profit_per_trade(df_minute_pct)
df_agg_minute_pct = calculate_profit_per_trade(df_agg_minute_pct)

# 准备目标值列
target_columns = ['net_sharpe_ratio', 'net_max_dd', 'profit_per_trade', 'hsr', 'net_calmar_ratio']

# 创建绘图函数，用于生成热图
def plot_heatmap(df, target_column, title):
    # 计算透视表（pivot table），行列分别是 factor_part1 和 factor_part2，数值是目标值
    pivot_table = df.pivot_table(index='factor_part1', columns='factor_part2', values=target_column, aggfunc=np.max)
    
    # 绘制热图
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt=".2f", cbar_kws={'label': target_column})
    plt.title(f'{title} - {target_column}')
    plt.xlabel('Factor Part 2')
    plt.ylabel('Factor Part 1')
    # plt.grid(True)
    plt.show()

# 分别绘制每个 DataFrame 和目标值的热图
for target_column in target_columns:
    plot_heatmap(df_minute_min_max, target_column, 'df_minute_min_max')
    plot_heatmap(df_agg_minute_min_max, target_column, 'df_agg_minute_min_max')
    plot_heatmap(df_minute_pct, target_column, 'df_minute_pct')
    plot_heatmap(df_agg_minute_pct, target_column, 'df_agg_minute_pct')
