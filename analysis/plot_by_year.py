# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:15:44 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import sys
from pathlib import Path
import pandas as pd
import toml
import copy
import pickle
import matplotlib.pyplot as plt


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from test_and_eval.scores import get_general_return_metrics


# %%
factor_data_dir = r'D:\CNIndexFutures\timeseries\factor_test\results\model\avg_agg_250203_by_trade_net_v6'
test_name = 'trade_ver0_futtwap_sp1min_s240d_icim'
pred_name = 'predict_avg_agg_250203_by_trade_net_v6'
fee = 0.00024


# %%
path_config = load_path_config(project_dir)
param_dir = Path(path_config['param'])


# %%
result_dir = Path(factor_data_dir)
summary_dir = result_dir / 'test' / test_name / 'plot_by_year'
summary_dir.mkdir(parents=True, exist_ok=True)


# %%
test_data_dir = result_dir / 'test' / test_name / 'data'

test_data = {}
for data_type in ('gpd', 'hsr'):
    data_path = test_data_dir / f'{data_type}_{pred_name}.pkl'
    with open(data_path, 'rb') as f:
        test_data[data_type] = pickle.load(f)
        
df_gp = test_data['gpd']['all']
df_hsr = test_data['hsr']['all']

net = (df_gp['return'] - fee * df_hsr['avg']).fillna(0)
metrics = get_general_return_metrics(net)
renamed_metrics = {f'net_{m}': v for m, v in metrics.items()}

profit_per_trade = df_gp["return"].sum() / df_hsr["avg"].sum()


# %%
# 转换 net 为 DataFrame，并添加年份列
net_df = pd.DataFrame({'return': net})
net_df.index = pd.to_datetime(net_df.index)
net_df['year'] = net_df.index.year

# 按年份分组，绘制每年一张图
years = net_df['year'].unique()

for year in years:
    yearly_net = net_df[net_df['year'] == year]['return']
    cumulative_return = (1 + yearly_net).cumprod() - 1  # 计算累计收益率

    # 创建新图表和子图布局
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])  # 1行2列，右边宽度为左边的1/3

    # 左侧子图：绘制累计收益率
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cumulative_return, label=f'Cumulative Return {year}', color='tab:blue', linewidth=2)
    ax1.set_title(f'Cumulative Return for {year}', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.tick_params(axis='x', rotation=45)

    # 右侧子图：显示 metrics 信息
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')  # 关闭坐标轴

    # 计算指标和 profit_per_trade
    metrics = get_general_return_metrics(yearly_net)
    profit_per_trade = yearly_net.sum() / df_hsr['avg'].sum()  # 每笔交易的利润
    renamed_metrics = {f'net_{m}': v for m, v in metrics.items()}
    
    # 格式化文本
    metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in renamed_metrics.items()])
    metrics_text += f'\nProfit per trade: {profit_per_trade:.6f}'

    # 将 metrics 文本添加到右侧子图
    ax2.text(0.05, 0.95, metrics_text, ha='left', va='top', fontsize=12, family='monospace')

    # 保存图表到指定目录
    summary_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    plt.tight_layout()
    plt.savefig(summary_dir / f'cumulative_returns_{year}.png')  # 保存为PNG文件
    plt.show()

    # 关闭当前图表，以便下一个年度图表不会重叠
    plt.close()
