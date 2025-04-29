# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:45:55 2024

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
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
import numpy as np
from functools import partial


from utils.datautils import align_and_sort_columns
from utils.market import index_to_futures
# from trans_operators.format import to_float32


from utils.timeutils import parse_time_string
from utils.trade_rules import *
from data_processing.ts_trans import *


# %%
factor_name = 'LargeOrderAmountByValue_p1.0_v40000-avg_imb06_dp2-org'
# factor_name = 'LargeOrderAmountByValue_p1.0_v40000-avg_side_dp2_Bid-org'
# factor_name = 'LargeOrderAmountByValue_p1.0_v40000-avg_side_dp2_Sum-org'
# raw_factor_list = [
#     'LargeOrderAmountByValue_p1.0_v40000-avg_side_dp2_Ask-org',
#     'LargeOrderAmountByValue_p1.0_v40000-avg_side_dp2_Bid-org',
#     ]
direction = 1
# factor_name = 'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb04_dp2-org'
# direction = -1

price_name = 't1min_fq1min_dl1min'

scale_method = 'minmax_scale'
scale_window = '240d'
scale_quantile = 0.02
sp = '1min'

trade_rule_name = 'trade_rule_by_trigger_v0'
trade_rule_param = {
    'openthres': 0.8,
    'closethres': 0,
    }


# %%
factor_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\sample_data\factors\1_2_org')
# factor_name = f'predict_{model_name}'
# factor_dir = Path(rf'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model\{model_name}\predict')
fut_dir = Path('/mnt/data1/future_twap')
analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\factor_intraday_pattern')
save_dir = analysis_dir / factor_name
save_dir.mkdir(parents=True, exist_ok=True)


# %%
if 'raw_factor_list' in globals():
    factor_data = pd.DataFrame()
    for raw_factor in raw_factor_list:
        raw_factor_data = pd.read_parquet(factor_dir / f'{raw_factor}.parquet')
        if factor_data.empty:
            factor_data = raw_factor_data
        else:
            factor_data += raw_factor_data
else:
    factor_data = pd.read_parquet(factor_dir / f'{factor_name}.parquet')
# factor_data = to_float32(factor_data)
price_data = pd.read_parquet(fut_dir / f'{price_name}.parquet')
factor_data = factor_data.rename(columns=index_to_futures)[['IC', 'IF', 'IM']]
factor_data, price_data = align_and_sort_columns([factor_data, price_data])

price_data = price_data.loc[factor_data.index.min():factor_data.index.max()] # 按factor头尾截取
factor_data = factor_data.reindex(price_data.index) # 按twap reindex，确保等长


# %%
# =============================================================================
# intraday_dir = save_dir / 'intraday'
# intraday_dir.mkdir(parents=True, exist_ok=True)
# 
# # 提取小时和分钟，创建"日内时间"列
# factor_data['intraday_time'] = factor_data.index.strftime('%H:%M')
# 
# for year, year_group in factor_data.groupby(pd.Grouper(freq='Y')):
#     # 提取当年的数据
#     year_group = year_group.copy()
#     
#     # 按“日内时间”分组，计算每分钟的因子均值
#     factor_mean = year_group.groupby('intraday_time').mean()
# 
#     # 绘制日内分钟因子均值的线图
#     plt.figure(figsize=(12, 6))
#     for col in factor_data.columns[:-1]:  # 遍历所有因子列（排除intraday_time）
#         plt.plot(factor_mean.index, factor_mean[col], label=col)
# 
#     # 设置标题和坐标轴标签
#     plt.title(f'Intraday Minute-Wise Factor Mean for Year {year.year}', fontsize=14)
#     plt.xlabel('Time of Day (HH:MM)')
#     plt.ylabel('Average Factor Value')
# 
#     # 调整x轴：每半小时显示一个刻度
#     ax = plt.gca()
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(30))  # 每30个点显示一个刻度
# 
#     # 旋转X轴标签，避免重叠
#     plt.xticks(rotation=45)
# 
#     # 添加网格和图例
#     plt.grid()
#     plt.legend()
#     plt.tight_layout()
# 
#     # 显示图表
#     plt.savefig(intraday_dir / f"{year.year}.jpg", dpi=300)
#     plt.show()
#     
# =============================================================================
    
# %%
intraday_dir = save_dir / 'intraday'
intraday_dir.mkdir(parents=True, exist_ok=True)

# 提取小时和分钟，创建"日内时间"列
factor_data['intraday_time'] = factor_data.index.strftime('%H:%M')

# 按年份分组
year_groups = list(factor_data.groupby(pd.Grouper(freq='Y')))
num_years = len(year_groups)

# 设置一行，每行最多4列的子图布局
cols_per_row = 4
num_rows = (num_years + cols_per_row - 1) // cols_per_row  # 向上取整

# 创建大图和子图
fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(20, 5 * num_rows), squeeze=False)

# 遍历每一年
for i, (year, year_group) in enumerate(year_groups):
    # 计算子图位置
    row_idx = i // cols_per_row
    col_idx = i % cols_per_row
    
    # 获取当前子图
    ax = axes[row_idx, col_idx]
    
    # 提取当年的数据
    year_group = year_group.copy()
    
    # 按"日内时间"分组，计算每分钟的因子均值
    factor_mean = year_group.groupby('intraday_time').mean()
    
    # 在子图中绘制日内分钟因子均值的线图
    for col in factor_data.columns[:-1]:  # 遍历所有因子列（排除intraday_time）
        ax.plot(factor_mean.index, factor_mean[col], label=col)
    
    # 设置标题和坐标轴标签
    ax.set_title(f'Year {year.year}', fontsize=12)
    
    # 如果是最后一行的子图，添加x轴标签
    if row_idx == num_rows - 1:
        ax.set_xlabel('Time of Day (HH:MM)')
    
    # 如果是每行的第一个子图，添加y轴标签
    if col_idx == 0:
        ax.set_ylabel('Average Factor Value')
    
    # 调整x轴：每半小时显示一个刻度
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    
    # 旋转X轴标签，避免重叠
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    
    # 添加网格
    ax.grid(True)
    
    # 只在第一个子图中添加图例
    if i == 0:
        ax.legend()

# 隐藏没有使用的子图
for i in range(num_years, num_rows * cols_per_row):
    row_idx = i // cols_per_row
    col_idx = i % cols_per_row
    fig.delaxes(axes[row_idx, col_idx])

# 调整布局，确保标签不会重叠
plt.tight_layout()

# 保存图片
plt.savefig(intraday_dir / "intraday_factors_by_year.jpg", dpi=300)
plt.show()
