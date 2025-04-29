# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 09:54:46 2025

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
model_name = 'avg_agg_250218_3_fix_tfe_by_trade_net_v4'
factor_name = f'predict_{model_name}'
factor_dir = Path(rf'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model\{model_name}\predict')
direction = 1

# factor_name = 'LargeOrderAmountByValue_p1.0_v40000-wavg_imb04_dpall-mean_w30min'
# direction = -1
# factor_dir = Path(r'D:/mnt/CNIndexFutures/timeseries/factor_test/sample_data/factors/low_freq')


# %%
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

fee = 0.00024


# %%
fut_dir = Path('/mnt/data1/future_twap')
analysis_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\results\analysis\predict_scale_limit')
summary_dir = analysis_dir / factor_name
summary_dir.mkdir(parents=True, exist_ok=True)


# %%
factor_data = pd.read_parquet(factor_dir / f'{factor_name}.parquet')


# %%
# 定义5%和95%分位数
lower_quantile = scale_quantile
upper_quantile = 1 - scale_quantile

df_lower = factor_data.rolling(window=4*240*240, min_periods=1, step=1).quantile(lower_quantile)
df_upper = factor_data.rolling(window=4*240*240, min_periods=1, step=1).quantile(upper_quantile)


# %%
# 重采样到日均值
df_lower_daily = df_lower.resample('D').mean()
df_upper_daily = df_upper.resample('D').mean()

# 绘制日均值时序图
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter

# 设置中文字体（如果需要）
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
# plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 创建子图
fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
columns = ['IC', 'IF', 'IM']
colors = ['#1f77b4', '#ff7f0e']  # 蓝色和橙色

# 生成渐变色填充
def gradient_fill(x, y1, y2, color, ax):
    # 创建渐变填充
    ax.fill_between(x, y1, y2, alpha=0.3, color=color)

# 绘制每个指标的时序图
for i, col in enumerate(columns):
    # 绘制上下边界线
    axes[i].plot(df_lower_daily.index, df_lower_daily[col], color=colors[0], linewidth=1.5, label=f'Lower {col}')
    axes[i].plot(df_upper_daily.index, df_upper_daily[col], color=colors[1], linewidth=1.5, label=f'Upper {col}')
    
    # 添加渐变填充
    gradient_fill(df_lower_daily.index, df_lower_daily[col], df_upper_daily[col], colors[0], axes[i])
    
    # Set title and grid
    axes[i].set_title(f'{col} Daily Mean Value Trend', fontsize=14)
    axes[i].legend(loc='best')
    axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # 设置Y轴格式
    axes[i].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.3f}'))
    
    # 添加水平参考线
    axes[i].axhline(y=0, color='r', linestyle='-', linewidth=0.8, alpha=0.5)

# 设置X轴格式
axes[2].xaxis.set_major_locator(mdates.YearLocator())
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[2].xaxis.set_minor_locator(mdates.MonthLocator())

# Add main title
fig.suptitle('IC, IF and IM Daily Mean Time Series (2016-2025)', fontsize=16, y=0.98)

# 设置布局
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.gcf().autofmt_xdate()  # 旋转日期标签

# 显示图表
plt.show()

# 如果需要保存图表
# plt.savefig('daily_means_time_series.png', dpi=300, bbox_inches='tight')


# %%
# =============================================================================
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# 
# # Ensure the index is datetime
# if not isinstance(factor_data.index, pd.DatetimeIndex):
#     factor_data.index = pd.to_datetime(factor_data.index)
# 
# # Group by year-month and calculate percentiles
# monthly_percentiles = {}
# columns = ['IC', 'IF', 'IM']
# 
# # Calculate 2% and 98% percentiles for each month and each column
# for col in columns:
#     # Group by year-month and calculate percentiles
#     percentile_2 = factor_data.groupby(factor_data.index.to_period('M'))[col].quantile(0.02)
#     percentile_98 = factor_data.groupby(factor_data.index.to_period('M'))[col].quantile(0.98)
#     
#     # Convert period index to datetime for plotting
#     percentile_2.index = percentile_2.index.to_timestamp()
#     percentile_98.index = percentile_98.index.to_timestamp()
#     
#     monthly_percentiles[col] = {
#         'p02': percentile_2,
#         'p98': percentile_98
#     }
# 
# # Create the visualization
# fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
# 
# # Define colors
# colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
# 
# # Plot each column
# for i, col in enumerate(columns):
#     # Get the percentiles
#     p02 = monthly_percentiles[col]['p02']
#     p98 = monthly_percentiles[col]['p98']
#     
#     # Plot the percentiles
#     axes[i].plot(p02.index, p02.values, color=colors[0], linewidth=1.5, label=f'2% Percentile ({col})')
#     axes[i].plot(p98.index, p98.values, color=colors[1], linewidth=1.5, label=f'98% Percentile ({col})')
#     
#     # Fill the area between the percentiles
#     axes[i].fill_between(p02.index, p02.values, p98.values, alpha=0.3, color=colors[0])
#     
#     # Set title and grid
#     axes[i].set_title(f'{col} Monthly 2% and 98% Percentiles', fontsize=14)
#     axes[i].legend(loc='best')
#     axes[i].grid(True, linestyle='--', alpha=0.7)
#     
#     # Format Y-axis
#     axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.3f}'))
#     
#     # Add horizontal reference line at y=0
#     axes[i].axhline(y=0, color='r', linestyle='-', linewidth=0.8, alpha=0.5)
# 
# # Format X-axis
# axes[2].xaxis.set_major_locator(mdates.YearLocator())
# axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# axes[2].xaxis.set_minor_locator(mdates.MonthLocator())
# 
# # Add main title
# fig.suptitle('Monthly 2% and 98% Percentiles for IC, IF and IM (2016-2025)', fontsize=16, y=0.98)
# 
# # Set layout
# plt.tight_layout()
# plt.subplots_adjust(top=0.95)
# plt.gcf().autofmt_xdate()  # Rotate date labels
# 
# # Show the plot
# plt.show()
# 
# # To save the figure if needed
# # plt.savefig('monthly_percentiles.png', dpi=300, bbox_inches='tight')
# =============================================================================
