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
# factor_name = 'LargeOrderAmountByValue_p1.0_v40000-avg_imb01_dp2-org'
# direction = 1
factor_name = 'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb04_dp2-org'
direction = -1

# model_name = 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18'
# direction = 1

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
analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\signals')
save_dir = analysis_dir / factor_name
save_dir.mkdir(parents=True, exist_ok=True)


# %%
factor_data = pd.read_parquet(factor_dir / f'{factor_name}.parquet')
# factor_data = to_float32(factor_data)
price_data = pd.read_parquet(fut_dir / f'{price_name}.parquet')
factor_data = factor_data.rename(columns=index_to_futures)[['IC', 'IF', 'IM']]
factor_data, price_data = align_and_sort_columns([factor_data, price_data])

price_data = price_data.loc[factor_data.index.min():factor_data.index.max()] # 按factor头尾截取
factor_data = factor_data.reindex(price_data.index) # 按twap reindex，确保等长


# %%
scale_func = globals()[scale_method]
scale_step = int(parse_time_string(scale_window) / parse_time_string(sp))
# factor_scaled = ts_quantile_scale(factor, window=scale_step, quantile=scale_quantile)
if scale_method in ['minmax_scale', 'minmax_scale_separate']:
    factor_scaled = scale_func(factor_data, window=scale_step, quantile=scale_quantile)
elif scale_method in ['minmax_scale_adj_by_his_rtn', 'zscore_adj_by_his_rtn_and_minmax']:
    factor_scaled = scale_func(factor, rtn_1p, window=scale_step, rtn_window=pp_by_sp, quantile=scale_quantile)
elif scale_method in ['rolling_percentile']:
    factor_scaled = scale_func(factor, window=scale_step)
elif scale_method in ['percentile_adj_by_his_rtn']:
    factor_scaled = scale_func(factor, rtn_1p, window=scale_step, rtn_window=pp_by_sp)

factor_scaled = (factor_scaled - 0.5) * 2 * direction


# %%
trade_rule_func = partial(globals()[trade_rule_name], **trade_rule_param)
actual_pos = factor_scaled.apply(lambda col: trade_rule_func(col.values), axis=0)


# %%
# =============================================================================
# fac_agg = factor_data.resample('30min').mean()
# fac_agg.plot()
# plt.show()
# =============================================================================


# %%
# =============================================================================
# by_month_dir = save_dir / 'by_month'
# by_month_dir.mkdir(parents=True, exist_ok=True)
# 
# # 检查列数
# factor_columns = factor_data.columns
# price_columns = price_data.columns
# 
# # 确保列的数量一致
# num_columns = min(len(factor_columns), len(price_columns))
# 
# # 按月份分组并绘制图表
# for month, factor_group in factor_data.groupby(pd.Grouper(freq='M')):
#     # 筛选价格数据的对应月份
#     price_group = price_data[((price_data.index.month == month.month)
#                               & (price_data.index.year == month.year))]
# 
#     # 如果某个月数据为空，则跳过
#     if factor_group.empty or price_group.empty:
#         continue
# 
#     # 生成顺序 x 轴 (arange)
#     x = np.arange(len(factor_group))  # 顺序索引
#     x_labels = factor_group.index.strftime('%Y-%m-%d %H:%M')  # 转换成时间标签
# 
#     # 找到每天9:30的位置
#     nine_thirty_indices = [i for i, t in enumerate(factor_group.index) if t.strftime('%H:%M') == '09:30']
# 
#     # 创建一个图形，每个因子-价格列占用2个子图
#     fig, axs = plt.subplots(num_columns * 2, 1, figsize=(12, 4 * num_columns), sharex=True)
#     fig.suptitle(f"Factor and Price Data for {month.strftime('%Y-%m')}", fontsize=16)
# 
#     # 遍历每一列，绘制子图
#     for i in range(num_columns):
#         factor_col = factor_columns[i]
#         price_col = price_columns[i]
# 
#         # 上方子图：因子数据
#         axs[i * 2].plot(x, factor_group[factor_col], label=f'Factor: {factor_col}')
#         axs[i * 2].set_ylabel('Factor Value')
#         axs[i * 2].set_title(f"Factor: {factor_col}")
#         axs[i * 2].legend()
#         axs[i * 2].grid(True)
# 
#         # 绘制红色虚线
#         for idx in nine_thirty_indices:
#             axs[i * 2].axvline(x=idx, color='red', linestyle='--', linewidth=1)
# 
#         # 下方子图：价格数据
#         price_data_trimmed = price_group[price_col][:len(x)]  # 确保长度一致
#         axs[i * 2 + 1].plot(x, price_data_trimmed, label=f'Price: {price_col}', color='orange')
#         axs[i * 2 + 1].set_ylabel('Price')
#         axs[i * 2 + 1].set_title(f"Price: {price_col}")
#         axs[i * 2 + 1].legend()
#         axs[i * 2 + 1].grid(True)
# 
#         # 绘制红色虚线
#         for idx in nine_thirty_indices:
#             axs[i * 2 + 1].axvline(x=idx, color='red', linestyle='--', linewidth=1)
# 
#     # 设置共享的 x 轴标签
#     tick_positions = np.linspace(0, len(x)-1, num=10, dtype=int)
#     axs[-1].set_xticks(tick_positions)
#     axs[-1].set_xticklabels([x_labels[i] for i in tick_positions], rotation=45)
# 
#     # 调整布局
#     plt.tight_layout(rect=[0, 0, 1, 0.96])  # 给标题留空间
#     plt.savefig(by_month_dir / f"{month.strftime('%Y-%m')}.jpg", dpi=300)
#     plt.show()
#     # break
# =============================================================================


# %%
# =============================================================================
# by_month_dir = save_dir / 'by_month'
# by_month_dir.mkdir(parents=True, exist_ok=True)
# 
# # 检查列数
# factor_columns = factor_data.columns
# price_columns = price_data.columns
# 
# # 确保列的数量一致
# num_columns = min(len(factor_columns), len(price_columns))
# 
# # 按月份分组并绘制图表
# for month, factor_group in factor_data.groupby(pd.Grouper(freq='M')):
#     if month.year < 2024:
#         continue
#     # 筛选价格数据的对应月份
#     price_group = price_data[((price_data.index.month == month.month)
#                               & (price_data.index.year == month.year))]
# 
#     # 筛选对应月份的仓位数据
#     actual_pos_month = actual_pos[((actual_pos.index.month == month.month) 
#                                    & (actual_pos.index.year == month.year))]
# 
#     # 如果某个月数据为空，则跳过
#     if factor_group.empty or price_group.empty or actual_pos_month.empty:
#         continue
# 
#     # 对齐 actual_pos_month 的索引，使其与 factor_group 的索引对齐
#     actual_pos_month_aligned = actual_pos_month.reindex(factor_group.index, method='ffill')
# 
#     # 生成顺序 x 轴 (arange)
#     x = np.arange(len(factor_group))  # 顺序索引
#     x_labels = factor_group.index.strftime('%Y-%m-%d %H:%M')  # 转换成时间标签
# 
#     # 找到每天9:30的位置
#     nine_thirty_indices = [i for i, t in enumerate(factor_group.index) if t.strftime('%H:%M') == '09:30']
# 
#     # 创建一个图形，每个因子-价格列占用2个子图
#     fig, axs = plt.subplots(num_columns * 2, 1, figsize=(12, 4 * num_columns), sharex=True)
#     fig.suptitle(f"Factor and Price Data for {month.strftime('%Y-%m')}", fontsize=16)
# 
#     # 遍历每一列，绘制子图
#     for i in range(num_columns):
#         factor_col = factor_columns[i]
#         price_col = price_columns[i]
#         actual_pos_col = actual_pos_month.columns[i]  # 对应的仓位列
# 
#         # 上方子图：因子数据
#         axs[i * 2].plot(x, factor_group[factor_col], label=f'Factor: {factor_col}')
#         axs[i * 2].set_ylabel('Factor Value')
#         axs[i * 2].set_title(f"Factor: {factor_col}")
#         axs[i * 2].legend()
#         axs[i * 2].grid(True)
# 
#         # 绘制红色虚线和实线
#         for idx in nine_thirty_indices:
#             axs[i * 2].axvline(x=idx, color='k', linestyle='--', linewidth=1)
# 
#         # 计算仓位变化的信号
#         actual_pos_col_aligned = actual_pos_month_aligned[actual_pos_col]
#         buy_changes = actual_pos_col_aligned[(actual_pos_col_aligned.diff() == 1) & (actual_pos_col_aligned == 1)].index  # 0 -> 1 (开多)
#         sell_changes = actual_pos_col_aligned[(actual_pos_col_aligned.diff() == -1) & (actual_pos_col_aligned == 0)].index  # 1 -> 0 (平多)
#         short_changes = actual_pos_col_aligned[(actual_pos_col_aligned.diff() == -1) & (actual_pos_col_aligned == -1)].index  # 0 -> -1 (开空)
#         cover_changes = actual_pos_col_aligned[(actual_pos_col_aligned.diff() == 1) & (actual_pos_col_aligned == 0)].index  # -1 -> 0 (平空)
# 
#         # 绘制仓位变化的竖线
#         for change in buy_changes:
#             idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
#             axs[i * 2].axvline(x=idx, color='red', linestyle='-', linewidth=1)  # 开多
#         for change in sell_changes:
#             idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
#             axs[i * 2].axvline(x=idx, color='red', linestyle='--', linewidth=1)  # 平多
#         for change in short_changes:
#             idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
#             axs[i * 2].axvline(x=idx, color='green', linestyle='-', linewidth=1)  # 开空
#         for change in cover_changes:
#             idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
#             axs[i * 2].axvline(x=idx, color='green', linestyle='--', linewidth=1)  # 平空
# 
#         # 下方子图：价格数据
#         price_data_trimmed = price_group[price_col][:len(x)]  # 确保长度一致
#         axs[i * 2 + 1].plot(x, price_data_trimmed, label=f'Price: {price_col}', color='orange')
#         axs[i * 2 + 1].set_ylabel('Price')
#         axs[i * 2 + 1].set_title(f"Price: {price_col}")
#         axs[i * 2 + 1].legend()
#         axs[i * 2 + 1].grid(True)
# 
#         # 绘制红色虚线和实线
#         for idx in nine_thirty_indices:
#             axs[i * 2 + 1].axvline(x=idx, color='k', linestyle='--', linewidth=1)
# 
#         # 绘制仓位变化的竖线
#         for change in buy_changes:
#             idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
#             axs[i * 2 + 1].axvline(x=idx, color='red', linestyle='-', linewidth=1)  # 开多
#         for change in sell_changes:
#             idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
#             axs[i * 2 + 1].axvline(x=idx, color='red', linestyle='--', linewidth=1)  # 平多
#         for change in short_changes:
#             idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
#             axs[i * 2 + 1].axvline(x=idx, color='green', linestyle='-', linewidth=1)  # 开空
#         for change in cover_changes:
#             idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
#             axs[i * 2 + 1].axvline(x=idx, color='green', linestyle='--', linewidth=1)  # 平空
# 
#     # 设置共享的 x 轴标签
#     tick_positions = np.linspace(0, len(x)-1, num=10, dtype=int)
#     axs[-1].set_xticks(tick_positions)
#     axs[-1].set_xticklabels([x_labels[i] for i in tick_positions], rotation=45)
# 
#     # 调整布局
#     plt.tight_layout(rect=[0, 0, 1, 0.96])  # 给标题留空间
#     plt.savefig(by_month_dir / f"{month.strftime('%Y-%m')}.jpg", dpi=300)
#     plt.show()
# =============================================================================
    
    
# %%
# =============================================================================
# by_week_dir = save_dir / 'by_week'
# by_week_dir.mkdir(parents=True, exist_ok=True)
# 
# # 检查列数
# factor_columns = factor_data.columns
# price_columns = price_data.columns
# 
# # 确保列的数量一致
# num_columns = min(len(factor_columns), len(price_columns))
# 
# # 按周分组并绘制图表
# for week_start, factor_group in factor_scaled.groupby(pd.Grouper(freq='W-MON', label='left', closed='left')):
#     if week_start < pd.Timestamp('2018-01-01 00:00:00') or week_start > pd.Timestamp('2020-02-17 00:00:00'):
#         continue
#     # 筛选价格数据的对应周
#     price_group = price_data[((price_data.index >= week_start) & (price_data.index < week_start + pd.Timedelta(weeks=1)))]
# 
#     # 筛选对应周的仓位数据
#     actual_pos_week = actual_pos[((actual_pos.index >= week_start) & (actual_pos.index < week_start + pd.Timedelta(weeks=1)))]
# 
#     # 如果某周数据为空，则跳过
#     if factor_group.empty or price_group.empty or actual_pos_week.empty:
#         continue
# 
#     # 对齐 actual_pos_week 的索引，使其与 factor_group 的索引对齐
#     actual_pos_week_aligned = actual_pos_week.reindex(factor_group.index, method='ffill')
# 
#     # 生成顺序 x 轴 (arange)
#     x = np.arange(len(factor_group))  # 顺序索引
#     x_labels = factor_group.index.strftime('%Y-%m-%d %H:%M')  # 转换成时间标签
# 
#     # 找到每天9:30的位置
#     nine_thirty_indices = [i for i, t in enumerate(factor_group.index) if t.strftime('%H:%M') == '09:30']
# 
#     # 创建一个图形，每个因子-价格列占用2个子图
#     fig, axs = plt.subplots(num_columns * 2, 1, figsize=(12, 4 * num_columns), sharex=True)
#     fig.suptitle(f"Factor and Price Data for Week Starting {week_start.strftime('%Y-%m-%d')}", fontsize=16)
# 
#     # 遍历每一列，绘制子图
#     for i in range(num_columns):
#         factor_col = factor_columns[i]
#         price_col = price_columns[i]
#         actual_pos_col = actual_pos_week.columns[i]  # 对应的仓位列
# 
#         # 上方子图：因子数据
#         axs[i * 2].plot(x, factor_group[factor_col], label=f'Factor: {factor_col}')
#         axs[i * 2].set_ylabel('Factor Value')
#         axs[i * 2].set_title(f"Factor: {factor_col}")
#         axs[i * 2].legend()
#         axs[i * 2].grid(True)
# 
#         # 绘制红色虚线和实线
#         for idx in nine_thirty_indices:
#             axs[i * 2].axvline(x=idx, color='k', linestyle='--', linewidth=1)
# 
#         # 计算仓位变化的信号
#         actual_pos_col_aligned = actual_pos_week_aligned[actual_pos_col]
#         buy_changes = actual_pos_col_aligned[(actual_pos_col_aligned.diff() == 1) & (actual_pos_col_aligned == 1)].index  # 0 -> 1 (开多)
#         sell_changes = actual_pos_col_aligned[(actual_pos_col_aligned.diff() == -1) & (actual_pos_col_aligned == 0)].index  # 1 -> 0 (平多)
#         short_changes = actual_pos_col_aligned[(actual_pos_col_aligned.diff() == -1) & (actual_pos_col_aligned == -1)].index  # 0 -> -1 (开空)
#         cover_changes = actual_pos_col_aligned[(actual_pos_col_aligned.diff() == 1) & (actual_pos_col_aligned == 0)].index  # -1 -> 0 (平空)
#         
#         # breakpoint()
#         # 绘制仓位变化的竖线
#         for change in buy_changes:
#             idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
#             axs[i * 2].axvline(x=idx, color='red', linestyle='-', linewidth=1)  # 开多
#         for change in sell_changes:
#             idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
#             axs[i * 2].axvline(x=idx, color='red', linestyle='--', linewidth=1)  # 平多
#         for change in short_changes:
#             idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
#             axs[i * 2].axvline(x=idx, color='green', linestyle='-', linewidth=1)  # 开空
#         for change in cover_changes:
#             idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
#             axs[i * 2].axvline(x=idx, color='green', linestyle='--', linewidth=1)  # 平空
# 
#         # 下方子图：价格数据
#         price_data_trimmed = price_group[price_col][:len(x)]  # 确保长度一致
#         axs[i * 2 + 1].plot(x, price_data_trimmed, label=f'Price: {price_col}', color='orange')
#         axs[i * 2 + 1].set_ylabel('Price')
#         axs[i * 2 + 1].set_title(f"Price: {price_col}")
#         axs[i * 2 + 1].legend()
#         axs[i * 2 + 1].grid(True)
# 
#         # 绘制红色虚线和实线
#         for idx in nine_thirty_indices:
#             axs[i * 2 + 1].axvline(x=idx, color='k', linestyle='--', linewidth=1)
# 
#         # 绘制仓位变化的竖线
#         for change in buy_changes:
#             idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
#             axs[i * 2 + 1].axvline(x=idx, color='red', linestyle='-', linewidth=1)  # 开多
#         for change in sell_changes:
#             idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
#             axs[i * 2 + 1].axvline(x=idx, color='red', linestyle='--', linewidth=1)  # 平多
#         for change in short_changes:
#             idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
#             axs[i * 2 + 1].axvline(x=idx, color='green', linestyle='-', linewidth=1)  # 开空
#         for change in cover_changes:
#             idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
#             axs[i * 2 + 1].axvline(x=idx, color='green', linestyle='--', linewidth=1)  # 平空
# 
#     # 设置共享的 x 轴标签
#     tick_positions = np.linspace(0, len(x)-1, num=10, dtype=int)
#     axs[-1].set_xticks(tick_positions)
#     axs[-1].set_xticklabels([x_labels[i] for i in tick_positions], rotation=45)
# 
#     # 调整布局
#     plt.tight_layout(rect=[0, 0, 1, 0.96])  # 给标题留空间
#     plt.savefig(by_week_dir / f"week_{week_start.strftime('%Y-%m-%d')}.jpg", dpi=300)
#     plt.show()
# =============================================================================


# %%
intraday_dir = save_dir / 'intraday'
intraday_dir.mkdir(parents=True, exist_ok=True)

# 提取小时和分钟，创建"日内时间"列
factor_data['intraday_time'] = factor_data.index.strftime('%H:%M')

for year, year_group in factor_data.groupby(pd.Grouper(freq='Y')):
    # 提取当年的数据
    year_group = year_group.copy()
    
    # 按“日内时间”分组，计算每分钟的因子均值
    factor_mean = year_group.groupby('intraday_time').mean()

    # 绘制日内分钟因子均值的线图
    plt.figure(figsize=(12, 6))
    for col in factor_data.columns[:-1]:  # 遍历所有因子列（排除intraday_time）
        plt.plot(factor_mean.index, factor_mean[col], label=col)

    # 设置标题和坐标轴标签
    plt.title(f'Intraday Minute-Wise Factor Mean for Year {year.year}', fontsize=14)
    plt.xlabel('Time of Day (HH:MM)')
    plt.ylabel('Average Factor Value')

    # 调整x轴：每半小时显示一个刻度
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))  # 每30个点显示一个刻度

    # 旋转X轴标签，避免重叠
    plt.xticks(rotation=45)

    # 添加网格和图例
    plt.grid()
    plt.legend()
    plt.tight_layout()

    # 显示图表
    plt.savefig(intraday_dir / f"{year.year}.jpg", dpi=300)
    plt.show()
