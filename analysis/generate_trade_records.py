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
model_name = 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18'
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
    'openthres': 0.6,
    'closethres': 0,
    }

fee = 0.00024


# %%
fut_dir = Path('/mnt/data1/futuretwap')
analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\trades')
summary_dir = analysis_dir / factor_name
summary_dir.mkdir(parents=True, exist_ok=True)


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
# 函数：根据仓位变化生成交易记录
def generate_trade_df(position_series, price_series):
    trade_info = []
    open_price = None  # 开仓价格（对应于价格序列）
    open_time = None  # 开仓时间戳（对应于时间索引）

    for i in range(1, len(position_series)):
        prev_position = position_series.iloc[i-1]
        current_position = position_series.iloc[i]
        current_price = price_series.iloc[i]  # 当前时刻的价格
        prev_price = price_series.iloc[i-1]  # 上一时刻的价格

        # 只有仓位发生变化时才记录交易
        if current_position != prev_position:
            if current_position == 1 and prev_position == 0:
                # 开多：记录开仓时间和价格
                open_price = current_price
                open_time = position_series.index[i]
            elif current_position == 0 and prev_position == 1:
                # 平多：记录平仓时间和价格
                trade_info.append({
                    'trade_type': 'long',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': current_price
                })
                open_price = None  # 清空开仓价格
                open_time = None  # 清空开仓时间
            elif current_position == -1 and prev_position == 0:
                # 开空：记录开仓时间和价格
                open_price = current_price
                open_time = position_series.index[i]
            elif current_position == 0 and prev_position == -1:
                # 平空：记录平仓时间和价格
                trade_info.append({
                    'trade_type': 'short',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': current_price
                })
                open_price = None  # 清空开仓价格
                open_time = None  # 清空开仓时间
            elif current_position == 1 and prev_position == -1:
                # 平空后开多：先平空，再开多
                trade_info.append({
                    'trade_type': 'short',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': prev_price
                })
                # 这里新的开仓会等平仓后再处理
                open_price = current_price  # 更新开仓价格
                open_time = position_series.index[i]  # 更新开仓时间
            elif current_position == -1 and prev_position == 1:
                # 平多后开空：先平多，再开空
                trade_info.append({
                    'trade_type': 'long',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': prev_price
                })
                # 这里新的开仓会等平仓后再处理
                open_price = current_price  # 更新开仓价格
                open_time = position_series.index[i]  # 更新开仓时间

    return pd.DataFrame(trade_info)

# 对每列标的单独生成交易记录
trade_dfs = {}

for col in actual_pos.columns:
    trade_df = generate_trade_df(actual_pos[col], price_data[col])
    trade_dfs[col] = trade_df
     
for col in trade_dfs:
    trade_df = trade_dfs[col]
    trade_df['direction'] = trade_df['trade_type'].apply(lambda x: 1 if x == 'long' else -1)
    trade_df['net_return'] = np.log(trade_df['close_price'] / trade_df['open_price']) * trade_df['direction'] - fee
    trade_df['holding_time'] = trade_df['close_time'] - trade_df['open_time']
    trade_dfs[col] = trade_df
    
    
# %%
import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(trade_df, symbol, summary_dir):
    # 将 'holding_time' 转换为分钟数
    trade_df['holding_time_minutes'] = trade_df['holding_time'].dt.total_seconds() / 60
    
    plt.figure(figsize=(10, 6))
    plt.scatter(trade_df['holding_time_minutes'], trade_df['net_return'], alpha=0.5, color='b')
    
    # 在 y=0 位置画一条横的虚线
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    
    # 设置图形标题与标签
    plt.title(f'Holding Time vs Net Return ({symbol})')
    plt.xlabel('Holding Time (minutes)')
    plt.ylabel('Net Return')
    plt.grid(True)
    
    # 保存图像
    plt.tight_layout()
    scatter_plot_path = summary_dir / f'{symbol}_holding_time_vs_net_return.png'
    plt.savefig(scatter_plot_path)
    # plt.close()

def plot_avg_return_by_minute(trade_df, symbol, summary_dir):
    # 提取日内时间（只取小时和分钟）
    trade_df['open_time_minute'] = trade_df['open_time'].dt.strftime('%H:%M')

    # 按分钟聚合计算平均收益
    avg_return_by_minute = trade_df.groupby('open_time_minute')['net_return'].sum()

    # 绘制条形图
    plt.figure(figsize=(10, 6))
    avg_return_by_minute.plot(kind='bar', color='skyblue')
    
    # 设置 X 轴标签为每隔30分钟一个刻度
    xticks_labels = avg_return_by_minute.index
    xticks_position = range(0, len(xticks_labels), 30)  # 每隔30个数据点
    plt.xticks(xticks_position, [xticks_labels[i] for i in xticks_position], rotation=45)
    
    plt.title(f'Average Net Return by Minute of Day ({symbol})')
    plt.xlabel('Time (Hour:Minute)')
    plt.ylabel('Average Net Return')
    plt.grid(True)
    
    # 保存图像
    plt.tight_layout()
    bar_plot_path = summary_dir / f'{symbol}_avg_net_return_by_minute.png'
    plt.savefig(bar_plot_path)
    plt.close()

def plot_avg_return_by_week_day_and_minute(trade_df, symbol, summary_dir):
    # 提取星期几和日内时间（小时:分钟）
    trade_df['week_day'] = trade_df['open_time'].dt.weekday  # 星期几：0=星期一，6=星期天
    trade_df['open_time_week_minute'] = trade_df['open_time'].dt.strftime('%H:%M')
    
    # 将星期几的数字转换为名字
    trade_df['week_day_name'] = trade_df['week_day'].map({
        0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
    })

    # 按星期几和分钟聚合计算平均收益，聚合到每五分钟
    trade_df['time_5min'] = trade_df['open_time'].dt.floor('5T').dt.strftime('%H:%M')  # 按5分钟聚合
    
    avg_return_by_5min = trade_df.groupby(['week_day', 'time_5min'])['net_return'].apply(np.nansum).unstack(fill_value=np.nan)
    
    # 重构为按周几和时间标签排序
    avg_return_by_5min = avg_return_by_5min.stack().reset_index()
    avg_return_by_5min.columns = ['Weekday', 'Time', 'Avg_Net_Return']
    
    # 拼接周几和时间为标签
    avg_return_by_5min['Label'] = avg_return_by_5min['Weekday'].map({
        0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'
    }) + ' ' + avg_return_by_5min['Time']

    # 仅保留有交易数据的时间点（去掉非交易时间）
    avg_return_by_5min = avg_return_by_5min[avg_return_by_5min['Avg_Net_Return'].notna()]

    # 选定固定的时间点：09:30 和 11:30
    fixed_labels = ['09:30', '11:25']

    # 创建一个新的时间标签列表，按星期几排序（确保按 Mon-Fri 顺序）
    labels = []
    for day in range(5):  # 星期一到星期五
        for time in fixed_labels:
            # 查找对应的时间点
            matching_row = avg_return_by_5min[(avg_return_by_5min['Weekday'] == day) & (avg_return_by_5min['Time'] == time)]
            if not matching_row.empty:
                labels.append(matching_row['Label'].iloc[0])

    # 绘制条形图
    plt.figure(figsize=(12, 8))
    plt.bar(avg_return_by_5min['Label'], avg_return_by_5min['Avg_Net_Return'], color='lightcoral')

    # 在每个 9:30 标签位置画一条竖线
    for label in labels:
        if '09:30' in label:  # 只在9:30位置画竖线
            # 获取9:30的x轴位置
            x_position = avg_return_by_5min[avg_return_by_5min['Label'] == label].index[0]
            plt.axvline(x=x_position, color='black', linestyle='--')

    # 设置 X 轴标签，只有在固定时间点（9:30和11:30）打标签，其他时间不显示标签
    plt.xticks(labels, rotation=45)
    
    # 添加图形标题与标签
    plt.title(f'Average Net Return by Weekday and Time of Day ({symbol})')
    plt.xlabel('Weekday and Time')
    plt.ylabel('Average Net Return')
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    week_day_plot_path = summary_dir / f'{symbol}_avg_net_return_by_week_day_and_minute.png'
    plt.savefig(week_day_plot_path)
    plt.close()

# 对每个标的生成并绘制所有图表
for symbol in trade_dfs.keys():
    trade_df = trade_dfs[symbol]
    
    # 1. Holding time vs net return 散点图
    plot_scatter(trade_df, symbol, summary_dir)
    
    # 2. 日内每分钟的平均收益条形图
    plot_avg_return_by_minute(trade_df, symbol, summary_dir)
    
    # 3. 按星期几+日内时间聚合的平均收益条形图
    plot_avg_return_by_week_day_and_minute(trade_df, symbol, summary_dir)
