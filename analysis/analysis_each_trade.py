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
analysis_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\results\analysis\trades')
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
    trade_dfs[col] = trade_df.dropna(subset=['net_return'])
    
    
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def generate_subtrade_df(trade_df, factor_scaled, price_data, open_threshold):
    """
    Generate a DataFrame of sub-trades by analyzing signal changes within each trade.
    
    Parameters:
    -----------
    trade_df : pandas.DataFrame
        DataFrame containing trade information with columns: open_time, close_time, direction
    factor_scaled : pandas.DataFrame
        DataFrame containing scaled factor values (signals)
    price_data : pandas.DataFrame
        DataFrame containing price data
    open_threshold : float
        Threshold value for determining if a signal is active (signal > threshold)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing subtrade information
    """
    subtrade_info = []
    
    # 遍历每笔交易
    for idx in trade_df.index:
        open_time, close_time, trade_direction = trade_df.loc[idx, ['open_time', 'close_time', 'direction']]
        trade_id = idx
        
        # 提取持仓期间的因子和价格数据
        holding_period_factor = factor_scaled.loc[open_time:close_time, trade_df.name]
        holding_period_price = price_data.loc[open_time:close_time, trade_df.name]
        
        # 创建二元信号（当信号活跃时为1，否则为0）
        signal = (holding_period_factor * trade_direction > open_threshold).astype(int)
        
        # 找出信号变化的点
        signal_diff = signal.diff().fillna(0)
        
        # 找出所有信号开始点（0->1转换）
        signal_starts_mask = signal_diff == -1
        signal_starts = signal.index[signal_starts_mask].tolist()
        
        # 找出所有信号结束点（1->0转换）
        signal_ends_mask = signal_diff == 1
        signal_ends = signal.index[signal_ends_mask].tolist()

        # 根据交易逻辑，交易开始时一定有信号(1)，结束时一定没有信号(0)

        # 确保最后一个信号段的结束点是交易结束时间
        # 如果信号开始点比结束点多，说明有一个信号没有对应的结束点
        # 这种情况下，使用交易结束时间作为最后一个信号的结束点
        if len(signal_starts) > len(signal_ends):
            signal_ends.append(close_time)
            
        # 按时间排序信号开始和结束点
        signal_starts.sort()
        signal_ends.sort()
            
        # 确保信号开始和结束的点数匹配
        min_pairs = min(len(signal_starts), len(signal_ends))
        signal_starts = signal_starts[:min_pairs]
        signal_ends = signal_ends[:min_pairs]
        
        # 创建子交易记录
        for i in range(len(signal_starts)):
            start_time = signal_starts[i]
            end_time = signal_ends[i]
            
            # 忽略相同的开始和结束时间
            if start_time == end_time:
                continue
                
            # 计算价格变化和持续时间
            start_price = price_data.loc[start_time, trade_df.name]
            end_price = price_data.loc[end_time, trade_df.name]
            
            # 判断出场方式
            # 如果结束时间是交易的关闭时间，则是信号平仓，标记为0
            # 否则是因为下一个信号触发而被截断，标记为1
            exit_type = 0 if end_time == close_time else 1
            # breakpoint()
            
            subtrade_info.append({
                'trade_id': trade_id,
                'subtrade_id': f"{trade_id}_{i}",
                'subtrade_idx': i,  # 添加整数类型的子交易索引
                'start_time': start_time,
                'end_time': end_time,
                'start_price': start_price,
                'end_price': end_price,
                'duration': end_time - start_time,
                'price_change': (end_price - start_price) * trade_direction,
                'pct_change': ((end_price / start_price) - 1) * trade_direction,
                'direction': trade_direction,
                'signal_strength': holding_period_factor.loc[start_time:end_time].mean() * trade_direction,
                'exit_type': exit_type  # 添加出场方式: 0-信号平仓, 1-被下一信号截断
            })
    
    return pd.DataFrame(subtrade_info)

def analyze_all_subtrades(trade_dfs, factor_scaled, price_data, open_threshold):
    """
    Analyze subtrades for all futures contracts.
    
    Parameters:
    -----------
    trade_dfs : dict
        Dictionary mapping futures names to trade DataFrames
    factor_scaled : pandas.DataFrame
        DataFrame containing scaled factor values
    price_data : pandas.DataFrame
        DataFrame containing price data
    open_threshold : float
        Threshold value for determining if a signal is active
        
    Returns:
    --------
    dict
        Dictionary mapping futures names to subtrade DataFrames
    """
    subtrade_dfs = {}
    
    for fut in trade_dfs:
        trade_df = trade_dfs[fut].dropna(subset=['net_return'])
        
        # 添加期货符号作为列名，以便在generate_subtrade_df函数中识别
        trade_df.name = fut
        
        subtrade_df = generate_subtrade_df(trade_df, factor_scaled, price_data, open_threshold)
        
        # 添加期货符号作为列
        subtrade_df['symbol'] = fut
        
        subtrade_dfs[fut] = subtrade_df
        
    return subtrade_dfs

def analyze_subtrade_statistics(subtrade_dfs):
    """
    Provide statistics on the subtrades.
    
    Parameters:
    -----------
    subtrade_dfs : dict
        Dictionary mapping futures names to subtrade DataFrames
        
    Returns:
    --------
    dict
        Dictionary containing various statistics for subtrades
    """
    # 合并所有子交易DataFrame
    if not subtrade_dfs:
        return {'error': 'No subtrades found'}
    
    all_subtrades = pd.concat(subtrade_dfs.values(), ignore_index=True)
    
    if len(all_subtrades) == 0:
        return {'error': 'No subtrades found after concatenation'}
    
    # 基本统计
    stats = {
        'total_subtrades': len(all_subtrades),
        'avg_duration': all_subtrades['duration'].mean(),
        'median_duration': all_subtrades['duration'].median(),
        'avg_price_change': all_subtrades['price_change'].mean(),
        'avg_pct_change': all_subtrades['pct_change'].mean(),
        'positive_subtrades': (all_subtrades['price_change'] > 0).sum(),
        'negative_subtrades': (all_subtrades['price_change'] <= 0).sum(),
        'positive_pct': (all_subtrades['price_change'] > 0).sum() / len(all_subtrades) * 100,
    }
    
    # 按符号分组统计
    symbol_stats = all_subtrades.groupby('symbol').agg({
        'subtrade_id': 'count',
        'duration': ['mean', 'median'],
        'price_change': ['mean', 'sum'],
        'pct_change': ['mean', 'sum']
    })
    
    # 按交易ID分组统计
    trade_stats = all_subtrades.groupby('trade_id').agg({
        'subtrade_id': 'count',
        'duration': 'sum',
        'price_change': 'sum',
        'pct_change': 'sum'
    })
    
    # 按子交易索引分组统计（查看不同序号的子交易表现）
    subtrade_idx_stats = all_subtrades.groupby('subtrade_idx').agg({
        'subtrade_id': 'count',
        'duration': ['mean', 'median'],
        'price_change': ['mean', 'sum'],
        'pct_change': ['mean', 'sum']
    })
    
    # 按出场方式分组统计
    exit_type_stats = all_subtrades.groupby('exit_type').agg({
        'subtrade_id': 'count',
        'duration': ['mean', 'median'],
        'price_change': ['mean', 'sum'],
        'pct_change': ['mean', 'sum']
    })
    
    return {
        'overall': stats,
        'by_symbol': symbol_stats,
        'by_trade': trade_stats,
        'by_subtrade_idx': subtrade_idx_stats,  # 添加按子交易索引分组的统计
        'by_exit_type': exit_type_stats,  # 添加按出场方式分组的统计
        'raw_data': all_subtrades
    }

# 使用示例
subtrade_dfs = analyze_all_subtrades(trade_dfs, factor_scaled, price_data, trade_rule_param['openthres'])
stats = analyze_subtrade_statistics(subtrade_dfs)


# %%
for fut in subtrade_dfs:
    subtrade_df = subtrade_dfs[fut]
    trade_df = trade_dfs[fut]
    
    for idx in subtrade_df.index:
        target_trade_id = subtrade_df.loc[idx, 'trade_id']
        subtrade_df.loc[idx, ['open_time', 'close_time']] = trade_df.loc[target_trade_id, ['open_time', 'close_time']]
        subtrade_df.loc[idx, ['open_price', 'close_price']] = trade_df.loc[target_trade_id, ['open_price', 'close_price']]
    subtrade_dfs[fut] = subtrade_df
    
    
# %%
for fut in subtrade_dfs:
    subtrade_df = subtrade_dfs[fut]
    for idx in subtrade_df.index:
        start_time, end_time, trade_direction = subtrade_df.loc[idx, ['start_time', 'end_time', 'direction']]
        holding_period_factor = factor_scaled.loc[start_time:end_time, fut]
        holding_period_price = price_data.loc[start_time:end_time, fut]
        break
    
    
# %%
# =============================================================================
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# # Assuming subtrade_dfs, factor_scaled, and price_data are already定义
# 
# # Create dictionaries to store the returns for long and short trades
# long_returns_dict = {}  # Key: minute index, Value: list of returns
# short_returns_dict = {}  # Key: minute index, Value: list of returns
# 
# # 追踪每个时间点的活跃交易数量（用于收益计算）
# long_active_counts = {}  # Key: minute index, Value: number of active trades
# short_active_counts = {}  # Key: minute index, Value: number of active trades
# 
# # 追踪实际活跃的交易数量（用于绘图）
# long_real_active_counts = {}  # Key: minute index, Value: number of actually active trades
# short_real_active_counts = {}  # Key: minute index, Value: number of actually active trades
# 
# # Iterate through each futures contract and its corresponding subtrades
# for fut in subtrade_dfs:
#     subtrade_df = subtrade_dfs[fut]
#     
#     # Process each subtrade
#     for idx in subtrade_df.index:
#         start_time, end_time, trade_direction = subtrade_df.loc[idx, ['start_time', 'end_time', 'direction']]
#         
#         # Extract holding period prices for this subtrade
#         holding_period_price = price_data.loc[start_time:end_time, fut]
#         
#         # Calculate per-minute returns
#         minute_returns = holding_period_price.pct_change().fillna(0)
#         
#         # 确定使用哪个字典（多头或空头）
#         if trade_direction > 0:  # Long position
#             returns_dict = long_returns_dict
#             active_counts = long_active_counts
#             real_active_counts = long_real_active_counts
#         else:  # Short position
#             returns_dict = short_returns_dict
#             active_counts = short_active_counts
#             real_active_counts = short_real_active_counts
#             minute_returns = -minute_returns  # Invert returns for short positions
#         
#         # 计算这个交易的持续时间（分钟数）
#         trade_duration = len(minute_returns)
#         
#         # Store returns at each minute index
#         for i, (time, ret) in enumerate(minute_returns.items()):
#             if i == 0:  # Skip the first entry which is NaN or 0 after fillna
#                 continue
#                 
#             if i not in returns_dict:
#                 returns_dict[i] = []
#             
#             returns_dict[i].append(ret)
#             
#             # 跟踪此分钟的活跃交易数量（用于收益计算）
#             if i not in active_counts:
#                 active_counts[i] = 0
#             active_counts[i] += 1
#             
#             # 同时跟踪实际活跃的交易数量（用于绘图）
#             if i not in real_active_counts:
#                 real_active_counts[i] = 0
#             real_active_counts[i] += 1
# 
# # 获取所有交易中的最大分钟数
# all_minutes = set(list(long_returns_dict.keys()) + list(short_returns_dict.keys()))
# max_minutes = max(all_minutes) if all_minutes else 0
# 
# # 填充每个分钟索引的所有交易收益
# # 对于已结束的交易，将其在后续分钟的收益填充为0
# for minute in range(1, max_minutes + 1):
#     # 处理多头交易
#     if minute in long_returns_dict:
#         active_count = long_active_counts.get(minute, 0)
#         current_trades_count = len(long_returns_dict[minute])
#         
#         # 计算已结束交易的数量
#         if minute > 1 and minute - 1 in long_returns_dict:
#             # 上一分钟的交易数 - 当前分钟活跃交易数 = 已结束交易数
#             ended_trades = len(long_returns_dict[minute - 1]) - active_count
#             # 为已结束的交易添加0收益
#             long_returns_dict[minute].extend([0] * ended_trades)
#     
#     # 处理空头交易
#     if minute in short_returns_dict:
#         active_count = short_active_counts.get(minute, 0)
#         current_trades_count = len(short_returns_dict[minute])
#         
#         # 计算已结束交易的数量
#         if minute > 1 and minute - 1 in short_returns_dict:
#             # 上一分钟的交易数 - 当前分钟活跃交易数 = 已结束交易数
#             ended_trades = len(short_returns_dict[minute - 1]) - active_count
#             # 为已结束的交易添加0收益
#             short_returns_dict[minute].extend([0] * ended_trades)
# 
# # Calculate average returns at each minute
# long_avg_returns = {minute: np.mean(returns) for minute, returns in long_returns_dict.items()}
# short_avg_returns = {minute: np.mean(returns) for minute, returns in short_returns_dict.items()}
# 
# # Convert to DataFrames for easier handling
# long_df = pd.DataFrame(list(long_avg_returns.items()), columns=['minute', 'avg_return']).sort_values('minute')
# short_df = pd.DataFrame(list(short_avg_returns.items()), columns=['minute', 'avg_return']).sort_values('minute')
# 
# # Calculate cumulative returns
# long_df['cum_return'] = long_df['avg_return'].cumsum()
# short_df['cum_return'] = short_df['avg_return'].cumsum()
# 
# # Preparation for combined visualization
# max_minutes = max(max(long_df['minute']), max(short_df['minute']))
# minute_range = range(1, int(max_minutes) + 1)
# 
# # Create DataFrames with all minutes for smooth plotting
# full_long_df = pd.DataFrame({'minute': minute_range})
# full_short_df = pd.DataFrame({'minute': minute_range})
# 
# # Merge with actual data
# full_long_df = full_long_df.merge(long_df, on='minute', how='left')
# full_short_df = full_short_df.merge(short_df, on='minute', how='left')
# 
# # Calculate cumulative returns, handling NaN values
# full_long_df['cum_return'] = full_long_df['avg_return'].fillna(0).cumsum()
# full_short_df['cum_return'] = full_short_df['avg_return'].fillna(0).cumsum()
# 
# # Calculate confidence intervals
# long_std = {minute: np.std(returns) for minute, returns in long_returns_dict.items()}
# short_std = {minute: np.std(returns) for minute, returns in short_returns_dict.items()}
# 
# long_std_df = pd.DataFrame(list(long_std.items()), columns=['minute', 'std'])
# short_std_df = pd.DataFrame(list(short_std.items()), columns=['minute', 'std'])
# 
# full_long_df = full_long_df.merge(long_std_df, on='minute', how='left')
# full_short_df = full_short_df.merge(short_std_df, on='minute', how='left')
# 
# # Correctly calculate the confidence intervals for cumulative returns
# # For long positions
# cumulative_variance_long = np.zeros(len(full_long_df))
# for i, minute in enumerate(full_long_df['minute']):
#     # Calculate variance for all minutes up to this point
#     valid_minutes = full_long_df['minute'].iloc[:i+1]
#     valid_std = full_long_df['std'].iloc[:i+1].fillna(0)
#     
#     # Get sample sizes for each minute
#     sample_sizes = np.array([len(long_returns_dict.get(int(m), [])) for m in valid_minutes])
#     
#     # Filter out minutes with zero samples
#     valid_indices = sample_sizes > 0
#     valid_variances = (valid_std[valid_indices]**2).values
#     valid_sample_sizes = sample_sizes[valid_indices]
#     
#     if len(valid_variances) > 0:
#         # Sum of variances divided by sample sizes for standard error
#         cumulative_variance_long[i] = np.sum(valid_variances / valid_sample_sizes)
# 
# # Standard error is sqrt of cumulative variance
# cumulative_se_long = np.sqrt(cumulative_variance_long)
# 
# # For short positions
# cumulative_variance_short = np.zeros(len(full_short_df))
# for i, minute in enumerate(full_short_df['minute']):
#     valid_minutes = full_short_df['minute'].iloc[:i+1]
#     valid_std = full_short_df['std'].iloc[:i+1].fillna(0)
#     
#     sample_sizes = np.array([len(short_returns_dict.get(int(m), [])) for m in valid_minutes])
#     
#     valid_indices = sample_sizes > 0
#     valid_variances = (valid_std[valid_indices]**2).values
#     valid_sample_sizes = sample_sizes[valid_indices]
#     
#     if len(valid_variances) > 0:
#         cumulative_variance_short[i] = np.sum(valid_variances / valid_sample_sizes)
# 
# cumulative_se_short = np.sqrt(cumulative_variance_short)
# 
# # Calculate 95% confidence intervals
# full_long_df['upper'] = full_long_df['cum_return'] + 1.96 * cumulative_se_long
# full_long_df['lower'] = full_long_df['cum_return'] - 1.96 * cumulative_se_long
# 
# full_short_df['upper'] = full_short_df['cum_return'] + 1.96 * cumulative_se_short
# full_short_df['lower'] = full_short_df['cum_return'] - 1.96 * cumulative_se_short
# 
# # Prepare sample count data for plotting (using real active counts instead of all returns)
# long_counts = long_real_active_counts
# short_counts = short_real_active_counts
# 
# long_count_df = pd.DataFrame(list(long_counts.items()), columns=['minute', 'count']).sort_values('minute')
# short_count_df = pd.DataFrame(list(short_counts.items()), columns=['minute', 'count']).sort_values('minute')
# 
# # Create the combined figure
# plt.figure(figsize=(15, 8))
# 
# # Plot cumulative returns with confidence intervals
# plt.plot(full_long_df['minute'], full_long_df['cum_return'] * 100, 'b-', linewidth=2, label='Long Positions')
# plt.fill_between(full_long_df['minute'], full_long_df['lower'] * 100, full_long_df['upper'] * 100, color='b', alpha=0.2)
# 
# plt.plot(full_short_df['minute'], full_short_df['cum_return'] * 100, 'r-', linewidth=2, label='Short Positions')
# plt.fill_between(full_short_df['minute'], full_short_df['lower'] * 100, full_short_df['upper'] * 100, color='r', alpha=0.2)
# 
# plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
# plt.xlabel('Holding Period (minutes)', fontsize=12)
# plt.ylabel('Cumulative Return (%)', fontsize=12)
# plt.title('Average Cumulative Returns by Trade Direction with Confidence Intervals', fontsize=14)
# plt.legend(loc='upper left', fontsize=10)
# plt.grid(True, alpha=0.3)
# 
# # Create twin axis for sample size
# ax2 = plt.twinx()
# ax2.plot(long_count_df['minute'], long_count_df['count'], 'b--', alpha=0.5, label='Long Count')
# ax2.plot(short_count_df['minute'], short_count_df['count'], 'r--', alpha=0.5, label='Short Count')
# ax2.set_ylabel('Number of Samples', fontsize=12)
# ax2.legend(loc='upper right', fontsize=10)
# 
# # Add annotations
# plt.annotate('Confidence interval: 95%', xy=(0.02, 0.03), xycoords='axes fraction', fontsize=10)
# plt.annotate('Note: Returns for ended trades are counted as 0', xy=(0.02, 0.06), xycoords='axes fraction', fontsize=10)
# plt.annotate('Sample count shows actual active trades only', xy=(0.02, 0.09), xycoords='axes fraction', fontsize=10)
# 
# plt.tight_layout()
# plt.show()
# 
# # 获取所有交易的真实样本数量（包括已结束的）
# total_long_samples = {minute: len(returns) for minute, returns in long_returns_dict.items()}
# total_short_samples = {minute: len(returns) for minute, returns in short_returns_dict.items()}
# 
# # Create a summary table
# summary_df = pd.DataFrame({
#     'Direction': ['Long', 'Short'],
#     'Max Holding Period (min)': [max(long_df['minute']), max(short_df['minute'])],
#     'Max Return (%)': [max(long_df['cum_return']) * 100, max(short_df['cum_return']) * 100],
#     'Min Return (%)': [min(long_df['cum_return']) * 100, min(short_df['cum_return']) * 100],
#     'End Return (%)': [long_df.iloc[-1]['cum_return'] * 100, short_df.iloc[-1]['cum_return'] * 100],
#     'Avg Active Trades Per Minute': [np.mean(list(long_counts.values())), np.mean(list(short_counts.values()))],
#     'Avg Total Samples Per Minute': [np.mean(list(total_long_samples.values())), np.mean(list(total_short_samples.values()))]
# })
# 
# print(summary_df)
# 
# =============================================================================



# %%
# =============================================================================
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# 
# def calculate_subtrade_returns(subtrade_dfs, factor_scaled, price_data):
#     """
#     Calculate maximum and minimum returns for each subtrade, considering trade direction.
#     
#     Parameters:
#     - subtrade_dfs: Dictionary with futures as keys and subtrade DataFrames as values
#     - factor_scaled: DataFrame with factor values
#     - price_data: DataFrame with price data
#     
#     Returns:
#     - DataFrame with subtrade_id, max_return, min_return for each subtrade
#     """
#     results = []
#     
#     for fut in subtrade_dfs:
#         subtrade_df = subtrade_dfs[fut]
#         
#         for idx in subtrade_df.index:
#             subtrade_id = subtrade_df.loc[idx, 'subtrade_id']
#             start_time = subtrade_df.loc[idx, 'start_time']
#             end_time = subtrade_df.loc[idx, 'end_time']
#             trade_direction = subtrade_df.loc[idx, 'direction']
#             
#             # Get price data for the holding period
#             holding_period_price = price_data.loc[start_time:end_time, fut]
#             
#             # Skip if no price data available
#             if len(holding_period_price) <= 1:
#                 continue
#                 
#             # Calculate entry price (first price in the series)
#             entry_price = holding_period_price.iloc[0]
#             
#             # Calculate percentage changes for each price point relative to entry
#             price_returns = (holding_period_price / entry_price - 1) * 100
#             
#             # Calculate final return (end to end percentage change)
#             final_price = holding_period_price.iloc[-1]
#             final_return = ((final_price / entry_price - 1) * 100)
#             
#             # Adjust returns based on trade direction
#             if trade_direction == -1:  # Short position
#                 adjusted_returns = -price_returns  # Negate returns for short positions
#                 final_return = -final_return
#             else:  # Long position (trade_direction == 1)
#                 adjusted_returns = price_returns
#                 
#             # Find max and min returns after considering direction
#             max_return = adjusted_returns.max()
#             min_return = adjusted_returns.min()
#             
#             results.append({
#                 'subtrade_id': subtrade_id,
#                 'direction': trade_direction,
#                 'max_return': max_return,
#                 'min_return': min_return,
#                 'final_return': final_return
#             })
#     
#     return pd.DataFrame(results)
# 
# def visualize_return_relationships(results_df):
#     """
#     Create visualizations showing relationships between max/min returns and final returns.
#     
#     Parameters:
#     - results_df: DataFrame with subtrade returns data
#     
#     Returns:
#     - None (displays visualizations)
#     """
#    # Set style
#     sns.set(style="whitegrid")
#     
#     # Create figure with 3 subplots (2x2 grid, using 3 positions)
#     fig, axes = plt.subplots(2, 2, figsize=(16, 14))
#     
#     # 1. Max Return vs Final Return (top left)
#     sns.scatterplot(
#         x='final_return', 
#         y='max_return',
#         hue='direction',
#         palette={1: 'green', -1: 'red'},
#         data=results_df,
#         ax=axes[0, 0],
#         alpha=0.7
#     )
#     axes[0, 0].set_title('Maximum Return vs Final Return')
#     axes[0, 0].set_xlabel('Final Return (%)')
#     axes[0, 0].set_ylabel('Maximum Return (%)')
#     axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
#     axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.3)
#     
#     # 2. Min Return vs Final Return (top right)
#     sns.scatterplot(
#         x='final_return', 
#         y='min_return',
#         hue='direction',
#         palette={1: 'green', -1: 'red'},
#         data=results_df,
#         ax=axes[0, 1],
#         alpha=0.7
#     )
#     axes[0, 1].set_title('Minimum Return vs Final Return')
#     axes[0, 1].set_xlabel('Final Return (%)')
#     axes[0, 1].set_ylabel('Minimum Return (%)')
#     axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
#     axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.3)
#     
#     # 3. Max Return vs Min Return (bottom span)
#     scatter_max_min = sns.scatterplot(
#         x='min_return', 
#         y='max_return',
#         hue='direction',
#         palette={1: 'green', -1: 'red'},
#         data=results_df,
#         ax=axes[1, 0],
#         alpha=0.7
#     )
#     axes[1, 0].set_title('Maximum Return vs Minimum Return')
#     axes[1, 0].set_xlabel('Minimum Return (%)')
#     axes[1, 0].set_ylabel('Maximum Return (%)')
#     axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
#     axes[1, 0].axvline(x=0, color='black', linestyle='--', alpha=0.3)
#     
#     # Remove the unused subplot
#     fig.delaxes(axes[1, 1])
#     
#     plt.tight_layout()
#     plt.show()
#  
#     
# # Example usage:
# # 1. Calculate returns
# results_df = calculate_subtrade_returns(subtrade_dfs, factor_scaled, price_data)
# # 
# # 2. Visualize relationships
# visualize_return_relationships(results_df)
# =============================================================================


# %%
# =============================================================================
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from typing import Dict, List, Tuple
# 
# def calculate_returns_with_sl_tp(
#     subtrade_dfs: Dict[str, pd.DataFrame],
#     price_data: pd.DataFrame,
#     take_profit_levels: List[float] = [0.001, 0.002, 0.003, 0.004, 0.005],
#     stop_loss_levels: List[float] = [0.001, 0.002, 0.003, 0.004, 0.005],
#     no_tp: bool = False,
#     no_sl: bool = False
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Calculate returns with various stop-loss and take-profit levels.
#     
#     Parameters:
#     -----------
#     subtrade_dfs : Dict[str, pd.DataFrame]
#         Dictionary of DataFrames containing subtrade data for each futures contract
#     price_data : pd.DataFrame
#         DataFrame containing price data for each futures contract
#     take_profit_levels : List[float]
#         List of take-profit levels to test (as decimal percentages)
#     stop_loss_levels : List[float]
#         List of stop-loss levels to test (as decimal percentages)
#     no_tp : bool
#         If True, don't apply take-profit rules
#     no_sl : bool
#         If True, don't apply stop-loss rules
#         
#     Returns:
#     --------
#     result_df : pd.DataFrame
#         DataFrame containing results for each combination of stop-loss and take-profit
#     heatmap_df : pd.DataFrame
#         DataFrame formatted for creating a heatmap of Sharpe ratios
#     """
#     # Initialize results DataFrame
#     results = []
#     
#     # If no_tp is True, use only a dummy value (None) for take_profit_levels
#     tp_levels = [None] if no_tp else take_profit_levels
#     
#     # If no_sl is True, use only a dummy value (None) for stop_loss_levels
#     sl_levels = [None] if no_sl else stop_loss_levels
#     
#     # Iterate through all combinations of stop-loss and take-profit levels
#     for tp in tp_levels:
#         for sl in sl_levels:
#             # Skip if both are None (no SL, no TP) as this would be equivalent to original results
#             if tp is None and sl is None:
#                 continue
#                 
#             # Initialize a list to store the returns for each subtrade
#             all_returns = []
#             
#             # Process each futures contract
#             for fut in subtrade_dfs:
#                 subtrade_df = subtrade_dfs[fut].copy()
#                 
#                 # Process each subtrade
#                 for idx in subtrade_df.index:
#                     start_time, end_time, trade_direction = subtrade_df.loc[idx, ['start_time', 'end_time', 'direction']]
#                     
#                     # Get the price series during the holding period
#                     holding_period_price = price_data.loc[start_time:end_time, fut]
#                     
#                     # Skip if there's no price data
#                     if holding_period_price.empty:
#                         continue
#                     
#                     # Get the entry price (first price in the holding period)
#                     entry_price = holding_period_price.iloc[0]
#                     
#                     # Initialize exit price to the last price in the holding period (in case no SL/TP is triggered)
#                     exit_price = holding_period_price.iloc[-1]
#                     exit_time = end_time
#                     
#                     # Flag to check if a stop was triggered
#                     stop_triggered = False
#                     
#                     # Apply stop-loss and take-profit logic to each price in the holding period
#                     for time, price in holding_period_price.items():
#                         # Skip the first price (entry price)
#                         if time == start_time:
#                             continue
#                         
#                         # Calculate the current return
#                         current_return = (price - entry_price) / entry_price
#                         
#                         # For short positions, multiply the return by -1
#                         if trade_direction == -1:
#                             current_return = -current_return
#                         
#                         # Check if take-profit is triggered
#                         if tp is not None and current_return >= tp:
#                             exit_price = price
#                             exit_time = time
#                             stop_triggered = True
#                             break
#                         
#                         # Check if stop-loss is triggered
#                         if sl is not None and current_return <= -sl:
#                             exit_price = price
#                             exit_time = time
#                             stop_triggered = True
#                             break
#                     
#                     # Calculate the return for this subtrade
#                     if trade_direction == 1:  # Long position
#                         pct_return = (exit_price - entry_price) / entry_price
#                     else:  # Short position
#                         pct_return = (entry_price - exit_price) / entry_price
#                     
#                     # Store the return along with metadata
#                     all_returns.append({
#                         'trade_id': subtrade_df.loc[idx, 'trade_id'],
#                         'subtrade_id': subtrade_df.loc[idx, 'subtrade_id'],
#                         'future': fut,
#                         'direction': trade_direction,
#                         'start_time': start_time,
#                         'end_time': exit_time,
#                         'entry_price': entry_price,
#                         'exit_price': exit_price,
#                         'pct_change': pct_return,
#                         'stop_triggered': stop_triggered
#                     })
#             
#             # Convert the returns to a DataFrame
#             returns_df = pd.DataFrame(all_returns)
#             
#             # Calculate the Sharpe ratio
#             if not returns_df.empty:
#                 mean_return = returns_df['pct_change'].mean()
#                 std_return = returns_df['pct_change'].std()
#                 sharpe_ratio = mean_return / std_return if std_return > 0 else 0
#                 
#                 # Calculate other metrics
#                 win_rate = (returns_df['pct_change'] > 0).mean()
#                 avg_win = returns_df.loc[returns_df['pct_change'] > 0, 'pct_change'].mean() if any(returns_df['pct_change'] > 0) else 0
#                 avg_loss = returns_df.loc[returns_df['pct_change'] < 0, 'pct_change'].mean() if any(returns_df['pct_change'] < 0) else 0
#                 profit_factor = abs(avg_win * win_rate / (avg_loss * (1 - win_rate))) if avg_loss != 0 and win_rate < 1 else float('inf')
#                 
#                 # Store the results
#                 results.append({
#                     'take_profit': tp if tp is not None else 'None',
#                     'stop_loss': sl if sl is not None else 'None',
#                     'sharpe_ratio': sharpe_ratio,
#                     'mean_return': mean_return,
#                     'std_return': std_return,
#                     'win_rate': win_rate,
#                     'avg_win': avg_win,
#                     'avg_loss': avg_loss,
#                     'profit_factor': profit_factor,
#                     'stop_triggered_rate': returns_df['stop_triggered'].mean()
#                 })
#     
#     # Convert results to DataFrame
#     result_df = pd.DataFrame(results)
#     
#     # Create a DataFrame formatted for a heatmap
#     if no_tp or no_sl:
#         heatmap_df = result_df.copy()
#     else:
#         heatmap_df = result_df.pivot_table(
#             index='stop_loss', 
#             columns='take_profit', 
#             values='sharpe_ratio'
#         )
#     
#     return result_df, heatmap_df
# 
# def plot_sharpe_heatmap(heatmap_df: pd.DataFrame, title: str = "Sharpe Ratio Heatmap") -> plt.Figure:
#     """
#     Plot a heatmap of Sharpe ratios for different combinations of stop-loss and take-profit levels.
#     
#     Parameters:
#     -----------
#     heatmap_df : pd.DataFrame
#         DataFrame containing Sharpe ratios for different combinations of stop-loss and take-profit
#     title : str
#         Title for the heatmap
#         
#     Returns:
#     --------
#     fig : plt.Figure
#         The matplotlib figure containing the heatmap
#     """
#     plt.figure(figsize=(12, 10))
#     
#     # Create the heatmap
#     ax = sns.heatmap(
#         heatmap_df,
#         annot=True,
#         cmap='viridis',
#         fmt='.3f',
#         linewidths=.5
#     )
#     
#     # Set the title and labels
#     plt.title(title, fontsize=16)
#     plt.xlabel('Take Profit Level', fontsize=14)
#     plt.ylabel('Stop Loss Level', fontsize=14)
#     
#     # Return the figure
#     return plt.gcf()
# 
# # Example usage:
# # Define parameters
# take_profit_levels = [None, 0.0025, 0.005, 0.01, 0.0125, 0.015, 0.02]
# stop_loss_levels = [None, 0.0025, 0.005, 0.01, 0.0125, 0.015, 0.02]
# 
# # Calculate returns with stop-loss and take-profit
# result_df, heatmap_df = calculate_returns_with_sl_tp(
#     subtrade_dfs=subtrade_dfs,
#     price_data=price_data,
#     take_profit_levels=take_profit_levels,
#     stop_loss_levels=stop_loss_levels
# )
# 
# # Plot the heatmap
# fig = plot_sharpe_heatmap(heatmap_df)
# plt.show()
# 
# # To run with no take-profit (only stop-loss)
# result_df_no_tp, heatmap_df_no_tp = calculate_returns_with_sl_tp(
#     subtrade_dfs=subtrade_dfs,
#     price_data=price_data,
#     stop_loss_levels=stop_loss_levels,
#     no_tp=True
# )
# 
# # To run with no stop-loss (only take-profit)
# result_df_no_sl, heatmap_df_no_sl = calculate_returns_with_sl_tp(
#     subtrade_dfs=subtrade_dfs,
#     price_data=price_data,
#     take_profit_levels=take_profit_levels,
#     no_sl=True
# )
# 
# =============================================================================
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator, AutoDateFormatter
import matplotlib.dates as mdates

# Function to calculate maximum drawdown for a single subtrade
def calculate_max_drawdown(prices, direction):
    """
    Calculate maximum drawdown during a trading period
    
    Parameters:
    prices (Series): Price series for the holding period
    direction: Direction indicator (1 for long, -1 for short)
    
    Returns:
    float: Maximum drawdown percentage
    """
    if direction == 1:
        # For long trades, we care about downward moves from entry
        entry_price = prices.iloc[0]
        min_price = prices.min()
        max_drawdown_pct = (min_price - entry_price) / entry_price * 100
        if max_drawdown_pct > 0:  # If prices only went up, there's no drawdown
            max_drawdown_pct = 0
    else:  # 'short'
        # For short trades, we care about upward moves from entry
        entry_price = prices.iloc[0]
        max_price = prices.max()
        max_drawdown_pct = (entry_price - max_price) / entry_price * 100
        if max_drawdown_pct > 0:  # If prices only went down, there's no drawdown
            max_drawdown_pct = 0
    
    return max_drawdown_pct

# Improved function to plot subtrade with significant drawdown
def plot_subtrade(factor, price, start_time, end_time, trade_id, subtrade_id, direction, max_drawdown, symbol):
    """
    Create a plot showing factor and price for a subtrade with significant drawdown
    
    Parameters:
    factor (Series): Factor values during the subtrade period
    price (Series): Price values during the subtrade period
    start_time: Start time of the subtrade
    end_time: End time of the subtrade
    trade_id: ID of the parent trade
    subtrade_id: ID of the subtrade
    direction: Trade direction (1 for long, -1 for short)
    max_drawdown: Maximum drawdown percentage
    symbol: Futures contract symbol
    """
    # Clean data - remove NaN values and align timestamps
    factor = factor.dropna()
    price = price.dropna()
    
    # Get common timestamps to ensure alignment
    common_timestamps = factor.index.intersection(price.index)
    factor = factor.loc[common_timestamps]
    price = price.loc[common_timestamps]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Convert index to datetime if it's not already
    if not isinstance(factor.index, pd.DatetimeIndex):
        factor.index = pd.to_datetime(factor.index)
    
    # Direction display format
    direction_str = "Long" if direction == 1 else "Short"
    
    # Plot factor - use simple line without connecting through missing data
    # Convert data to simple sequential plot
    x_seq1 = np.arange(len(factor))
    ax1.plot(x_seq1, factor.values, 'b-', linewidth=2)
    ax1.set_ylabel('Factor Value', fontsize=12)
    ax1.set_title(f'Future: {symbol} - Trade {trade_id}, Subtrade {subtrade_id} - Direction: {direction_str}, Max Drawdown: {max_drawdown:.2f}%', 
                  fontsize=14, fontweight='bold', pad=40)
    ax1.grid(True, alpha=0.3)
    
    # Plot price - also use sequential x-axis
    x_seq2 = np.arange(len(price))
    ax2.plot(x_seq2, price.values, 'r-', linewidth=2)
    ax2.set_ylabel('Price', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Mark entry and exit points
    ax2.plot(x_seq2[0], price.iloc[0], 'go', markersize=8, label='Entry')
    ax2.plot(x_seq2[-1], price.iloc[-1], 'ro', markersize=8, label='Exit')
    
    # Calculate and mark the point of maximum drawdown
    if direction == 1:  # Long
        worst_idx = price.argmin()
    else:  # Short
        worst_idx = price.argmax()
    
    ax2.plot(x_seq2[worst_idx], price.iloc[worst_idx], 'mo', markersize=8, label='Max Drawdown')
    ax2.legend(loc='best')
    
    # Create custom x-tick positions and labels
    # For longer sequences, select a reasonable number of ticks
    num_ticks = min(10, len(factor))
    tick_positions = np.linspace(0, len(factor)-1, num_ticks, dtype=int)
    tick_labels = [factor.index[pos].strftime('%Y-%m-%d %H:%M') for pos in tick_positions]
    
    # Set the tick positions and labels
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Add date markers
    # Get unique dates in the dataset
    dates = [idx.date() for idx in factor.index]
    unique_dates = sorted(set(dates))
    
    # Find the first index for each date
    date_change_indices = []
    current_date = None
    for i, idx in enumerate(factor.index):
        if idx.date() != current_date:
            current_date = idx.date()
            date_change_indices.append((i, current_date))
    
    # Add vertical lines at date changes
    for idx, date in date_change_indices[1:]:  # Skip the first one as it's the start
        ax1.axvline(x=idx, color='gray', linestyle='--', alpha=0.7)
        ax2.axvline(x=idx, color='gray', linestyle='--', alpha=0.7)
        
        # Add date label at the top of each line
        ax1.annotate(date.strftime('%Y-%m-%d'), 
                  xy=(idx, ax1.get_ylim()[1]),
                  xytext=(0, 5),  # Offset text to be above line
                  textcoords='offset points',
                  ha='center', va='bottom',
                  fontsize=9, fontweight='bold',
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Add futures symbol watermark
    fig.text(0.5, 0.5, symbol, fontsize=60, color='gray', alpha=0.15,
             ha='center', va='center', rotation=30)
    
    # Set x-axis label
    ax2.set_xlabel('Time', fontsize=12)
    
    plt.tight_layout()
    
    return fig

# Main analysis function
def analyze_subtrades(subtrade_dfs, factor_scaled, price_data, threshold=-2.0):
    """
    Analyze each subtrade for maximum drawdown and plot those exceeding the threshold
    
    Parameters:
    subtrade_dfs (dict): Dictionary of dataframes containing subtrade information
    factor_scaled (DataFrame): DataFrame of factor values
    price_data (DataFrame): DataFrame of price data
    threshold (float): Threshold for significant drawdown percentage (negative for losses)
    
    Returns:
    DataFrame: Summary of maximum drawdowns for all subtrades
    """
    results = []
    plots = []
    
    for fut in subtrade_dfs:
        subtrade_df = subtrade_dfs[fut]
        
        for idx in subtrade_df.index:
            trade_id = subtrade_df.loc[idx, 'trade_id']
            subtrade_id = subtrade_df.loc[idx, 'subtrade_id']
            start_time = subtrade_df.loc[idx, 'start_time']
            end_time = subtrade_df.loc[idx, 'end_time']
            direction = subtrade_df.loc[idx, 'direction']
            
            # Extract factor and price data for this subtrade's time period
            holding_period_factor = factor_scaled.loc[start_time:end_time, fut]
            holding_period_price = price_data.loc[start_time:end_time, fut]
            
            # Skip if data is empty
            if holding_period_factor.empty or holding_period_price.empty:
                continue
                
            # Calculate maximum drawdown
            max_drawdown = calculate_max_drawdown(holding_period_price, direction)
            
            # Store result
            results.append({
                'trade_id': trade_id,
                'subtrade_id': subtrade_id,
                'symbol': fut,
                'direction': direction,
                'start_time': start_time,
                'end_time': end_time,
                'max_drawdown_pct': max_drawdown
            })
            
            # Plot subtrades with significant drawdowns
            if max_drawdown < threshold:
                fig = plot_subtrade(
                    holding_period_factor, 
                    holding_period_price, 
                    start_time, 
                    end_time, 
                    trade_id, 
                    subtrade_id, 
                    direction, 
                    max_drawdown,
                    fut  # Pass the futures symbol to the plot function
                )
                plots.append({
                    'trade_id': trade_id,
                    'subtrade_id': subtrade_id,
                    'fig': fig
                })
    
    results_df = pd.DataFrame(results)
    return results_df, plots

# Example usage:
results_df, plots = analyze_subtrades(subtrade_dfs, factor_scaled, price_data)

# Display summary statistics
print(f"Total subtrades analyzed: {len(results_df)}")
print(f"Subtrades with drawdown > 2%: {sum(results_df['max_drawdown_pct'] < -2.0)}")

# View the worst drawdowns
worst_drawdowns = results_df.sort_values('max_drawdown_pct').head(10)
print(worst_drawdowns)

maxdd_subtrade_dir = summary_dir / 'maxdd_subtrade'
maxdd_subtrade_dir.mkdir(parents=True, exist_ok=True)

# Save the plots for significant drawdowns
for plot_info in plots:
    trade_id = plot_info['trade_id']
    subtrade_id = plot_info['subtrade_id']
    fig = plot_info['fig']
    fig.savefig(maxdd_subtrade_dir / f"drawdown_trade_{trade_id}_subtrade_{subtrade_id}.png")
    plt.close(fig)