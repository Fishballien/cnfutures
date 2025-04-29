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
# model_name = 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18'
# factor_name = f'predict_{model_name}'
# factor_dir = Path(rf'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model\{model_name}\predict')
# direction = 1

# factor_name = 'LargeOrderAmountByValue_p1.0_v40000-avg_imb01_dp2-org'
factor_name = 'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb04_dp2-org'
direction = -1
factor_dir = Path(r'D:/mnt/CNIndexFutures/timeseries/factor_test/sample_data/factors/1_2_org')


# factor_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\sample_data\factors\tf\typical_trade_factor')
# factor_name = 'order_amount_Dollar_LX_R3_dp2_SumIntraRm5_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15'
# direction = 1
# factor_name = 'order_amount_Dollar_R3_dp2_SumIntraRm5_LXPct_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15'
# direction = 1
# factor_name = 'trade_amount_Dollar_LX_R3_dp2_SumIntraRm5_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15'
# direction = 1
# factor_name = 'trade_amount_Dollar_R3_dp2_SumIntraRm5_LXPct_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15'
# direction = 1

# factor_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\sample_data\factors\lxy')
# factor_name = 'IntraRm_m30_IntraRelQtl_d20_q002_dp2_wstr_jump5mVwap'
# direction = 1


# %%
version_name = 'v0'
feature_dir = Path(rf'D:\mnt\idx_opt_processed\{version_name}_features')
feature_name = 'atm_vol'
feature_col_name = 'MO'


# %%
# =============================================================================
# feature_dir = Path(r'D:\mnt\idx_opt_processed\realized_vol')
# feature_name = 'realized_vol_multi_wd'
# feature_col_name = ''
# 
# =============================================================================

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
fut_dir = Path('/mnt/data1/futuretwap')
analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\trades_with_vol_trend_reverse')
summary_dir = analysis_dir / f'{feature_name}{feature_col_name}' / factor_name
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
feature_path = feature_dir / f'{feature_name}.parquet'
feature = pd.read_parquet(feature_path).loc['20160101':]


# %%
def generate_trade_df(position_series, price_series, feature_series, index_series=None):
    trade_info = []
    open_price = None  # 开仓价格（对应于价格序列）
    open_time = None  # 开仓时间戳（对应于时间索引）
    open_feature = None  # 开仓时刻的feature值
    
    for i in range(1, len(position_series)):
        prev_position = position_series.iloc[i-1]
        current_position = position_series.iloc[i]
        current_price = price_series.iloc[i]  # 当前时刻的价格
        prev_price = price_series.iloc[i-1]  # 上一时刻的价格
        current_feature = feature_series.iloc[i] if i < len(feature_series) else None  # 当前时刻的feature值
        
        # 只有仓位发生变化时才记录交易
        if current_position != prev_position:
            # 计算指数收益率函数
            def calculate_index_returns(open_timestamp, index_data):
                if index_data is None:
                    return None, None, None
                
                # 找到开仓时间在指数数据中的位置
                # try:
                idx_pos = index_data.index.get_loc(open_timestamp) - 1
                
                # 计算前15分钟、30分钟和60分钟的位置
                # 注意：这里假设数据是按时间顺序的，且频率足够高
                # 如果是分钟级数据，可以根据实际情况调整步长
                steps_15min = 15  # 对应于15分钟的数据点数量
                steps_30min = 30  # 对应于30分钟的数据点数量
                steps_60min = 60  # 对应于60分钟的数据点数量
                
                # 确保不超出数据范围
                idx_15min = max(0, idx_pos - steps_15min)
                idx_30min = max(0, idx_pos - steps_30min)
                idx_60min = max(0, idx_pos - steps_60min)
                
                # 计算收益率
                return_15min = (index_data.iloc[idx_pos] / index_data.iloc[idx_15min] - 1) * 100 if idx_pos != idx_15min else 0
                return_30min = (index_data.iloc[idx_pos] / index_data.iloc[idx_30min] - 1) * 100 if idx_pos != idx_30min else 0
                return_60min = (index_data.iloc[idx_pos] / index_data.iloc[idx_60min] - 1) * 100 if idx_pos != idx_60min else 0
                return return_15min, return_30min, return_60min
                # except:
                #     # 如果发生错误（如数据不足），返回None
                #     return None, None, None
            
            if current_position == 1 and prev_position == 0:  # 开多：记录开仓时间和价格
                open_price = current_price
                open_time = position_series.index[i]
                open_feature = current_feature
                
            elif current_position == 0 and prev_position == 1:  # 平多：记录平仓时间和价格
                # 计算开仓前的指数收益率
                index_return_15min, index_return_30min, index_return_60min = calculate_index_returns(open_time, price_series)
                # breakpoint()
                trade_info.append({
                    'trade_type': 'long',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': current_price,
                    'open_feature': open_feature,
                    'index_return_15min': index_return_15min,
                    'index_return_30min': index_return_30min,
                    'index_return_60min': index_return_60min
                })
                
                open_price = None  # 清空开仓价格
                open_time = None  # 清空开仓时间
                open_feature = None  # 清空开仓时刻的feature值
                
            elif current_position == -1 and prev_position == 0:  # 开空：记录开仓时间和价格
                open_price = current_price
                open_time = position_series.index[i]
                open_feature = current_feature
                
            elif current_position == 0 and prev_position == -1:  # 平空：记录平仓时间和价格
                # 计算开仓前的指数收益率
                # breakpoint()
                index_return_15min, index_return_30min, index_return_60min = calculate_index_returns(open_time, price_series)
                
                trade_info.append({
                    'trade_type': 'short',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': current_price,
                    'open_feature': open_feature,
                    'index_return_15min': index_return_15min,
                    'index_return_30min': index_return_30min,
                    'index_return_60min': index_return_60min
                })
                
                open_price = None  # 清空开仓价格
                open_time = None  # 清空开仓时间
                open_feature = None  # 清空开仓时刻的feature值
                
            elif current_position == 1 and prev_position == -1:  # 平空后开多：先平空，再开多
                # 计算开仓前的指数收益率
                index_return_15min, index_return_30min, index_return_60min = calculate_index_returns(open_time, price_series)
                
                trade_info.append({
                    'trade_type': 'short',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': prev_price,
                    'open_feature': open_feature,
                    'index_return_15min': index_return_15min,
                    'index_return_30min': index_return_30min,
                    'index_return_60min': index_return_60min
                })
                
                # 这里新的开仓会等平仓后再处理
                open_price = current_price  # 更新开仓价格
                open_time = position_series.index[i]  # 更新开仓时间
                open_feature = current_feature  # 更新开仓时刻的feature值
                
            elif current_position == -1 and prev_position == 1:  # 平多后开空：先平多，再开空
                # 计算开仓前的指数收益率
                index_return_15min, index_return_30min, index_return_60min = calculate_index_returns(open_time, price_series)
                
                trade_info.append({
                    'trade_type': 'long',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': prev_price,
                    'open_feature': open_feature,
                    'index_return_15min': index_return_15min,
                    'index_return_30min': index_return_30min,
                    'index_return_60min': index_return_60min
                })
                
                # 这里新的开仓会等平仓后再处理
                open_price = current_price  # 更新开仓价格
                open_time = position_series.index[i]  # 更新开仓时间
                open_feature = current_feature  # 更新开仓时刻的feature值
    
    return pd.DataFrame(trade_info)
        
# 对每列标的单独生成交易记录
trade_dfs = {}

# 确保feature数据有相同的时间索引
feature = feature.reindex(price_data.index)

for col in actual_pos.columns:
    # 如果feature是多列数据，需要确保对应的列存在
    if isinstance(feature, pd.DataFrame) and col in feature.columns:
        feature_col = feature[col]
    else:
        # 如果feature只有一列或者没有对应列，使用整个feature
        feature_col = feature[feature_col_name]
    
    trade_df = generate_trade_df(actual_pos[col].fillna(0), price_data[col], feature_col)
    trade_dfs[col] = trade_df
     
for col in trade_dfs:
    trade_df = trade_dfs[col]
    trade_df['direction'] = trade_df['trade_type'].apply(lambda x: 1 if x == 'long' else -1)
    trade_df['net_return'] = np.log(trade_df['close_price'] / trade_df['open_price']) * trade_df['direction'] - fee
    trade_df['holding_time'] = trade_df['close_time'] - trade_df['open_time']
    # 重命名open_feature列为feature_name的值
    if 'open_feature' in trade_df.columns:
        trade_df[feature_name] = trade_df['open_feature']
        trade_df = trade_df.drop(columns=['open_feature'])
    trade_df['index_return_avg'] = trade_df[['index_return_15min', 'index_return_30min', 'index_return_60min']].mean(axis=1)
    trade_dfs[col] = trade_df.dropna(subset=['net_return'])
    

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import json
from datetime import datetime

def analyze_trades_by_volatility(trade_dfs, summary_dir):
    """
    Analyze trading strategies (trend-following vs reversal) across different volatility levels
    
    Parameters:
    trade_dfs - Dictionary of DataFrames with trading data for different instruments
    summary_dir - pathlib.Path object pointing to directory where results should be saved
    
    Each DataFrame should contain:
    - trade_type: 'long' or 'short'
    - direction: 1 (long) or -1 (short)
    - index_return_15min, index_return_30min, index_return_60min: pre-trade returns
    - net_return: trade returns
    - atm_vol: at-the-money volatility
    """
    # Ensure the summary directory exists
    summary_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    for symbol, df in trade_dfs.items():
        print(f"Analyzing {symbol}...")
        
        # Create volatility bins starting from 0.0075 with step size of 0.0025
        df_analysis = df.copy()
        min_vol = 0.005
        max_vol = 0.03
        
        # Create bin edges including the min and max values
        bin_edges = np.arange(min_vol, max_vol + 0.0001, 0.0025)
        
        # Add an extra bin for values greater than max_vol
        bin_edges = np.append(bin_edges, np.inf)
        
        # Create bin labels (including one for values > max_vol)
        bin_labels = [f'{edge:.4f}-{bin_edges[i+1]:.4f}' for i, edge in enumerate(bin_edges[:-2])]
        bin_labels.append(f'{max_vol:.4f}+')  # Special label for the last bin
        
        # Apply custom binning
        df_analysis['vol_bin'] = pd.cut(
            df_analysis[feature_name], 
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True
        )
        symbol_results = {}
        
        # Analyze each time period
        for period in ['15min', '30min', '60min', 'avg']:
            return_col = f'index_return_{period}'
            
            # Define trend-following vs reversal vs neutral
            # If pre-trade return absolute value < 0.1%, it's neutral (no significant movement)
            # If pre-trade return > 0.1% and matches direction, it's trend-following
            # If pre-trade return < -0.1% against direction, it's a reversal trade
            df_analysis[f'strategy_{period}'] = 'Neutral'  # Default is neutral

            df_analysis[f'strategy_{period}'] = 'Neutral'  # Default is neutral

            # Identify Long Trend scenarios with different magnitude
            # Small Long Trend: return between 0.1% and 0.25% with positive direction
            trend_s_long_mask = (df_analysis[return_col] > 0.1) & (df_analysis[return_col] <= 0.3) & (df_analysis['direction'] > 0)
            df_analysis.loc[trend_s_long_mask, f'strategy_{period}'] = 'TrendSLong'
            
            # Large Long Trend: return > 0.25% with positive direction
            trend_l_long_mask = (df_analysis[return_col] > 0.3) & (df_analysis['direction'] > 0)
            df_analysis.loc[trend_l_long_mask, f'strategy_{period}'] = 'TrendLLong'
            
            # Identify Short Trend scenarios with different magnitude
            # Small Short Trend: return between -0.25% and -0.1% with negative direction
            trend_s_short_mask = (df_analysis[return_col] < -0.1) & (df_analysis[return_col] >= -0.3) & (df_analysis['direction'] < 0)
            df_analysis.loc[trend_s_short_mask, f'strategy_{period}'] = 'TrendSShort'
            
            # Large Short Trend: return < -0.25% with negative direction
            trend_l_short_mask = (df_analysis[return_col] < -0.3) & (df_analysis['direction'] < 0)
            df_analysis.loc[trend_l_short_mask, f'strategy_{period}'] = 'TrendLShort'
            
            # Identify Long Reversal scenarios with different magnitude
            # Small Long Reversal: return between -0.25% and -0.1% with positive direction
            reversal_s_long_mask = (df_analysis[return_col] < -0.1) & (df_analysis[return_col] >= -0.3) & (df_analysis['direction'] > 0)
            df_analysis.loc[reversal_s_long_mask, f'strategy_{period}'] = 'ReversalSLong'
            
            # Large Long Reversal: return < -0.25% with positive direction
            reversal_l_long_mask = (df_analysis[return_col] < -0.3) & (df_analysis['direction'] > 0)
            df_analysis.loc[reversal_l_long_mask, f'strategy_{period}'] = 'ReversalLLong'
            
            # Identify Short Reversal scenarios with different magnitude
            # Small Short Reversal: return between 0.1% and 0.25% with negative direction
            reversal_s_short_mask = (df_analysis[return_col] > 0.1) & (df_analysis[return_col] <= 0.3) & (df_analysis['direction'] < 0)
            df_analysis.loc[reversal_s_short_mask, f'strategy_{period}'] = 'ReversalSShort'
            
            # Large Short Reversal: return > 0.25% with negative direction
            reversal_l_short_mask = (df_analysis[return_col] > 0.3) & (df_analysis['direction'] < 0)
            df_analysis.loc[reversal_l_short_mask, f'strategy_{period}'] = 'ReversalLShort'
            
            # Mark winning trades
            df_analysis['is_win'] = df_analysis['net_return'] > 0
            
            # Group by volatility bin and strategy
            grouped = df_analysis.groupby(['vol_bin', f'strategy_{period}'])
            
            # Calculate metrics
            count_data = grouped.size().unstack(fill_value=0)
            win_rate_data = grouped['is_win'].mean().unstack(fill_value=0) * 100  # as percentage
            avg_return_data = grouped['net_return'].mean().unstack(fill_value=0) * 100  # as percentage
            
            # Store in results
            symbol_results[period] = {
                'trade_count': count_data,
                'win_rate': win_rate_data,
                'avg_return': avg_return_data
            }
            
            # Create heatmaps
            plot_heatmaps(symbol, period, count_data, win_rate_data, avg_return_data, summary_dir)
        
        results[symbol] = symbol_results
    
    # Save summary results as JSON (converting DataFrames to dict for serialization)
    summary_results = {}
    for symbol, symbol_data in results.items():
        summary_results[symbol] = {}
        for period, period_data in symbol_data.items():
            summary_results[symbol][period] = {}
            for metric, df in period_data.items():
                summary_results[symbol][period][metric] = df.to_dict()
    
    # Save to JSON file
    result_path = summary_dir / 'strategy_analysis_summary.json'
    with open(result_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    # Also save a simple text summary
    with open(summary_dir / 'strategy_analysis_summary.txt', 'w') as f:
        f.write(f"Trade Strategy Analysis Summary\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for symbol, symbol_data in results.items():
            f.write(f"Symbol: {symbol}\n")
            f.write("=" * 50 + "\n")
            
            for period in ['15min', '30min', '60min', 'avg']:
                f.write(f"\nPeriod: {period}\n")
                f.write("-" * 30 + "\n")
                
                period_data = symbol_data[period]
                
                # Overall statistics by strategy
                trend_count = period_data['trade_count']['Trend'].sum() if 'Trend' in period_data['trade_count'] else 0
                reversal_count = period_data['trade_count']['Reversal'].sum() if 'Reversal' in period_data['trade_count'] else 0
                neutral_count = period_data['trade_count']['Neutral'].sum() if 'Neutral' in period_data['trade_count'] else 0
                total_count = trend_count + reversal_count + neutral_count
                
                trend_win_rate = period_data['win_rate']['Trend'].mean() if 'Trend' in period_data['win_rate'] else 0
                reversal_win_rate = period_data['win_rate']['Reversal'].mean() if 'Reversal' in period_data['win_rate'] else 0
                neutral_win_rate = period_data['win_rate']['Neutral'].mean() if 'Neutral' in period_data['win_rate'] else 0
                
                trend_return = period_data['avg_return']['Trend'].mean() if 'Trend' in period_data['avg_return'] else 0
                reversal_return = period_data['avg_return']['Reversal'].mean() if 'Reversal' in period_data['avg_return'] else 0
                neutral_return = period_data['avg_return']['Neutral'].mean() if 'Neutral' in period_data['avg_return'] else 0
                
                f.write(f"Trend Strategy: {trend_count} trades ({trend_count/total_count*100:.1f}%)\n")
                f.write(f"  Average Win Rate: {trend_win_rate:.2f}%\n")
                f.write(f"  Average Return: {trend_return:.4f}%\n\n")
                
                f.write(f"Reversal Strategy: {reversal_count} trades ({reversal_count/total_count*100:.1f}%)\n")
                f.write(f"  Average Win Rate: {reversal_win_rate:.2f}%\n")
                f.write(f"  Average Return: {reversal_return:.4f}%\n\n")
                
                f.write(f"Neutral Strategy: {neutral_count} trades ({neutral_count/total_count*100:.1f}%)\n")
                f.write(f"  Average Win Rate: {neutral_win_rate:.2f}%\n")
                f.write(f"  Average Return: {neutral_return:.4f}%\n\n")
    
    return results

def plot_heatmaps(symbol, period, count_data, win_rate_data, avg_return_data, summary_dir):
    """Plot heatmaps for the three metrics with three strategy types"""
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    
    # Ensure consistent colormap scales across all strategies
    count_vmax = count_data.values.max() if not count_data.empty else 1
    
    # Trade count heatmap
    sns.heatmap(count_data, annot=True, fmt='d', cmap='Blues', ax=axes[0], vmin=0, vmax=count_vmax)
    axes[0].set_title(f'{symbol} - Trade Count - {period}')
    
    # Win rate heatmap
    sns.heatmap(win_rate_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1], vmin=0, vmax=100)
    axes[1].set_title(f'{symbol} - Win Rate (%) - {period}')
    
    # Average return heatmap
    sns.heatmap(avg_return_data, annot=True, fmt='.4f', cmap='RdYlGn', center=0, ax=axes[2])
    axes[2].set_title(f'{symbol} - Avg Return (%) - {period}')
    
    plt.tight_layout()
    
    # Save to the summary directory
    output_path = summary_dir / f'{symbol}_{period}_strategy_analysis.png'
    plt.savefig(output_path)
    plt.close()

# Main execution code
if __name__ == "__main__":
    # Analyze each instrument in the trade_dfs dictionary
    results = analyze_trades_by_volatility(trade_dfs, summary_dir)
    
    print(f"Analysis complete. Results saved to {summary_dir.absolute()}")

