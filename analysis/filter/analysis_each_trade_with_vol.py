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
# model_name = 'avg_agg_250218_3_fix_tfe_by_trade_net_v4'
# factor_name = f'predict_{model_name}'
# factor_dir = Path(rf'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model\{model_name}\predict')
# direction = 1

# factor_name = 'LargeOrderAmountByValue_p1.0_v40000-avg_imb01_dp2-org'
factor_name = 'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb04_dp2-org'
direction = -1
factor_dir = Path(r'D:/mnt/CNIndexFutures/timeseries/factor_test/sample_data/factors/1_2_org')

version_name = 'v0'
feature_dir = Path(rf'D:\mnt\idx_opt_processed\{version_name}_features')
feature_name = 'atm_vol'
feature_col_name = 'MO'


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
feature_path = feature_dir / f'{feature_name}.parquet'
feature = pd.read_parquet(feature_path)


# %%
# 函数：根据仓位变化生成交易记录，并记录开仓时刻的feature值
def generate_trade_df(position_series, price_series, feature_series):
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
            if current_position == 1 and prev_position == 0:
                # 开多：记录开仓时间和价格
                open_price = current_price
                open_time = position_series.index[i]
                open_feature = current_feature
            elif current_position == 0 and prev_position == 1:
                # 平多：记录平仓时间和价格
                trade_info.append({
                    'trade_type': 'long',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': current_price,
                    'open_feature': open_feature
                })
                open_price = None  # 清空开仓价格
                open_time = None  # 清空开仓时间
                open_feature = None  # 清空开仓时刻的feature值
            elif current_position == -1 and prev_position == 0:
                # 开空：记录开仓时间和价格
                open_price = current_price
                open_time = position_series.index[i]
                open_feature = current_feature
            elif current_position == 0 and prev_position == -1:
                # 平空：记录平仓时间和价格
                trade_info.append({
                    'trade_type': 'short',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': current_price,
                    'open_feature': open_feature
                })
                open_price = None  # 清空开仓价格
                open_time = None  # 清空开仓时间
                open_feature = None  # 清空开仓时刻的feature值
            elif current_position == 1 and prev_position == -1:
                # 平空后开多：先平空，再开多
                trade_info.append({
                    'trade_type': 'short',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': prev_price,
                    'open_feature': open_feature
                })
                # 这里新的开仓会等平仓后再处理
                open_price = current_price  # 更新开仓价格
                open_time = position_series.index[i]  # 更新开仓时间
                open_feature = current_feature  # 更新开仓时刻的feature值
            elif current_position == -1 and prev_position == 1:
                # 平多后开空：先平多，再开空
                trade_info.append({
                    'trade_type': 'long',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': prev_price,
                    'open_feature': open_feature
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
    
    trade_df = generate_trade_df(actual_pos[col], price_data[col], feature_col)
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
    trade_dfs[col] = trade_df.dropna(subset=['net_return'])
    

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Assuming trade_dfs is your dictionary containing the three dataframes
# If you're running this code separately, you'll need to load your data first

# Set the style
sns.set(style="whitegrid")

# Create a figure with 3 subplots horizontally
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define the instruments
instruments = ['IC', 'IF', 'IM']

# Plot each instrument
for i, instrument in enumerate(instruments):
    # Get the dataframe for the current instrument
    df = trade_dfs[instrument]
    
    # Drop rows with NaN values in either atm_vol or net_return
    df_clean = df.dropna(subset=['atm_vol', 'net_return'])
    
    # Create scatter plot
    sns.scatterplot(
        x='atm_vol', 
        y='net_return',
        hue='trade_type',  # Color by trade type (long/short)
        data=df_clean,
        alpha=0.6,
        s=40,
        ax=axes[i]
    )
    
    # Add a trend line
    sns.regplot(
        x='atm_vol',
        y='net_return',
        data=df_clean,
        scatter=False,
        ax=axes[i],
        line_kws={'color': 'red', 'linewidth': 2}
    )
    
    # Calculate correlation
    corr = df_clean['atm_vol'].corr(df_clean['net_return'])
    
    # Set title and labels
    axes[i].set_title(f'{instrument} (n={len(df_clean)}, corr={corr:.4f})', fontsize=14)
    axes[i].set_xlabel('ATM Volatility', fontsize=12)
    
    # Only set y-label for the first subplot
    if i == 0:
        axes[i].set_ylabel('Net Return', fontsize=12)
    else:
        axes[i].set_ylabel('')

# Adjust layout
plt.tight_layout()
plt.suptitle('Relationship between ATM Volatility and Net Return by Instrument', fontsize=16, y=1.05)

# Show the plot
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelweight': 'bold'
})

instruments = ['IC', 'IF', 'IM']
metrics = ['Trade Count', 'Win Rate', 'Sum Return']

bin_edges = np.arange(0.005, 0.07, 0.0025)
bin_labels = [f'{edge*100:.2f}' for edge in bin_edges[:-1]]

palette_dict = {
    'Trade Count': '#6FA8DC',
    'Win Rate': '#F6A623',
    'Sum Return': '#7AC29A'
}

fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex='col')

for i, instrument in enumerate(instruments):
    df = trade_dfs[instrument]
    df_clean = df.dropna(subset=['atm_vol', 'net_return'])
    df_clean['vol_bin'] = pd.cut(df_clean['atm_vol'], bins=bin_edges, labels=bin_labels)
    
    bin_stats = []
    for bin_label in bin_labels:
        bin_data = df_clean[df_clean['vol_bin'] == bin_label]
        if len(bin_data) > 0:
            trade_count = len(bin_data)
            win_count = len(bin_data[bin_data['net_return'] > 0])
            win_rate = win_count / trade_count if trade_count > 0 else 0
            sum_return = bin_data['net_return'].sum() * 100
            bin_stats.append({
                'Vol Bin': bin_label,
                'Trade Count': trade_count,
                'Win Rate': win_rate * 100,
                'Sum Return': sum_return
            })
    
    bin_stats_df = pd.DataFrame(bin_stats)
    if len(bin_stats) == 0:
        bin_stats_df = pd.DataFrame(columns=['Vol Bin', 'Trade Count', 'Win Rate', 'Sum Return'])
        
    for j, metric in enumerate(metrics):
        ax = axes[i, j]
        
        if not bin_stats_df.empty:
            sns.barplot(
                x='Vol Bin', 
                y=metric, 
                data=bin_stats_df, 
                color=palette_dict[metric],
                ax=ax, 
                width=0.7
            )
            for bar in ax.patches:
                label_value = f"{int(bar.get_height())}" if metric == 'Trade Count' else f"{int(bar.get_height())}%"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    label_value,
                    ha='center', va='bottom',
                    fontsize=9
                )
        
        # 调整标题和坐标轴
        if i == 0:
            ax.set_title(metric, fontsize=14, fontweight='bold', pad=20)  # 加大pad
        if j == 0:
            ax.set_ylabel(f'{instrument}', fontsize=13, fontweight='bold')
        else:
            ax.set_ylabel('')
        if i == 2:
            ax.set_xlabel('Volatility Bin', fontsize=11)
        else:
            ax.set_xlabel('')
        
        ax.tick_params(axis='x', rotation=45)
        
        if metric == 'Win Rate':
            ax.set_ylim(0, 100)
        elif metric == 'Sum Return':
            min_return = min([-1, bin_stats_df['Sum Return'].min() if not bin_stats_df.empty else 0]) - 0.5
            max_return = max([1, bin_stats_df['Sum Return'].max() if not bin_stats_df.empty else 0]) + 0.5
            ax.set_ylim(min_return, max_return)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# 让列标题和图之间更松
plt.tight_layout(pad=3.5)
plt.subplots_adjust(top=0.90, bottom=0.1, hspace=0.1, wspace=0.1)
plt.suptitle(f'Factor: {factor_name}\n  by  feature: {feature_name} {feature_col_name} bin', 
             fontsize=18, 
             fontweight='bold', 
             y=0.99)

plt.show()

