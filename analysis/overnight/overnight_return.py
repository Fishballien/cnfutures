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
test_name = 'trade_ver3_futtwap_sp1min_s240d_icim_v6'
factor_name = f'predict_{model_name}'
pos_dir = Path(rf'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model\{model_name}\test') / test_name / 'data'
direction = 1

# factor_name = 'LargeOrderAmountByValue_p1.0_v40000-wavg_imb04_dpall-mean_w30min'
# direction = -1
# factor_dir = Path(r'D:/mnt/CNIndexFutures/timeseries/factor_test/sample_data/factors/low_freq')


# %%
price_name = 't1min_fq1min_dl1min'


# %%
fut_dir = Path('/mnt/data1/futuretwap')
analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\overnight')
summary_dir = analysis_dir / factor_name / test_name
summary_dir.mkdir(parents=True, exist_ok=True)


# %%
pos_data = pd.read_parquet(pos_dir / f'pos_{factor_name}.parquet')
# pos_data = to_float32(pos_data)
price_data = pd.read_parquet(fut_dir / f'{price_name}.parquet')
pos_data = pos_data.rename(columns=index_to_futures)[['IC', 'IM']]
pos_data, price_data = align_and_sort_columns([pos_data, price_data])

price_data = price_data.loc[pos_data.index.min():pos_data.index.max()] # 按factor头尾截取
pos_data = pos_data.reindex(price_data.index) # 按twap reindex，确保等长


# %%
rtn_1p = price_data.pct_change(1, fill_method=None).shift(-1)


# %%
gp = pos_data * rtn_1p
gp_neg = gp.clip(upper=0)
gp_neg['return'] = gp_neg.mean(axis=1)
gp_neg["minute"] = gp_neg.index.time  # 提取分钟部分
avg_per_minute = gp_neg.groupby("minute")["return"].mean()


# %%
overnight_positions = pos_data[pos_data.index.strftime('%H:%M:%S') == '14:55:00'].copy()
overnight_rtn = rtn_1p[rtn_1p.index.strftime('%H:%M:%S') == '14:55:00'].copy()


# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pandas as pd

# Get column names
assets = overnight_positions.columns

# Create scatter plots for each asset
for asset in assets:
    plt.figure(figsize=(10, 6))
    
    # Separate long and short positions
    long_mask = overnight_positions[asset] > 0
    short_mask = overnight_positions[asset] < 0
    
    # Plot long positions (red)
    plt.scatter(overnight_positions[asset][long_mask], 
                overnight_rtn[asset][long_mask],
                color='red', label='Long', alpha=0.7)
    
    # Plot short positions (blue)
    plt.scatter(overnight_positions[asset][short_mask], 
                overnight_rtn[asset][short_mask],
                color='blue', label='Short', alpha=0.7)
    
    # Add reference lines
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    
    # Set chart properties
    plt.xlabel('Overnight Positions')
    plt.ylabel('Overnight Returns')
    plt.title(f'Overnight Positions vs Returns - {asset}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save chart to specified directory
    fig_path = summary_dir / f"{asset}_overnight_scatter.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {fig_path}")
    
    plt.show()

# Calculate and save correlation results
print("Correlations between overnight positions and returns:")
correlation_results = []

for asset in assets:
    try:
        # Overall correlation
        all_corr = overnight_positions[asset].corr(overnight_rtn[asset])
        
        # Long positions correlation
        long_mask = overnight_positions[asset] > 0
        if sum(long_mask) > 1:
            long_corr = overnight_positions[asset][long_mask].corr(overnight_rtn[asset][long_mask])
        else:
            long_corr = np.nan
        
        # Short positions correlation
        short_mask = overnight_positions[asset] < 0
        if sum(short_mask) > 1:
            short_corr = overnight_positions[asset][short_mask].corr(overnight_rtn[asset][short_mask])
        else:
            short_corr = np.nan
        
        result_str = f"{asset}: Overall={all_corr:.4f}, Long={long_corr:.4f}, Short={short_corr:.4f}"
        print(result_str)
        
        # Add results to list
        correlation_results.append({
            'asset': asset,
            'all_correlation': all_corr,
            'long_correlation': long_corr,
            'short_correlation': short_corr
        })
    except Exception as e:
        print(f"Error calculating correlation for {asset}: {str(e)}")

# Save correlation results to CSV
corr_df = pd.DataFrame(correlation_results)
corr_csv_path = summary_dir / "overnight_correlations.csv"
corr_df.to_csv(corr_csv_path, index=False)
print(f"Correlation results saved to: {corr_csv_path}")
    
    
# %% dist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# 注意：这里假设summary_dir是一个已经定义好的pathlib.Path对象
# 如果它不存在，则需要创建它

# 假设overnight_positions和overnight_rtn是已经定义好的DataFrame对象
# 如果你单独运行这段代码，你需要先加载数据

# 创建副本以避免修改原始数据
positions = overnight_positions.copy()
returns = overnight_rtn.copy()

# 按仓位方向分析收益的函数
def analyze_returns_by_position(positions_df, returns_df, instrument):
    # 获取品种数据
    pos = positions_df[instrument].copy()
    ret = returns_df[instrument].copy()
    
    # 创建一个结合仓位和收益的DataFrame
    combined = pd.DataFrame({
        'position': pos,
        'return': ret
    }).dropna()  # 删除仓位或收益为NaN的行
    
    # 分离多头和空头仓位
    long_returns = combined[combined['position'] > 0]['return']
    short_returns = combined[combined['position'] < 0]['return']
    zero_returns = combined[combined['position'] == 0]['return']
    
    return {
        'long': long_returns,
        'short': short_returns,
        'zero': zero_returns
    }

# 分析两个品种
ic_analysis = analyze_returns_by_position(positions, returns, 'IC')
im_analysis = analyze_returns_by_position(positions, returns, 'IM')

# ---------- 创建直方图和统计数据的组合图 ----------
# 创建统计数据文本
summary_text = [
    "IC Long Positions:",
    f"  Count: {len(ic_analysis['long'])}",
    f"  Mean: {ic_analysis['long'].mean():.6f}",
    f"  Median: {ic_analysis['long'].median():.6f}",
    f"  Std Dev: {ic_analysis['long'].std():.6f}",
    f"  Min: {ic_analysis['long'].min():.6f}",
    f"  Max: {ic_analysis['long'].max():.6f}",
    "",
    "IC Short Positions:",
    f"  Count: {len(ic_analysis['short'])}",
    f"  Mean: {ic_analysis['short'].mean():.6f}",
    f"  Median: {ic_analysis['short'].median():.6f}",
    f"  Std Dev: {ic_analysis['short'].std():.6f}",
    f"  Min: {ic_analysis['short'].min():.6f}",
    f"  Max: {ic_analysis['short'].max():.6f}",
    "",
    "IM Long Positions:",
    f"  Count: {len(im_analysis['long'])}",
    f"  Mean: {im_analysis['long'].mean():.6f}",
    f"  Median: {im_analysis['long'].median():.6f}",
    f"  Std Dev: {im_analysis['long'].std():.6f}",
    f"  Min: {im_analysis['long'].min():.6f}",
    f"  Max: {im_analysis['long'].max():.6f}",
    "",
    "IM Short Positions:",
    f"  Count: {len(im_analysis['short'])}",
    f"  Mean: {im_analysis['short'].mean():.6f}",
    f"  Median: {im_analysis['short'].median():.6f}",
    f"  Std Dev: {im_analysis['short'].std():.6f}",
    f"  Min: {im_analysis['short'].min():.6f}",
    f"  Max: {im_analysis['short'].max():.6f}"
]

# 将统计数据保存到文本文件
with open(summary_dir / "overnight_returns_summary.txt", "w") as f:
    f.write('\n'.join(summary_text))

# 创建一个大图，左侧放置直方图，右侧放置统计数据
fig = plt.figure(figsize=(20, 12))

# 设置网格，左侧占2/3，右侧占1/3
gs = plt.GridSpec(2, 3, figure=fig)

# 创建左侧的直方图子图（占据左侧2/3区域）
ax1 = fig.add_subplot(gs[0, 0])  # 左上
ax2 = fig.add_subplot(gs[0, 1])  # 右上
ax3 = fig.add_subplot(gs[1, 0])  # 左下
ax4 = fig.add_subplot(gs[1, 1])  # 右下

# 创建右侧的统计数据子图（占据右侧1/3区域）
ax_stats = fig.add_subplot(gs[:, 2])  # 右侧完整区域

# IC多头直方图
sns.histplot(ic_analysis['long'], kde=True, color='green', ax=ax1)
ax1.set_title(f'IC Long Positions (n={len(ic_analysis["long"])})')
ax1.set_xlabel('Overnight Return')
ax1.grid(True, alpha=0.3)

# IC空头直方图
sns.histplot(ic_analysis['short'], kde=True, color='red', ax=ax2)
ax2.set_title(f'IC Short Positions (n={len(ic_analysis["short"])})')
ax2.set_xlabel('Overnight Return')
ax2.grid(True, alpha=0.3)

# IM多头直方图
sns.histplot(im_analysis['long'], kde=True, color='green', ax=ax3)
ax3.set_title(f'IM Long Positions (n={len(im_analysis["long"])})')
ax3.set_xlabel('Overnight Return')
ax3.grid(True, alpha=0.3)

# IM空头直方图
sns.histplot(im_analysis['short'], kde=True, color='red', ax=ax4)
ax4.set_title(f'IM Short Positions (n={len(im_analysis["short"])})')
ax4.set_xlabel('Overnight Return')
ax4.grid(True, alpha=0.3)

# 统计数据文本框
ax_stats.axis('off')
ax_stats.text(0.1, 0.5, '\n'.join(summary_text), fontsize=12, family='monospace', transform=ax_stats.transAxes, verticalalignment='center')
ax_stats.set_title('Overnight Returns Summary Statistics', fontsize=14)

plt.tight_layout()
# 保存组合图
fig.savefig(summary_dir / "overnight_returns_combined.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ---------- 保存小提琴图 ----------
fig3 = plt.figure(figsize=(14, 8))

# IC小提琴图
plt.subplot(1, 2, 1)
ic_long_data = pd.DataFrame({'Return': ic_analysis['long'], 'Position': 'Long'})
ic_short_data = pd.DataFrame({'Return': ic_analysis['short'], 'Position': 'Short'})
ic_combined = pd.concat([ic_long_data, ic_short_data])
sns.violinplot(x='Position', y='Return', data=ic_combined, palette={'Long': 'green', 'Short': 'red'})
plt.title('IC Overnight Returns by Position Type')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)

# IM小提琴图
plt.subplot(1, 2, 2)
im_long_data = pd.DataFrame({'Return': im_analysis['long'], 'Position': 'Long'})
im_short_data = pd.DataFrame({'Return': im_analysis['short'], 'Position': 'Short'})
im_combined = pd.concat([im_long_data, im_short_data])
sns.violinplot(x='Position', y='Return', data=im_combined, palette={'Long': 'green', 'Short': 'red'})
plt.title('IM Overnight Returns by Position Type')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)

plt.tight_layout()
fig3.savefig(summary_dir / "overnight_returns_violin.png", dpi=300, bbox_inches='tight')
plt.close(fig3)

print(f"所有图形和数据已保存到 '{summary_dir}' 目录:")
print("- overnight_returns_combined.png (直方图和统计数据的组合图)")
print("- overnight_returns_violin.png (小提琴图)")
print("- overnight_returns_summary.txt (统计数据文本文件)")
    
    
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_yearly_stats(pos_df, rtn_df, position_filter='negative'):
    """
    Calculate yearly statistics for overnight returns based on position filter.
    
    Parameters:
    -----------
    pos_df : pandas.DataFrame
        DataFrame containing position data with tickers as columns
    rtn_df : pandas.DataFrame
        DataFrame containing return data with tickers as columns
    position_filter : str, optional
        Filter positions by 'positive', 'negative', or 'all' (default: 'negative')
    
    Returns:
    --------
    dict
        Dictionary of DataFrames with yearly statistics for each ticker
    """
    # Ensure index is datetime
    pos_df.index = pd.to_datetime(pos_df.index)
    rtn_df.index = pd.to_datetime(rtn_df.index)
    
    results = {}
    
    # Process each ticker
    for ticker in pos_df.columns:
        if ticker not in rtn_df.columns:
            print(f"Warning: Ticker {ticker} not found in return data")
            continue
            
        # Apply position filter
        if position_filter.lower() == 'negative':
            filtered_returns = rtn_df[ticker][pos_df[ticker] < 0]
            title_suffix = "Position < 0"
        elif position_filter.lower() == 'positive':
            filtered_returns = rtn_df[ticker][pos_df[ticker] > 0]
            title_suffix = "Position > 0"
        else:  # 'all'
            filtered_returns = rtn_df[ticker]
            title_suffix = "All Positions"
        
        if len(filtered_returns) == 0:
            print(f"No data for {ticker} with {position_filter} position filter")
            continue
        
        # Create a DataFrame with the filtered returns
        filtered_df = pd.DataFrame({'return': filtered_returns})
        
        # Extract year and group by year
        filtered_df['year'] = filtered_df.index.year
        yearly_avg = filtered_df.groupby('year')['return'].agg(['mean', 'count', 'std'])
        yearly_avg.columns = ['avg_return', 'count', 'std_dev']
        
        results[ticker] = {
            'stats': yearly_avg,
            'title_suffix': title_suffix,
            'overall_avg': filtered_returns.mean()
        }
    
    return results

def plot_yearly_stats(yearly_results, figsize=(14, 5), color_map=None, summary_dir=None):
    """
    Visualize yearly statistics for each ticker.
    
    Parameters:
    -----------
    yearly_results : dict
        Dictionary of results from calculate_yearly_stats
    figsize : tuple, optional
        Figure size multiplier (width, height) per ticker
    color_map : dict, optional
        Dictionary mapping tickers to colors
    summary_dir : str, optional
        Directory to save the figure. If None, figure is not saved.
    
    Returns:
    --------
    tuple
        (fig, axes) - matplotlib figure and axes objects
    """
    # Count number of tickers to plot
    n_tickers = len(yearly_results)
    
    if n_tickers == 0:
        print("No data to plot")
        return None, None
    
    # Default colors if not provided
    if color_map is None:
        default_colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
        color_map = {ticker: default_colors[i % len(default_colors)] 
                     for i, ticker in enumerate(yearly_results.keys())}
    
    # Create figure with appropriate size
    fig, axes = plt.subplots(n_tickers, 1, figsize=(figsize[0], figsize[1] * n_tickers))
    
    # Handle case with only one ticker
    if n_tickers == 1:
        axes = [axes]
    
    # Get position filter for filename
    sample_ticker = list(yearly_results.keys())[0]
    position_type = yearly_results[sample_ticker]['title_suffix']
    
    # Create a safe filename by replacing problematic characters
    if "< 0" in position_type:
        safe_position_type = "negative"
    elif "> 0" in position_type:
        safe_position_type = "positive"
    else:
        safe_position_type = "all"
    
    # Plot each ticker
    for i, (ticker, data) in enumerate(yearly_results.items()):
        ax = axes[i]
        yearly_data = data['stats']
        title_suffix = data['title_suffix']
        overall_avg = data['overall_avg']
        
        years = yearly_data.index
        avg_returns = yearly_data['avg_return']
        counts = yearly_data['count']
        
        bar_width = 0.8
        bars = ax.bar(years, avg_returns, width=bar_width, alpha=0.7, color=color_map.get(ticker, 'blue'))
        
        # Find y-axis limits to position the labels properly
        y_min, y_max = ax.get_ylim()
        y_pos = y_min + (y_max - y_min) * 0.05  # Position text 5% above the bottom
        
        # Add count labels near y=0
        for j, year in enumerate(years):
            bar = bars[j]
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'n={counts.loc[year]}', ha='center', va='bottom', fontsize=12)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Average Overnight Return')
        ax.set_title(f'{ticker}: Yearly Average Overnight Returns when {title_suffix}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Add overall average line
        ax.axhline(y=overall_avg, color='r', linestyle='-', alpha=0.5, 
                  label=f'Overall Avg: {overall_avg:.6f}')
        ax.legend()
    
    plt.tight_layout()
    
    # Save figure if summary_dir is provided
    if summary_dir is not None:
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
            
        # Create filename with safe characters
        tickers_str = '_'.join(yearly_results.keys())
        filename = f"overnight_returns_{safe_position_type}_{tickers_str}.png"
        filepath = os.path.join(summary_dir, filename)
        
        # Save with high DPI for better quality
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {filepath}")
    
    plt.show()
    
    return fig, axes

def create_combined_summary(yearly_results, summary_dir=None):
    """
    Create a combined summary table for all tickers.
    
    Parameters:
    -----------
    yearly_results : dict
        Dictionary of results from calculate_yearly_stats
    summary_dir : str, optional
        Directory to save the summary CSV. If None, CSV is not saved.
    
    Returns:
    --------
    pandas.DataFrame
        Combined summary table
    """
    if not yearly_results:
        return None
    
    combined_df = None
    
    for ticker, data in yearly_results.items():
        stats = data['stats'].copy()
        # Rename columns to include ticker
        renamed = stats.rename(columns={
            'avg_return': f'{ticker}_avg_return',
            'count': f'{ticker}_count',
            'std_dev': f'{ticker}_std_dev'
        })
        
        if combined_df is None:
            combined_df = renamed
        else:
            combined_df = pd.merge(combined_df, renamed, left_index=True, right_index=True, how='outer')
    
    # Print overall statistics
    print("\nOverall Statistics:")
    for ticker, data in yearly_results.items():
        print(f"{ticker} overall average return: {data['overall_avg']:.6f}")
    
    # Save summary to CSV if directory is provided
    if summary_dir is not None and combined_df is not None:
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
            
        # Get position filter for filename
        sample_ticker = list(yearly_results.keys())[0]
        position_type = yearly_results[sample_ticker]['title_suffix']
        
        # Create a safe filename by replacing problematic characters
        if "< 0" in position_type:
            safe_position_type = "negative"
        elif "> 0" in position_type:
            safe_position_type = "positive"
        else:
            safe_position_type = "all"
        
        # Create filename
        tickers_str = '_'.join(yearly_results.keys())
        filename = f"overnight_returns_summary_{safe_position_type}_{tickers_str}.csv"
        filepath = os.path.join(summary_dir, filename)
        
        # Save to CSV
        combined_df.to_csv(filepath)
        print(f"Summary saved to: {filepath}")
    
    return combined_df

# Main function to run the analysis
def analyze_overnight_returns(pos_df, rtn_df, position_filter='negative', 
                             figsize=(14, 5), color_map=None, summary_dir=None):
    """
    Analyze overnight returns based on position filter.
    
    Parameters:
    -----------
    pos_df : pandas.DataFrame
        DataFrame containing position data with tickers as columns
    rtn_df : pandas.DataFrame
        DataFrame containing return data with tickers as columns
    position_filter : str, optional
        Filter positions by 'positive', 'negative', or 'all' (default: 'negative')
    figsize : tuple, optional
        Figure size multiplier (width, height) per ticker
    color_map : dict, optional
        Dictionary mapping tickers to colors
    summary_dir : str, optional
        Directory to save results (figures and CSV). If None, results are not saved.
    
    Returns:
    --------
    dict
        Dictionary containing results and summary
    """
    # Calculate yearly statistics
    yearly_results = calculate_yearly_stats(pos_df, rtn_df, position_filter)
    
    # Print statistics for each ticker
    for ticker, data in yearly_results.items():
        print(f"{ticker} - Yearly Average Returns when {data['title_suffix']}:")
        print(data['stats'])
        print("\n")
    
    # Plot results
    fig, axes = plot_yearly_stats(yearly_results, figsize, color_map, summary_dir)
    
    # Create combined summary
    combined_summary = create_combined_summary(yearly_results, summary_dir)
    if combined_summary is not None:
        print("\nCombined Yearly Statistics:")
        print(combined_summary)
    
    return {
        'yearly_results': yearly_results,
        'summary': combined_summary,
        'figure': fig,
        'axes': axes
    }

# Example usage:
result = analyze_overnight_returns(overnight_positions, overnight_rtn, position_filter='negative', 
                                  summary_dir=summary_dir)
result = analyze_overnight_returns(overnight_positions, overnight_rtn, position_filter='positive', 
                                  summary_dir=summary_dir)


# %% week
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import PercentFormatter

def analyze_overnight_returns_by_weekday(pos_df, rtn_df, summary_dir=None):
    """
    Analyze overnight returns by weekday, separating by positive and negative positions.
    
    Parameters:
    -----------
    pos_df : pandas.DataFrame
        DataFrame containing position data with tickers as columns
    rtn_df : pandas.DataFrame
        DataFrame containing return data with tickers as columns
    summary_dir : str, optional
        Directory to save results (figures and CSV). If None, results are not saved.
    
    Returns:
    --------
    dict
        Dictionary containing results and summary
    """
    # Ensure index is datetime
    pos_df.index = pd.to_datetime(pos_df.index)
    rtn_df.index = pd.to_datetime(rtn_df.index)
    
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    results = {}
    
    # Process each ticker
    for ticker in pos_df.columns:
        if ticker not in rtn_df.columns:
            print(f"Warning: Ticker {ticker} not found in return data")
            continue
            
        # Create combined DataFrame for analysis
        combined_df = pd.DataFrame({
            'position': pos_df[ticker],
            'return': rtn_df[ticker],
            'weekday': pos_df.index.weekday,
            'weekday_name': pos_df.index.weekday.map(lambda x: weekday_names[x])
        })
        
        # Remove rows with NaN values
        combined_df = combined_df.dropna()
        
        # Split by position sign
        positive_pos = combined_df[combined_df['position'] > 0]
        negative_pos = combined_df[combined_df['position'] < 0]
        zero_pos = combined_df[combined_df['position'] == 0]
        
        # Group by weekday for each position type
        pos_weekday_stats = positive_pos.groupby('weekday').agg({
            'return': ['mean', 'median', 'std', 'count'],
            'position': 'mean'
        })
        
        neg_weekday_stats = negative_pos.groupby('weekday').agg({
            'return': ['mean', 'median', 'std', 'count'],
            'position': 'mean'
        })
        
        zero_weekday_stats = zero_pos.groupby('weekday').agg({
            'return': ['mean', 'median', 'std', 'count'],
            'position': 'mean'
        }) if not zero_pos.empty else None
        
        # Add weekday names
        pos_weekday_stats.index = [weekday_names[i] for i in pos_weekday_stats.index]
        neg_weekday_stats.index = [weekday_names[i] for i in neg_weekday_stats.index]
        if zero_weekday_stats is not None:
            zero_weekday_stats.index = [weekday_names[i] for i in zero_weekday_stats.index]
        
        # Store results
        results[ticker] = {
            'positive': pos_weekday_stats,
            'negative': neg_weekday_stats,
            'zero': zero_weekday_stats,
            'all_data': combined_df
        }
        
        # Print summary statistics
        print(f"\n--- {ticker} Overnight Return Statistics by Weekday ---")
        
        print("\nPositive Positions:")
        print(pos_weekday_stats)
        
        print("\nNegative Positions:")
        print(neg_weekday_stats)
        
        if zero_weekday_stats is not None and not zero_weekday_stats.empty:
            print("\nZero Positions:")
            print(zero_weekday_stats)
    
    # Create visualizations
    fig, axes = plot_weekday_returns(results, summary_dir)
    
    # Save detailed statistics to CSV if requested
    if summary_dir is not None:
        save_weekday_stats_to_csv(results, summary_dir)
    
    return {
        'results': results,
        'figure': fig,
        'axes': axes
    }

def plot_weekday_returns(results, summary_dir=None):
    """
    Create visualizations for weekday return analysis.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_overnight_returns_by_weekday
    summary_dir : str, optional
        Directory to save figures
        
    Returns:
    --------
    tuple
        (fig, axes) - matplotlib figure and axes objects
    """
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    n_tickers = len(results)
    
    # Create figure with subplots - 2 rows (positive/negative) per ticker
    fig, axes = plt.subplots(n_tickers * 2, 1, figsize=(12, 5 * n_tickers))
    
    # Flatten axes array if there's only one ticker
    if n_tickers == 1:
        axes = axes.reshape(-1)
    
    for i, (ticker, data) in enumerate(results.items()):
        # Index for positive and negative axes
        pos_idx = i * 2
        neg_idx = i * 2 + 1
        
        # Positive positions
        if not data['positive'].empty:
            # Get days that exist in the data
            available_days = [day for day in weekdays if day in data['positive'].index]
            
            # Extract means and counts for available days
            pos_means = []
            pos_counts = []
            
            for day in available_days:
                # Get return mean and count for the day
                day_data = data['positive'].loc[day]
                pos_means.append(day_data[('return', 'mean')])
                pos_counts.append(day_data[('return', 'count')])
            
            # Handle missing days
            if len(available_days) < len(weekdays):
                missing_days = [day for day in weekdays if day not in data['positive'].index]
                print(f"Warning: {ticker} has no positive position data for {', '.join(missing_days)}")
            
            # Plot positive positions
            if available_days:  # Only plot if we have data
                ax_pos = axes[pos_idx]
                bars_pos = ax_pos.bar(available_days, pos_means, alpha=0.7, color='green')
                
                # Add count labels
                for j, bar in enumerate(bars_pos):
                    ax_pos.text(bar.get_x() + bar.get_width()/2., 0.0001, 
                               f'n={int(pos_counts[j])}', ha='center', va='bottom', fontsize=10)
                
                ax_pos.set_title(f'{ticker}: Average Overnight Return by Weekday (Positive Positions)')
                ax_pos.set_ylabel('Average Return')
                ax_pos.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax_pos.grid(True, alpha=0.3)
                
                # Add overall average line
                overall_avg = data['positive'][('return', 'mean')].mean()
                ax_pos.axhline(y=overall_avg, color='r', linestyle='-', alpha=0.5, 
                              label=f'Overall Avg: {overall_avg:.6f}')
                ax_pos.legend()
                
                # Format y-axis as percentage
                ax_pos.yaxis.set_major_formatter(PercentFormatter(1.0))
        
        # Negative positions
        if not data['negative'].empty:
            # Get days that exist in the data
            available_days = [day for day in weekdays if day in data['negative'].index]
            
            # Extract means and counts for available days
            neg_means = []
            neg_counts = []
            
            for day in available_days:
                # Get return mean and count for the day
                day_data = data['negative'].loc[day]
                neg_means.append(day_data[('return', 'mean')])
                neg_counts.append(day_data[('return', 'count')])
            
            # Handle missing days
            if len(available_days) < len(weekdays):
                missing_days = [day for day in weekdays if day not in data['negative'].index]
                print(f"Warning: {ticker} has no negative position data for {', '.join(missing_days)}")
            
            # Plot negative positions
            if available_days:  # Only plot if we have data
                ax_neg = axes[neg_idx]
                bars_neg = ax_neg.bar(available_days, neg_means, alpha=0.7, color='red')
                
                # Add count labels
                for j, bar in enumerate(bars_neg):
                    ax_neg.text(bar.get_x() + bar.get_width()/2., 0.0001, 
                               f'n={int(neg_counts[j])}', ha='center', va='bottom', fontsize=10)
                
                ax_neg.set_title(f'{ticker}: Average Overnight Return by Weekday (Negative Positions)')
                ax_neg.set_ylabel('Average Return')
                ax_neg.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax_neg.grid(True, alpha=0.3)
                
                # Add overall average line
                overall_avg = data['negative'][('return', 'mean')].mean()
                ax_neg.axhline(y=overall_avg, color='r', linestyle='-', alpha=0.5, 
                              label=f'Overall Avg: {overall_avg:.6f}')
                ax_neg.legend()
                
                # Format y-axis as percentage
                ax_neg.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.tight_layout()
    
    # Save figure if directory is provided
    if summary_dir is not None:
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
            
        # Create filename
        tickers_str = '_'.join(results.keys())
        filename = f"overnight_returns_by_weekday_{tickers_str}.png"
        filepath = os.path.join(summary_dir, filename)
        
        # Save with high DPI for better quality
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {filepath}")
    
    plt.show()
    
    return fig, axes

def save_weekday_stats_to_csv(results, summary_dir):
    """
    Save weekday statistics to CSV files.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_overnight_returns_by_weekday
    summary_dir : str
        Directory to save CSV files
    """
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    
    for ticker, data in results.items():
        # Save positive positions stats
        if not data['positive'].empty:
            pos_filename = f"overnight_returns_weekday_positive_{ticker}.csv"
            pos_filepath = os.path.join(summary_dir, pos_filename)
            data['positive'].to_csv(pos_filepath)
            print(f"Positive position stats saved to: {pos_filepath}")
        
        # Save negative positions stats
        if not data['negative'].empty:
            neg_filename = f"overnight_returns_weekday_negative_{ticker}.csv"
            neg_filepath = os.path.join(summary_dir, neg_filename)
            data['negative'].to_csv(neg_filepath)
            print(f"Negative position stats saved to: {neg_filepath}")
        
        # Save zero positions stats
        if data['zero'] is not None and not data['zero'].empty:
            zero_filename = f"overnight_returns_weekday_zero_{ticker}.csv"
            zero_filepath = os.path.join(summary_dir, zero_filename)
            data['zero'].to_csv(zero_filepath)
            print(f"Zero position stats saved to: {zero_filepath}")
    
    # Also create a summary file with all tickers' weekday stats
    summary_pos = pd.DataFrame()
    summary_neg = pd.DataFrame()
    
    for ticker, data in results.items():
        if not data['positive'].empty:
            ticker_pos = data['positive'][('return', 'mean')].copy()
            ticker_pos.name = ticker
            summary_pos = pd.concat([summary_pos, ticker_pos], axis=1)
        
        if not data['negative'].empty:
            ticker_neg = data['negative'][('return', 'mean')].copy()
            ticker_neg.name = ticker
            summary_neg = pd.concat([summary_neg, ticker_neg], axis=1)
    
    # Save summary files
    if not summary_pos.empty:
        summary_pos_path = os.path.join(summary_dir, "overnight_returns_weekday_positive_summary.csv")
        summary_pos.to_csv(summary_pos_path)
        print(f"Positive position summary saved to: {summary_pos_path}")
    
    if not summary_neg.empty:
        summary_neg_path = os.path.join(summary_dir, "overnight_returns_weekday_negative_summary.csv")
        summary_neg.to_csv(summary_neg_path)
        print(f"Negative position summary saved to: {summary_neg_path}")

# Example usage:
weekday_analysis = analyze_overnight_returns_by_weekday(overnight_positions, overnight_rtn, 
                                                      summary_dir=summary_dir)