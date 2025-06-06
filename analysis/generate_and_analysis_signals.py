# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:45:55 2024
Refactored on Fri June 06 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker
import numpy as np
from functools import partial
import os
import logging
from datetime import datetime

# Add sys path
import sys
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))

# Import local modules
from utils.datautils import align_and_sort_columns
from utils.market import index_to_futures
from utils.timeutils import parse_time_string
from utils.trade_rules import *
from data_processing.ts_trans import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(factor_dir, factor_name, fut_dir, price_name):
    """Load factor and price data"""
    logger.info(f"Loading data for factor: {factor_name}")
    factor_data = pd.read_parquet(factor_dir / f'{factor_name}.parquet')
    price_data = pd.read_parquet(fut_dir / f'{price_name}.parquet')
    
    # Rename columns and align data
    factor_data = factor_data.rename(columns=index_to_futures)[['IC', 'IF', 'IM']]
    factor_data, price_data = align_and_sort_columns([factor_data, price_data])
    
    # Align time frames
    price_data = price_data.loc[factor_data.index.min():factor_data.index.max()]
    factor_data = factor_data.reindex(price_data.index)
    
    return factor_data, price_data

def scale_factor(factor_data, scale_method, scale_window, sp, scale_quantile, direction):
    """Scale factor data using specified method"""
    logger.info(f"Scaling factor using {scale_method} method")
    scale_func = globals()[scale_method]
    scale_step = int(parse_time_string(scale_window) / parse_time_string(sp))
    
    if scale_method in ['minmax_scale', 'minmax_scale_separate']:
        factor_scaled = scale_func(factor_data, window=scale_step, quantile=scale_quantile)
    elif scale_method in ['minmax_scale_adj_by_his_rtn', 'zscore_adj_by_his_rtn_and_minmax']:
        factor_scaled = scale_func(factor_data, rtn_1p, window=scale_step, rtn_window=pp_by_sp, quantile=scale_quantile)
    elif scale_method in ['rolling_percentile']:
        factor_scaled = scale_func(factor_data, window=scale_step)
    elif scale_method in ['percentile_adj_by_his_rtn']:
        factor_scaled = scale_func(factor_data, rtn_1p, window=scale_step, rtn_window=pp_by_sp)
    
    factor_scaled = (factor_scaled - 0.5) * 2 * direction
    return factor_scaled

def apply_trade_rules(factor_scaled, trade_rule_name, trade_rule_param, trade_rule_input='array'):
    """Apply trade rules to the scaled factor with support for 'series' input"""
    logger.info(f"Applying trade rule: {trade_rule_name} with input type: {trade_rule_input}")
    
    trade_rule_func = globals()[trade_rule_name]
    
    if trade_rule_input == 'array':
        # Original approach - passing numpy arrays to the trade rule
        actual_pos = factor_scaled.apply(lambda col: trade_rule_func(col.values, **trade_rule_param), axis=0)
    elif trade_rule_input == 'series':
        # New approach - passing Series objects to the trade rule
        actual_pos = pd.DataFrame(index=factor_scaled.index, columns=factor_scaled.columns)
        
        for col in factor_scaled.columns:
            # Pass the Series directly to the trade rule function
            actual_pos[col] = trade_rule_func(factor_scaled[col], **trade_rule_param)
    else:
        raise ValueError(f"Unsupported trade_rule_input: {trade_rule_input}")
    
    return actual_pos

def generate_trade_df(position_series, price_series):
    """Generate trade records based on position changes"""
    trade_info = []
    open_price = None
    open_time = None

    for i in range(1, len(position_series)):
        prev_position = position_series.iloc[i-1]
        current_position = position_series.iloc[i]
        current_price = price_series.iloc[i]
        prev_price = price_series.iloc[i-1]

        # Record trades only when position changes
        if current_position != prev_position:
            if current_position == 1 and prev_position == 0:
                # Open long position
                open_price = current_price
                open_time = position_series.index[i]
            elif current_position == 0 and prev_position == 1:
                # Close long position
                trade_info.append({
                    'trade_type': 'long',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': current_price
                })
                open_price = None
                open_time = None
            elif current_position == -1 and prev_position == 0:
                # Open short position
                open_price = current_price
                open_time = position_series.index[i]
            elif current_position == 0 and prev_position == -1:
                # Close short position
                trade_info.append({
                    'trade_type': 'short',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': current_price
                })
                open_price = None
                open_time = None
            elif current_position == 1 and prev_position == -1:
                # Close short and open long
                trade_info.append({
                    'trade_type': 'short',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': prev_price
                })
                open_price = current_price
                open_time = position_series.index[i]
            elif current_position == -1 and prev_position == 1:
                # Close long and open short
                trade_info.append({
                    'trade_type': 'long',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': prev_price
                })
                open_price = current_price
                open_time = position_series.index[i]

    return pd.DataFrame(trade_info)

def process_trade_dfs(trade_dfs, fee):
    """Process trade DataFrames to calculate metrics"""
    for col in trade_dfs:
        trade_df = trade_dfs[col]
        if trade_df.empty:
            continue
        
        trade_df['direction'] = trade_df['trade_type'].apply(lambda x: 1 if x == 'long' else -1)
        trade_df['net_return'] = np.log(trade_df['close_price'] / trade_df['open_price']) * trade_df['direction'] - fee
        trade_df['holding_time'] = trade_df['close_time'] - trade_df['open_time']
        
        trade_dfs[col] = trade_df.dropna(subset=['net_return'])
    
    return trade_dfs

def plot_scatter(trade_df, symbol, summary_dir):
    """Plot holding time vs net return scatter plot"""
    if trade_df.empty:
        logger.warning(f"Empty trade_df for {symbol}, skipping scatter plot")
        return
    
    # Convert 'holding_time' to minutes
    trade_df['holding_time_minutes'] = trade_df['holding_time'].dt.total_seconds() / 60
    
    plt.figure(figsize=(10, 6))
    plt.scatter(trade_df['holding_time_minutes'], trade_df['net_return'], alpha=0.5, color='b')
    
    # Draw horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    
    # Set title and labels
    plt.title(f'Holding Time vs Net Return ({symbol})')
    plt.xlabel('Holding Time (minutes)')
    plt.ylabel('Net Return')
    plt.grid(True)
    
    # Save plot
    plt.tight_layout()
    scatter_plot_path = summary_dir / f'{symbol}_holding_time_vs_net_return.png'
    plt.savefig(scatter_plot_path)
    plt.close()
    logger.info(f"Saved scatter plot for {symbol}")

def plot_avg_return_by_minute(trade_df, symbol, summary_dir):
    """Plot average return by minute of day"""
    if trade_df.empty:
        logger.warning(f"Empty trade_df for {symbol}, skipping minute plot")
        return
    
    # Extract intraday time (hour and minute only)
    trade_df['open_time_minute'] = trade_df['open_time'].dt.strftime('%H:%M')

    # Aggregate by minute to calculate average return
    avg_return_by_minute = trade_df.groupby('open_time_minute')['net_return'].sum()

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    avg_return_by_minute.plot(kind='bar', color='skyblue')
    
    # Set X-axis labels to show every 30 minutes
    xticks_labels = avg_return_by_minute.index
    xticks_position = range(0, len(xticks_labels), 30)
    plt.xticks(xticks_position, [xticks_labels[i] for i in xticks_position], rotation=45)
    
    plt.title(f'Average Net Return by Minute of Day ({symbol})')
    plt.xlabel('Time (Hour:Minute)')
    plt.ylabel('Average Net Return')
    plt.grid(True)
    
    # Save plot
    plt.tight_layout()
    bar_plot_path = summary_dir / f'{symbol}_avg_net_return_by_minute.png'
    plt.savefig(bar_plot_path)
    plt.close()
    logger.info(f"Saved minute plot for {symbol}")

def plot_avg_return_by_week_day_and_minute(trade_df, symbol, summary_dir):
    """Plot average return by weekday and minute"""
    if trade_df.empty:
        logger.warning(f"Empty trade_df for {symbol}, skipping weekday plot")
        return
    
    # Extract weekday and intraday time
    trade_df['week_day'] = trade_df['open_time'].dt.weekday  # 0=Monday, 6=Sunday
    trade_df['open_time_week_minute'] = trade_df['open_time'].dt.strftime('%H:%M')
    
    # Convert weekday numbers to names
    trade_df['week_day_name'] = trade_df['week_day'].map({
        0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'
    })

    # Aggregate by weekday and 5-minute intervals
    trade_df['time_5min'] = trade_df['open_time'].dt.floor('5T').dt.strftime('%H:%M')
    
    avg_return_by_5min = trade_df.groupby(['week_day', 'time_5min'])['net_return'].apply(np.nansum).unstack(fill_value=np.nan)
    
    # Restructure data
    avg_return_by_5min = avg_return_by_5min.stack().reset_index()
    avg_return_by_5min.columns = ['Weekday', 'Time', 'Avg_Net_Return']
    
    # Create combined labels
    avg_return_by_5min['Label'] = avg_return_by_5min['Weekday'].map({
        0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'
    }) + ' ' + avg_return_by_5min['Time']

    # Keep only trading data (remove NaN values)
    avg_return_by_5min = avg_return_by_5min[avg_return_by_5min['Avg_Net_Return'].notna()]

    # Fixed time points for reference
    fixed_labels = ['09:30', '11:25']

    # Create labels for each weekday
    labels = []
    for day in range(5):  # Monday to Friday
        for time in fixed_labels:
            matching_row = avg_return_by_5min[(avg_return_by_5min['Weekday'] == day) & (avg_return_by_5min['Time'] == time)]
            if not matching_row.empty:
                labels.append(matching_row['Label'].iloc[0])

    # Plot bar chart
    plt.figure(figsize=(12, 8))
    plt.bar(avg_return_by_5min['Label'], avg_return_by_5min['Avg_Net_Return'], color='lightcoral')

    # Draw vertical lines at 9:30 positions
    for i, label in enumerate(avg_return_by_5min['Label']):
        if '09:30' in label:
            plt.axvline(x=i, color='black', linestyle='--')

    # Set X-axis labels
    plt.xticks(rotation=45)
    
    # Add title and labels
    plt.title(f'Average Net Return by Weekday and Time of Day ({symbol})')
    plt.xlabel('Weekday and Time')
    plt.ylabel('Average Net Return')
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    week_day_plot_path = summary_dir / f'{symbol}_avg_net_return_by_week_day_and_minute.png'
    plt.savefig(week_day_plot_path)
    plt.close()
    logger.info(f"Saved weekday plot for {symbol}")

def plot_return_scatter_by_minute(trade_df, symbol, summary_dir):
    """Plot scatter plot of returns by minute of day"""
    if trade_df.empty:
        logger.warning(f"Empty trade_df for {symbol}, skipping return scatter plot")
        return
    
    # Extract intraday time in minutes from market open
    # Chinese stock index futures trading hours:
    # Morning: 09:30-11:30 (120 minutes)
    # Afternoon: 13:00-15:00 (120 minutes)
    trade_df = trade_df.copy()
    trade_df['open_time_minute'] = trade_df['open_time'].dt.strftime('%H:%M')
    
    # Convert time to minutes from 09:30, handling lunch break
    def time_to_minutes_from_open(time_str):
        hour, minute = map(int, time_str.split(':'))
        total_minutes = hour * 60 + minute
        
        # Morning session: 09:30-11:30 (570-690 minutes from 00:00)
        if 570 <= total_minutes <= 690:  # 09:30 to 11:30
            return total_minutes - 570  # 0-120 minutes
        
        # Afternoon session: 13:00-15:00 (780-900 minutes from 00:00)
        elif 780 <= total_minutes <= 900:  # 13:00 to 15:00
            return total_minutes - 780 + 120  # 120-240 minutes (continuing from morning)
        
        # Outside trading hours
        else:
            return -1  # Mark as invalid
    
    trade_df['minutes_from_open'] = trade_df['open_time_minute'].apply(time_to_minutes_from_open)
    
    # Filter to trading hours only (remove invalid times)
    trade_df_filtered = trade_df[trade_df['minutes_from_open'] >= 0]
    
    if trade_df_filtered.empty:
        logger.warning(f"No trading data during market hours for {symbol}")
        return
    
    # Create scatter plot
    plt.figure(figsize=(14, 8))
    plt.scatter(trade_df_filtered['minutes_from_open'], 
               trade_df_filtered['net_return'], 
               alpha=0.6, 
               s=30,
               c='blue',
               edgecolors='black',
               linewidth=0.5)
    
    # Draw horizontal line at y=0
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # Add vertical line to separate morning and afternoon sessions
    plt.axvline(x=120, color='gray', linestyle='-', linewidth=2, alpha=0.7, label='Lunch Break')
    
    # Set title and labels
    plt.title(f'Trade Returns by Time of Day - Scatter Plot ({symbol})')
    plt.xlabel('Trading Minutes (Morning: 0-120, Afternoon: 120-240)')
    plt.ylabel('Net Return')
    plt.grid(True, alpha=0.3)
    
    # Add x-axis labels for key times
    key_minutes = [0, 30, 60, 90, 120, 150, 180, 210, 240]
    key_times = []
    for mins in key_minutes:
        if mins <= 120:  # Morning session
            total_mins = 570 + mins  # 570 is 09:30 in minutes
        else:  # Afternoon session
            total_mins = 780 + (mins - 120)  # 780 is 13:00, minus 120 to adjust for afternoon
        
        hours = total_mins // 60
        minutes = total_mins % 60
        key_times.append(f"{hours:02d}:{minutes:02d}")
    
    plt.xticks(key_minutes, key_times, rotation=45)
    plt.legend()
    
    # Add session labels
    plt.text(60, plt.ylim()[1] * 0.9, 'Morning\n09:30-11:30', 
             ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    plt.text(180, plt.ylim()[1] * 0.9, 'Afternoon\n13:00-15:00', 
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Add statistics text
    mean_return = trade_df_filtered['net_return'].mean()
    std_return = trade_df_filtered['net_return'].std()
    num_trades = len(trade_df_filtered)
    win_rate = (trade_df_filtered['net_return'] > 0).mean()
    
    # Separate statistics for morning and afternoon
    morning_trades = trade_df_filtered[trade_df_filtered['minutes_from_open'] <= 120]
    afternoon_trades = trade_df_filtered[trade_df_filtered['minutes_from_open'] > 120]
    
    stats_text = f'Total Trades: {num_trades}\nMean: {mean_return:.6f}\nStd: {std_return:.6f}\nWin Rate: {win_rate:.3f}\n\n'
    stats_text += f'Morning: {len(morning_trades)} trades\n'
    stats_text += f'Afternoon: {len(afternoon_trades)} trades'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save plot
    plt.tight_layout()
    scatter_minute_plot_path = summary_dir / f'{symbol}_return_scatter_by_minute.png'
    plt.savefig(scatter_minute_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved return scatter by minute plot for {symbol}")

def generate_all_plots(trade_dfs, summary_dir):
    """Generate all plots for each symbol"""
    logger.info("Generating all plots")
    for symbol in trade_dfs.keys():
        trade_df = trade_dfs[symbol]
        
        if trade_df.empty:
            logger.warning(f"Empty trade_df for {symbol}, skipping all plots")
            continue
        
        # 1. Holding time vs net return scatter plot
        plot_scatter(trade_df, symbol, summary_dir)
        
        # 2. Average return by minute bar chart
        plot_avg_return_by_minute(trade_df, symbol, summary_dir)
        
        # 3. Average return by weekday and time bar chart
        plot_avg_return_by_week_day_and_minute(trade_df, symbol, summary_dir)
        
        # 4. Return scatter plot by minute of day
        plot_return_scatter_by_minute(trade_df, symbol, summary_dir)

def generate_all_plots(trade_dfs, summary_dir):
    """Generate all plots for each symbol"""
    logger.info("Generating all plots")
    for symbol in trade_dfs.keys():
        trade_df = trade_dfs[symbol]
        
        if trade_df.empty:
            logger.warning(f"Empty trade_df for {symbol}, skipping all plots")
            continue
        
        # 1. Holding time vs net return scatter plot
        plot_scatter(trade_df, symbol, summary_dir)
        
        # 2. Average return by minute bar chart
        plot_avg_return_by_minute(trade_df, symbol, summary_dir)
        
        # 3. Average return by weekday and time bar chart
        plot_avg_return_by_week_day_and_minute(trade_df, symbol, summary_dir)
        
        # 4. Return scatter plot by minute of day
        plot_return_scatter_by_minute(trade_df, symbol, summary_dir)

def save_trade_dfs(trade_dfs, save_dir, factor_name):
    """Save trade DataFrames to the specified directory"""
    factor_save_dir = save_dir / factor_name
    factor_save_dir.mkdir(parents=True, exist_ok=True)
    
    for col, df in trade_dfs.items():
        if df.empty:
            logger.warning(f"Empty DataFrame for {col}, skipping")
            continue
        
        save_path = factor_save_dir / f"{col}_trades.parquet"
        df.to_parquet(save_path)
        logger.info(f"Saved trade records for {col} to {save_path}")
    
    # Save combined data
    all_trades = pd.concat([df.assign(instrument=col) for col, df in trade_dfs.items() if not df.empty])
    if not all_trades.empty:
        all_trades.to_parquet(factor_save_dir / "all_trades.parquet")
        logger.info(f"Saved combined trade records to {factor_save_dir / 'all_trades.parquet'}")

def run_factor_analysis(config):
    """Run the complete factor analysis pipeline for a single factor configuration"""
    logger.info(f"Starting analysis for factor: {config['factor_name']}")
    
    # Load data
    factor_data, price_data = load_data(
        config['factor_dir'], 
        config['factor_name'], 
        config['fut_dir'], 
        config['price_name']
    )
    
    # Scale factor
    factor_scaled = scale_factor(
        factor_data, 
        config['scale_method'], 
        config['scale_window'], 
        config['sp'], 
        config['scale_quantile'], 
        config['direction']
    )
    
    # Apply trade rules
    actual_pos = apply_trade_rules(
        factor_scaled, 
        config['trade_rule_name'], 
        config['trade_rule_param'],
        config.get('trade_rule_input', 'array')
    )
    
    # Generate trade records for each instrument
    trade_dfs = {}
    for col in actual_pos.columns:
        trade_df = generate_trade_df(actual_pos[col], price_data[col])
        trade_dfs[col] = trade_df
    
    # Process trade records
    trade_dfs = process_trade_dfs(trade_dfs, config['fee'])
    
    # Create summary directory for plots
    summary_dir = config['analysis_dir'] / config['factor_name']
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    generate_all_plots(trade_dfs, summary_dir)
    
    # Save trade records
    save_trade_dfs(trade_dfs, config['save_dir'], config['factor_name'])
    
    return trade_dfs

def run_multiple_factors(factor_configs, base_config):
    """Run analysis for multiple factor configurations"""
    results = {}
    
    for factor_config in factor_configs:
        # Combine base config with specific factor config
        config = {**base_config, **factor_config}
        
        # Create a unique identifier for this run
        run_id = f"{config['factor_name']}_{config['direction']}"
        
        try:
            # Run analysis
            trade_dfs = run_factor_analysis(config)
            results[run_id] = trade_dfs
            
            # Log summary statistics
            if any(not df.empty for df in trade_dfs.values()):
                all_trades = pd.concat([df for df in trade_dfs.values() if not df.empty])
                logger.info(f"Factor {run_id} - Total trades: {len(all_trades)}")
                logger.info(f"Factor {run_id} - Avg return: {all_trades['net_return'].mean():.6f}")
                logger.info(f"Factor {run_id} - Win rate: {(all_trades['net_return'] > 0).mean():.4f}")
            else:
                logger.warning(f"No trades generated for {run_id}")
                
        except Exception as e:
            logger.error(f"Error processing factor {run_id}: {str(e)}", exc_info=True)
    
    return results


if __name__ == "__main__":
    # Base configuration - all common parameters including trade rules
    base_config = {
        # Data paths
        'fut_dir': Path('/mnt/nfs/30.132_xt_data1/futuretwap'),
        'price_name': 't1min_fq1min_dl1min',
        
        # Scaling parameters
        'scale_method': 'minmax_scale',
        'scale_window': '240d',
        'scale_quantile': 0.02,
        'sp': '1min',
        
        # Trading parameters - all factors will use these by default
        'trade_rule_name': 'trade_rule_by_trigger_v3_4',  # Use newer trade rule as default
        'trade_rule_input': 'series',  # Use 'series' input for newer trade rules
        'trade_rule_param': {
            'threshold_combinations': [[0.8, 0.0]],
            'time_threshold_minutes': 240,
            'close_long': True,
            'close_short': True
        },
        'fee': 0.00024,
        
        # Output paths
        'analysis_dir': Path(r'/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/analysis/trade_signals'),
        'save_dir': Path(r'/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/analysis/trade_signals')
    }
    
    # Alternative trade rule configurations (comment/uncomment as needed)
    # For older trade rules:
    # base_config.update({
    #     'trade_rule_name': 'trade_rule_by_trigger_v0',
    #     'trade_rule_input': 'array',
    #     'trade_rule_param': {
    #         'openthres': 0.6,
    #         'closethres': 0,
    #     }
    # })
    
    # Factor configurations to run - only specify factor-specific parameters
    factor_configs = [
        {
            'factor_name': 'avg_predict_210401_250401',
            'direction': 1,
            'factor_dir': Path(rf'/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/merge_selected_factors/batch_till20_newma_batch_test_v3_icim_nsr22_m2/210401_250401')
        },
        # Add more factor configurations as needed - all will use the same trade rules from base_config
    ]
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_config['save_dir'] / f"batch_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    file_handler = logging.FileHandler(run_dir / "run_log.txt")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting batch analysis for {len(factor_configs)} factor configurations")
    
    # Update save_dir to use the timestamped directory
    base_config['save_dir'] = run_dir
    base_config['analysis_dir'] = run_dir
    
    # Run analysis
    results = run_multiple_factors(factor_configs, base_config)
    
    logger.info("Batch analysis completed")