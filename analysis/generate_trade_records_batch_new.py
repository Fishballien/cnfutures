# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 17:20:18 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
from pathlib import Path
import pandas as pd
import numpy as np
from functools import partial
import os
import logging
from datetime import datetime

#  add sys path
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
        # For trade_rule_by_trigger_v3_4, we need to pass the Series with datetime index
        actual_pos = pd.DataFrame(index=factor_scaled.index, columns=factor_scaled.columns)
        
        for col in factor_scaled.columns:
            # Pass the Series directly to the trade rule function
            actual_pos[col] = trade_rule_func(factor_scaled[col], **trade_rule_param)
    else:
        raise ValueError(f"Unsupported trade_rule_input: {trade_rule_input}")
    
    return actual_pos

def generate_trade_df(position_series, price_series, feature_series):
    """Generate trade records based on position changes, and record feature values at entry"""
    trade_info = []
    open_price = None
    open_time = None
    open_feature = None

    for i in range(1, len(position_series)):
        prev_position = position_series.iloc[i-1]
        current_position = position_series.iloc[i]
        current_price = price_series.iloc[i]
        prev_price = price_series.iloc[i-1]
        current_feature = feature_series.iloc[i] if i < len(feature_series) else None

        # Record trades only when position changes
        if current_position != prev_position:
            if current_position == 1 and prev_position == 0:
                # Open long position
                open_price = current_price
                open_time = position_series.index[i]
                open_feature = current_feature
            elif current_position == 0 and prev_position == 1:
                # Close long position
                trade_info.append({
                    'trade_type': 'long',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': current_price,
                    'open_feature': open_feature
                })
                open_price = None
                open_time = None
                open_feature = None
            elif current_position == -1 and prev_position == 0:
                # Open short position
                open_price = current_price
                open_time = position_series.index[i]
                open_feature = current_feature
            elif current_position == 0 and prev_position == -1:
                # Close short position
                trade_info.append({
                    'trade_type': 'short',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': current_price,
                    'open_feature': open_feature
                })
                open_price = None
                open_time = None
                open_feature = None
            elif current_position == 1 and prev_position == -1:
                # Close short and open long
                trade_info.append({
                    'trade_type': 'short',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': prev_price,
                    'open_feature': open_feature
                })
                open_price = current_price
                open_time = position_series.index[i]
                open_feature = current_feature
            elif current_position == -1 and prev_position == 1:
                # Close long and open short
                trade_info.append({
                    'trade_type': 'long',
                    'open_time': open_time,
                    'close_time': position_series.index[i],
                    'open_price': open_price,
                    'close_price': prev_price,
                    'open_feature': open_feature
                })
                open_price = current_price
                open_time = position_series.index[i]
                open_feature = current_feature

    return pd.DataFrame(trade_info)

def process_trade_dfs(trade_dfs, feature_name, fee):
    """Process trade DataFrames to calculate metrics"""
    for col in trade_dfs:
        trade_df = trade_dfs[col]
        if trade_df.empty:
            continue
        
        trade_df['direction'] = trade_df['trade_type'].apply(lambda x: 1 if x == 'long' else -1)
        trade_df['net_return'] = np.log(trade_df['close_price'] / trade_df['open_price']) * trade_df['direction'] - fee
        trade_df['holding_time'] = trade_df['close_time'] - trade_df['open_time']
        
        # Rename open_feature column to feature_name value
        if 'open_feature' in trade_df.columns:
            trade_df[feature_name] = trade_df['open_feature']
            trade_df = trade_df.drop(columns=['open_feature'])
        
        trade_dfs[col] = trade_df.dropna(subset=['net_return'])
    
    return trade_dfs

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
    
    # Apply trade rules with the specified input type
    actual_pos = apply_trade_rules(
        factor_scaled, 
        config['trade_rule_name'], 
        config['trade_rule_param'],
        config.get('trade_rule_input', 'array')  # Default to 'array' for backward compatibility
    )
    
    # Load feature data
    feature = pd.read_parquet(config['feature_dir'] / f"{config['feature_name']}.parquet")
    feature = feature.reindex(price_data.index)
    
    # Generate trade records for each instrument
    trade_dfs = {}
    for col in actual_pos.columns:
        if isinstance(feature, pd.DataFrame) and col in feature.columns:
            feature_col = feature[col]
        else:
            feature_col = feature[config['feature_col_name']]
        
        trade_df = generate_trade_df(actual_pos[col], price_data[col], feature_col)
        trade_dfs[col] = trade_df
    
    # Process trade records
    trade_dfs = process_trade_dfs(trade_dfs, config['feature_name'], config['fee'])
    
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
    # Base configuration
    base_config = {
        'fut_dir': Path('/mnt/30.132_xt_data1/futuretwap'),
        'price_name': 't1min_fq1min_dl1min',
        'feature_dir': Path(r'/mnt/30.132_xt_data1/idx_opt_processed/v0_features'),
        'feature_name': 'atm_vol',
        'feature_col_name': 'IO',
        'scale_method': 'minmax_scale',
        'scale_window': '240d',
        'scale_quantile': 0.02,
        'sp': '1min',
        'trade_rule_name': 'trade_rule_by_trigger_v3_4',
        'trade_rule_input': 'series',  # Using 'series' input mode
        'trade_rule_param': {
            'threshold_combinations': [[0.8, 0.0]],
            'time_threshold_minutes': 240,
            'close_long': True,
            'close_short': True
        },
        'fee': 0.00024,
        'save_dir': Path(r'/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/analysis/trades')
    }
    
    # Factor configurations to run
    factor_configs = [
        {
            'factor_name': 'TimeRangeValueOrderAmount_p1.0_v200000_t30-wavg_imb01_dp2-rollingAggMinutePctl_w245d_i5',
            'direction': 1,
            'factor_dir': Path(r'/mnt/Data/xintang/index_factors/Batch18_250425/normal_trans_for_test_250414')
        },
        # Add more factor configurations as needed
    ]
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_config['save_dir'] # / f"batch_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    file_handler = logging.FileHandler(run_dir / "run_log.txt")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting batch analysis for {len(factor_configs)} factor configurations")
    
    # Update save_dir to use the timestamped directory
    base_config['save_dir'] = run_dir
    
    # Run analysis
    results = run_multiple_factors(factor_configs, base_config)
    
    logger.info("Batch analysis completed")