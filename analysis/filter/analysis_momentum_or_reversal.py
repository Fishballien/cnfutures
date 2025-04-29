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
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from functools import partial
from typing import Dict, Tuple, List
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from tqdm import tqdm


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[2]
sys.path.append(str(project_dir))


# %%
from utils.datautils import align_and_sort_columns
from utils.market import index_to_futures
# from trans_operators.format import to_float32


from utils.timeutils import parse_time_string
from utils.trade_rules import *
from data_processing.ts_trans import *


# %%
def process_factor_data(factor_data: pd.DataFrame, 
                        price_data: pd.DataFrame, 
                        direction: int = 1,
                        scale_method: str = 'minmax_scale',
                        scale_window: str = '240d',
                        scale_quantile: float = 0.02,
                        sp: str = '1min',
                        trade_rule_name: str = 'trade_rule_by_trigger_v0',
                        trade_rule_param: Dict = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process factor data, scale it, and generate positions.
    
    Parameters:
    -----------
    factor_data : pd.DataFrame
        DataFrame containing factor data
    price_data : pd.DataFrame
        DataFrame containing price data
    direction : int, default 1
        Direction of the factor (1 or -1)
    scale_method : str, default 'minmax_scale'
        Method to scale the factor
    scale_window : str, default '240d'
        Window size for scaling
    scale_quantile : float, default 0.02
        Quantile for scaling
    sp : str, default '1min'
        Sampling period
    trade_rule_name : str, default 'trade_rule_by_trigger_v0'
        Name of the trade rule function
    trade_rule_param : Dict, default None
        Parameters for the trade rule
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Processed price data, scaled factor data, and actual positions
    """
    if trade_rule_param is None:
        trade_rule_param = {
            'openthres': 0.8,
            'closethres': 0,
        }
    
    # Align data
    factor_data = factor_data.rename(columns=index_to_futures)[['IC', 'IF', 'IM']]
    factor_data, price_data = align_and_sort_columns([factor_data, price_data])
    price_data = price_data.loc[factor_data.index.min():factor_data.index.max()]  # Trim price data to factor range
    factor_data = factor_data.reindex(price_data.index)  # Reindex factor data to price datartn_1p = twap_price.pct_change(1, fill_method=None).shift(-1) / 1
    rtn_1p = price_data.pct_change(1, fill_method=None).shift(-1) / 1
    rtn_1p = rtn_1p.replace([np.inf, -np.inf], 0)
    rtn_1p = rtn_1p.loc[factor_data.index.min():factor_data.index.max()]
    
    # Scale the factor
    scale_step = int(parse_time_string(scale_window) / parse_time_string(sp))
    
    if scale_method in ['minmax_scale', 'minmax_scale_separate']:
        factor_scaled = globals()[scale_method](factor_data, window=scale_step, quantile=scale_quantile)
    elif scale_method in ['minmax_scale_adj_by_his_rtn', 'zscore_adj_by_his_rtn_and_minmax']:
        rtn_1p = price_data.pct_change(1, fill_method=None)
        pp_by_sp = int(1440 / parse_time_string(sp))  # Assuming 1 day = 1440 minutes
        factor_scaled = globals()[scale_method](factor_data, rtn_1p, window=scale_step, rtn_window=pp_by_sp, quantile=scale_quantile)
    elif scale_method in ['rolling_percentile']:
        factor_scaled = globals()[scale_method](factor_data, window=scale_step)
    elif scale_method in ['percentile_adj_by_his_rtn']:
        rtn_1p = price_data.pct_change(1, fill_method=None)
        pp_by_sp = int(1440 / parse_time_string(sp))
        factor_scaled = globals()[scale_method](factor_data, rtn_1p, window=scale_step, rtn_window=pp_by_sp)
    
    # Transform to [-1, 1] range
    factor_scaled = (factor_scaled - 0.5) * 2
    
    # Generate positions
    trade_rule_func = partial(globals()[trade_rule_name], **trade_rule_param)
    actual_pos = factor_scaled.apply(lambda col: trade_rule_func(col.values), axis=0)
    
    # direction
    gp = (factor_scaled * rtn_1p)
    gp['return'] = gp.mean(axis=1)
    direction = 1 if gp['return'].sum() > 0 else -1
    
    return price_data, factor_scaled * direction, actual_pos * direction


# 将处理函数移到主函数外部，使其可以被 pickle
def process_single_factor(factor_name, price_data, factor_dir, lookback_windows):
    try:
        # Load factor data
        factor_data = pd.read_parquet(factor_dir / f'{factor_name}.parquet')
        
        # Process data
        price_data_proc, factor_scaled, actual_pos = process_factor_data(
            factor_data=factor_data, 
            price_data=price_data
        )
        
        factor_results = {}
        
        # Calculate correlations for each lookback window
        for window in lookback_windows:
            # Calculate historical returns for current window
            his_rtn = price_data_proc.pct_change(window, fill_method=None)
            
            # Calculate correlations
            # 将两个序列中的 NA 值都填充为 0
            his_rtn_filled = his_rtn.fillna(0)
            factor_scaled_filled = factor_scaled.fillna(0)
            
            # 然后计算相关系数
            corr_cont = his_rtn_filled.corrwith(factor_scaled_filled, drop=True)
             
            # Rename columns to include window size
            for col in corr_cont.index:
                factor_results[f"{col}_{window}"] = corr_cont[col]
        
        # Return results as DataFrame
        return pd.DataFrame([factor_results], index=[factor_name])
    except Exception as e:
        # Handle exceptions in worker processes
        print(f"Error processing factor {factor_name}: {str(e)}")
        return pd.DataFrame([], index=[factor_name])

def calculate_correlations(factors_info: List[str], 
                          price_data: pd.DataFrame,
                          factor_dir: Path,
                          lookback_windows: List[int] = [240],
                          max_workers: int = None) -> pd.DataFrame:
    """
    Calculate correlations between factors and historical returns for multiple factors and instruments
    using parallel processing with progress bar.
    
    Parameters:
    -----------
    factors_info : List[str]
        List of factor names
    price_data : pd.DataFrame
        DataFrame containing price data
    factor_dir : Path
        Directory containing factor data files
    lookback_windows : List[int], default [240]
        List of lookback windows for historical returns
    max_workers : int, default None
        Maximum number of worker processes. If None, uses default based on system.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with correlations for each factor and instrument and each lookback window
    """
    # Use ProcessPoolExecutor to process factors in parallel
    results = []
    total_factors = len(factors_info)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks using partial to pass additional arguments
        futures = []
        for factor_name in factors_info:
            futures.append(
                executor.submit(
                    process_single_factor, 
                    factor_name, 
                    price_data, 
                    factor_dir, 
                    lookback_windows
                )
            )
        
        # Process completed tasks as they complete with tqdm progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=total_factors,
                          desc="Processing factors"):
            try:
                result = future.result()
                if not result.empty:
                    results.append(result)
            except Exception as e:
                print(f"Error retrieving result: {str(e)}")
    
    # Combine results
    if results:
        result_df = pd.concat(results)
        return result_df
    else:
        return pd.DataFrame()
    
    
# %%
# Example usage:
factor_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\sample_data\factors\1_2')
fut_dir = Path('/mnt/data1/futuretwap')
price_name = 't1min_fq1min_dl1min'
price_data = pd.read_parquet(fut_dir / f'{price_name}.parquet')
save_dir = Path()
save_dir.mkdir(exist_ok=True, parents=True)

# 获取factor_dir目录中所有的.parquet文件并提取文件名作为factors_to_test
factors_to_test = [f.stem for f in factor_dir.glob('*.parquet')]
print(f"发现的因子文件: {factors_to_test}")

# 定义多个回看窗口
lookback_windows = [30, 60, 120, 240, 480]

# =============================================================================
# correlation_table = calculate_correlations(
#     factors_info=factors_to_test,
#     price_data=price_data,
#     factor_dir=factor_dir,
#     lookback_windows=lookback_windows
# )
# print(correlation_table)
# correlation_table.to_csv(save_dir / '1_2_factor_hisrtn_corr.csv')
# =============================================================================

factor_name = factors_to_test[7]
res = process_single_factor(factor_name, price_data, factor_dir, lookback_windows)