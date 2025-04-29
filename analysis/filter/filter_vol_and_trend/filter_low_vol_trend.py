# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:42:15 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
"""
Modular Factor Processing with Parameter Combinations

Refactored version of the original code with:
1) Modular functions for each component
2) Support for parameter combinations (vol_T, vol_k, trend_k, trend_th)
3) Support for processing multiple factors from a dictionary input
"""
# %% imports
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from typing import Dict, List, Tuple, Union, Optional, Any
import warnings
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[3]
sys.path.append(str(project_dir))


# %%
from utils.datautils import align_and_sort_columns
from utils.market import index_to_futures
from utils.timeutils import parse_time_string
from data_processing.ts_trans import *
from test_and_eval.factor_tester import FactorTesterByDiscrete


# %% Utility functions
def vol_filter_sigmoid(vol, T=0.01, k=800):
    """
    Sigmoid function transformation for volatility filtering
    
    Parameters:
    -----------
    vol : float or numpy.ndarray
        Volatility values
    T : float, optional
        Threshold value, default is 0.01
    k : float, optional
        Steepness coefficient, default is 800
        
    Returns:
    --------
    float or numpy.ndarray
        Filtered values between 0 and 1
    """
    return 1 / (1 + np.exp(-k * (vol - T)))


def trend_consistency_filter(trend_score, k=2000, trend_th=0.0005):
    """
    Calculate trend consistency gate value
    
    Parameters:
    -----------
    trend_score : float or numpy.ndarray
        Trend consistency score
    k : float, optional
        Sigmoid function steepness coefficient, default is 2000
    trend_th : float, optional
        Trend threshold, default is 0.0005
        
    Returns:
    --------
    float or numpy.ndarray
        Trend consistency gate value between 0 and 1
    """
    return 1 / (1 + np.exp(k * (trend_score - trend_th)))


def soft_and(vol_filter, trend_filter):
    """
    Soft AND operation: 1 - (1 - vol_filter) * (1 - trend_filter)
    
    Parameters:
    -----------
    vol_filter : float or numpy.ndarray
        Volatility filter values
    trend_filter : float or numpy.ndarray
        Trend filter values
        
    Returns:
    --------
    float or numpy.ndarray
        Combined filter values
    """
    return 1 - (1 - vol_filter) * (1 - trend_filter)


def load_factor_data(factor_dir: Path, factor_name: str, direction: int = 1) -> pd.DataFrame:
    """
    Load and prepare factor data
    
    Parameters:
    -----------
    factor_dir : Path
        Directory containing factor data
    factor_name : str
        Name of the factor file (without extension)
    direction : int, optional
        Direction of the factor (1 or -1), default is 1
        
    Returns:
    --------
    pd.DataFrame
        Loaded factor data
    """

    factor_data = pd.read_parquet(factor_dir / f'{factor_name}.parquet')
    factor_data = factor_data.rename(columns=index_to_futures)[['IC', 'IF', 'IM']]
    return factor_data


def load_price_data(price_dir: Path, price_name: str, factor_data: pd.DataFrame) -> pd.DataFrame:
    """
    Load and align price data with factor data
    
    Parameters:
    -----------
    price_dir : Path
        Directory containing price data
    price_name : str
        Name of the price file (without extension)
    factor_data : pd.DataFrame
        Factor data to align with
        
    Returns:
    --------
    pd.DataFrame
        Aligned price data
    """

    price_data = pd.read_parquet(price_dir / f'{price_name}.parquet')
    factor_data, price_data = align_and_sort_columns([factor_data, price_data])
    
    # Trim price data to factor data range and ensure equal length
    price_data = price_data.loc[factor_data.index.min():factor_data.index.max()]
    factor_data_aligned = factor_data.reindex(price_data.index)
    
    return price_data, factor_data_aligned


def scale_factor(factor_data: pd.DataFrame, 
                scale_method: str, 
                scale_window: str, 
                scale_quantile: float,
                sp: str,
                direction: int) -> pd.DataFrame:
    """
    Scale factor data using the specified method
    
    Parameters:
    -----------
    factor_data : pd.DataFrame
        Factor data to scale
    scale_method : str
        Scaling method to use
    scale_window : str
        Time window for scaling
    scale_quantile : float
        Quantile for scaling
    sp : str
        Sampling period
    direction : int
        Direction of the factor (1 or -1)
        
    Returns:
    --------
    pd.DataFrame
        Scaled factor data
    """

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


def load_volatility_feature(feature_dir: Path, 
                           feature_name: str, 
                           feature_col_name: str,
                           price_data: pd.DataFrame) -> pd.Series:
    """
    Load volatility feature and align with price data
    
    Parameters:
    -----------
    feature_dir : Path
        Directory containing feature data
    feature_name : str
        Name of the feature file
    feature_col_name : str
        Column name of the feature
    price_data : pd.DataFrame
        Price data to align with
        
    Returns:
    --------
    pd.Series
        Volatility feature aligned with price data
    """
    feature_path = feature_dir / f'{feature_name}.parquet'
    feature = pd.read_parquet(feature_path)
    feature_series = feature[feature_col_name].reindex(price_data.index)
    return feature_series


def calculate_vol_filter(feature_series: pd.Series, vol_T: float, vol_k: float) -> pd.Series:
    """
    Calculate volatility filter
    
    Parameters:
    -----------
    feature_series : pd.Series
        Volatility feature series
    vol_T : float
        Volatility threshold
    vol_k : float
        Volatility steepness coefficient
        
    Returns:
    --------
    pd.Series
        Volatility filter
    """
    return vol_filter_sigmoid(feature_series, T=vol_T, k=vol_k)


def calculate_trend_score(price_data: pd.DataFrame, factor_scaled: pd.DataFrame, 
                         trend_weight_dict: Dict[int, float]) -> pd.DataFrame:
    """
    Calculate trend score based on price changes and factor direction
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Price data
    factor_scaled : pd.DataFrame
        Scaled factor data
    trend_weight_dict : Dict[int, float]
        Dictionary of lookback periods and their weights
        
    Returns:
    --------
    pd.DataFrame
        Trend score
    """
    # Calculate percentage changes for different lookback periods
    price_data_pct_change = {lookback: price_data.shift(2).pct_change(lookback) 
                            for lookback in trend_weight_dict}
    
    # Calculate total weight
    total_weight = sum(trend_weight_dict.values())
    
    # Initialize weighted average percentage change DataFrame
    sample_key = list(price_data_pct_change.keys())[0]
    columns = price_data_pct_change[sample_key].columns
    index = price_data_pct_change[sample_key].index
    weighted_avg_pct_change = pd.DataFrame(0, index=index, columns=columns)
    
    # Calculate weighted average percentage change
    for lookback, weight in trend_weight_dict.items():
        normalized_weight = weight / total_weight
        weighted_avg_pct_change = weighted_avg_pct_change + price_data_pct_change[lookback] * normalized_weight
    
    # Calculate trend score
    factor_direction = factor_scaled.apply(np.sign)
    trend_score = weighted_avg_pct_change * factor_direction
    
    return trend_score


def calculate_trend_filter(trend_score: pd.DataFrame, trend_k: float, trend_th: float) -> pd.DataFrame:
    """
    Calculate trend filter
    
    Parameters:
    -----------
    trend_score : pd.DataFrame
        Trend score
    trend_k : float
        Trend steepness coefficient
    trend_th : float
        Trend threshold
        
    Returns:
    --------
    pd.DataFrame
        Trend filter
    """
    return trend_consistency_filter(trend_score, k=trend_k, trend_th=trend_th)


def calculate_combined_filter(vol_filter: pd.Series, trend_filter: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate combined filter using soft AND operation
    
    Parameters:
    -----------
    vol_filter : pd.Series
        Volatility filter
    trend_filter : pd.DataFrame
        Trend filter
        
    Returns:
    --------
    pd.DataFrame
        Combined filter
    """
    # Create a new DataFrame with the same structure as trend_filter
    combined_filter = pd.DataFrame(index=trend_filter.index, columns=trend_filter.columns)
    
    # Apply soft_and function to each column of trend_filter with vol_filter
    for column in trend_filter.columns:
        # Handle NaN values properly
        valid_mask = ~(pd.isna(vol_filter) | pd.isna(trend_filter[column]))
        result = pd.Series(index=trend_filter.index, dtype='float64')
        
        # Only calculate soft_and where both inputs are valid numbers
        result.loc[valid_mask] = soft_and(
            vol_filter.loc[valid_mask],
            trend_filter.loc[valid_mask, column]
        )
        
        # Assign to the result DataFrame
        combined_filter[column] = result
    
    return combined_filter


def apply_filter_and_save(factor_scaled: pd.DataFrame, 
                         combined_filter: pd.DataFrame,
                         org_fac_dir: Path,
                         filtered_fac_dir: Path,
                         factor_config: Dict,
                         params: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply combined filter to scaled factor and save results
    
    Parameters:
    -----------
    factor_scaled : pd.DataFrame
        Scaled factor data
    combined_filter : pd.DataFrame
        Combined filter
    filtered_fac_dir : Path
        Directory to save filtered factor data
    factor_config : Dict
        Factor configuration
    params : Dict
        Filter parameters
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Filtered factor and original scaled factor
    """
    simplified_name = factor_config['simplified_name']
    vol_T = params['vol_T']
    vol_k = params['vol_k']
    trend_k = params['trend_k']
    trend_th = params['trend_th']
    
    # Apply filter and save
    filtered_factor = factor_scaled * combined_filter
    filtered_factor.to_parquet(filtered_fac_dir / f'{simplified_name}_vT{vol_T}_vk{vol_k}_tk{trend_k}_tth{trend_th}.parquet')
    
    # Drop NaN rows and reindex original factor
    filtered_factor = filtered_factor.dropna(how='all')
    factor_scaled_org = factor_scaled.reindex(index=filtered_factor.index)
    factor_scaled_org.to_parquet(org_fac_dir / f'{simplified_name}_org.parquet')
    
    return filtered_factor, factor_scaled_org


def run_factor_tests(org_fac_dir: Path, filtered_fac_dir: Path, 
                    save_dir: Path,
                    simplified_name: str,
                    test_name: str,
                    test_name_org: str,
                    test_workers: int) -> None:
    """
    Run factor tests on filtered and original factors
    
    Parameters:
    -----------
    filtered_fac_dir : Path
        Directory containing filtered factor data
    save_dir : Path
        Directory to save test results
    simplified_name : str
        Simplified name of the factor
    test_name : str
        Name for the filtered factor test
    test_name_org : str
        Name for the original factor test
    test_workers : int
        Number of workers for testing
    """

    # Test original factor
    tester = FactorTesterByDiscrete(
        None, None, org_fac_dir, 
        test_name=test_name_org, 
        result_dir=save_dir, 
        n_workers=test_workers
    )
    tester.test_one_factor(f'{simplified_name}_org')
    
    # Test filtered factors
    tester = FactorTesterByDiscrete(
        None, None, filtered_fac_dir, 
        test_name=test_name, 
        result_dir=save_dir, 
        n_workers=test_workers
    )
    tester.test_multi_factors(skip_exists=False)


def process_factor(factor_config: Dict, 
                  param_grid: Dict[str, List[float]],
                  price_dir: Path,
                  price_name: str,
                  feature_dir: Path,
                  feature_name: str,
                  feature_col_name: str,
                  scale_method: str,
                  scale_window: str,
                  scale_quantile: float,
                  sp: str,
                  trend_weight_dict: Dict[int, float],
                  save_base_dir: Path,
                  test_name: str,
                  test_name_org: str,
                  test_workers: int) -> None:
    """
    Process a single factor with all parameter combinations
    
    Parameters:
    -----------
    factor_config : Dict
        Factor configuration
    param_grid : Dict[str, List[float]]
        Grid of parameters for filtering
    price_dir : Path
        Directory containing price data
    price_name : str
        Name of the price file
    feature_dir : Path
        Directory containing feature data
    feature_name : str
        Name of the feature file
    feature_col_name : str
        Column name of the feature
    scale_method : str
        Scaling method to use
    scale_window : str
        Time window for scaling
    scale_quantile : float
        Quantile for scaling
    sp : str
        Sampling period
    trend_weight_dict : Dict[int, float]
        Dictionary of lookback periods and their weights
    save_base_dir : Path
        Base directory to save results
    test_name : str
        Name for the filtered factor test
    test_name_org : str
        Name for the original factor test
    test_workers : int
        Number of workers for testing
    """
    factor_name = factor_config['factor_name']
    factor_dir = Path(factor_config['factor_dir'])
    direction = factor_config['direction']
    simplified_name = factor_config['simplified_name']
    
    # Create save directories
    save_dir = save_base_dir / feature_name / simplified_name
    filtered_fac_dir = save_dir / 'filtered_fac'
    org_fac_dir = save_dir / 'org_fac'
    filtered_fac_dir.mkdir(parents=True, exist_ok=True)
    org_fac_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    factor_data = load_factor_data(factor_dir, factor_name, direction)
    price_data, factor_data_aligned = load_price_data(price_dir, price_name, factor_data)
    
    # Scale factor
    factor_scaled = scale_factor(
        factor_data_aligned, scale_method, scale_window, scale_quantile, sp, direction
    )
    
    # Load volatility feature
    feature_series = load_volatility_feature(feature_dir, feature_name, feature_col_name, price_data)
    
    # Calculate trend score
    trend_score = calculate_trend_score(price_data, factor_scaled, trend_weight_dict)
    
    # Generate all parameter combinations
    params_list = [
        dict(zip(param_grid.keys(), values)) 
        for values in itertools.product(*param_grid.values())
    ]
    
    # Process each parameter combination
    for params in tqdm(params_list, 'generating filtered'):
        vol_filter = calculate_vol_filter(feature_series, params['vol_T'], params['vol_k'])
        trend_filter = calculate_trend_filter(trend_score, params['trend_k'], params['trend_th'])
        combined_filter = calculate_combined_filter(vol_filter, trend_filter)
        
        filtered_factor, factor_scaled_org = apply_filter_and_save(
            factor_scaled, combined_filter, org_fac_dir, filtered_fac_dir, factor_config, params
        )
    
    # Run factor tests
    run_factor_tests(
        org_fac_dir, filtered_fac_dir, save_dir, simplified_name, test_name, test_name_org, test_workers
    )


def main():
    """Main function to process all factors with parameter combinations"""
    # List of factors to process
    factors = [
        # {
        #     'factor_name': 'trade1',
        #     'factor_dir': r'/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/model/trade1',
        #     'direction': 1,
        #     'simplified_name': 'trade1'
        # },
        {
            'factor_name': 'predict_avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18',
            'factor_dir': r'/mnt/30.132_xt_data1/xintang/CNIndexFutures/timeseries/factor_test/results/model/avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18/predict',
            'direction': 1,
            'simplified_name': 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18'
        },
        # {
        #     'factor_name': 'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb01_dp2-rollingAggMinuteMinMaxScale_w30d_q0_i10',
        #     'factor_dir': r'/mnt/30.132_xt_data1/xintang/index_factors/Batch10_fix_best_241218_selected_f64/v1.2_all_trans_3',
        #     'direction': 1,
        #     'simplified_name': 'VTDOA_imb01_dp2'
        # },
        # {
        #     'factor_name': 'amount_Dollar_LX_R3_dp2_SumIntraRm1_R4_Imb1_R5_IntraQtl_30_R6_IntraRm_20',
        #     'factor_dir': r'/mnt/30.156_tf_trade_order/01_addcomb_R3_dp2_basicIntraRm_R4_imb_R5_ts_basic_R6_ts_basic',
        #     'direction': 1,
        #     'simplified_name': 'trade_01'
        # },
        # {
        #     'factor_name': 'LargeOrderAmountByValue_p1.0_v40000-avg_imb01_dp2-org',
        #     'factor_dir': r'/mnt/30.132_xt_data1/xintang/index_factors/Batch10_fix_best_241218_selected_f64/v1.2_all_trans_3',
        #     'direction': 1,
        #     'simplified_name': 'LOA'
        # },
        # {
        #     'factor_name': 'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb04_dp2-org',
        #     'factor_dir': r'/mnt/30.132_xt_data1/xintang/index_factors/Batch10_fix_best_241218_selected_f64/v1.2_all_trans_3',
        #     'direction': -1,
        #     'simplified_name': 'VTDOA'
        # },
        # {
        #     'factor_name': 'IntraRm_m30_IntraRelQtl_d20_q002_dp2_wstr_jump5mVwap',
        #     'factor_dir': r'/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/sample_factors',
        #     'direction': 1,
        #     'simplified_name': 'Jump'
        # },
        # {
        #     'factor_name': 'trade_amount_Dollar_LX_R3_dp2_SumIntraRm5_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15',
        #     'factor_dir': r'/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/sample_factors',
        #     'direction': 1,
        #     'simplified_name': 'TA_Dollar_LX'
        # },
        # {
        #     'factor_name': 'trade_amount_Dollar_R3_dp2_SumIntraRm5_LXPct_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15',
        #     'factor_dir': r'/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/sample_factors',
        #     'direction': 1,
        #     'simplified_name': 'TA_Dollar_LXPct'
        # },
        # {
        #     'factor_name': 'order_amount_Dollar_LX_R3_dp2_SumIntraRm5_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15',
        #     'factor_dir': r'/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/sample_factors',
        #     'direction': 1,
        #     'simplified_name': 'OA_Dollar_LX'
        # },
        # {
        #     'factor_name': 'order_amount_Dollar_R3_dp2_SumIntraRm5_LXPct_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15',
        #     'factor_dir': r'/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/sample_factors',
        #     'direction': 1,
        #     'simplified_name': 'OA_Dollar_LXPct'
        # },
        # Add more factors as needed
    ]
    
    # Parameter grid
    param_grid = {
        'vol_T': [0.006, 0.007, 0.008, 0.009, 0.01, 0.0125],
        'vol_k': [600, 800, 1000],
        'trend_k': [200, 1000, 2000, 3000],
        'trend_th': [-0.02, 0, 0.0005, 0.002]
    }
    
    # Common parameters
    price_dir = Path('/mnt/30.132_xt_data1/futuretwap')
    price_name = 't1min_fq1min_dl1min'
    
# =============================================================================
#     version_name = 'v0'
#     feature_dir = Path(rf'/mnt/30.132_xt_data1/idx_opt_processed/{version_name}_features')
#     feature_name = 'atm_vol'
#     feature_col_name = 'IO'
# =============================================================================
    
    feature_dir = Path(r'/mnt/30.132_xt_data1/idx_opt_processed/realized_vol')
    feature_name = 'realized_vol_multi_wd'
    feature_col_name = 'IC'
    
    scale_method = 'minmax_scale'
    scale_window = '240d'
    scale_quantile = 0.02
    sp = '1min'
    
    trend_weight_dict = {
        15: 1,
        30: 1,
        60: 1,
        120: 1,
    }
    
    # test_name = 'trade_ver0_futtwap_sp1min_s240d_icim'
    # test_name_org = 'trade_ver0_futtwap_sp1min_noscale_icim'
    test_name = 'trade_ver3_futtwap_sp1min_s240d_icim_v6'
    test_name_org = 'trade_ver3_futtwap_sp1min_noscale_icim_v6'
    # test_name = 'trade_ver3_futtwap_sp1min_s240d_icim_vtrade'
    # test_name_org = 'trade_ver3_futtwap_sp1min_noscale_icim_vtrade'
    test_workers = 50
    
    # analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\filter_vol_and_trend')
    analysis_dir = Path(r'/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/analysis/filter_vol_and_trend')
    
    # Process each factor
    for factor_config in factors:
        process_factor(
            factor_config=factor_config,
            param_grid=param_grid,
            price_dir=price_dir,
            price_name=price_name,
            feature_dir=feature_dir,
            feature_name=feature_name,
            feature_col_name=feature_col_name,
            scale_method=scale_method,
            scale_window=scale_window,
            scale_quantile=scale_quantile,
            sp=sp,
            trend_weight_dict=trend_weight_dict,
            save_base_dir=analysis_dir,
            test_name=test_name,
            test_name_org=test_name_org,
            test_workers=test_workers
        )


if __name__ == "__main__":
    main()