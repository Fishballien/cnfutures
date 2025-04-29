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
# %% imports
from pathlib import Path
import numpy as np
import pandas as pd


from utils.datautils import align_and_sort_columns
from utils.market import index_to_futures
from utils.timeutils import parse_time_string
from data_processing.ts_trans import *
from test_and_eval.factor_tester import FactorTesterByDiscrete


# %%
def vol_filter_sigmoid(vol, T=0.01, k=800):
    """Sigmoid function transformation"""
    return 1 / (1 + np.exp(-k * (vol - T)))


def trend_consistency_filter(trend_score, k=2000, trend_th=0.0005):
    """
    计算趋势一致门控值
    
    参数:
    trend_score : float 或 numpy.ndarray
        趋势一致性得分
    k : float, 可选
        sigmoid函数的陡度系数，默认为1000
    trend_th : float, 可选
        趋势阈值，默认为0.001
        
    返回:
    float 或 numpy.ndarray
        趋势一致门控值，范围在0到1之间
    """
    # 计算sigmoid
    trend_filter = 1 / (1 + np.exp(k * (trend_score - trend_th)))
    
    return trend_filter


# Define the soft_and function
def soft_and(vol_filter, trend_filter):
    """Soft AND operation: 1 - (1 - vol_filter) * (1 - trend_filter)"""
    return 1 - (1 - vol_filter) * (1 - trend_filter)


# %%
# model_name = 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18'
# factor_name = f'predict_{model_name}'
# factor_dir = Path(rf'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model\{model_name}\predict')
# direction = 1

# factor_name = 'LargeOrderAmountByValue_p1.0_v40000-avg_imb01_dp2-org'
# simplified_name = 'LOA'
# factor_name = 'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb04_dp2-org'
# simplified_name = 'VTDOA'
# direction = 1
# factor_dir = Path(r'D:/mnt/CNIndexFutures/timeseries/factor_test/sample_data/factors/1_2_org')


factor_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\sample_data\factors\tf\typical_trade_factor')
# factor_name = 'order_amount_Dollar_LX_R3_dp2_SumIntraRm5_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15'
# direction = 1
# factor_name = 'order_amount_Dollar_R3_dp2_SumIntraRm5_LXPct_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15'
# direction = 1
factor_name = 'trade_amount_Dollar_LX_R3_dp2_SumIntraRm5_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15'
simplified_name = 'TA_Dollar_LX'
direction = 1
# factor_name = 'trade_amount_Dollar_R3_dp2_SumIntraRm5_LXPct_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15'
# direction = 1

# factor_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\sample_data\factors\lxy')
# factor_name = 'IntraRm_m30_IntraRelQtl_d20_q002_dp2_wstr_jump5mVwap'
# direction = 1


# %%
version_name = 'v0'
feature_dir = Path(rf'D:\mnt\idx_opt_processed\{version_name}_features')
feature_name = 'atm_vol'
feature_col_name = 'IO'


# %%
price_name = 't1min_fq1min_dl1min'

scale_method = 'minmax_scale'
scale_window = '240d'
scale_quantile = 0.02
sp = '1min'


vol_T = 0.01
vol_k = 800
trend_k = 2000
trend_th = 0.0005

test_name = 'trade_ver0_futtwap_sp1min_s240d_icim'
test_name_org = 'trade_ver0_futtwap_sp1min_noscale_icim'
test_workers = 1


# %%
fut_dir = Path('/mnt/data1/futuretwap')
analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\filter_vol_and_trend')
save_dir = analysis_dir / simplified_name
filtered_fac_dir = save_dir / 'filtered_fac'
filtered_fac_dir.mkdir(parents=True, exist_ok=True)
org_fac_dir = save_dir / 'org_fac'
org_fac_dir.mkdir(parents=True, exist_ok=True)


# %%
factor_data = pd.read_parquet(factor_dir / f'{factor_name}.parquet')
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


# %% load vol
feature_path = feature_dir / f'{feature_name}.parquet'
feature = pd.read_parquet(feature_path)
feature_series = feature[feature_col_name].reindex(price_data.index)


# %% vol filter
vol_filter = vol_filter_sigmoid(feature_series, T=vol_T, k=vol_T)


# %% trend
trend_weight_dict = {
    15: 1,
    30: 1,
    60: 1,
    120: 1,
    }
price_data_pct_change = {lookback: price_data.shift(2).pct_change(lookback) for lookback in trend_weight_dict}
# 计算加权平均
# 首先计算总权重
total_weight = sum(trend_weight_dict.values())

# 假设所有DataFrame有相同的列和索引
# 获取第一个DataFrame的列，以便初始化结果
sample_key = list(price_data_pct_change.keys())[0]
columns = price_data_pct_change[sample_key].columns
index = price_data_pct_change[sample_key].index


weighted_avg_pct_change = pd.DataFrame(0, index=index, columns=columns)

for lookback, weight in trend_weight_dict.items():
    # 对每个回溯期，应用权重
    normalized_weight = weight / total_weight
    # 加权后累加到结果DataFrame
    weighted_avg_pct_change = weighted_avg_pct_change + price_data_pct_change[lookback] * normalized_weight
    

# %% trend filter
factor_direction = factor_scaled.apply(np.sign)
trend_score = weighted_avg_pct_change * factor_direction
trend_filter = trend_consistency_filter(trend_score, k=trend_k, trend_th=trend_th)


# %% soft and
# Create a new DataFrame with the same structure as trend_filter
soft_and_filter = pd.DataFrame(index=trend_filter.index, columns=trend_filter.columns)

# Apply soft_and function to each column of trend_filter with vol_filter
for column in trend_filter.columns:
    # Handle NaN values properly by using NumPy's where function
    # If either vol_filter or trend_filter[column] is NaN, the result will be NaN
    valid_mask = ~(pd.isna(vol_filter) | pd.isna(trend_filter[column]))
    result = pd.Series(index=trend_filter.index, dtype='float64')
    
    # Only calculate soft_and where both inputs are valid numbers
    result.loc[valid_mask] = soft_and(
        vol_filter.loc[valid_mask],
        trend_filter.loc[valid_mask, column]
    )
    
    # Assign to the result DataFrame
    soft_and_filter[column] = result


# %% scaled factor
filtered_factor = factor_scaled * soft_and_filter
filtered_factor.to_parquet(filtered_fac_dir / f'{simplified_name}_vT{vol_T}_vk{vol_k}_tk{trend_k}_tth{trend_th}.parquet')
filtered_factor = filtered_factor.dropna(how='all')
factor_scaled_org = factor_scaled.reindex(index=filtered_factor.index)
factor_scaled_org.to_parquet(filtered_fac_dir / f'{simplified_name}_org.parquet')


# %%
tester = FactorTesterByDiscrete(None, None, filtered_fac_dir, test_name=test_name_org, 
                                result_dir=save_dir, n_workers=test_workers)
tester.test_one_factor(f'{simplified_name}_org')

tester = FactorTesterByDiscrete(None, None, filtered_fac_dir, test_name=test_name, 
                                result_dir=save_dir, n_workers=test_workers)
tester.test_multi_factors(skip_exists=False)