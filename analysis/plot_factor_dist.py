# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:18:00 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
#%% imports
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


from utils.market import index_to_futures
from data_processing.ts_trans import *
from utils.timeutils import parse_time_string
from utils.datautils import align_columns, align_index, align_and_sort_columns


# %%
# factor_name = 'l_amount_wavg_imb01'
# factor_name = 'amount_Quantile_R1_org_R2_org_R3_Sum_LXPct_R4_Imb2IntraRmDodPctChg_10'
# factor_name = 'tsstd_2h_csmean_closeprice_taylor_240m'
# factor_name = 'ActBuyAmt'
factor_name = 'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-imb04_wavg-rollingMinuteQuantileScale_w245d_q0.02'

price_name = 't1min_fq1min_dl1min'
sp = '1min'
pp = '60min'
scale_window = '240d'
scale_quantile = 0.02
scale_method = 'minmax_scale'


# %%
sample_factor_dir = Path(r'D:\CNIndexFutures\timeseries\factor_factory\sample_data\factors\batch10')
sample_price_dir = Path(r'D:\mnt\data1\future_twap')
save_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\analysis')
save_dir.mkdir(parents=True, exist_ok=True)


# %%
midprice = pd.read_parquet(sample_price_dir / f'{price_name}.parquet')
factor = pd.read_parquet(sample_factor_dir / f'{factor_name}.parquet')
factor = factor.rename(columns=index_to_futures)[['IF', 'IM', 'IC']]


# %%
pp_by_sp = int(parse_time_string(pp) / parse_time_string(sp))
rtn_1p = midprice.pct_change(pp_by_sp, fill_method=None).shift(-pp_by_sp) / pp_by_sp
rtn_1p = rtn_1p.replace([np.inf, -np.inf], 0)
main_col = rtn_1p.columns


# %%
factor, rtn_1p, midprice = align_and_sort_columns([factor, rtn_1p, midprice])

midprice = midprice.loc[factor.index.min():factor.index.max()] # 按factor头尾截取
rtn_1p = rtn_1p.loc[factor.index.min():factor.index.max()] # 按factor头尾截取
factor = factor.reindex(rtn_1p.index) # 按twap reindex，确保等长


# %%
data = factor

# 创建更大的图形
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# 上图：时序图
for col in data.columns:
    axes[0].plot(data.index, data[col], label=col, alpha=0.6)
axes[0].axhline(y=0, color='red', linestyle='--', label="y=0")
axes[0].set_title(f"{factor_name}", fontsize=16, pad=15)
axes[0].legend(fontsize=12)
axes[0].grid(True)

# 下图：直方图
for col in data.columns:
    axes[1].hist(data[col], bins=100, alpha=0.6, label=col, histtype='stepfilled')
axes[1].axvline(x=0, color='red', linestyle='--', label="x=0")
# axes[1].set_title("Histogram", fontsize=16)
axes[1].legend(fontsize=12)
axes[1].grid(True)

plt.tight_layout()
# 生成文件名
plot_file_path = save_dir / f"{factor_name}.jpg"
# 保存图表到 sample_data_dir
plt.savefig(plot_file_path, bbox_inches="tight", dpi=300)

plt.show()


# %% adf test
def check_stationarity(timeseries, significance_level=0.05):
    """
    检验时间序列的平稳性（使用ADF检验）。
    
    参数：
    timeseries (pd.Series): 时间序列数据
    significance_level (float): 显著性水平，默认0.05
    
    返回：
    dict: 包含检验统计量、p值、滞后数、ADF检验结论等信息的字典
    """
    result = adfuller(timeseries)
    test_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    # 判断平稳性
    is_stationary = p_value < significance_level
    
    return {
        "Test Statistic": test_statistic,
        "P-Value": p_value,
        "Lags Used": result[2],
        "Number of Observations": result[3],
        "Critical Values": critical_values,
        "Is Stationary": is_stationary
    }


adf_test_res = {fut: check_stationarity(factor[fut].resample('1d').mean().dropna()) 
                for fut in factor.columns}


# %% 多空比
def calculate_positive_ratio(df):
    """
    计算 DataFrame 每列大于 0 的比例。
    
    参数：
    df (pd.DataFrame): 输入的 DataFrame
    
    返回：
    pd.Series: 每列大于 0 的比例
    """
    return (df > 0).mean()


scale_func = globals()[scale_method]
scale_step = int(parse_time_string(scale_window) / parse_time_string(sp))
if scale_method in ['minmax_scale', 'minmax_scale_separate']:
    factor_scaled = scale_func(factor, window=scale_step, quantile=scale_quantile)
elif scale_method in ['minmax_scale_adj_by_his_rtn', 'zscore_adj_by_his_rtn_and_minmax']:
    factor_scaled = scale_func(factor, rtn_1p, window=scale_step, rtn_window=pp_by_sp, quantile=scale_quantile)
elif scale_method in ['rolling_percentile']:
    factor_scaled = scale_func(factor, window=scale_step)
elif scale_method in ['percentile_adj_by_his_rtn']:
    factor_scaled = scale_func(factor, rtn_1p, window=scale_step, rtn_window=pp_by_sp)
factor_scaled = (factor_scaled - 0.5) * 2

pos_ratio = calculate_positive_ratio(factor_scaled)
rtn_ratio = calculate_positive_ratio(rtn_1p)
if pos_ratio.mean() > 0.5:
    pos_ratio = 1 - pos_ratio
ratio_diff = rtn_ratio - pos_ratio
print(pos_ratio, ratio_diff)


# %% check if valid
adf_valid = all([adf_test_res[fut]["Is Stationary"] for fut in adf_test_res])
pos_ratio_valid = all(ratio_diff < 0.1)
valid = adf_valid and pos_ratio_valid
print(valid)