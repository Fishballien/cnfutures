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
strategy_name = "pyelf_lob_sif_1_0"

model_name = 'avg_agg_250203_by_trade_net_v6'
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


# %%
# factor_dir = Path(r'D:\CNIndexFutures\timeseries\factor_factory\sample_data\factors\batch10')
factor_dir = Path(rf'D:\CNIndexFutures\timeseries\factor_test\results\model\{model_name}\predict')
fut_dir = Path(r'D:\Data\cnfutures\futuretwap')
res_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\导回测')
save_dir = res_dir / model_name
save_dir.mkdir(parents=True, exist_ok=True)


# %%
factor_data = pd.read_parquet(factor_dir / f'predict_{model_name}.parquet')
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

factor_scaled = (factor_scaled - 0.5) * 2


# %%
trade_rule_func = partial(globals()[trade_rule_name], **trade_rule_param)
actual_pos = factor_scaled.apply(lambda col: trade_rule_func(col.values), axis=0)


# %%
import os

# 定义策略名称和文件夹路径
folder_dir = save_dir/ strategy_name
folder_dir.mkdir(parents=True, exist_ok=True)


# 读取 actual_pos 并转换格式
actual_pos.index.name = "stockdate"
actual_pos.reset_index(inplace=True)

# 遍历每个品种并存储为 CSV 文件
file_paths = []
for symbol in actual_pos.columns[1:]:  # 跳过 "stockdate" 列
    df_symbol = actual_pos[["stockdate", symbol]]
    file_name = f"strategy_{symbol}.csv"
    file_path = folder_dir / file_name
    df_symbol.to_csv(file_path, index=False, date_format="%Y-%m-%d %H:%M:%S")



