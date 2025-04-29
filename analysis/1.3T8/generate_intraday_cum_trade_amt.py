# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:41:50 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
from pathlib import Path
import pandas as pd


# %%
org_fac_dir = Path('D:/mnt/CNIndexFutures/timeseries/factor_test/sample_data/filters/org_indicators')


# %%
SA_name = 'trade_SA_amount_R3_dp2_Sum'
BA_name = 'trade_BA_amount_R3_dp2_Sum'
SA_path = org_fac_dir / f'{SA_name}.parquet'
BA_path = org_fac_dir / f'{BA_name}.parquet'
SA = pd.read_parquet(SA_path)
BA = pd.read_parquet(BA_path)
trade_amt = SA + BA


# %%
# 创建一个新的数据框来存储结果
result = pd.DataFrame(index=trade_amt.index)

# 为每个指数计算日内累积成交量
for column in trade_amt.columns:
    # 提取当前列数据
    col_data = trade_amt[column]
    
    # 创建日期列(不添加到原始数据框中)
    dates = col_data.index.date
    
    # 按日期分组并计算累积和
    grouped = col_data.groupby(dates).cumsum()
    
    # 将结果添加到结果数据框
    result[column] = grouped

# 显示结果
result.to_parquet(org_fac_dir / 'TradeAmount_intraday_cumsum.parquet')