# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 13:18:29 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import pandas as pd


# %%
# old_path = 'D:/mnt/CNIndexFutures/timeseries/factor_test/results/model/avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18/test/trade_ver3_futtwap_sp1min_s240d_icim_v6/data/scaled_predict_avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18.parquet'
# new_path = 'D:/mnt/CNIndexFutures/timeseries/factor_test/results/model/avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18_debug_new/test/trade_ver3_futtwap_sp1min_s240d_icim_v6/data/scaled_predict_avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18.parquet'


old_path = 'D:/mnt/CNIndexFutures/timeseries/factor_test/results/model/avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18/predict/predict_avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18.parquet'
new_path = 'D:/mnt/CNIndexFutures/timeseries/factor_test/results/model/avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18_debug_new/predict/predict_avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18.parquet'

# %%
old_pred = pd.read_parquet(old_path)
new_pred = pd.read_parquet(new_path)