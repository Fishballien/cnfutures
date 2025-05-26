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


# %%
strategy_name = "pyelf_lob_sif_1_2_8"

pos_path = r'D:/mnt/CNIndexFutures/timeseries/factor_test/results/model/avg_agg_250515_by_trade_net_v18_02/rolling_model/valid_range_v2/pos/pos_avg_agg_250515_by_trade_net_v18_02_valid_range_v2.parquet'


# %%
res_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\导回测')
save_dir = res_dir / strategy_name
save_dir.mkdir(parents=True, exist_ok=True)


# %%
actual_pos = pd.read_parquet(pos_path)


# %%
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



