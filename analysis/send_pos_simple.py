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
strategy_name = "pyelf_lob_sif_1_2_4"

model_name = 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18'
test_name = 'trade_ver3_1_futtwap_sp1min_s240d_icim_v6'


# %%
pos_dir = Path(rf'D:/mnt/CNIndexFutures/timeseries/factor_test/results/model\{model_name}\test\{test_name}\data')
res_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\导回测')
save_dir = res_dir / model_name
save_dir.mkdir(parents=True, exist_ok=True)


# %%
pos_file_name = f'pos_predict_{model_name}.parquet'
actual_pos = pd.read_parquet(pos_dir / pos_file_name)


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



