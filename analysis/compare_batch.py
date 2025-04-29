# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 20:09:42 2024

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


# %%
eval_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\results\factor_evaluation')


# %%
eval_1 = 'agg_batch10_241231_fix_wgt'
eval_2 = 'agg_batch10_241231_wrong_wgt'
eval_name = 'factor_eval_151201_240901'


# %%
eval_1_path = eval_dir / eval_1 / f'{eval_name}.csv'
eval_2_path = eval_dir / eval_2 / f'{eval_name}.csv'


# %%
eval_1_data = pd.read_csv(eval_1_path)
eval_2_data = pd.read_csv(eval_2_path)

eval_1_data['net_sharpe_ratio'].hist(bins=20, alpha=.5, label='fixed')
eval_2_data['net_sharpe_ratio'].hist(bins=20, alpha=.5, label='wrong')
plt.legend()