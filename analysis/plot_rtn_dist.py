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


from utils.market import index_to_futures
from utils.timeutils import parse_time_string


# %%
price_name = 't1min_fq1min_dl1min'
sp = '1min'
pp = '60min'


# %%
sample_price_dir = Path(r'D:\mnt\data1\future_twap')
save_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\analysis')
save_dir.mkdir(parents=True, exist_ok=True)


# %%
midprice = pd.read_parquet(sample_price_dir / f'{price_name}.parquet')[['IF', 'IM', 'IC']]
pp_by_sp = int(parse_time_string(pp) / parse_time_string(sp))
rtn_1p = midprice.pct_change(pp_by_sp, fill_method=None).shift(-pp_by_sp) / pp_by_sp
rtn_1p = rtn_1p.replace([np.inf, -np.inf], 0)


# %%
data = rtn_1p

# 创建更大的图形
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# 上图：时序图
for col in data.columns:
    axes[0].plot(data.index, data[col], label=col, alpha=0.6)
axes[0].axhline(y=0, color='red', linestyle='--', label="y=0")
axes[0].set_title(f"{price_name}", fontsize=16, pad=15)
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
plot_file_path = save_dir / f"{price_name}_sp{sp}_pp{pp}.jpg"
# 保存图表到 sample_data_dir
plt.savefig(plot_file_path, bbox_inches="tight", dpi=300)

plt.show()
