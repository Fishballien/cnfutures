# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 14:22:49 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datetime import datetime


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.datautils import compute_dataframe_dict_average, add_dataframe_to_dataframe_reindex
from utils.timeutils import RollingPeriods, period_shortcut


# %%
fstart = '20150101'
pstart = '20170101'
puntil = '20250201'
window_kwargs = {'months': 96}
rrule_kwargs = {'freq': 'M', 'interval': 1}
end_by = 'date'


# %%
rolling = RollingPeriods(
    fstart=datetime.strptime(fstart, '%Y%m%d'),
    pstart=datetime.strptime(pstart, '%Y%m%d'),
    puntil=datetime.strptime(puntil, '%Y%m%d'),
    window_kwargs=window_kwargs,
    rrule_kwargs=rrule_kwargs,
    end_by=end_by,
    )


# %%
pos_all = pd.DataFrame()
for fp, pp in tqdm(list(zip(rolling.fit_periods, rolling.predict_periods)), desc='rolling predict'):
    fit_period = period_shortcut(*fp)
    filtered = pd.read_csv(filtered_dir / f'filtered_{version_name}_{fit_period}.csv')
    pos_dict, weight_dict = {}, {}
    for pred_name in filtered['pred_name']:
        pos_path = pos_dir / f'{pred_name}.parquet'
        pos = pd.read_parquet(pos_path)
        pos_dict[pred_name] = pos.loc[pp[0]:pp[1]]
        weight_dict[pred_name] = 1
    pos_avg_period = compute_dataframe_dict_average(pos_dict, weight_dict)
    pos_all = add_dataframe_to_dataframe_reindex(pos_all, pos_avg_period)
    
pos_all.to_csv(predict_dir / f'pos_{version_name}.csv')
pos_all.to_parquet(predict_dir / f'pos_{version_name}.parquet')


# %%
