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
model_name = 'avg_agg_250218_3_fix_tfe_by_trade_net_v4'
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
pp_by_sp = 1


# %%
factor_name = f'predict_{model_name}'
# factor_dir = Path(r'D:\CNIndexFutures\timeseries\factor_factory\sample_data\factors\batch10')
factor_dir = Path(rf'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model\{model_name}\predict')
fut_dir = Path('/mnt/data1/future_twap')
analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\signals')
save_dir = analysis_dir / factor_name
save_dir.mkdir(parents=True, exist_ok=True)


# %%
factor_data = pd.read_parquet(factor_dir / f'{factor_name}.parquet')
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
rtn_1p = price_data.pct_change(pp_by_sp, fill_method=None).shift(-pp_by_sp) / pp_by_sp
rtn_1p = rtn_1p.replace([np.inf, -np.inf], 0)


# %%
gp = (actual_pos * rtn_1p).shift(pp_by_sp)


# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns


# 提取工作日和时间组件
gp['weekday'] = gp.index.weekday  # 0是星期一，6是星期日
gp['hour'] = gp.index.hour
gp['minute'] = gp.index.minute
gp['minute_of_day'] = gp['hour'] * 60 + gp['minute']

# 计算平均收益（三列的平均值）
gp['avg_return'] = gp[['IC', 'IF', 'IM']].mean(axis=1)

# 按星期几和每日分钟进行分组并计算平均值
grouped = gp.groupby(['weekday', 'minute_of_day']).mean()
grouped = grouped.reset_index()

# 创建更可读的标签
weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
grouped['weekday_name'] = grouped['weekday'].apply(lambda x: weekday_names[x])

# 从minute_of_day重新计算小时和分钟
grouped['hour'] = grouped['minute_of_day'] // 60
grouped['minute'] = grouped['minute_of_day'] % 60
grouped['time_label'] = grouped.apply(lambda row: f"{int(row['hour']):02d}:{int(row['minute']):02d}", axis=1)
grouped['x_label'] = grouped['weekday_name'] + ' ' + grouped['time_label']

# 设置图表样式
plt.figure(figsize=(15, 8))
sns.set_style("whitegrid")

# 创建条形图
bar_plot = sns.barplot(x='x_label', y='avg_return', data=grouped, 
                        errorbar=None, palette='viridis')

# 设置标题和标签
plt.title('Average Per-Minute Returns by Weekday and Time', fontsize=16)
plt.xlabel('Weekday and Time', fontsize=12)
plt.ylabel('Average Return', fontsize=12)

# 旋转x轴标签以提高可读性
plt.xticks(rotation=90)

# 只显示部分x轴标签，以避免拥挤
n = len(grouped)
step = max(1, n // 20)  # 显示大约20个标签
indices = np.arange(0, n, step)
plt.xticks(indices, [grouped['x_label'].iloc[i] for i in indices])

# 添加网格线以提高可读性
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()

# 如果需要，还可以按星期几进行子图分析
plt.figure(figsize=(20, 12))
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 12), sharey=True)
axes = axes.flatten()

for i, day in enumerate(sorted(grouped['weekday'].unique())):
    if i < len(axes):
        day_data = grouped[grouped['weekday'] == day]
        sns.barplot(x='time_label', y='avg_return', data=day_data, 
                   errorbar=None, ax=axes[i], palette='viridis')
        axes[i].set_title(f'{weekday_names[day]}', fontsize=14)
        axes[i].set_xlabel('Time', fontsize=10)
        axes[i].set_ylabel('Average Return', fontsize=10)
        axes[i].tick_params(axis='x', rotation=90)
        
        # 只显示部分x轴标签
        n_day = len(day_data)
        step_day = max(1, n_day // 10)
        indices_day = np.arange(0, n_day, step_day)
        axes[i].set_xticks(indices_day)
        axes[i].set_xticklabels([day_data['time_label'].iloc[j] for j in indices_day])

# 移除多余的子图
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()