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
import numpy as np
from functools import partial


from utils.datautils import align_and_sort_columns
from utils.market import index_to_futures
# from trans_operators.format import to_float32


from utils.timeutils import parse_time_string
from utils.trade_rules import *
from data_processing.ts_trans import *


# %%
model_name = 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18'
factor_name = f'predict_{model_name}'
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

start = pd.Timestamp('2020-01-01 00:00:00')
end = pd.Timestamp('2025-03-22 00:00:00')


# %%
feature_dir = Path(rf'D:\mnt\idx_opt_processed\realized_vol')
feature_name = 'realized_vol_multi_wd'


# %%
factor_dir = Path(rf'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model\{model_name}\predict')
new_fac_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\sample_data\filters\org_indicators')
fut_dir = Path('/mnt/data1/futuretwap')
analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\signals')
save_dir = analysis_dir / feature_name / factor_name
save_dir.mkdir(parents=True, exist_ok=True)


# %%
# version_name = 'v0'
# feature_dir = Path(rf'D:\mnt\idx_opt_processed\{version_name}_features')
# feature_name = 'atm_vol'
# feature_col_name = 'IO'


# %%
factor_data = pd.read_parquet(factor_dir / f'{factor_name}.parquet')
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
# feature_path = feature_dir / f'{feature_name}.parquet'
# feature = pd.read_parquet(feature_path)
# new_fac_dict = {}
# new_fac_dict['atm_vol_io'] = feature['IO'].reindex(index=actual_pos.index)


# %%
feature_path = feature_dir / f'{feature_name}.parquet'
feature = pd.read_parquet(feature_path)
new_fac_dict = {}
new_fac_dict['realized_vol'] = feature.reindex(index=actual_pos.index)


# %%
by_week_dir = save_dir / 'by_week'
by_week_dir.mkdir(parents=True, exist_ok=True)

# 检查列数
factor_columns = factor_data.columns
price_columns = price_data.columns

# 确保列的数量一致
num_columns = min(len(factor_columns), len(price_columns))

# 按周分组并绘制图表
for week_start, factor_group in factor_scaled.groupby(pd.Grouper(freq='W-MON', label='left', closed='left')):
    if week_start < start or week_start > end:
        continue
    # 筛选价格数据的对应周
    price_group = price_data[((price_data.index >= week_start) & (price_data.index < week_start + pd.Timedelta(weeks=1)))]
    
    # 筛选对应周的仓位数据
    actual_pos_week = actual_pos[((actual_pos.index >= week_start) & (actual_pos.index < week_start + pd.Timedelta(weeks=1)))]
    
    # 如果某周数据为空，则跳过
    if factor_group.empty or price_group.empty or actual_pos_week.empty:
        continue
    
    # 对齐 actual_pos_week 的索引，使其与 factor_group 的索引对齐
    actual_pos_week_aligned = actual_pos_week.reindex(factor_group.index, method='ffill')

    # 生成顺序 x 轴 (arange)
    x = np.arange(len(factor_group))  # 顺序索引
    x_labels = factor_group.index.strftime('%Y-%m-%d %H:%M')  # 转换成时间标签

    # 找到每天9:30的位置
    nine_thirty_indices = [i for i, t in enumerate(factor_group.index) if t.strftime('%H:%M') == '09:30']

    # 遍历每一列，绘制单独的图片
    for i in range(num_columns):
        factor_col = factor_columns[i]
        price_col = price_columns[i]
        actual_pos_col = actual_pos_week.columns[i]  # 对应的仓位列

        # 计算需要的子图行数：2行 + 新的信号的子图数量
        num_new_fac_plots = len(new_fac_dict)  # 计算新信号的数量
        total_rows = 2 + num_new_fac_plots  # 两行基础图 + 每个新信号对应的子图

        # 创建一个图形，2行基础图 + 新信号数量的子图
        fig, axs = plt.subplots(total_rows, 1, figsize=(12, 4 * total_rows), sharex=True)
        fig.suptitle(f"Factor and Price Data for Week Starting {week_start.strftime('%Y-%m-%d')} - {factor_col}", fontsize=16)
        
        # 上方子图：价格数据
        price_data_trimmed = price_group[price_col][:len(factor_group)]  # 确保长度一致
        axs[0].plot(x, price_data_trimmed, label=f'Price: {price_col}', color='orange')
        axs[0].set_ylabel('Price')
        axs[0].set_title(f"Price: {price_col}")
        axs[0].legend()
        axs[0].grid(True)

        # 绘制红色虚线和实线
        for idx in nine_thirty_indices:
            axs[0].axvline(x=idx, color='k', linestyle='--', linewidth=1)

        # 计算仓位变化的信号
        actual_pos_col_aligned = actual_pos_week_aligned[actual_pos_col]
        buy_changes = actual_pos_col_aligned[(actual_pos_col_aligned.diff() == 1) & (actual_pos_col_aligned == 1)].index  # 0 -> 1 (开多)
        sell_changes = actual_pos_col_aligned[(actual_pos_col_aligned.diff() == -1) & (actual_pos_col_aligned == 0)].index  # 1 -> 0 (平多)
        short_changes = actual_pos_col_aligned[(actual_pos_col_aligned.diff() == -1) & (actual_pos_col_aligned == -1)].index  # 0 -> -1 (开空)
        cover_changes = actual_pos_col_aligned[(actual_pos_col_aligned.diff() == 1) & (actual_pos_col_aligned == 0)].index  # -1 -> 0 (平空)
        
        # 绘制仓位变化的竖线
        for change in buy_changes:
            idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
            axs[0].axvline(x=idx, color='red', linestyle='-', linewidth=1)  # 开多
        for change in sell_changes:
            idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
            axs[0].axvline(x=idx, color='red', linestyle='--', linewidth=1)  # 平多
        for change in short_changes:
            idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
            axs[0].axvline(x=idx, color='green', linestyle='-', linewidth=1)  # 开空
        for change in cover_changes:
            idx = factor_group.index.get_loc(change)  # 获取对应的顺序索引位置
            axs[0].axvline(x=idx, color='green', linestyle='--', linewidth=1)  # 平空

        # 下方子图：因子数据
        axs[1].plot(x, factor_group[factor_col], label=f'Factor: {factor_col}')
        axs[1].set_ylabel('Factor Value')
        axs[1].set_title(f"Factor: {factor_col}")
        axs[1].legend()
        axs[1].grid(True)

        # 绘制红色虚线和实线
        for idx in nine_thirty_indices:
            axs[1].axvline(x=idx, color='k', linestyle='--', linewidth=1)

        # 绘制仓位变化的竖线 (for factor subplot)
        for change in buy_changes:
            idx = factor_group.index.get_loc(change)
            axs[1].axvline(x=idx, color='red', linestyle='-', linewidth=1)  # 开多
        for change in sell_changes:
            idx = factor_group.index.get_loc(change)
            axs[1].axvline(x=idx, color='red', linestyle='--', linewidth=1)  # 平多
        for change in short_changes:
            idx = factor_group.index.get_loc(change)
            axs[1].axvline(x=idx, color='green', linestyle='-', linewidth=1)  # 开空
        for change in cover_changes:
            idx = factor_group.index.get_loc(change)
            axs[1].axvline(x=idx, color='green', linestyle='--', linewidth=1)  # 平空

        # 添加新的信号子图
        row_idx = 2  # 新信号从第三行开始
        for new_fac_name, new_fac_df in new_fac_dict.items():
            if new_fac_name in new_fac_dict:
                if isinstance(new_fac_df, pd.Series):
                    new_factor_series = new_fac_df.loc[actual_pos_week_aligned.index]
                else:
                    new_factor_series = new_fac_df.loc[actual_pos_week_aligned.index, factor_col]  # 对应列的信号数据
                axs[row_idx].plot(x, new_factor_series, label=f"New Signal: {new_fac_name}", linestyle='--')
                axs[row_idx].set_ylabel(new_fac_name)
                axs[row_idx].set_title(f"New Signal: {new_fac_name}")
                axs[row_idx].legend()
                axs[row_idx].grid(True)
                
                # 绘制红色虚线和实线
                for idx in nine_thirty_indices:
                    axs[row_idx].axvline(x=idx, color='k', linestyle='--', linewidth=1)
                
                # 绘制仓位变化的竖线 (for new signal subplot)
                for change in buy_changes:
                    idx = factor_group.index.get_loc(change)
                    axs[row_idx].axvline(x=idx, color='red', linestyle='-', linewidth=1)  # 开多
                for change in sell_changes:
                    idx = factor_group.index.get_loc(change)
                    axs[row_idx].axvline(x=idx, color='red', linestyle='--', linewidth=1)  # 平多
                for change in short_changes:
                    idx = factor_group.index.get_loc(change)
                    axs[row_idx].axvline(x=idx, color='green', linestyle='-', linewidth=1)  # 开空
                for change in cover_changes:
                    idx = factor_group.index.get_loc(change)
                    axs[row_idx].axvline(x=idx, color='green', linestyle='--', linewidth=1)  # 平空
                
                row_idx += 1

        # 设置共享的 x 轴标签
        tick_positions = np.linspace(0, len(x)-1, num=10, dtype=int)
        axs[-1].set_xticks(tick_positions)
        axs[-1].set_xticklabels([x_labels[i] for i in tick_positions], rotation=45)

        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 给标题留空间
        plt.savefig(by_week_dir / f"{factor_col}_week_{week_start.strftime('%Y-%m-%d')}.jpg", dpi=300)
        plt.close(fig)  # 关闭图形，释放内存

