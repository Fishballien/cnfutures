# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:41:56 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates


from utils.timeutils import parse_time_string
from utils.algos import ts_quantile_scale, ts_quantile_scale_separate, clip_values
from utils.algos import ts_quantile_scale_adj_by_his_rtn, ts_quantile_scale_adj_zscore_and_minmax
from utils.algos import rolling_percentile, ts_adj_by_his_rtn_percentile, ts_adj_by_his_rtn_percentile_simple
from utils.datautils import align_columns, align_index, align_and_sort_columns
from utils.market import index_to_futures


# %%
factor_name = 'l_amount_wavg_imb01'
# factor_name = 'amount_Quantile_R1_org_R2_org_R3_Sum_LXPct_R4_Imb2IntraRmDodPctChg_10'
# factor_name = 'tsstd_2h_csmean_closeprice_taylor_240m'
# factor_name = 'ActBuyAmt'
price_name = 't1min_fq1min_dl1min'
sp = '1min'
pp = '60min'
scale_window = '240d'
scale_quantile = 0.1
direction_choices = ['all', 'pos', 'neg']
scale_method = 'ts_quantile_scale'


# %%
sample_factor_dir = Path(r'D:\CNIndexFutures\timeseries\factor_factory\sample_data\factors')
# sample_price_dir = Path(r'D:\CNIndexFutures\timeseries\index_price\sample_data')
sample_price_dir = Path(r'D:\mnt\data1\future_twap')
save_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\sample_data\test_results')
sample_save_dir = save_dir / scale_method
sample_save_dir.mkdir(parents=True, exist_ok=True)


# %%
midprice = pd.read_parquet(sample_price_dir / f'{price_name}.parquet')
factor = pd.read_parquet(sample_factor_dir / f'{factor_name}.parquet')
# factor = -factor
factor = factor.rename(columns=index_to_futures)


# %% plot factor
# =============================================================================
# factor.resample('4h').mean().plot()
# plt.legend(loc='best')
# plt.gcf().set_size_inches(10, 6)  # 宽 10 英寸，高 6 英寸
# plt.grid(True)
# plt.show()
# 
# =============================================================================

# %% plot scaled factor
# =============================================================================
# factor_scaled.resample('4h').mean().plot()
# plt.legend(loc='best')
# plt.gcf().set_size_inches(10, 6)  # 宽 10 英寸，高 6 英寸
# plt.grid(True)
# plt.show()
# =============================================================================


# %%
# =============================================================================
# midprice = midprice.reindex(columns=factor.columns)
# midprice = midprice.loc[factor.index.min():factor.index.max()] # 按factor头尾截取
# factor = factor.reindex(midprice.index) # 按twap reindex，确保等长
# 
# =============================================================================

# %%
pp_by_sp = int(parse_time_string(pp) / parse_time_string(sp))
rtn_1p = midprice.pct_change(pp_by_sp, fill_method=None).shift(-pp_by_sp) / pp_by_sp
rtn_1p = rtn_1p.replace([np.inf, -np.inf], 0)
main_col = rtn_1p.columns


# %%
factor, rtn_1p, midprice = align_and_sort_columns([factor, rtn_1p, midprice])

midprice = midprice.loc[factor.index.min():factor.index.max()] # 按factor头尾截取
rtn_1p = rtn_1p.loc[factor.index.min():factor.index.max()] # 按factor头尾截取
factor = factor.reindex(rtn_1p.index) # 按twap reindex，确保等长


# %%
scale_func = globals()[scale_method]
scale_step = int(parse_time_string(scale_window) / parse_time_string(sp))
if scale_method in ['ts_quantile_scale', 'ts_quantile_scale_separate']:
    factor_scaled = scale_func(factor, window=scale_step, quantile=scale_quantile)
elif scale_method in ['ts_quantile_scale_adj_by_his_rtn', 'ts_quantile_scale_adj_zscore_and_minmax']:
    factor_scaled = scale_func(factor, rtn_1p, window=scale_step, rtn_window=pp_by_sp, quantile=scale_quantile)
elif scale_method in ['rolling_percentile']:
    factor_scaled = scale_func(factor, window=scale_step)
elif scale_method in ['ts_adj_by_his_rtn_percentile', 'ts_adj_by_his_rtn_percentile_simple']:
    factor_scaled = scale_func(factor, rtn_1p, window=scale_step, rtn_window=pp_by_sp)


# %% 计算不同 direction 下的 factor_scaled
factor_scaled_dict = {}
for direction in direction_choices:
    factor_scaled_to_pos = (factor_scaled - 0.5) * 2
    if direction == 'all':
        factor_scaled_direction = factor_scaled_to_pos.copy()
    elif direction == 'pos':
        factor_scaled_direction = factor_scaled_to_pos.clip(lower=0)
    elif direction == 'neg':
        factor_scaled_direction = factor_scaled_to_pos.clip(upper=0)

    factor_scaled_dict[direction] = factor_scaled_direction
    

# %% 计算不同 direction 下的 gp
gp_dict = {}
gpd_dict = {}
for direction in direction_choices:
    factor_scaled_direction = factor_scaled_dict[direction]
    gp = (factor_scaled_direction * rtn_1p).fillna(0)
    gp['return'] = gp.mean(axis=1)
    gpd = gp.resample('D').sum(min_count=1).dropna()
    gpd['return'] = gpd.mean(axis=1)
    gp_dict[direction] = gp
    gpd_dict[direction] = gpd
    
    
# %% 计算不同 direction 下的 hsr
hsr_dict = {}
for direction in direction_choices:
    factor_scaled_direction = factor_scaled_dict[direction]
    hsr = ((factor_scaled_direction - factor_scaled_direction.shift(pp_by_sp)) / 2).abs().replace(
        [np.inf, -np.inf, np.nan], 0)
    hsrd = hsr.resample('1d').mean()
    hsrd['avg'] = hsrd.mean(axis=1)
    hsr_dict[direction] = hsrd
    
        
# %% plot
# 创建图形并设置大小
fig = plt.figure(figsize=(48, 24))  # 增大图形尺寸
gs = gridspec.GridSpec(4, 3, height_ratios=[4, 1, 1, 1], hspace=0.2, wspace=0.15)  # 调整间距为 4 行布局

# 获取 gpd_dict 所有列的最大和最小 Y 值以对齐 Y 轴
max_cum_return = max(
    [
        gpd_dict[direction].cumsum().max().max()  # 对所有列求累计和的最大值
        for direction in direction_choices
    ]
)
min_cum_return = min(
    [
        gpd_dict[direction].cumsum().min().min()  # 对所有列求累计和的最小值
        for direction in direction_choices
    ]
)

direction_choices = direction_choices.copy()

mul = 1 if gpd_dict['all']["return"].sum() > 0 else -1
if mul == -1:
    direction_choices[1] = 'neg'
    direction_choices[2] = 'pos'
plot_titles = ['all', 'long_only', 'short_only']

# 遍历方向并绘制子图
for i, direction in enumerate(direction_choices):
    col = i  # 控制左右排列的列索引
    
    # gp 图
    ax_gp = fig.add_subplot(gs[0, col])
    gpd = gpd_dict[direction]
    
    for i_c, column in enumerate(gpd.columns):
        if column != "return":
            ax_gp.plot(gpd.index, gpd[column].cumsum()*mul, alpha=0.7, label=column, color=plt.cm.tab10(i_c))
    
    ax_gp.plot(gpd.index, gpd["return"].cumsum()*mul, color="black", linewidth=4, label="Return")
    ax_gp.set_title(f"Direction: {plot_titles[i].capitalize()}", fontsize=28, pad=20)
    
    # 设置 Y 轴范围和标签
    ylim = ([min_cum_return * 1.1, max_cum_return * 1.1] if mul == 1
            else [max_cum_return * 1.1 * mul, min_cum_return * 1.1 * mul])
    ax_gp.set_ylim(ylim)  # 动态调整 Y 轴范围
    ax_gp.set_ylabel("Cumulative Return", fontsize=22, labelpad=20)
    if i == 0:
        ax_gp.yaxis.set_label_position('right')
    else:
        ax_gp.yaxis.label.set_visible(False)
    ax_gp.grid(True, linestyle="--", linewidth=0.8)
    ax_gp.tick_params(axis="y", labelsize=22)

    # 使用 AutoDateLocator 和 AutoDateFormatter 动态调整刻度
    locator = mdates.AutoDateLocator()  # 自动选择合适的刻度单位
    formatter = mdates.AutoDateFormatter(locator)  # 自动格式化刻度
    ax_gp.xaxis.set_major_locator(locator)  # 应用主刻度定位器
    ax_gp.xaxis.set_major_formatter(formatter)  # 应用主刻度格式化器
    
    # 调整主刻度字体和样式
    ax_gp.tick_params(axis="x", which="major", labelsize=16, pad=15)  # 主刻度字体更大
    
    if i == 2:  # 只显示一次 legend
        ax_gp.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=20, frameon=True, shadow=True)
    else:
        ax_gp.legend().remove()
        
    # midprice 图（新增的第四幅子图，位于第二行）
    ax_midprice = fig.add_subplot(gs[1, col], sharex=ax_gp)
    mid_log_return = np.log(midprice).diff().resample('1d').sum()  # 转为对数收益率
    mid_cum_log_return = mid_log_return.cumsum()  # 累计对数收益率

    # 绘制累计对数收益
    for i_c, column in enumerate(mid_cum_log_return.columns):
        ax_midprice.plot(mid_cum_log_return.index, mid_cum_log_return[column], 
                         alpha=0.7, label=column, color=plt.cm.tab10(i_c))
    ax_midprice.set_ylabel("Index CumRtn", fontsize=22, labelpad=20)
    ax_midprice.yaxis.set_label_position('right')
    ax_midprice.grid(True, linestyle="--", linewidth=0.8)
    ax_midprice.tick_params(axis="y", labelsize=22)
    
    plt.setp(ax_midprice.get_xticklabels(), visible=False)

        
    # hsr 图
    ax_hsr = fig.add_subplot(gs[2, col], sharex=ax_gp)
    hsrd = hsr_dict[direction]
    ax_hsr.bar(hsrd.index, hsrd["avg"], linewidth=2, label="HSR Avg")
    
    # 设置 HSR 图的 Y 轴范围和标签
    ax_hsr.set_ylim([0, 0.25])
    ax_hsr.set_ylabel("HSR Avg", fontsize=22, labelpad=20)
    ax_hsr.yaxis.set_label_position('right')
    ax_hsr.grid(True, linestyle="--", linewidth=0.8)
    ax_hsr.tick_params(axis="y", labelsize=22)
    
    # 移除 HSR 图的 X 轴刻度
    plt.setp(ax_hsr.get_xticklabels(), visible=False)

    # 每分钟收益的平均值图
    ax_minute = fig.add_subplot(gs[3, col])
    gp = gp_dict[direction]
    
    # 按时间聚合，提取每分钟的时间部分并求均值
    gp.index = pd.to_datetime(gp.index)  # 确保索引为 datetime 类型
    gp["minute"] = gp.index.time  # 提取分钟部分
    avg_per_minute = gp.groupby("minute")["return"].mean()  # 按分钟分组求平均收益
    
    # 将 `datetime.time` 转换为数值索引（例如分钟序号）
    minute_labels = [t.hour * 60 + t.minute for t in avg_per_minute.index]
    
    # 绘制条形图
    ax_minute.bar(minute_labels, avg_per_minute*mul, color=plt.cm.Set2(3), label="Avg Return")
    
    # 设置 X 轴为原始时间格式
    ax_minute.set_xticks(minute_labels[::30])  # 每隔 30 分钟显示一次刻度
    ax_minute.set_xticklabels([f"{t // 60:02}:{t % 60:02}" for t in minute_labels[::30]], rotation=45, fontsize=16)
    
    # 设置 Y 轴
    ax_minute.set_ylabel("Avg Minute Return", fontsize=22, labelpad=20)
    ax_minute.yaxis.set_label_position('right')
    ax_minute.grid(True, linestyle="--", linewidth=0.8)
    ax_minute.tick_params(axis="y", labelsize=22)
    
# 设置全局标题
fig.suptitle(f"{factor_name}  {scale_method}\nsp: {sp}  pp: {pp}  scale: {scale_window}  direction: {mul}", fontsize=40, y=0.98)

# 生成文件名
plot_file_path = sample_save_dir / f"{factor_name}_sp{sp}_pp{pp}_scale{scale_window}.png"

# 保存图表到 sample_data_dir
plt.savefig(plot_file_path, bbox_inches="tight", dpi=300)

# 显示图表
plt.show()

    
    
# %% plot
# =============================================================================
# # 创建图形并设置大小
# fig = plt.figure(figsize=(36, 18))  # 增大图形尺寸
# gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1], hspace=0.2, wspace=0.2)  # 调整间距
# 
# 
# # 获取 gpd_dict 所有列的最大和最小 Y 值以对齐 Y 轴
# max_cum_return = max(
#     [
#         gpd_dict[direction].cumsum().max().max()  # 对所有列求累计和的最大值
#         for direction in direction_choices
#     ]
# )
# min_cum_return = min(
#     [
#         gpd_dict[direction].cumsum().min().min()  # 对所有列求累计和的最小值
#         for direction in direction_choices
#     ]
# )
# 
# 
# # 遍历方向并绘制子图
# for i, direction in enumerate(direction_choices):
#     col = i  # 控制左右排列的列索引
#     
#     # gp 图
#     ax_gp = fig.add_subplot(gs[0, col])
#     gpd = gpd_dict[direction]
#     
#     for column in gpd.columns:
#         if column != "return":
#             ax_gp.plot(gpd.index, gpd[column].cumsum(), alpha=0.6, label=column)
#     
#     ax_gp.plot(gpd.index, gpd["return"].cumsum(), color="black", linewidth=4, label="Return")
#     ax_gp.set_title(f"Direction: {direction.capitalize()}", fontsize=28, pad=20)
#     
#     # 设置 Y 轴范围和标签
#     ax_gp.set_ylim([min_cum_return * 1.1, max_cum_return * 1.1])  # 动态调整 Y 轴范围
#     ax_gp.set_ylabel("Cumulative Return", fontsize=26, labelpad=20)
#     ax_gp.grid(True, linestyle="--", linewidth=0.8)
#     ax_gp.tick_params(axis="y", labelsize=22)
#  
#     # 使用 AutoDateLocator 和 AutoDateFormatter 动态调整刻度
#     locator = mdates.AutoDateLocator()  # 自动选择合适的刻度单位
#     formatter = mdates.AutoDateFormatter(locator)  # 自动格式化刻度
#     ax_gp.xaxis.set_major_locator(locator)  # 应用主刻度定位器
#     ax_gp.xaxis.set_major_formatter(formatter)  # 应用主刻度格式化器
#     
#     # 调整主刻度字体和样式
#     ax_gp.tick_params(axis="x", which="major", labelsize=16, pad=15)  # 主刻度字体更大
#     
#     if i == 1:  # 只显示一次 legend
#         ax_gp.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=20, frameon=True, shadow=True)
#     else:
#         ax_gp.legend().remove()
#         
#     # hsr 图
#     ax_hsr = fig.add_subplot(gs[1, col], sharex=ax_gp)
#     hsrd = hsr_dict[direction]
#     ax_hsr.bar(hsrd.index, hsrd["avg"], linewidth=2, label="HSR Avg")
#     
#     # 设置 HSR 图的 Y 轴范围和标签
#     ax_hsr.set_ylim([0, 0.25])
#     ax_hsr.set_ylabel("HSR Avg", fontsize=26, labelpad=20)
#     ax_hsr.grid(True, linestyle="--", linewidth=0.8)
#     ax_hsr.tick_params(axis="y", labelsize=22)
#     
#     # 移除 HSR 图的 X 轴刻度
#     plt.setp(ax_hsr.get_xticklabels(), visible=False)
# 
# # 设置全局标题
# fig.suptitle(f"{factor_name}  sp: {sp}  pp: {pp}  scale: {scale_window}", fontsize=40, y=0.98)
# 
# # 生成文件名
# plot_file_path = sample_save_dir / f"{factor_name}_sp{sp}_pp{pp}_scale{scale_window}.png"
# 
# # 保存图表到 sample_data_dir
# plt.savefig(plot_file_path, bbox_inches="tight", dpi=300)
# 
# # 显示图表
# plt.show()
# =============================================================================
