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
from tqdm import tqdm
import pickle
import seaborn as sns


from utils.datautils import align_and_sort_columns
from utils.market import index_to_futures
# from trans_operators.format import to_float32


from utils.timeutils import parse_time_string
from utils.trade_rules import *
from data_processing.ts_trans import *
from test_and_eval.factor_tester import FactorTesterByDiscrete
from test_and_eval.factor_evaluation import eval_one_factor_one_period_net_public


# %%
sl_tp_name = "v0"
model_name = 'avg_agg_250218_3_fix_tfe_by_trade_net_v4'
factor_name = f'predict_{model_name}'
factor_dir = Path(rf'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model\{model_name}\predict')
direction = 1

# factor_name = 'LargeOrderAmountByValue_p1.0_v40000-wavg_imb04_dpall-mean_w30min'
# direction = -1
# factor_dir = Path(r'D:/mnt/CNIndexFutures/timeseries/factor_test/sample_data/factors/low_freq')


# %%
test_name = 'traded_futtwap_sp1min_s240d_icim_v6_noscale'
price_name = 't1min_fq1min_dl1min'

scale_method = 'minmax_scale'
scale_window = '240d'
scale_quantile = 0.02
sp = '1min'

trade_rule_name = 'trade_rule_by_trigger_v5'
trade_rule_param = {
    'openthres': 0.8,
    'closethres': 0,
    }

fee = 0.00024

sl_list = [100, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
tp_list = [100, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]

# sl_list = [100, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
# tp_list = [100, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]

date_start = '20160101'
date_end = '20250101'


# %%
fut_dir = Path('/mnt/data1/future_twap')
analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\sl_tp')
pos_dir = analysis_dir / 'pos' / factor_name / trade_rule_name / sl_tp_name
pos_dir.mkdir(parents=True, exist_ok=True)
test_dir = analysis_dir / 'test' / factor_name / trade_rule_name / sl_tp_name
test_dir.mkdir(parents=True, exist_ok=True)
summary_dir = analysis_dir / 'summary' / factor_name / trade_rule_name / sl_tp_name
summary_dir.mkdir(parents=True, exist_ok=True)


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

factor_scaled = (factor_scaled - 0.5) * 2 * direction


# %%
# =============================================================================
# for sl in tqdm(sl_list, desc='sl'):
#     for tp in tp_list:
#         trade_rule_param_spec = {k: v for k, v in trade_rule_param.items()}
#         trade_rule_param_spec['stoploss_pct'] = sl
#         trade_rule_param_spec['takeprofit_pct'] = tp
#         trade_rule_func = partial(globals()[trade_rule_name], **trade_rule_param_spec)
#         actual_pos = factor_scaled.apply(lambda col: trade_rule_func(col.values, price_data[col.name].shift(2).values), axis=0)
#         file_name = f'sl{sl}_tp{tp}'
#         actual_pos.to_parquet(pos_dir / f'{file_name}.parquet')
# 
# =============================================================================

# %%
# =============================================================================
# tester = FactorTesterByDiscrete(None, None, pos_dir, test_name=test_name, 
#                                 result_dir=test_dir)
# tester.test_multi_factors()
# =============================================================================


# %%
# =============================================================================
# test_data_dir = test_dir / 'test' / test_name / 'data'
# 
# res_list = []
# for sl in sl_list:
#     for tp in tp_list:
#         res_info = {
#             'sl': sl, 
#             'tp': tp, 
#             }
#         pred_name = f'sl{sl}_tp{tp}'
#         res_dict = eval_one_factor_one_period_net_public(
#             pred_name, res_info, test_data_dir, date_start, date_end, fee)
#         
#         res_list.append(res_dict)
#         
# res_df = pd.DataFrame(res_list)
# =============================================================================


# %%
# =============================================================================
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# # 假设res_df已经加载到环境中
# # 提取 x, y 和 heatmap 指标
# x = "sl"
# y = "tp"
# heatmap_metrics = [col for col in res_df.columns if col not in [x, y]]
# 
# # 遍历所有指标绘制heatmap
# for metric in heatmap_metrics:
#     pivot_table = res_df.pivot(index=y, columns=x, values=metric)
# 
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
#     plt.title(f"Heatmap of {metric}")
#     plt.xlabel(x)
#     plt.ylabel(y)
# 
#     # 保存图片
#     img_path = os.path.join(summary_dir, f"heatmap_{metric}.png")
#     plt.savefig(img_path, dpi=300, bbox_inches="tight")
#     plt.close()
# 
# # 显示完成消息
# summary_dir
# =============================================================================


# %%
test_data_dir = test_dir / 'test' / test_name / 'data'

net_list = []
for sl in sl_list:
    for tp in tp_list:
        res_info = {
            'sl': sl, 
            'tp': tp, 
            }
        pred_name = f'sl{sl}_tp{tp}'
        test_data = {}
        for data_type in ('gpd', 'hsr'):
            data_path = test_data_dir / f'{data_type}_{pred_name}.pkl'
            with open(data_path, 'rb') as f:
                test_data[data_type] = pickle.load(f)
                
        df_gp = test_data['gpd']['all']
        df_hsr = test_data['hsr']['all']
        
        net = (df_gp['return'] - fee * df_hsr['avg']).fillna(0)
        
        net_list.append(net)
        
# 将 net_list 转换为 DataFrame
# 每个 net_list 元素是一个 pandas Series，所以我们将它们组合成一个 DataFrame
net_df = pd.DataFrame({f'net_{i}': net for i, net in enumerate(net_list)})

# 计算两两之间的相关性
corr_matrix = net_df.corr()

# 绘制相关性热力图
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', 
            xticklabels=False, yticklabels=False)

# 添加标题
plt.title(f'Correlation Heatmap for {model_name} {trade_rule_name}')

# 调整图像布局，避免标签遮挡
plt.tight_layout()

# 保存图像
corr_img_filename = summary_dir / 'correlation_heatmap.png'
plt.savefig(corr_img_filename)
plt.show()
plt.close()  # 关闭当前图像，防止重叠


# %%
test_data_dir = test_dir / 'test' / test_name / 'data'
net_list = []
# 保存每个网格对应的sl和tp值
param_labels = []

for sl in sl_list:
    for tp in tp_list:
        res_info = {
            'sl': sl, 
            'tp': tp, 
            }
        param_labels.append(f'sl{sl}_tp{tp}')
        pred_name = f'sl{sl}_tp{tp}'
        test_data = {}
        for data_type in ('gpd', 'hsr'):
            data_path = test_data_dir / f'{data_type}_{pred_name}.pkl'
            with open(data_path, 'rb') as f:
                test_data[data_type] = pickle.load(f)
                
        df_gp = test_data['gpd']['all']
        df_hsr = test_data['hsr']['all']
        
        net = (df_gp['return'] - fee * df_hsr['avg']).fillna(0)
        
        net_list.append(net)
        
# 将 net_list 转换为 DataFrame，并使用参数标签作为列名
net_df = pd.DataFrame({label: net for label, net in zip(param_labels, net_list)})

# 计算两两之间的相关性
corr_matrix = net_df.corr()

# 创建一个更大的图，以便有足够空间显示标签
plt.figure(figsize=(16, 14))

# 绘制热力图，这次包含标签
ax = sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f',
               xticklabels=param_labels, yticklabels=param_labels)

# 设置标签旋转角度，提高可读性
plt.xticks(rotation=90)
plt.yticks(rotation=0)

# 添加网格线来分隔不同的sl区域
sl_count = len(sl_list)
tp_count = len(tp_list)
total_count = sl_count * tp_count

# 添加垂直和水平分隔线以区分不同的sl组
for i in range(1, sl_count):
    plt.axhline(y=i*tp_count, color='black', linestyle='-', linewidth=2)
    plt.axvline(x=i*tp_count, color='black', linestyle='-', linewidth=2)

# 添加标题和标签
plt.title(f'Correlation Heatmap for {model_name} {trade_rule_name}', fontsize=16)
plt.xlabel('Parameter Combinations', fontsize=14)
plt.ylabel('Parameter Combinations', fontsize=14)


# 添加sl和tp的区域标注
# 为每个sl组添加文本标签
for i, sl in enumerate(sl_list):
    # 为热力图边缘添加sl组标签
    plt.text(-0.05, i*tp_count + tp_count/2, f'SL={sl}', 
             horizontalalignment='right', verticalalignment='center',
             fontsize=12, fontweight='bold')
    
    plt.text(i*tp_count + tp_count/2, -0.05, f'SL={sl}', 
             horizontalalignment='center', verticalalignment='top',
             fontsize=12, fontweight='bold', rotation=90)

# 调整图像布局，避免标签遮挡
plt.tight_layout()

# 保存图像
corr_img_filename = summary_dir / 'correlation_heatmap_annotated.png'
plt.savefig(corr_img_filename, bbox_inches='tight', dpi=300)
plt.show()
plt.close()  # 关闭当前图像，防止重叠

# 如果需要更清晰地查看不同参数组合的相关性，可以添加另一个热力图
# 这个热力图使用平均相关性来显示不同sl组之间的关系
sl_groups = []
for i, sl in enumerate(sl_list):
    start_idx = i * tp_count
    end_idx = start_idx + tp_count
    sl_group = corr_matrix.iloc[start_idx:end_idx, start_idx:end_idx].mean().mean()
    sl_groups.append((sl, sl_group))

# 创建一个新的热力图，显示不同sl组之间的平均相关性
sl_corr = np.zeros((sl_count, sl_count))
for i in range(sl_count):
    for j in range(sl_count):
        start_i, end_i = i * tp_count, (i + 1) * tp_count
        start_j, end_j = j * tp_count, (j + 1) * tp_count
        sl_corr[i, j] = corr_matrix.iloc[start_i:end_i, start_j:end_j].mean().mean()

plt.figure(figsize=(10, 8))
ax = sns.heatmap(sl_corr, annot=True, cmap='coolwarm', fmt='.2f',
               xticklabels=[f'SL={sl}' for sl in sl_list],
               yticklabels=[f'SL={sl}' for sl in sl_list])

plt.title(f'Average Correlation Between SL Groups for {model_name} {trade_rule_name}', fontsize=14)
plt.tight_layout()

# 保存图像
sl_corr_img_filename = summary_dir / 'sl_group_correlation_heatmap.png'
plt.savefig(sl_corr_img_filename, bbox_inches='tight', dpi=300)
plt.show()
plt.close()