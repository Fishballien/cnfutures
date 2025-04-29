# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:17:19 2025

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
import pickle
import pandas as pd


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[2]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from test_and_eval.scores import get_general_return_metrics
from test_and_eval.factor_tester import FactorTesterByDiscrete
from test_and_eval.factor_evaluation import eval_one_factor_one_period_net_public, eval_one_factor_one_period_by_net


# %%
merge_name = 'v1.2_new_and_old_test'
model_list = [
    ['avg_agg_250218_3_by_trade_net_v6', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    ['avg_agg_250218_3_old_test_by_trade_net_v6', 'trade_ver3_futtwap_sp1min_s240d_icim_v6'],
    ]
fee = 0.00024
date_start = '20160101'
date_end = '20250101'
test_name = 'trade_ver3_futtwap_sp1min_s240d_icim_v6'
test_name_no_scale = 'trade_ver3_futtwap_sp1min_s240d_icim_v6_noscale'


# %%
path_config = load_path_config(project_dir)
result_dir = Path(path_config['result'])
model_dir = result_dir / 'model'
summary_dir = result_dir / 'analysis' / 'merge_model'
merge_dir = summary_dir / merge_name
merge_dir.mkdir(parents=True, exist_ok=True)


# %%
res_list = []


# %% 各自
for model_name, test_name in model_list:
    test_data_dir = model_dir / model_name / 'test' / test_name / 'data'
    pred_name = f'predict_{model_name}'
    
    res_info = {
        'model_name': model_name, 
        'test_name': test_name, 
        }
    
    res_dict = eval_one_factor_one_period_net_public(
        pred_name, res_info, test_data_dir, date_start, date_end, fee)
    
    res_list.append(res_dict)


# %% 权益等权
# =============================================================================
# net_dict = {}
# for model_name, test_name in model_list:
#     test_data_dir = model_dir / model_name / 'test' / test_name / 'data'
#     pred_name = f'predict_{model_name}'
#     
#     res_dict = {
#         'model_name': model_name, 
#         'test_name': test_name, 
#         }
#     
#     test_data = {}
#     for data_type in ('gpd', 'hsr'):
#         data_path = test_data_dir / f'{data_type}_{pred_name}.pkl'
#         with open(data_path, 'rb') as f:
#             test_data[data_type] = pickle.load(f)
#             
#     df_gp = test_data['gpd']['all']
#     df_hsr = test_data['hsr']['all']
#     
#     df_gp = df_gp[(df_gp.index >= date_start) & (df_gp.index <= date_end)]
#     df_hsr = df_hsr[(df_hsr.index >= date_start) & (df_hsr.index <= date_end)]
#     
#     net = (df_gp['return'] - fee * df_hsr['avg']).fillna(0)
#     net_dict[(model_name, test_name)] = net
# 
# 
# dfs = [df for df in net_dict.values()]
# combined_df = pd.concat(dfs, axis=1)
# average_df = combined_df.mean(axis=1)
# 
# res_dict = {'model_name': f'pnl_avg_{merge_name}'}
# res_dict.update(eval_one_factor_one_period_by_net(average_df))
# res_list.append(res_dict)
# =============================================================================
        
        
# %%
predict_dict = {}
for model_name, test_name in model_list:
    pred_dir = model_dir / model_name / 'predict'
    filename = f'predict_{model_name}'
    pred_path = pred_dir / f'{filename}.parquet'
    
    predict = pd.read_parquet(pred_path)
    predict_dict[model_name] = predict
    
# 假设 net_dict 是已经存在的字典
# 获取所有 DataFrame 的列名
columns = list(predict_dict.values())[0].columns  # 假设所有 df 的列名一致

# 获取第一个 DataFrame 的 index 作为结果的 index
index = list(predict_dict.values())[0].index

# 对每一列进行平均
averages = {}
for col in columns:
    # 获取所有 df 对应列的值并求平均
    averages[col] = sum([df[col] for df in predict_dict.values()]) / len(predict_dict)

# 将结果转换为一个 DataFrame，设置 index 为第一个 df 的 index
average_df = pd.DataFrame(averages, index=index)
factor_name = f'merge_{merge_name}'
average_df.to_parquet(merge_dir / f'{factor_name}.parquet')

tester = FactorTesterByDiscrete(None, None, merge_dir, test_name=test_name, 
                                result_dir=merge_dir)
tester.test_one_factor(f'merge_{merge_name}')

res_info = {'model_name': f'pos_merge_{merge_name}', 'test_name': test_name}
res_dict = eval_one_factor_one_period_net_public(
    factor_name, res_info, merge_dir / 'test' / test_name / 'data', date_start, date_end, fee)
res_list.append(res_dict)


# %%
final_res = pd.DataFrame(res_list)
final_res.to_csv(merge_dir / f'merge_compare_{merge_name}.csv', index=None)

