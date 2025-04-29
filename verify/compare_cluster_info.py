# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:32:31 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import pandas as pd
import pickle


# %%
org_cluster_path = 'D:/2025-02-10  原先D盘/CNIndexFutures/timeseries/factor_test/results/cluster/agg_250203_by_trade_net_double3m_v6/cluster_info_210201_250201.csv'
new_cluster_path = 'D:/CNIndexFutures/timeseries/factor_test/results/cluster/agg_250203_by_trade_net_double3m_v6/cluster_info_210201_250201.csv'
eval_path = 'D:/CNIndexFutures/timeseries/factor_test/results/factor_evaluation/agg_batch10_250203_downscale_final/factor_eval_210201_250201.csv'


# %%
org_cluster = pd.read_csv(org_cluster_path)
new_cluster = pd.read_csv(new_cluster_path)
eval_data = pd.read_csv(eval_path)
eval_data = eval_data.set_index('factor')


# %%
# 假设org_cluster和new_cluster已经是DataFrame
org_factors = org_cluster['factor'].unique()  # 获取org_cluster中的所有唯一的factor
new_factors = new_cluster['factor'].unique()  # 获取new_cluster中的所有唯一的factor

# 在org_cluster有但new_cluster没有的factor
org_only_factors = set(org_factors) - set(new_factors)

# 在new_cluster有但org_cluster没有的factor
new_only_factors = set(new_factors) - set(org_factors)


org_only_eval = eval_data.loc[list(org_only_factors)]
new_only_eval = eval_data.loc[list(new_only_factors)]


# %% ValueTimeDecayOrderAmount_p1.0_v40000_d0.01-avg_imb04_dpall-org
old_path = 'D:/CNIndexFutures/timeseries/factor_factory/sample_data/verify/250213/old/ValueTimeDecayOrderAmount_p1.0_v40000_d0.01-avg_imb04_dpall-org.parquet'
new_path = 'D:/CNIndexFutures/timeseries/factor_factory/sample_data/verify/250213/new/ValueTimeDecayOrderAmount_p1.0_v40000_d0.01-avg_imb04_dpall-org.parquet'

old_fac = pd.read_parquet(old_path)
new_fac = pd.read_parquet(new_path)

def compare_dfs(df1, df2):
    # 首先检查两个 DataFrame 是否完全相同
    if df1.equals(df2):
        print("两个 DataFrame 完全相同！")
        return None
    
    # 查找 df1 中有但 df2 中没有的行
    df1_only = df1[~df1.apply(tuple, 1).isin(df2.apply(tuple, 1))]
    
    # 查找 df2 中有但 df1 中没有的行
    df2_only = df2[~df2.apply(tuple, 1).isin(df1.apply(tuple, 1))]
    
    return df1_only, df2_only


res = compare_dfs(old_fac, new_fac.loc[:'20250210'])


old_gp_path = 'D:/CNIndexFutures/timeseries/factor_factory/sample_data/verify/250213/old/gpd_ValueTimeDecayOrderAmount_p1.0_v40000_d0.01-avg_imb04_dpall-org.pkl'
old_hsr_path = 'D:/CNIndexFutures/timeseries/factor_factory/sample_data/verify/250213/old/hsr_ValueTimeDecayOrderAmount_p1.0_v40000_d0.01-avg_imb04_dpall-org.pkl'
new_gp_path = 'D:/CNIndexFutures/timeseries/factor_factory/sample_data/verify/250213/new/gpd_ValueTimeDecayOrderAmount_p1.0_v40000_d0.01-avg_imb04_dpall-org.pkl'
new_hsr_path = 'D:/CNIndexFutures/timeseries/factor_factory/sample_data/verify/250213/new/hsr_ValueTimeDecayOrderAmount_p1.0_v40000_d0.01-avg_imb04_dpall-org.pkl'

# Dictionary to hold the loaded data
loaded_data = {}

# Load pickle files
with open(old_gp_path, 'rb') as file:
    loaded_data['old_gp'] = pickle.load(file)

with open(old_hsr_path, 'rb') as file:
    loaded_data['old_hsr'] = pickle.load(file)

with open(new_gp_path, 'rb') as file:
    loaded_data['new_gp'] = pickle.load(file)

with open(new_hsr_path, 'rb') as file:
    loaded_data['new_hsr'] = pickle.load(file)
    
old_gp_all = loaded_data['old_gp']['all']
new_gp_all = loaded_data['new_gp']['all']

# 找到原因：gp将当日最后一个切片至跨日的收益 从 记在当日 改为 记到下一日