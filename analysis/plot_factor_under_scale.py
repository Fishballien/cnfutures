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
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import toml


from utils.market import index_to_futures
from utils.scores import get_general_return_metrics


# %%
# factor_name = 'l_amount_wavg_imb01'
# factor_name = 'amount_Quantile_R1_org_R2_org_R3_Sum_LXPct_R4_Imb2IntraRmDodPctChg_10'
# factor_name = 'tsstd_2h_csmean_closeprice_taylor_240m'
# factor_name = 'ActBuyAmt'
factor_name_list = [
    'l_amount_wavg_imb01',
    'amount_Quantile_R1_org_R2_org_R3_Sum_LXPct_R4_Imb2IntraRmDodPctChg_10',
    'tsstd_2h_csmean_closeprice_taylor_240m',
    'ActBuyAmt',
    ]
test_name_list = [
    'regular_futtwap_sp1min_pp60min_s240d',
    's02_futtwap_sp1min_pp60min_s240d',
    's03_futtwap_sp1min_pp60min_s240d',
    's04_futtwap_sp1min_pp60min_s240d',
    's05_futtwap_sp1min_pp60min_s240d',
    's06_futtwap_sp1min_pp60min_s240d',
    ]


# %%
param_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\params\test')
test_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\results\test')
save_dir = Path(r'D:\CNIndexFutures\timeseries\factor_test\analysis\factor_under_scale')
save_dir.mkdir(parents=True, exist_ok=True)


# %%
factor_sharpe = pd.DataFrame()

for factor_name in factor_name_list:
    scale_perf = {}
    for test_name in test_name_list:
        test_param_path = param_dir / f'{test_name}.toml'
        test_param = toml.load(test_param_path)
        scale_method = test_param.get('scale_method', 'minmax_scale')
        gp_path = test_dir / test_name / 'zxt' / 'factors_for_scale_test' / 'data' / f'gp_{factor_name}.pkl'
        with open(gp_path, 'rb') as f:
            gp = pickle.load(f)
        scale_perf[scale_method] = gp['all']['return'].loc[:'20241001']
        
    # 绘制图形
    plt.figure(figsize=(14, 8))
    
    for test_name, series in scale_perf.items():
        metrics = get_general_return_metrics(series.values)
        factor_sharpe.loc[factor_name, test_name] = metrics['sharpe_ratio']
        label = f"{test_name} - Sharpe: {metrics['sharpe_ratio']:.2f}"
        plt.plot(series.index, series.cumsum(), label=label)
    
    plt.title(f"{factor_name}", fontsize=16, pad=15)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Cumulative Return", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plot_file_path = save_dir / f"{factor_name}.jpg"
    plt.savefig(plot_file_path, bbox_inches="tight", dpi=300)
    plt.show()
    
factor_sharpe.to_csv(save_dir / 'sharpe_factors_under_scale.csv')
