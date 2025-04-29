# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:30:26 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import sys
import pickle
from collections import defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import list_pattern_matches
from test_and_eval.scores import get_general_return_metrics


# %%
# dir_dict = {
#     'dp0': r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\test\trade_ver0_futtwap_sp1min_s240d_icim\zxt\Batch10_fix_best_241218_selected_f64\test_downscale_dp0\data',
#     'dp2': r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\test\trade_ver0_futtwap_sp1min_s240d_icim\zxt_select_250203\Batch10_fix_best_241218_selected_f64\test_downscale_dp2\data',
#     }
dir_dict = {
    'dp0': '/mnt/data1/xintang/CNIndexFutures/timeseries/factor_test/results/test/trade_ver0_futtwap_sp1min_s240d_icim/zxt/Batch10_fix_best_241218_selected_f64/test_downscale_dp0/data',
    'dp2': '/mnt/data1/xintang/CNIndexFutures/timeseries/factor_test/results/test/trade_ver0_futtwap_sp1min_s240d_icim/zxt_select_250203/Batch10_fix_best_241218_selected_f64/test_downscale_dp2/data',
    }
fee = 0.00024
summary_dir = Path('/mnt/data1/xintang/CNIndexFutures/timeseries/factor_test/results/analysis/compare_dp_by_year')
summary_dir.mkdir(parents=True, exist_ok=True)


# %%
res_dict = defaultdict(dict)

for dir_name in dir_dict:
    res_dict_by_year = defaultdict(list)
    factors = list_pattern_matches(dir_dict[dir_name], 'gpd_*.pkl')
    for factor_name in tqdm(factors, desc=f'{dir_name}'):
        test_data = {}
        for data_type in ('gpd', 'hsr'):
            data_path = Path(dir_dict[dir_name]) / f'{data_type}_{factor_name}.pkl'
            with open(data_path, 'rb') as f:
                test_data[data_type] = pickle.load(f)
        
        df_gp = test_data['gpd']['all']
        df_hsr = test_data['hsr']['all']
        
        # Extract year from the date column (assuming 'date' exists in df_gp and df_hsr)
        df_gp['year'] = df_gp.index.year
        df_hsr['year'] = df_hsr.index.year
        
        cumrtn = df_gp['return'].sum()
        direction = 1 if cumrtn > 0 else -1
        
        for year in df_gp['year'].unique():
            # Filter data for the specific year
            df_gp_year = df_gp[df_gp['year'] == year]
            df_hsr_year = df_hsr[df_hsr['year'] == year]
        
            net = (df_gp_year['return'] * direction - fee * df_hsr_year['avg']).fillna(0)
            metrics = get_general_return_metrics(net.values)
            renamed_metrics = {f'net_{m}': v for m, v in metrics.items()}
        
            # Create dictionary for each year
            _res_dict = {
                'factor_name': factor_name,
                'year': year,
            }
            _res_dict.update(renamed_metrics)
        
            profit_per_trade = df_gp_year["return"].sum() / df_hsr_year["avg"].sum()
            _res_dict.update({'profit_per_trade': profit_per_trade * 1000})
            res_dict_by_year[year].append(_res_dict)
    
    for year in res_dict_by_year:
        res_dict[dir_name][year] = pd.DataFrame(res_dict_by_year[year])
        
        
# %%
import numpy as np
import matplotlib.pyplot as plt

# 提取每年每个路径的 Sharpe Ratio
years = sorted(res_dict['dp0'].keys())  # 假设每年都有数据

# 设置边缘颜色
edge_colors = ['#0000B3', '#FF007F']  # 深蓝色、深粉色

for year in years:
    dp0_sharpe_by_year = res_dict['dp0'][year]['net_sharpe_ratio'].values
    dp2_sharpe_by_year = res_dict['dp2'][year]['net_sharpe_ratio'].values
    
    # 创建每年单独的图表
    plt.figure(figsize=(10, 6))
    
    # 绘制 dp0 和 dp2 的 Sharpe Ratio 分布对比
    plt.hist(dp0_sharpe_by_year, bins=50, edgecolor=edge_colors[0], linewidth=1.5, histtype='step', 
             label=f'dp0 {year} (mean={np.mean(dp0_sharpe_by_year):.2f})')
    plt.hist(dp2_sharpe_by_year, bins=50, edgecolor=edge_colors[1], linewidth=1.5, histtype='step', 
             label=f'dp2 {year} (mean={np.mean(dp2_sharpe_by_year):.2f})')
    
    # 设置图表属性
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Frequency")
    plt.title(f"Sharpe Ratio Distribution Comparison (dp0 vs dp2) - Year {year}")
    plt.legend()
    plt.grid(True)
    
    # 保存每年图表到 summary_dir 目录
    plot_filename = summary_dir / f"sharpe_ratio_comparison_{year}.png"
    plt.savefig(plot_filename)
    print(f"Plot for {year} saved to {plot_filename}")
    
    # 显示图表
    plt.show()