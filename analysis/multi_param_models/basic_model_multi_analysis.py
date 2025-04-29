# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 15:36:08 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import sys
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[2]
sys.path.append(str(project_dir))


# %% import self-defined
from utils.dirutils import load_path_config
from test_and_eval.scores import get_general_return_metrics

          
# %% main
def extract_initials(s: str) -> str:
    parts = s.split('_')
    initials = ''.join(part[0] for part in parts if part)
    return initials


# %%
base_model_name = 'avg_agg_250409'
test_name = 'trade_ver3_intraday_futtwap_sp1min_s240d_icim_v6'

sharpe_list = [1.6, 1.8, 2.0]
min_count_list = [50]
burke_list = [10, 15, 100]
sort_target_list = ['net_sharpe_ratio']
distance_list = [0.1, 0.2, 0.4]

fee = 0.00024


# %%
path_config = load_path_config(project_dir)
result_dir = Path(path_config['result'])
model_dir = result_dir / 'model' / base_model_name
analysis_dir = result_dir / 'analysis' / 'multi_models' / base_model_name
analysis_dir.mkdir(parents=True, exist_ok=True)


# %%
# 用于存储结果的变量
all_results = []
all_net_series = {}

# 遍历所有参数组合
for sharpe in sharpe_list:
    for min_count in min_count_list:
        for burke in burke_list:
            for sort_target in sort_target_list:
                for distance in distance_list:
                    # 生成模型名称
                    suffix = f'nsr{sharpe}_top{min_count}by{extract_initials(sort_target)}_nbr{burke}_dist{distance}'
                    model_name = f'{base_model_name}_{suffix}'
                    
                    # 加载数据
                    try:
                        gp_path = model_dir / f'{model_name}/test/{test_name}/data/gpd_predict_{model_name}.pkl'
                        hsr_path = model_dir / f'{model_name}/test/{test_name}/data/hsr_predict_{model_name}.pkl'
                        
                        with open(gp_path, 'rb') as f:
                            gp = pickle.load(f)
                        with open(hsr_path, 'rb') as f:
                            hsr = pickle.load(f)
                        
                        # 计算净收益率
                        net = gp['all']['return'] - hsr['all']['avg'] * fee
                        
                        # 计算指标
                        metrics = get_general_return_metrics(net.values)
                        profit_per_trade = gp['all']['return'].sum() / hsr['all']['avg'].sum()
                        
                        # 添加profit_per_trade到metrics
                        metrics['profit_per_trade'] = profit_per_trade
                        
                        # 保存结果
                        result = {
                            'model_name': model_name,
                            'sharpe': sharpe,
                            'min_count': min_count,
                            'burke': burke,
                            'sort_target': sort_target,
                            'distance': distance,
                            **metrics
                        }
                        all_results.append(result)
                        
                        # 保存净收益率序列
                        all_net_series[model_name] = net
                        
                        print(f"Processed {model_name}")
                    except Exception as e:
                        print(f"Error processing {model_name}: {e}")

# 将结果转换为DataFrame
results_df = pd.DataFrame(all_results)

# 1. 绘制所有net的cumsum，并标注sharpe_ratio前三
if all_net_series:
    plt.figure(figsize=(12, 8))
    
    # 找出sharpe_ratio前三的模型
    top3_sharpe = results_df.sort_values('sharpe_ratio', ascending=False).head(3)
    top3_models = top3_sharpe['model_name'].tolist()
    
    # 绘制所有模型的累积收益
    for model_name, net in all_net_series.items():
        cum_returns = net.cumsum()
        if model_name in top3_models:
            plt.plot(cum_returns.index, cum_returns.values, linewidth=2, label=f"{model_name} (Sharpe: {results_df[results_df['model_name'] == model_name]['sharpe_ratio'].values[0]:.2f})")
        else:
            plt.plot(cum_returns.index, cum_returns.values, linewidth=0.8, alpha=0.5)
    
    plt.title('Cumulative Returns for All Parameter Combinations')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(analysis_dir / 'cumulative_returns.png', dpi=300)
    plt.close()

# 2. 创建热力图
# 需要考虑的指标
metrics_to_plot = [
    'return', 'return_annualized', 'max_dd', 'sharpe_ratio', 
    'calmar_ratio', 'sortino_ratio', 'sterling_ratio', 
    'burke_ratio', 'ulcer_index', 'drawdown_recovery_ratio',
    'profit_per_trade'
]

# 参数对1: sharpe 和 distance
for metric in metrics_to_plot:
    # 创建一个透视表计算平均指标值
    pivot_df = results_df.pivot_table(
        index='sharpe', 
        columns='distance',
        values=metric,
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f')
    plt.title(f'Average {metric} by Sharpe and Distance')
    plt.tight_layout()
    plt.savefig(analysis_dir / f'heatmap_sharpe_distance_{metric}.png', dpi=300)
    plt.close()

# 参数对2: min_count 和 burke
for metric in metrics_to_plot:
    # 创建一个透视表计算平均指标值
    pivot_df = results_df.pivot_table(
        index='min_count', 
        columns='burke',
        values=metric,
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f')
    plt.title(f'Average {metric} by Min Count and Burke')
    plt.tight_layout()
    plt.savefig(analysis_dir / f'heatmap_mincount_burke_{metric}.png', dpi=300)
    plt.close()

print("Analysis complete. All results saved to analysis_dir.")