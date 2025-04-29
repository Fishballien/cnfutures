# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:27:24 2025

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
import toml
import concurrent.futures
from tqdm import tqdm
import traceback
import warnings
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from utils.dirutils import load_path_config
from synthesis.rolling_cluster import RollingCluster
from scripts.rolling_fit_pred_backtest import main as rolling_fit_pred_backtest

          
# %% main
def extract_initials(s: str) -> str:
    parts = s.split('_')
    initials = ''.join(part[0] for part in parts if part)
    return initials


# %%
base_cluster_name = 'agg_250409'
base_model_name = 'avg_agg_250409'
fstart = '20150101'
pstart = '20160101'
puntil = '20250401'
mode = 'rolling'
model_n_workers = 20
total_n_workers = 20

sharpe_list = [1.4, 1.6, 1.8, 2.0]
min_count_list = [10, 25, 50, 100, 200]
burke_list = [10, 15, 30, 100]
sort_target_list = ['net_sharpe_ratio']
distance_list = [0.1, 0.2, 0.4]

to_run_cluster = False
to_run_model = True


# %%
path_config = load_path_config(project_dir)
param_dir = Path(path_config['param'])

base_cluster_params = toml.load(param_dir / 'cluster' / f'{base_cluster_name}.toml')
base_model_params = toml.load(param_dir / 'model' / f'{base_model_name}.toml')

cluster_dir = param_dir / 'cluster' / base_cluster_name
model_dir = param_dir / 'model' / base_model_name
cluster_dir.mkdir(exist_ok=True, parents=True)
model_dir.mkdir(exist_ok=True, parents=True)

cluster_name_list = []
model_name_list = []

for sharpe in sharpe_list:
    for min_count in min_count_list:
        for burke in burke_list:
            for sort_target in sort_target_list:
                for distance in distance_list:
                    suffix = f'nsr{sharpe}_top{min_count}by{extract_initials(sort_target)}_nbr{burke}_dist{distance}'
                    cluster_name = f'{base_cluster_name}_{suffix}'
                    model_name = f'{base_model_name}_{suffix}'
                    cluster_path = cluster_dir / f'{cluster_name}.toml'
                    model_path = model_dir / f'{model_name}.toml'
                    cluster_pr_to_update = {
                        'filter_func': 'filter_func_dynamic',
                        'filter_func_param': {
                            'conditions': [
                                {'target': 'net_sharpe_ratio', 'operator': 'greater', 'threshold': sharpe, 'is_multiplier': False},
                                {'target': 'net_burke_ratio', 'operator': 'less', 'threshold': burke, 'is_multiplier': False},
                            ],
                            'min_count': min_count,
                            'sort_target': sort_target,
                            },
                        'cluster_params': {
                            'criterion': 'distance',
                            't': distance,
                            },
                        }
                    cluster_param = base_cluster_params.copy()
                    cluster_param.update(cluster_pr_to_update)
                    model_param = base_model_params.copy()
                    model_param['preprocess_params']['cluster'] = cluster_name
                    model_param['preprocess_params']['base_cluster'] = base_cluster_name
                    with open(cluster_path, "w", encoding="utf-8") as f:
                        toml.dump(cluster_param, f)
                    with open(model_path, "w", encoding="utf-8") as f:
                        toml.dump(model_param, f)
                    cluster_name_list.append(cluster_name)
                    model_name_list.append(model_name)
                

# %% Run clusters with parallel processing
def run_cluster(cluster_name):
    params = {
        'cluster_name': cluster_name,
        'pstart': pstart,
        'puntil': puntil,
        'cluster_type': mode,
        'base_cluster_name': base_cluster_name,
    }
    c = RollingCluster(**params)
    c.run()
    return cluster_name

def run_model(model_name):
    params = {
        'test_name': model_name,
        'fstart': fstart,
        'pstart': pstart,
        'puntil': puntil,
        'mode': mode,
        'n_workers': model_n_workers,
        'base_model_name': base_model_name,
    }
    rolling_fit_pred_backtest(**params)
    return model_name

# Run clusters in parallel
if to_run_cluster:
    print(f"Running {len(cluster_name_list)} clusters in parallel with {total_n_workers} workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=total_n_workers) as executor:
        futures = [executor.submit(run_cluster, cluster_name) for cluster_name in cluster_name_list]
        
        # Track progress with tqdm
        for completed_cluster in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing clusters"
        ):
            try:
                cluster_name = completed_cluster.result()
                print(f"Completed cluster: {cluster_name}")
            except Exception as e:
                e = traceback.format_exc()
                print(f"Error in cluster processing: {e}")
                raise

# Run models in parallel
if to_run_model:
    print(f"Running {len(model_name_list)} models in parallel with {total_n_workers} workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=total_n_workers) as executor:
        futures = [executor.submit(run_model, model_name) for model_name in model_name_list]
        
        # Track progress with tqdm
        for completed_model in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing models"
        ):
            try:
                model_name = completed_model.result()
                print(f"Completed model: {model_name}")
            except Exception as e:
                e = traceback.format_exc()
                print(f"Error in model processing: {e}")
                raise
            