# -*- coding: utf-8 -*-
"""
Created on Thu May 29 10:14:14 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))

# %% import self-defined
from apply_filter.rolling_apply_filters_on_merged import run_rolling_apply_filters_on_merged

# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-af', '--apply_filters_name', type=str, help='apply_filters_name')
    parser.add_argument('-m', '--merge_name', type=str, help='merge_name')
    parser.add_argument('-rn', '--rolling_name', type=str, help='rolling_name')
    parser.add_argument('-pst', '--pstart', type=str, default='20230701', help='pstart')
    parser.add_argument('-pu', '--puntil', type=str, help='puntil')
    parser.add_argument('-mo', '--mode', type=str, default='rolling', help='mode')
    parser.add_argument('-wkr', '--max_workers', type=int, default=1, help='max_workers for filter processing')
    parser.add_argument('-nw', '--n_workers', type=int, default=1, help='n_workers for parallel periods')
    args = parser.parse_args()
    
    run_rolling_apply_filters_on_merged(
        apply_filters_name=args.apply_filters_name,
        merge_name=args.merge_name,
        rolling_name=args.rolling_name,
        pstart=args.pstart,
        puntil=args.puntil,
        mode=args.mode,
        max_workers=args.max_workers,
        n_workers=args.n_workers,
    )
    

# %% main
if __name__=='__main__':
    main()
    
    
# =============================================================================
# # /param/apply_filters_on_merged/{apply_filters_name}.yaml
# filter_configs:
#   - filter_path: 'filters/momentum'
#     filter_names: ['momentum_1d', 'momentum_5d']
#     apply_filter_func: 'mul_filter'
#     save_name: 'momentum_filtering'
#   
#   - filter_path: 'filters/volatility'
#     filter_names: 
#       - {'pos': 'bull_market', 'neg': 'bear_market'}
#     suffix_list: ['1d', '5d', '20d']
#     apply_filter_func: 'conditional_mul_filter'
#     save_name: 'volatility_filtering'
# =============================================================================
