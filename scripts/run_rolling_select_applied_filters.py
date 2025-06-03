# -*- coding: utf-8 -*-
"""
Created on Wed May 29 2025

@author: Xintang Zheng

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
from apply_filter.rolling_select_applied_filters import run_rolling_select_applied_filters

# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser(description='Run rolling selection for applied filters results')
    parser.add_argument('-s', '--select_name', type=str, required=True, 
                        help='select configuration name')
    parser.add_argument('-m', '--merge_name', type=str, required=True,
                        help='merge configuration name')
    parser.add_argument('-t', '--test_eval_filtered_alpha_name', type=str, required=True,
                        help='test eval filtered alpha configuration name')
    parser.add_argument('-r', '--rolling_name', type=str, required=True,
                        help='rolling configuration name')
    parser.add_argument('-pst', '--pstart', type=str, default='20230701', 
                        help='start date for processing (default: 20230701)')
    parser.add_argument('-pu', '--puntil', type=str, 
                        help='end date for processing (default: current date)')
    parser.add_argument('-md', '--mode', type=str, default='rolling', 
                        choices=['rolling', 'update'],
                        help='processing mode: rolling (all periods) or update (latest only) (default: rolling)')
    parser.add_argument('-nw', '--n_workers', type=int, default=1,
                        help='number of parallel workers for period processing (default: 1)')
    
    args = parser.parse_args()
    
    print("Running rolling selection for applied filters with arguments:")
    print(f"  Select name: {args.select_name}")
    print(f"  Merge name: {args.merge_name}")
    print(f"  Test eval filtered alpha name: {args.test_eval_filtered_alpha_name}")
    print(f"  Rolling name: {args.rolling_name}")
    print(f"  Start date: {args.pstart}")
    print(f"  End date: {args.puntil}")
    print(f"  Mode: {args.mode}")
    print(f"  Number of workers: {args.n_workers}")
    print("-" * 60)
    
    run_rolling_select_applied_filters(
        select_name=args.select_name,
        merge_name=args.merge_name,
        test_eval_filtered_alpha_name=args.test_eval_filtered_alpha_name,
        rolling_name=args.rolling_name,
        pstart=args.pstart,
        puntil=args.puntil,
        mode=args.mode,
        n_workers=args.n_workers
    )

# %% main
if __name__ == '__main__':
    main()