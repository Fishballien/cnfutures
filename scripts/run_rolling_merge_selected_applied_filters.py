# -*- coding: utf-8 -*-
"""
Created on Wed May 29 2025

@author: Xintang Zheng

滚动合并选定的应用过滤器因子脚本
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
from apply_filter.rolling_merge_selected_applied_filters import run_rolling_merge_selected_applied_filters

# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser(description='Rolling merge for selected applied filters')
    parser.add_argument('-fm', '--fac_merge_name', type=str, required=True,
                        help='Factor merge name (determines where to read filtered factors)')
    parser.add_argument('-te', '--test_eval_filtered_alpha_name', type=str, required=True,
                        help='Test eval filtered alpha name')
    parser.add_argument('-s', '--select_name', type=str, required=True,
                        help='Select name')
    parser.add_argument('-ffm', '--filter_merge_name', type=str, required=True,
                        help='Filter merge name (determines merge method)')
    parser.add_argument('-rm', '--rolling_merge_name', type=str, required=True,
                        help='Rolling merge config name')
    parser.add_argument('-pst', '--pstart', type=str, default='20230701',
                        help='Start date (YYYYMMDD format)')
    parser.add_argument('-pu', '--puntil', type=str, default=None,
                        help='End date (YYYYMMDD format, default: current date)')
    parser.add_argument('-mo', '--mode', type=str, default='rolling',
                        choices=['rolling', 'update'],
                        help='Processing mode: rolling or update')
    parser.add_argument('-mw', '--max_workers', type=int, default=None,
                        help='Max workers for factor merging within period')
    parser.add_argument('-nw', '--n_workers', type=int, default=1,
                        help='Number of workers for parallel period processing')
    
    args = parser.parse_args()
    
    run_rolling_merge_selected_applied_filters(
        fac_merge_name=args.fac_merge_name,
        test_eval_filtered_alpha_name=args.test_eval_filtered_alpha_name,
        select_name=args.select_name,
        filter_merge_name=args.filter_merge_name,
        rolling_merge_name=args.rolling_merge_name,
        pstart=args.pstart,
        puntil=args.puntil,
        mode=args.mode,
        max_workers=args.max_workers,
        n_workers=args.n_workers
    )
    

# %% main
if __name__=='__main__':
    main()