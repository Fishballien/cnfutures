# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:19 2024

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
from synthesis.rolling_merge_selected_basic_features import run_rolling_merge_selected_basic_features

# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--merge_name', type=str, help='merge_name')
    parser.add_argument('-rm', '--rolling_merge_name', type=str, help='rolling_merge_name')
    parser.add_argument('-pst', '--pstart', type=str, default='20230701', help='pstart')
    parser.add_argument('-pu', '--puntil', type=str, help='puntil')
    parser.add_argument('-mo', '--mode', type=str, default='rolling', help='mode')
    parser.add_argument('-wkr', '--max_workers', type=int, default=1, help='mode')
    args = parser.parse_args()
    
    run_rolling_merge_selected_basic_features(
        merge_name=args.merge_name,
        rolling_merge_name=args.rolling_merge_name,
        pstart=args.pstart,
        puntil=args.puntil,
        mode=args.mode,
        max_workers=args.max_workers,
    )
    

# %% main
if __name__=='__main__':
    main()

