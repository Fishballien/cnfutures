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
from synthesis.rolling_select_factors import run_rolling_factor_select

# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--select_name', type=str, help='select_name')
    parser.add_argument('-rs', '--rolling_select_name', type=str, help='rolling_select_name')
    parser.add_argument('-e', '--eval_name', type=str, help='evaluation name')
    parser.add_argument('-pst', '--pstart', type=str, default='20230701', help='pstart')
    parser.add_argument('-pu', '--puntil', type=str, help='puntil')
    parser.add_argument('-m', '--mode', type=str, default='rolling', help='mode')
    args = parser.parse_args()
    
    run_rolling_factor_select(
        select_name=args.select_name,
        rolling_select_name=args.rolling_select_name,
        pstart=args.pstart,
        puntil=args.puntil,
        mode=args.mode,
        eval_name=args.eval_name
    )

# %% main
if __name__=='__main__':
    main()