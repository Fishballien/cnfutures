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
from synthesis.rolling_select_trade_method import run_rolling_trade_select

# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser(description='Run rolling trade selection evaluation')
    parser.add_argument('-s', '--select_name', type=str, required=True, help='Trade selection method name')
    parser.add_argument('-rs', '--rolling_select_name', type=str, required=True, help='Name of rolling configuration file')
    parser.add_argument('-pst', '--pstart', type=str, default='20180101', help='Start date in YYYYMMDD format')
    parser.add_argument('-pu', '--puntil', type=str, help='End date in YYYYMMDD format (default: current date)')
    parser.add_argument('-m', '--mode', type=str, default='rolling', choices=['rolling', 'update'], 
                        help='Evaluation mode: rolling (all periods) or update (latest period only)')
    args = parser.parse_args()
    
    run_rolling_trade_select(
        select_name=args.select_name,
        rolling_select_name=args.rolling_select_name,
        pstart=args.pstart,
        puntil=args.puntil,
        mode=args.mode
    )

# %% main
if __name__=='__main__':
    main()
        
        
    

