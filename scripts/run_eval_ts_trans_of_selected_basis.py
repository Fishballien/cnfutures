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
from test_and_eval.eval_ts_trans_of_selected_basic import run_dynamic_evaluation

# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser(description='Run dynamic evaluation with specified parameters')
    parser.add_argument('-d', '--dynamic_eval_name', required=True, type=str, help='Dynamic evaluation name')
    parser.add_argument('-f', '--fstart', type=str, default='20150101', help='Factor start date (YYYYMMDD)')
    parser.add_argument('-p', '--pstart', type=str, default='20180101', help='Prediction start date (YYYYMMDD)')
    parser.add_argument('-u', '--puntil', type=str, default='20200101', help='Prediction until date (YYYYMMDD)')
    parser.add_argument('-w', '--n_workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('-e', '--eval_type', type=str, default='rolling', choices=['rolling', 'update'], 
                        help='Evaluation type (rolling or update)')
    parser.add_argument('-c', '--check_consistency', action='store_true', help='Check consistency of data')
    args = parser.parse_args()
    
    run_dynamic_evaluation(
        dynamic_eval_name=args.dynamic_eval_name,
        fstart=args.fstart,
        pstart=args.pstart,
        puntil=args.puntil,
        n_workers=args.n_workers,
        eval_type=args.eval_type,
        check_consistency=args.check_consistency
    )

# %% main
if __name__=='__main__':
    main()
    

