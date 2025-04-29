# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:19 2024

@author: Xintang Zheng

"""
# %% imports
import sys
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from test_and_eval.factor_evaluation import FactorEvaluation

          
# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval_name', type=str, help='eval_name')
    parser.add_argument('-st', '--start', type=str, default='20150101', help='start')
    parser.add_argument('-ed', '--end', type=str, help='end')
    parser.add_argument('-wkr', '--n_workers', type=int, help='n_workers')
    args = parser.parse_args()
    
    eval_name = args.eval_name
    n_workers = args.n_workers
    date_start = datetime.strptime(args.start, '%Y%m%d')
    date_end = datetime.strptime(args.end, '%Y%m%d')
    fe = FactorEvaluation(eval_name, n_workers)
    fe.eval_one_period(date_start, date_end)
        
        
# %% main
if __name__=='__main__':
    main()
        
        
        
    

