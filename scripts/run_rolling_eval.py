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
from test_and_eval.rolling_eval import RollingEval

# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval_name', type=str, help='eval_name')
    parser.add_argument('-er', '--eval_rolling_name', type=str, help='eval_rolling_name')
    parser.add_argument('-pst', '--pstart', type=str, default='20230701', help='pstart')
    parser.add_argument('-pu', '--puntil', type=str, help='puntil')
    parser.add_argument('-et', '--eval_type', type=str, default='rolling', help='eval_type')
    parser.add_argument('-nw', '--n_workers', type=int, default=1, help='n_workers')
    parser.add_argument('-cc', '--check_consistency', action='store_true', help='check_consistency')
    args = parser.parse_args()
    args_dict = vars(args)
    
    re = RollingEval(**args_dict)
    re.run()

# %% main
if __name__=='__main__':
    main()
        
        
        
    

