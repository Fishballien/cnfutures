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
from synthesis.merge_model_pos import MergePos

          
# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--merge_name', type=str, help='merge_name')
    args = parser.parse_args()
    
    
    merge_name = args.merge_name
    
    mg = MergePos(merge_name)
    mg.run()
        
        
# %% main
if __name__=='__main__':
    main()
        
        
        
    

