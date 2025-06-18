# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 2025

@author: Analysis Script

Run factor selection trend analysis
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
from synthesis.summarize_selected_factors import run_trend_analysis

# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser(description='Run factor selection trend analysis')
    parser.add_argument('-e', '--eval_name', type=str, required=True, 
                       help='Evaluation name')
    parser.add_argument('-s', '--select_name', type=str, required=True, 
                       help='Selection configuration name')
    parser.add_argument('--no_save_plots', action='store_true', 
                       help='Do not save plots (default: save plots)')
    args = parser.parse_args()
    
    # Convert no_save_plots flag to save_plots
    save_plots = not args.no_save_plots
    
    print("🚀 Starting factor selection trend analysis...")
    print(f"📊 Evaluation name: {args.eval_name}")
    print(f"⚙️ Selection name: {args.select_name}")
    print(f"📈 Save plots: {save_plots}")
    
    # Run the analysis
    results = run_trend_analysis(
        eval_name=args.eval_name,
        select_name=args.select_name,
        save_plots=save_plots
    )
    
    if results and results[0] is not None:
        print("\n✅ Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed. Please check the input parameters and data files.")
        sys.exit(1)

# %% main
if __name__ == '__main__':
    main()