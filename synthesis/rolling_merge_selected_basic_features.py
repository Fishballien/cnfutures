# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:37:19 2024

@author: Xintang Zheng

"""
# %% imports
import sys
from pathlib import Path
from datetime import datetime, timedelta
import toml
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.timeutils import get_lb_fit_periods, get_rolling_dates
from synthesis.merge_selected_basic_features import FactorMerger

                
# %%
def run_rolling_merge_selected_basic_features(merge_name, rolling_merge_name, pstart='20230701', puntil=None, 
                                              mode='rolling', max_workers=None):
    """
    Executes a rolling or update merge operation for selected time series transformations.
    
    This function performs time series transformations by either processing all periods in a rolling
    window approach or updating only the most recent period. It loads configuration parameters from
    a TOML file and uses the FactorMerger class to execute the operations.
    
    Parameters
    ----------
    merge_name : str
        Name of the merge configuration to use
    rolling_merge_name : str
        Name of the rolling merge configuration file (without .toml extension)
    pstart : str, optional
        Start date for the processing period in 'YYYYMMDD' format (default: '20230701')
    puntil : str, optional
        End date for the processing period in 'YYYYMMDD' format (default: current date)
    mode : str, optional
        Processing mode, either 'rolling' (process all periods) or 'update' (process only latest period)
        (default: 'rolling')
        
    Returns
    -------
    None
        Results are saved to the configured output location
    
    Notes
    -----
    The function relies on configuration files in the 'param/rolling_merge_selected_ts_trans' directory
    and uses the FactorMerger class to perform the actual merging operations.
    """
    # Initialize variables
    if puntil is None:
        puntil = datetime.utcnow().date().strftime('%Y%m%d')
    
    print(f"Starting {mode} merge with parameters:")
    print(f"- Merge name: {merge_name}")
    print(f"- Rolling merge config: {rolling_merge_name}")
    print(f"- Period: {pstart} to {puntil}")
    
    # Load path configuration
    path_config = load_path_config(project_dir)
    print(f"Loaded path configuration from {project_dir}")
    
    # Initialize directories
    param_dir = Path(path_config['param']) / 'rolling_single_period'
    print(f"Using parameter directory: {param_dir}")
    
    # Load parameters
    params_file = param_dir / f'{rolling_merge_name}.toml'
    print(f"Loading parameters from {params_file}")
    params = toml.load(params_file)
    
    # Get rolling dates
    fstart = params['fstart']
    rolling_dates = get_rolling_dates(fstart, pstart, puntil)
    
    # Get lookback fit periods
    rolling_params = params['rolling_params']
    lb = params['lb']
    lb_fit_periods = get_lb_fit_periods(rolling_dates, rolling_params, lb)
    print("Created lookback fit periods with lookback windows")
    
    # Run factor evaluation
    fm = FactorMerger(merge_name, max_workers=max_workers)
    print(f"Initialized FactorMerger with configuration: {merge_name}")
    
    if mode == 'rolling':
        print(f"Running in ROLLING mode - processing all {len(lb_fit_periods)} periods")
        for i, fp in enumerate(tqdm(lb_fit_periods, desc='Rolling merge progress')):
            print(f"Processing period {i+1}/{len(lb_fit_periods)}: {fp}")
            fm.run_one_period(*fp)
            print(f"Completed period {i+1}/{len(lb_fit_periods)}")
    elif mode == 'update':
        print("Running in UPDATE mode - processing only the latest period")
        fp = lb_fit_periods[-1]
        print(f"Processing latest period: {fp}")
        fm.run_one_period(*fp)
        print("Completed processing of latest period")
    
    print(f"Finished {mode} merge operation for {rolling_merge_name}")
            
    