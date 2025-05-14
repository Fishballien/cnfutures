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
import concurrent.futures
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.timeutils import get_lb_fit_periods, get_rolling_dates
from synthesis.merge_selected_factors import FactorMerger


# %% Helper function for concurrent processing
def process_period(fm, period_info, period_idx, total_periods):
    """
    Helper function to process a single period for parallel execution.
    
    Parameters
    ----------
    fm : FactorMerger
        Instance of FactorMerger class to use for processing
    period_info : tuple
        Tuple containing the period information
    period_idx : int
        Index of the current period (1-based)
    total_periods : int
        Total number of periods to process
        
    Returns
    -------
    dict
        Dictionary containing the processing result information
    """
    print(f"Processing period {period_idx}/{total_periods}: {period_info}")
    try:
        fm.run_one_period(*period_info)
        status = "success"
        message = f"Completed period {period_idx}/{total_periods}"
    except Exception as e:
        status = "error"
        message = f"Error processing period {period_idx}/{total_periods}: {str(e)}"
    
    print(message)
    return {
        "period_idx": period_idx,
        "period_info": period_info,
        "status": status,
        "message": message
    }

                
# %%
def run_rolling_merge_selected_factors(merge_name, select_name, rolling_merge_name, pstart='20230701', puntil=None, 
                                     mode='rolling', max_workers=None, n_workers=1):
    """
    Executes a rolling or update merge operation for selected factors.
    
    This function performs factor transformations by either processing all periods in a rolling
    window approach or updating only the most recent period. It loads configuration parameters from
    a TOML file and uses the FactorMerger class to execute the operations.
    
    Parameters
    ----------
    merge_name : str
        Name of the merge configuration to use
    select_name : str
        Name of the selection configuration to use
    rolling_merge_name : str
        Name of the rolling merge configuration file (without .toml extension)
    pstart : str, optional
        Start date for the processing period in 'YYYYMMDD' format (default: '20230701')
    puntil : str, optional
        End date for the processing period in 'YYYYMMDD' format (default: current date)
    mode : str, optional
        Processing mode, either 'rolling' (process all periods) or 'update' (process only latest period)
        (default: 'rolling')
    max_workers : int, optional
        Maximum number of worker processes to use for factor merging operations within a period (default: None)
    n_workers : int, optional
        Number of worker processes to use for parallel period processing (default: 1).
        If set to 1, periods are processed sequentially as in the original implementation.
        
    Returns
    -------
    None
        Results are saved to the configured output location
    
    Notes
    -----
    The function relies on configuration files in the 'param/rolling_merge_selected_factors' directory
    and uses the FactorMerger class to perform the actual merging operations.
    """
    # Initialize variables
    if puntil is None:
        puntil = datetime.utcnow().date().strftime('%Y%m%d')
    
    print(f"Starting {mode} merge with parameters:")
    print(f"- Merge name: {merge_name}")
    print(f"- Select name: {select_name}")
    print(f"- Rolling merge config: {rolling_merge_name}")
    print(f"- Period: {pstart} to {puntil}")
    print(f"- Parallel workers: {n_workers}")
    
    # Load path configuration
    path_config = load_path_config(project_dir)
    print(f"Loaded path configuration from {project_dir}")
    
    # Initialize directories
    param_dir = Path(path_config['param'])
    print(f"Using parameter directory: {param_dir}")
    
    # Load parameters
    params_file = param_dir / 'rolling_single_period' / f'{rolling_merge_name}.toml'
    print(f"Loading parameters from {params_file}")
    params = toml.load(params_file)
    
    # Get rolling dates
    fstart = params['fstart']
    rolling_dates = get_rolling_dates(fstart, pstart, puntil)
    
    # Get lookback fit periods
    rolling_params = params['rolling_params']
    lb = params['lb']
    lb_fit_periods = get_lb_fit_periods(rolling_dates, rolling_params, lb)
    print(f"Created {len(lb_fit_periods)} lookback fit periods with lookback windows")
    
    # Initialize the factor merger
    fm = FactorMerger(merge_name, select_name, max_workers=max_workers)
    print(f"Initialized FactorMerger with configuration: {merge_name}, select_name: {select_name}")
    
    if mode == 'rolling':
        print(f"Running in ROLLING mode - processing all {len(lb_fit_periods)} periods")
        
        # Sequential processing (original behavior)
        if n_workers == 1:
            print("Using sequential processing (n_workers=1)")
            for i, fp in enumerate(tqdm(lb_fit_periods, desc='Rolling merge progress')):
                process_period(fm, fp, i+1, len(lb_fit_periods))
        
        # Parallel processing
        else:
            print(f"Using parallel processing with {n_workers} workers")
            n_workers = min(n_workers, len(lb_fit_periods))  # Don't use more workers than periods
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Prepare the tasks
                future_to_period = {
                    executor.submit(
                        process_period, 
                        FactorMerger(merge_name, select_name, max_workers=max_workers),  # Create new instance for each worker
                        fp, i+1, len(lb_fit_periods)
                    ): i for i, fp in enumerate(lb_fit_periods)
                }
                
                # Process results as they complete
                results = []
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_period), 
                    total=len(future_to_period),
                    desc='Rolling merge progress'
                ):
                    period_idx = future_to_period[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Period {period_idx+1} processing failed: {str(e)}")
                
                # Print summary of results
                success_count = sum(1 for r in results if r['status'] == 'success')
                print(f"Completed {success_count}/{len(lb_fit_periods)} periods successfully")
    
    elif mode == 'update':
        print("Running in UPDATE mode - processing only the latest period")
        fp = lb_fit_periods[-1]
        print(f"Processing latest period: {fp}")
        fm.run_one_period(*fp)
        print("Completed processing of latest period")
    
    print(f"Finished {mode} merge operation for {rolling_merge_name}")