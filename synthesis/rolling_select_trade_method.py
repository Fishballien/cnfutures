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
from utils.timeutils import get_rolling_periods, get_rolling_dates
from synthesis.select_trade_method import TradeSelector

                
# %%
def run_rolling_trade_select(select_name, rolling_select_name, pstart='20180101', puntil=None, 
                             mode='rolling'):
    """
    Run rolling evaluation using trade selection methods.
    
    This function performs rolling or update evaluation of trading strategies based on
    configuration parameters. It processes data over multiple time periods to evaluate 
    trading performance and generate predictions.
    
    Parameters:
    -----------
    select_name : str
        Name of the trade selection method to be evaluated
    rolling_select_name : str
        Name of the rolling configuration file (without .toml extension)
    pstart : str, optional
        Period start date in 'YYYYMMDD' format, defaults to '20180101'
    puntil : str, optional
        Period end date in 'YYYYMMDD' format, defaults to current date if None
    mode : str, optional
        Evaluation mode, either 'rolling' (evaluate all periods) or 'update' (only latest period),
        defaults to 'rolling'
        
    Returns:
    --------
    None
        Results are saved to the configured output locations
    
    Notes:
    ------
    - The function reads parameters from a TOML file in the param/rolling_single_period directory
    - For 'rolling' mode, it processes all periods sequentially
    - For 'update' mode, it only processes the most recent period
    """
    # Initialize variables and validate inputs
    print(f"Starting {mode} evaluation for '{select_name}' using '{rolling_select_name}' configuration")
    
    if puntil is None:
        puntil = datetime.utcnow().date().strftime('%Y%m%d')
        print(f"End date not specified, using current date: {puntil}")
    
    print(f"Evaluation period: {pstart} to {puntil}")
    
    # Load path configuration
    print("Loading path configuration...")
    path_config = load_path_config(project_dir)
    
    # Initialize directories
    param_dir = Path(path_config['param']) / 'rolling_single_period'
    print(f"Using parameter directory: {param_dir}")
    
    # Load parameters from TOML file
    param_file = param_dir / f'{rolling_select_name}.toml'
    print(f"Loading parameters from: {param_file}")
    params = toml.load(param_file)
    
    # Get rolling dates
    fstart = params['fstart']
    print(f"Feature start date: {fstart}")
    rolling_dates = get_rolling_dates(fstart, pstart, puntil)
    print(f"Generated {len(rolling_dates)} rolling dates")
    
    # Get lookback fit periods
    rolling_params = params['rolling_params']
    lb = params['lb']
    print(f"Using lookback period: {lb}")
    
    # Calculate rolling periods for fitting and prediction
    print("Calculating rolling periods...")
    rolling_periods = get_rolling_periods(rolling_dates, rolling_params, lb)
    fit_periods = rolling_periods.fit_periods
    predict_periods = rolling_periods.predict_periods
    print(f"Generated {len(fit_periods)} fit periods and {len(predict_periods)} prediction periods")
    
    # Initialize the trade selector
    print(f"Initializing TradeSelector with method: {select_name}")
    ts = TradeSelector(select_name)
    
    # Run factor evaluation based on mode
    if mode == 'rolling':
        print(f"Running in 'rolling' mode - evaluating all {len(fit_periods)} periods")
        for i, (fp, pp) in enumerate(tqdm(list(zip(fit_periods, predict_periods)), 
                                         desc='Rolling trade selection progress')):
            print(f"\nProcessing period {i+1}/{len(fit_periods)}")
            print(f"Fit period: {fp[0]} to {fp[1]}")
            print(f"Prediction period: {pp[0]} to {pp[1]}")
            ts.run_one_period(*fp, *pp)
            
    elif mode == 'update':
        print("Running in 'update' mode - evaluating only the latest period")
        fp = fit_periods[-1]
        pp = predict_periods[-1]
        print(f"Fit period: {fp[0]} to {fp[1]}")
        print(f"Prediction period: {pp[0]} to {pp[1]}")
        ts.run_one_period(*fp, *pp)
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be either 'rolling' or 'update'")
        
    ts.test_predicted()
    
    print(f"\nCompleted {mode} trade selection evaluation for {select_name}")