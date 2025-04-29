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
from utils.timeutils import RollingPeriods, get_lb_list_fit_periods, get_rolling_dates
from synthesis.select_ts_trans import FactorSelector

                
# %%
def run_rolling_factor_select(select_name, rolling_select_name, pstart='20230701', puntil=None, 
                              mode='rolling'):
    """
    Run rolling evaluation using factor evaluation.
    
    Parameters:
    -----------
    eval_name : str
        Evaluation name
    eval_rolling_name : str
        Name of the rolling evaluation configuration
    pstart : str, optional
        Period start date in 'YYYYMMDD' format, defaults to '20230701'
    puntil : str, optional
        Period end date in 'YYYYMMDD' format, defaults to current date
    eval_type : str, optional
        Evaluation type, either 'rolling' or 'update', defaults to 'rolling'
    n_workers : int, optional
        Number of workers, defaults to 1
    check_consistency : bool, optional
        Whether to check consistency, defaults to True
        
    Returns:
    --------
    None
    """
    # Initialize variables
    if puntil is None:
        puntil = datetime.utcnow().date().strftime('%Y%m%d')
    
    # Load path configuration
    path_config = load_path_config(project_dir)
    
    # Initialize directories
    param_dir = Path(path_config['param']) / 'rolling_select_ts_trans'
    
    # Load parameters
    params = toml.load(param_dir / f'{rolling_select_name}.toml')

    # Get rolling dates
    fstart = params['fstart']
    rolling_dates = get_rolling_dates(fstart, pstart, puntil)
    
    # Get lookback fit periods
    rolling_params = params['rolling_params']
    lb_list = params['lb_list']
    lb_fit_periods_list = get_lb_list_fit_periods(rolling_dates, rolling_params, lb_list)
    # Run factor evaluation
    fs = FactorSelector(select_name)
    fit_periods = list(zip(*lb_fit_periods_list))
    if mode == 'rolling':
        for fp in tqdm(fit_periods, desc='rolling select ts trans'):
            fs.run_one_period(fp)
    elif mode == 'update':
        fp = fit_periods[-1]
        fs.run_one_period(fp)
            
    