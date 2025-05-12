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
from synthesis.select_factors import FactorSelector

                
# %%
def run_rolling_factor_select(select_name, rolling_select_name, pstart='20230701', puntil=None, 
                              mode='rolling', eval_name=None):
    """
    Run rolling evaluation using factor evaluation.
    
    Parameters:
    -----------
    select_name : str
        Selection configuration name
    rolling_select_name : str
        Name of the rolling selection configuration
    pstart : str, optional
        Period start date in 'YYYYMMDD' format, defaults to '20230701'
    puntil : str, optional
        Period end date in 'YYYYMMDD' format, defaults to current date
    mode : str, optional
        Selection mode, either 'rolling' or 'update', defaults to 'rolling'
    eval_name : str, optional
        Evaluation name, if None, it will be read from the selection configuration
        
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
    
    # Run factor selection
    fs = FactorSelector(select_name, eval_name)
    fit_periods = list(zip(*lb_fit_periods_list))
    if mode == 'rolling':
        for fp in tqdm(fit_periods, desc='rolling select ts trans'):
            fs.run_one_period(fp)
    elif mode == 'update':
        fp = fit_periods[-1]
        fs.run_one_period(fp)