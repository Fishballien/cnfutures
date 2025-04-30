# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 15:30:23 2025

@author: AI Assistant

Function to visualize cumulative returns of different factor categories.
"""
# %% import public
import sys
from pathlib import Path
import toml
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# %% add sys path (similar to batch_test.py)
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[2] 
sys.path.append(str(project_dir))

# %% import self-defined
from utils.dirutils import load_path_config
from utils.timeutils import period_shortcut


def load_gp_data(data_dir, factor_name, return_type='gp', direction_type='all', fee=4e-4):
    """
    Load and process GP (group performance) or net data based on specified parameters.
    
    Parameters:
    -----------
    data_dir : Path
        Directory path containing the factor data files
    factor_name : str
        Name of the factor to load
    return_type : str
        Type of return to calculate: 'gp' for group performance or 'net' for net returns
    direction_type : str
        Direction type: 'all', 'long_only', or 'short_only'
    fee : float
        Fee to apply when calculating net returns
        
    Returns:
    -----------
    pd.Series
        Series containing the return data
    """
    try:
        # Load GPD data
        gpd_path = data_dir / f'gpd_{factor_name}.pkl'
        with open(gpd_path, 'rb') as f:
            gpd_data = pickle.load(f)
        
        # Determine direction based on all data
        df_gp_all = gpd_data['all']
        cumrtn = df_gp_all['return'].sum()
        direction = 1 if cumrtn > 0 else -1
        
        # Select appropriate dataframe based on direction_type
        if direction_type == 'all':
            df_gp = df_gp_all
            direction_sign = direction
        elif direction_type == 'long_only':
            value_sign = 'pos' if direction == 1 else 'neg'
            df_gp = gpd_data[value_sign]
            direction_sign = direction
        elif direction_type == 'short_only':
            value_sign = 'neg' if direction == 1 else 'pos'
            df_gp = gpd_data[value_sign]
            direction_sign = direction
        else:
            raise ValueError(f"Invalid direction_type: {direction_type}")
        
        # Calculate returns
        if return_type == 'gp':
            return df_gp['return'] * direction_sign
        elif return_type == 'net':
            # Load HSR data for net return calculation
            hsr_path = data_dir / f'hsr_{factor_name}.pkl'
            with open(hsr_path, 'rb') as f:
                hsr_data = pickle.load(f)
            
            # Get appropriate HSR data based on direction_type
            if direction_type == 'all':
                df_hsr = hsr_data['all']
            elif direction_type == 'long_only':
                value_sign = 'pos' if direction == 1 else 'neg'
                df_hsr = hsr_data[value_sign]
            elif direction_type == 'short_only':
                value_sign = 'neg' if direction == 1 else 'pos'
                df_hsr = hsr_data[value_sign]
            
            # Calculate net returns
            net = (df_gp['return'] * direction_sign - fee * df_hsr['avg']).fillna(0)
            return net
        else:
            raise ValueError(f"Invalid return_type: {return_type}")
            
    except Exception as e:
        print(f"Error loading data for {factor_name}: {str(e)}")
        return pd.Series()


def visualize_factor_returns(ind_cate_name, final_path_name, factor_data_dir, batch_test_name, tag_name,
                            date_start=None, date_end=None, return_type='gp', direction_type='all',
                            fee=4e-4, n_workers=1, max_factors_per_group=50, figsize=(20, 15),
                            alpha=0.2, linewidth=1):
    """
    Visualize cumulative returns for different factor groups based on process_info_list.
    Each test_name gets its own figure with subplots for each final_path.
    
    Parameters:
    -----------
    ind_cate_name : str
        Industry category name
    final_path_name : str
        Final path name for loading factor paths
    factor_data_dir : str
        Directory containing factor data
    batch_test_name : str
        Name of the batch test configuration
    tag_name : str
        Tag name used to locate the factor test paths
    date_start : str, optional
        Start date in format 'YYYYMMDD'
    date_end : str, optional
        End date in format 'YYYYMMDD'
    return_type : str, optional
        Type of return to plot: 'gp' or 'net'
    direction_type : str, optional
        Direction type: 'all', 'long_only', or 'short_only'
    fee : float, optional
        Fee to apply when calculating net returns
    n_workers : int, optional
        Number of worker processes for parallel loading
    max_factors_per_group : int, optional
        Maximum number of factors to plot per group (to avoid overcrowding)
    figsize : tuple, optional
        Figure size as (width, height)
    alpha : float, optional
        Transparency of individual factor lines
    linewidth : float, optional
        Width of factor lines
    """
    # Load path configuration
    path_config = load_path_config(project_dir)
    final_path_dir = Path(path_config['final_ts_path'])
    param_dir = Path(path_config['param'])
    batch_test_param_dir = param_dir / 'batch_test'
    test_dir = Path(path_config['result']) / 'test'
    
    # Load batch test configuration
    config = toml.load(batch_test_param_dir / f'{batch_test_name}.toml')
    test_name_list = [(single_test_param['mode'], single_test_param['test_name'])
                      for single_test_param in config['test_list']]
    
    # Load final path list
    final_path_file = final_path_dir / f'{final_path_name}.json'
    with open(final_path_file, 'r') as f:
        data = json.load(f)
        final_path_list = data['final_path']
    
    # Generate process names
    process_name_list = [f'{ind_cate_name}/{final_path}' for final_path in final_path_list]
    
    # Map process names to their corresponding final_path for labeling
    process_to_final_path = {f'{ind_cate_name}/{final_path}': final_path for final_path in final_path_list}
    
    # Generate process_info_list similar to batch_test_by_path_and_eval_one_period
    process_info_list = [(factor_data_dir, tag_name, process_name, mode, test_name)
                        for process_name in process_name_list
                        for mode, test_name in test_name_list]
    
    # Group process_info by test_name
    test_name_to_process_info = {}
    for info in process_info_list:
        _, _, _, mode, test_name = info
        if test_name not in test_name_to_process_info:
            test_name_to_process_info[test_name] = []
        test_name_to_process_info[test_name].append(info)
    
    # Process each test_name group
    for test_name, process_infos in test_name_to_process_info.items():
        print(f"Processing test: {test_name}")
        
        # Extract unique process names (which represent different final paths)
        unique_process_names = list(set(info[2] for info in process_infos))
        n_processes = len(unique_process_names)
        
        # Calculate grid layout
        n_cols = min(3, n_processes)  # Max 3 columns
        n_rows = (n_processes + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        fig.suptitle(f"{ind_cate_name} - Cumulative {return_type.upper()} Returns ({direction_type}) - {test_name}", 
                     fontsize=16)
        
        # Process each unique process name (final path)
        for i, process_name in enumerate(unique_process_names):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            # Get the final_path for this process_name for labeling
            final_path = process_to_final_path.get(process_name, process_name)
            
            # Set subplot title with the final_path
            ax.set_title(f"Final Path: {final_path}", fontsize=12)
            
            # Find the process_info for this process_name and test_name
            info = next((info for info in process_infos if info[2] == process_name), None)
            if info is None:
                continue
                
            factor_data_dir, tag_name, process_name, mode, _ = info
            
            # Locate test results directory
            process_dir = test_dir / test_name / tag_name / process_name if tag_name is not None else test_dir / test_name / process_name
            data_dir = process_dir / 'data'
            
            # Get factor list
            factor_path = Path(factor_data_dir) / process_name
            try:
                factor_name_list = [path.stem for path in factor_path.glob('*.parquet')]
                
                # Limit number of factors to avoid overcrowding
                if len(factor_name_list) > max_factors_per_group:
                    print(f"Limiting {process_name} from {len(factor_name_list)} to {max_factors_per_group} factors")
                    factor_name_list = factor_name_list[:max_factors_per_group]
                    
                # Load and plot data for each factor
                load_func = partial(load_gp_data, data_dir, 
                                   return_type=return_type, 
                                   direction_type=direction_type,
                                   fee=fee)
                
                # Use parallel processing if specified
                if n_workers > 1:
                    with ProcessPoolExecutor(max_workers=n_workers) as executor:
                        futures = [executor.submit(load_func, factor_name) for factor_name in factor_name_list]
                        factor_returns = []
                        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Loading {process_name}"):
                            factor_returns.append(future.result())
                else:
                    factor_returns = []
                    for factor_name in tqdm(factor_name_list, desc=f"Loading {process_name}"):
                        factor_returns.append(load_func(factor_name))
                
                # Filter out empty series
                factor_returns = [r for r in factor_returns if not r.empty]
                
                if not factor_returns:
                    print(f"No valid return data found for {process_name}")
                    continue
                
                # Filter by date if specified
                if date_start or date_end:
                    for i, returns in enumerate(factor_returns):
                        mask = pd.Series(True, index=returns.index)
                        if date_start:
                            mask &= returns.index >= pd.Timestamp(date_start)
                        if date_end:
                            mask &= returns.index <= pd.Timestamp(date_end)
                        factor_returns[i] = returns[mask]
                
                # Plot cumulative returns for each factor
                for returns in factor_returns:
                    cum_returns = (1 + returns).cumprod() - 1
                    ax.plot(cum_returns.index, cum_returns.values, alpha=alpha, linewidth=linewidth)
                
                # Plot mean cumulative return
                all_returns = pd.concat(factor_returns, axis=1)
                mean_returns = all_returns.mean(axis=1)
                cum_mean_returns = (1 + mean_returns).cumprod() - 1
                ax.plot(cum_mean_returns.index, cum_mean_returns.values, color='red', 
                       linewidth=2, label=f'Mean ({len(factor_returns)} factors)')
                
                # Add a label with the number of factors and final cumulative return
                final_mean_return = cum_mean_returns.iloc[-1] if not cum_mean_returns.empty else 0
                ax.text(0.05, 0.95, 
                        f"Factors: {len(factor_returns)}\nFinal Return: {final_mean_return:.2%}", 
                        transform=ax.transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as e:
                print(f"Error processing {process_name}: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Format axis
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper left')
            ax.set_ylabel('Cumulative Return')
            
            # Format x-axis dates
            ax.tick_params(axis='x', rotation=45)
        
        # Hide empty subplots
        for i in range(len(unique_process_names), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        # Save figure
        period_name = period_shortcut(
            pd.Timestamp(date_start) if date_start else None,
            pd.Timestamp(date_end) if date_end else None
        )
        
        save_dir = Path(path_config['result']) / 'batch_path_factor_visualization'
        save_dir.mkdir(exist_ok=True, parents=True)
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)  # Make room for suptitle
        
        save_path = save_dir / f"factor_returns_{ind_cate_name}_{tag_name}_{final_path_name}_{test_name}_{return_type}_{direction_type}_{period_name}.png"
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
        plt.close(fig)


# Example usage
if __name__ == "__main__":
    visualize_factor_returns(
        ind_cate_name="basis_pct_250416",
        final_path_name="org_batch_250419",
        factor_data_dir="/mnt/Data/xintang/index_factors",
        batch_test_name="batch_test_v1",
        tag_name="zxt",  # Added required parameter
        date_start="20160101",
        date_end="20250401",
        return_type="gp",  # 'gp' or 'net'
        direction_type="long_only",  # 'all', 'long_only', or 'short_only'
        fee=2.4e-4,
        n_workers=50,
        max_factors_per_group=30
    )
