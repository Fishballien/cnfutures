# -*- coding: utf-8 -*-
"""
Trading Method Comparison Script

This script analyzes and compares the performance of different trading method combinations:
- Reads test results for each combination of select_name, merge_name, and select_trade_method_name
- Calculates performance metrics (net returns, Sharpe ratios, etc.)
- Creates comparative visualizations
- Presents key metrics in a table format

Author: Claude
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import itertools
from datetime import datetime
import json

# Add project root to path
file_path = Path(__file__).resolve()
project_dir = file_path.parents[2]
sys.path.append(str(project_dir))

# Import project-specific utilities
from utils.dirutils import load_path_config
from test_and_eval.scores import get_general_return_metrics, calc_sharpe

# Set up matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_test_data(base_dir, test_name, factor_name):
    """
    Load test data (gpd and hsr) for a specific factor from given path.
    """
    test_data = {}
    data_dir = base_dir / 'test' / test_name / 'data'
    
    for data_type in ('gpd', 'hsr'):
        data_path = data_dir / f'{data_type}_{factor_name}.pkl'
        try:
            with open(data_path, 'rb') as f:
                test_data[data_type] = pickle.load(f)
        except Exception as e:
            print(f"Error loading {data_type} for {factor_name} from {data_path}: {e}")
            return None
    
    return test_data


def calculate_net_returns(test_data, date_start=None, date_end=None, fee=4e-4, direction=None):
    """
    Calculate net returns for a given test data.
    
    Parameters:
    -----------
    test_data : dict
        Dictionary containing 'gpd' and 'hsr' data
    date_start, date_end : datetime, optional
        Date range for filtering
    fee : float, default=4e-4
        Trading fee
    direction : int, optional
        Trading direction (1 for long, -1 for short), if None will be determined from data
        
    Returns:
    --------
    dict
        Dictionary containing 'net_returns', 'metrics', and 'direction'
    """
    df_gp = test_data['gpd']['all']
    df_hsr = test_data['hsr']['all']
    
    # Apply date filtering if provided
    if date_start is not None and date_end is not None:
        df_gp = df_gp[(df_gp.index >= date_start) & (df_gp.index <= date_end)]
        df_hsr = df_hsr[(df_hsr.index >= date_start) & (df_hsr.index <= date_end)]
    
    # Determine direction if not provided
    if direction is None:
        cumrtn = df_gp['return'].sum()
        direction = 1 if cumrtn > 0 else -1
    
    # Calculate net returns
    net = (df_gp['return'] * direction - fee * df_hsr['avg']).fillna(0)
    
    # Calculate metrics using the imported function
    metrics = get_general_return_metrics(net)
    
    # Calculate profit per trade
    hsr_sum = df_hsr['avg'].sum()
    profit_per_trade = df_gp['return'].sum() / hsr_sum if hsr_sum > 0 else 0
    metrics['profit_per_trade'] = profit_per_trade * 1000  # Convert to basis points
    
    # Calculate turnover (hsr) metrics
    metrics['hsr'] = df_hsr['avg'].mean()
    
    return {
        'net_returns': net,
        'metrics': metrics,
        'direction': direction
    }


def process_combination(combination, result_dir, fee=4e-4, date_start=None, date_end=None, test_name_list=None):
    """
    Process a single combination of select_name, merge_name, and select_trade_method_name.
    
    Parameters:
    -----------
    combination : tuple
        (eval_name, select_name, merge_name, select_trade_method_name)
    result_dir : Path
        Base directory for results
    fee : float, default=4e-4
        Trading fee
    date_start, date_end : datetime, optional
        Date range for filtering
    test_name_list : list, optional
        List of test names to process. If None, defaults to ["test_default", "trade_default"]
        
    Returns:
    --------
    dict
        Metrics and other information for this combination
    """
    eval_name, select_name, merge_name, trade_select_name = combination
    
    # Default test names if not provided
    if test_name_list is None:
        test_name_list = ["test_default", "trade_default"]
    
    # Construct the full path according to your code structure
    full_select_name = f"{eval_name}_{select_name}"
    trade_merge_name = f"{full_select_name}_{merge_name}"
    
    # Path to select_trade_method results
    base_dir = result_dir / 'select_trade_method' / f'{trade_merge_name}_{trade_select_name}'
    
    # Check if directory exists
    if not base_dir.exists():
        print(f"Warning: Directory not found: {base_dir}")
        return None
    
    # The factor name is standardized in your code as 'pos_{trade_select_name}'
    factor_name = f'pos_{trade_select_name}'
    
    # Load test data
    test_data_results = {}
    
    for test_name in test_name_list:
        test_data = load_test_data(base_dir, test_name, factor_name)
        
        if test_data is None:
            continue
            
        # Calculate net returns
        result = calculate_net_returns(test_data, date_start=date_start, date_end=date_end, fee=fee)
        test_data_results[test_name] = result
    
    if not test_data_results:
        return None
    
    # Compile results into a standardized format
    combo_key = f"{eval_name}_{select_name}_{merge_name}_{trade_select_name}"
    result = {
        'combination': combo_key,
        'eval_name': eval_name,
        'select_name': select_name,
        'merge_name': merge_name,
        'trade_select_name': trade_select_name,
        'test_data': test_data_results
    }
    
    return result


def analyze_combinations(eval_name, select_name_list, merge_name_list, select_trade_method_name_list, 
                         test_name_list=None, project_dir=None, n_workers=4, fee=4e-4, date_start=None, date_end=None):
    """
    Analyze multiple combinations of trading method parameters.
    
    Parameters:
    -----------
    eval_name : str
        Evaluation name
    select_name_list : list
        List of select_name values
    merge_name_list : list
        List of merge_name values
    select_trade_method_name_list : list
        List of select_trade_method_name values
    test_name_list : list, optional
        List of test_name values to analyze (e.g., "test_default", "trade_default")
        If None, will use ["test_default", "trade_default"]
    project_dir : Path, optional
        Project directory, if None will use current directory's parent
    n_workers : int, default=4
        Number of worker processes for parallel processing
    fee : float, default=4e-4
        Trading fee
    date_start, date_end : datetime, optional
        Date range for filtering
        
    Returns:
    --------
    dict
        Results for all combinations
    """
    # Set up project directory
    if project_dir is None:
        project_dir = Path(__file__).resolve().parents[1]
    
    # Set default test names if not provided
    if test_name_list is None:
        test_name_list = ["test_default", "trade_default"]
    
    # Load path configuration
    path_config = load_path_config(project_dir)
    result_dir = Path(path_config['result'])
    
    # Generate all combinations
    combinations = list(itertools.product(
        [eval_name], 
        select_name_list, 
        merge_name_list, 
        select_trade_method_name_list
    ))
    
    print(f"Analyzing {len(combinations)} combinations...")
    
    # Process combinations in parallel
    results = []
    
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Create a list of future objects
            futures = [executor.submit(
                process_combination, 
                combo, 
                result_dir, 
                fee=fee,
                date_start=date_start,
                date_end=date_end,
                test_name_list=test_name_list
            ) for combo in combinations]
            
            # Process as they complete
            for future in tqdm(futures, total=len(futures), desc="Processing combinations"):
                result = future.result()
                if result is not None:
                    results.append(result)
    else:
        # Sequential processing
        for combo in tqdm(combinations, desc="Processing combinations"):
            result = process_combination(
                combo, 
                result_dir, 
                fee=fee,
                date_start=date_start,
                date_end=date_end,
                test_name_list=test_name_list
            )
            if result is not None:
                results.append(result)
    
    print(f"Successfully processed {len(results)} out of {len(combinations)} combinations")
    
    return results


def plot_cumulative_returns(results, output_dir, test_name_list=None):
    """
    Plot cumulative returns for all combinations.
    
    Parameters:
    -----------
    results : list
        List of result dictionaries
    output_dir : Path
        Directory to save plots
    test_name_list : list, optional
        List of test names to plot. If None, will try to detect from results.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If test_name_list is not provided, detect from results
    if test_name_list is None:
        test_name_list = set()
        for result in results:
            test_name_list.update(result['test_data'].keys())
        test_name_list = list(test_name_list)
        
    print(f"Found test names: {test_name_list}")
    
    # Plot each test name in a separate figure
    for test_name in test_name_list:
        plt.figure(figsize=(15, 10))
        
        # Track if we've added any lines to the plot
        lines_added = False
        
        for result in results:
            if test_name not in result['test_data']:
                continue
                
            combo_name = result['combination']
            net_returns = result['test_data'][test_name]['net_returns']
            
            try:
                # Calculate cumulative returns
                cum_returns = (1 + net_returns).cumprod()
                cum_returns.plot(label=combo_name)
                lines_added = True
            except Exception as e:
                print(f"Error plotting {combo_name} for {test_name}: {e}")
                continue
        
        # Only add labels and save if we actually plotted something
        if lines_added:
            plt.title(f'Cumulative Returns - {test_name}')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()
            plt.savefig(output_dir / f'cumulative_returns_{test_name}.png', dpi=300, bbox_inches='tight')
        else:
            print(f"Warning: No data to plot for {test_name}, skipping chart")
        
        plt.close()


def plot_monthly_returns(results, output_dir):
    """
    Plot monthly returns heatmap for all combinations.
    
    Parameters:
    -----------
    results : list
        List of result dictionaries
    output_dir : Path
        Directory to save plots
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For each combination, create a monthly returns heatmap
    for result in results:
        for test_name, test_result in result['test_data'].items():
            combo_name = result['combination']
            net_returns = test_result['net_returns']
            
            # Resample to monthly returns
            monthly_returns = net_returns.resample('M').sum()
            
            # Create a pivot table with years as rows and months as columns
            monthly_returns.index = pd.MultiIndex.from_arrays([
                monthly_returns.index.year, 
                monthly_returns.index.month
            ], names=['Year', 'Month'])
            
            monthly_pivot = monthly_returns.unstack('Month')
            
            # Plot heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(monthly_pivot, annot=True, fmt='.2%', cmap='RdYlGn', center=0)
            plt.title(f'Monthly Returns - {combo_name} - {test_name}')
            plt.tight_layout()
            plt.savefig(output_dir / f'monthly_returns_{combo_name}_{test_name}.png', 
                      dpi=300, bbox_inches='tight')
            plt.close()


def plot_drawdowns(results, output_dir, test_name_list=None):
    """
    Plot drawdowns for all combinations.
    
    Parameters:
    -----------
    results : list
        List of result dictionaries
    output_dir : Path
        Directory to save plots
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure for each test type
    for test_name in test_name_list:
        plt.figure(figsize=(15, 10))
        
        for result in results:
            if test_name not in result['test_data']:
                continue
                
            combo_name = result['combination']
            net_returns = result['test_data'][test_name]['net_returns']
            
            # Calculate drawdowns
            cum_returns = (1 + net_returns).cumprod()
            running_max = cum_returns.cummax()
            drawdowns = (cum_returns / running_max) - 1
            
            drawdowns.plot(label=combo_name)
        
        plt.title(f'Drawdowns - {test_name}')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(output_dir / f'drawdowns_{test_name}.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_metrics_table(results):
    """
    Create a table of key metrics for all combinations.
    
    Parameters:
    -----------
    results : list
        List of result dictionaries
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing key metrics for all combinations
    """
    metrics_rows = []
    
    for result in results:
        for test_name, test_result in result['test_data'].items():
            metrics = test_result['metrics']
            
            row = {
                'combination': result['combination'],
                'eval_name': result['eval_name'],
                'select_name': result['select_name'],
                'merge_name': result['merge_name'],
                'trade_select_name': result['trade_select_name'],
                'test_name': test_name,
                'sharpe_ratio': metrics.get('sharpe_ratio', np.nan),
                'annualized_return': metrics.get('return_annualized', np.nan),
                'max_dd': metrics.get('max_dd', np.nan),
                'win_rate': metrics.get('win_rate', np.nan) if 'win_rate' in metrics else (metrics['return'] > 0).mean(),
                'profit_per_trade': metrics.get('profit_per_trade', np.nan),
                'turnover': metrics.get('hsr', np.nan),
                'sortino_ratio': metrics.get('sortino_ratio', np.nan),
                'calmar_ratio': metrics.get('calmar_ratio', np.nan),
                'sterling_ratio': metrics.get('sterling_ratio', np.nan),
                'burke_ratio': metrics.get('burke_ratio', np.nan),
                'ulcer_index': metrics.get('ulcer_index', np.nan),
                'drawdown_recovery_ratio': metrics.get('drawdown_recovery_ratio', np.nan),
            }
            
            metrics_rows.append(row)
    
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_rows)
    
    return metrics_df


def plot_metrics_comparison(metrics_df, output_dir):
    """
    Create bar charts comparing key metrics across combinations.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame containing metrics
    output_dir : Path
        Directory to save plots
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For each test type, create comparative plots
    for test_name in metrics_df['test_name'].unique():
        test_metrics = metrics_df[metrics_df['test_name'] == test_name]
        
        # Key metrics to plot
        metrics = [
            ('sharpe_ratio', 'Sharpe Ratio'),
            ('annualized_return', 'Annualized Return'),
            ('max_dd', 'Maximum Drawdown'),
            ('win_rate', 'Win Rate'),
            ('profit_per_trade', 'Profit Per Trade (bp)'),
            ('calmar_ratio', 'Calmar Ratio'),
            ('sortino_ratio', 'Sortino Ratio'),
            ('sterling_ratio', 'Sterling Ratio'),
        ]
        
        for metric_col, metric_name in metrics:
            if metric_col not in test_metrics.columns:
                print(f"Warning: Metric '{metric_col}' not found in data, skipping plot")
                continue
                
            plt.figure(figsize=(14, 8))
            
            # For grouped bar charts, group by select_name and merge_name
            pivot_data = test_metrics.pivot_table(
                index=['select_name', 'merge_name'],
                columns='trade_select_name',
                values=metric_col
            )
            
            pivot_data.plot(kind='bar')
            
            plt.title(f'{metric_name} Comparison - {test_name}')
            plt.xlabel('Select Name, Merge Name')
            plt.ylabel(metric_name)
            plt.legend(title='Trade Select Name')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Format y-axis based on metric
            if metric_col in ['annualized_return', 'max_dd', 'win_rate']:
                plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
            
            plt.savefig(output_dir / f'{metric_col}_{test_name}.png', dpi=300, bbox_inches='tight')
            plt.close()


def main(eval_name=None, select_name_list=None, merge_name_list=None, select_trade_method_name_list=None, 
         test_name_list=None, fee=4e-4, date_start=None, date_end=None):
    """
    Main function to run the analysis.
    
    Parameters:
    -----------
    eval_name : str
        Evaluation name
    select_name_list : list
        List of select_name values
    merge_name_list : list
        List of merge_name values
    select_trade_method_name_list : list
        List of select_trade_method_name values
    test_name_list : list, optional
        List of test names to process
    fee : float, default=4e-4
        Trading fee
    date_start, date_end : datetime, optional
        Date range for filtering
    """
    # Set default parameters if not provided
    if eval_name is None:
        eval_name = "your_eval_name"  # Replace with default eval name
    
    if select_name_list is None:
        select_name_list = ["select_v1", "select_v2"]  # Replace with default select names
        
    if merge_name_list is None:
        merge_name_list = ["merge_v1", "merge_v2"]  # Replace with default merge names
        
    if select_trade_method_name_list is None:
        select_trade_method_name_list = ["trade_v1", "trade_v2"]  # Replace with default trade method names
    
    if test_name_list is None:
        test_name_list = ["test_default", "trade_default"]  # Default test names
    
    # Set up project directory
    project_dir = Path.cwd().parents[1]  # Adjust as needed
    
    # Load path configuration
    path_config = load_path_config(project_dir)
    result_dir = Path(path_config['result'])
    
    # Create timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up output directory
    output_dir = result_dir / "analysis" / "compare_merge_results" / eval_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save input parameters as JSON
    params = {
        "eval_name": eval_name,
        "select_name_list": select_name_list,
        "merge_name_list": merge_name_list,
        "select_trade_method_name_list": select_trade_method_name_list,
        "test_name_list": test_name_list,
        "fee": fee,
        "date_start": str(date_start) if date_start else None,
        "date_end": str(date_end) if date_end else None,
        "timestamp": timestamp
    }
    
    with open(output_dir / "analysis_params.json", "w") as f:
        json.dump(params, f, indent=4)
    
    # Run analysis - ONLY CALL THIS ONCE
    results = analyze_combinations(
        eval_name=eval_name,
        select_name_list=select_name_list,
        merge_name_list=merge_name_list,
        select_trade_method_name_list=select_trade_method_name_list,
        test_name_list=test_name_list,
        project_dir=project_dir,
        n_workers=4,  # Adjust as needed
        fee=fee,
        date_start=date_start,
        date_end=date_end
    )
    
    # Create metrics table
    metrics_df = create_metrics_table(results)
    
    # Save metrics to CSV
    metrics_df.to_csv(output_dir / "metrics_comparison.csv", index=False)
    
    # Generate summary prints
    print("\n===== METRICS SUMMARY =====")
    for test_name in metrics_df['test_name'].unique():
        print(f"\n--- {test_name} ---")
        test_metrics = metrics_df[metrics_df['test_name'] == test_name]
        
        # Sort by Sharpe ratio
        sorted_metrics = test_metrics.sort_values('sharpe_ratio', ascending=False)
        
        # Display top 5 combinations
        print("Top 5 combinations by Sharpe ratio:")
        for _, row in sorted_metrics.head(5).iterrows():
            print(f"{row['combination']}: Sharpe={row['sharpe_ratio']:.2f}, "
                  f"Return={row['annualized_return']:.2%}, "
                  f"MaxDD={row['max_dd']:.2%}, "
                  f"PPT={row['profit_per_trade']:.2f}bp")
    
    # Create plots
    plot_cumulative_returns(results, output_dir, test_name_list=test_name_list)
    plot_drawdowns(results, output_dir, test_name_list=test_name_list)
    plot_monthly_returns(results, output_dir)
    plot_metrics_comparison(metrics_df, output_dir)
    
    print(f"\nAnalysis completed. Results saved to: {output_dir}")



if __name__ == "__main__":
    # Example usage with command line arguments (you could add argparse for better CLI)
    from datetime import datetime
    
    # Parse command line arguments here if needed
    # For now, using default values
    
    # You can import or specify dates as needed
    date_start = datetime(2017, 1, 1)
    date_end = datetime(2025, 4, 1)
    
    # Define your parameters here
    eval_name = "batch_18_v1_ma_batch_test_v2_icim"  # Replace with your actual eval name
    select_name_list = ["s0", "s1", "s6"]  # Replace with your actual select names
    merge_name_list = ["m0", "m1"]  # Replace with your actual merge names
    select_trade_method_name_list = ["st0", "st1"]  # Replace with your actual trade method names
    test_name_list = ["icim_traded"]  # Add other test names if needed
    fee = 2.4e-4  # Default fee value
    
    # Run the analysis
    # Call main with parameters
    main(
        eval_name=eval_name,
        select_name_list=select_name_list,
        merge_name_list=merge_name_list,
        select_trade_method_name_list=select_trade_method_name_list,
        test_name_list=test_name_list,
        fee=fee,
        date_start=date_start,
        date_end=date_end
    )