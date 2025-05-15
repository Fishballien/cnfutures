# -*- coding: utf-8 -*-
"""
Cluster Factor Performance Analysis
This script analyzes the performance of clustered factors by:
1. Loading cluster information for a given period
2. Reading factor test results (gpd and hsr files)
3. Computing net returns for each factor
4. Plotting factor performance grouped by cluster
5. Analyzing underperforming factors
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import pickle
from datetime import datetime
import seaborn as sns
from functools import partial

# %% add sys path
import sys
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[2]
sys.path.append(str(project_dir))

# Import necessary functions from provided modules
from utils.dirutils import load_path_config
from utils.timeutils import period_shortcut
from test_and_eval.scores import get_general_return_metrics

def load_cluster_info(cluster_dir, cluster_name, period_name):
    """
    Load cluster information from CSV file.
    
    Parameters:
    -----------
    cluster_dir : Path
        Path to the cluster directory
    cluster_name : str
        Name of the cluster
    period_name : str
        Period name in the format "YYMMDD_YYMMDD"
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing cluster information
    """
    cluster_path = cluster_dir / cluster_name / f'cluster_info_{period_name}.csv'
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster file not found at {cluster_path}")
    
    return pd.read_csv(cluster_path)

def read_factor_data(test_dir, test_name, tag_name, process_name, factor_name):
    """
    Read gpd and hsr data for a given factor.
    
    Parameters:
    -----------
    test_dir : Path
        Path to the test directory
    test_name : str
        Name of the test
    tag_name : str
        Tag name
    process_name : str
        Process name
    factor_name : str
        Factor name
        
    Returns:
    --------
    tuple
        (gpd_data, hsr_data) dictionaries
    """
    data_dir = test_dir / test_name
    if tag_name is not None and tag_name != 'None':
        data_dir = data_dir / tag_name
    
    data_dir = data_dir / process_name / 'data'
    
    gpd_path = data_dir / f'gpd_{factor_name}.pkl'
    hsr_path = data_dir / f'hsr_{factor_name}.pkl'
    
    try:
        with open(gpd_path, 'rb') as f:
            gpd_data = pickle.load(f)
            
        with open(hsr_path, 'rb') as f:
            hsr_data = pickle.load(f)
            
        return gpd_data, hsr_data
    except Exception as e:
        print(f"Error reading data for {factor_name}: {e}")
        return None, None

def calculate_net_return(gpd_data, hsr_data, direction, fee, date_start, date_end):
    """
    Calculate net return for a factor.
    
    Parameters:
    -----------
    gpd_data : dict
        Dictionary containing gross profit data
    hsr_data : dict
        Dictionary containing hit-switch ratio data
    direction : int
        Direction of the factor (1 or -1)
    fee : float
        Transaction fee
    date_start : datetime or str
        Start date for the analysis
    date_end : datetime or str
        End date for the analysis
        
    Returns:
    --------
    pd.Series
        Net return series
    """
    if gpd_data is None or hsr_data is None:
        return None
    
    df_gp = gpd_data['all']
    df_hsr = hsr_data['all']
    
    df_gp = df_gp[(df_gp.index >= date_start) & (df_gp.index <= date_end)]
    df_hsr = df_hsr[(df_hsr.index >= date_start) & (df_hsr.index <= date_end)]
    
    net = (df_gp['return'] * direction - fee * df_hsr['avg']).fillna(0)
    return net

def analyze_clusters(cluster_name, period_name, date_start, date_end, fee=2.4e-4):
    """
    Analyze factor performance by cluster groups.
    
    Parameters:
    -----------
    cluster_name : str
        Name of the cluster
    period_name : str
        Period name in the format "YYMMDD_YYMMDD" that identifies the cluster file
    date_start : str
        Start date in 'YYYY-MM-DD' format for analysis
    date_end : str
        End date in 'YYYY-MM-DD' format for analysis
    fee : float
        Transaction fee
        
    Returns:
    --------
    None
    """
    path_config = load_path_config(project_dir)
    
    # Setup directories
    result_dir = Path(path_config['result'])
    cluster_dir = result_dir / 'cluster'
    test_dir = result_dir / 'test'
    
    # Create analysis directory
    analysis_period = period_shortcut(
        datetime.strptime(date_start, '%Y-%m-%d'),
        datetime.strptime(date_end, '%Y-%m-%d')
    )
    analysis_dir = result_dir / 'analysis' / 'fac_rec_perf_review' / cluster_name / analysis_period
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load cluster info
    cluster_info = load_cluster_info(cluster_dir, cluster_name, period_name)
    
    # Calculate net returns for each factor
    net_returns = {}
    cumulative_returns = {}
    
    for _, row in cluster_info.iterrows():
        test_name = row['test_name']
        tag_name = row['tag_name']
        process_name = row['process_name']
        factor_name = row['factor']
        direction = row['direction']
        group = row['group']
        
        # Read factor data
        gpd_data, hsr_data = read_factor_data(test_dir, test_name, tag_name, process_name, factor_name)
        
        if gpd_data is not None and hsr_data is not None:
            # Calculate net return
            net = calculate_net_return(gpd_data, hsr_data, direction, fee, date_start, date_end)
            
            if net is not None:
                factor_id = f"{process_name}_{factor_name}"
                net_returns[factor_id] = {
                    'net': net,
                    'group': group,
                    'process_name': process_name,
                    'factor_name': factor_name,
                    'direction': direction
                }
                
                # Calculate cumulative return
                cum_ret = net.cumsum()
                cumulative_returns[factor_id] = cum_ret
    
    # Convert to DataFrame for easier analysis
    net_df = pd.DataFrame({k: v['net'] for k, v in net_returns.items()})
    cum_ret_df = pd.DataFrame(cumulative_returns)
    
    # Add group information
    group_info = {factor_id: data['group'] for factor_id, data in net_returns.items()}
    group_df = pd.Series(group_info)
    
    # Calculate group average net returns
    group_avg_returns = {}
    for group in cluster_info['group'].unique():
        group_factors = group_df[group_df == group].index
        if len(group_factors) > 0:
            group_avg = cum_ret_df[group_factors].mean(axis=1)
            group_avg_returns[f'Group {group}'] = group_avg
    
    # Plot individual factor and group average cumulative returns
    plt.figure(figsize=(15, 10))
    
    # Plot individual factor returns
    for factor_id, cum_ret in cumulative_returns.items():
        group = net_returns[factor_id]['group']
        plt.plot(cum_ret.index, cum_ret.values, alpha=0.3, linewidth=0.8, 
                 color=f'C{group % 10}', label='_nolegend_')
    
    # Plot group average returns
    for group_name, avg_ret in group_avg_returns.items():
        plt.plot(avg_ret.index, avg_ret.values, linewidth=2.5, label=group_name)
    
    plt.title(f'Cumulative Net Returns by Cluster Group - {cluster_name} ({analysis_period})')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(analysis_dir / 'cluster_group_returns.png', dpi=300, bbox_inches='tight')
    
    # Analyze underperforming factors
    total_returns = {factor_id: net.sum() for factor_id, net in net_df.items()}
    factor_performance = pd.DataFrame({
        'Factor': total_returns.keys(),
        'Total Return': total_returns.values(),
        'Group': [net_returns[f]['group'] for f in total_returns.keys()],
        'Process': [net_returns[f]['process_name'] for f in total_returns.keys()],
        'Factor Name': [net_returns[f]['factor_name'] for f in total_returns.keys()]
    })
    
    # Sort by performance (ascending)
    factor_performance = factor_performance.sort_values('Total Return')
    
    # Save worst performing factors
    worst_factors = factor_performance.head(10)
    worst_factors.to_csv(analysis_dir / 'worst_performing_factors.csv', index=False)
    
    # Save best performing factors
    best_factors = factor_performance.tail(10).iloc[::-1]
    best_factors.to_csv(analysis_dir / 'best_performing_factors.csv', index=False)
    
    # Plot group performance distribution
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Group', y='Total Return', data=factor_performance)
    plt.title(f'Distribution of Factor Returns by Group - {cluster_name}')
    plt.grid(True, alpha=0.3)
    plt.savefig(analysis_dir / 'group_return_distribution.png', dpi=300, bbox_inches='tight')
    
    # Calculate and save summary statistics
    group_stats = factor_performance.groupby('Group').agg({
        'Total Return': ['mean', 'std', 'min', 'max', 'count'],
    })
    
    group_stats.to_csv(analysis_dir / 'group_statistics.csv')
    
    # Create a summary of all factors
    factor_performance.to_csv(analysis_dir / 'all_factor_performance.csv', index=False)
    
    # Plot cumulative return of all factors as heatmap
    plt.figure(figsize=(20, 10))
    
    # Create a pivot table with time vs factor
    pivot_data = cum_ret_df.copy()
    pivot_data.index = pivot_data.index.strftime('%Y-%m-%d')
    
    # Select every Nth date for readability
    dates = pivot_data.index.tolist()
    n = max(1, len(dates) // 20)  # Show about 20 dates
    selected_dates = dates[::n]
    
    # Sort factors by group
    sorted_factors = factor_performance.sort_values(['Group', 'Total Return']).loc[:, 'Factor'].tolist()
    heatmap_data = pivot_data.loc[selected_dates, sorted_factors]
    
    # Plot heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(heatmap_data.T, cmap='RdBu_r', center=0)
    plt.title(f'Cumulative Returns Over Time - {cluster_name}')
    plt.xlabel('Date')
    plt.ylabel('Factor')
    plt.tight_layout()
    plt.savefig(analysis_dir / 'return_heatmap.png', dpi=300, bbox_inches='tight')
    
    # Generate summary report
    with open(analysis_dir / 'analysis_summary.txt', 'w') as f:
        f.write(f"Cluster Analysis Summary: {cluster_name}\n")
        f.write(f"Cluster Period: {period_name}\n")
        f.write(f"Analysis Period: {date_start} to {date_end}\n")
        f.write(f"Number of Factors: {len(net_returns)}\n")
        f.write(f"Number of Groups: {len(cluster_info['group'].unique())}\n\n")
        
        f.write("Group Statistics:\n")
        f.write(group_stats.to_string())
        f.write("\n\n")
        
        f.write("Best Performing Factors:\n")
        f.write(best_factors.to_string())
        f.write("\n\n")
        
        f.write("Worst Performing Factors:\n")
        f.write(worst_factors.to_string())
    
    print(f"Analysis completed. Results saved to {analysis_dir}")

if __name__ == "__main__":
    # Example usage
    analyze_clusters(
        cluster_name="agg_250218_3_fix_fut_fr15_by_trade_net_double3m_v18",
        period_name="210401_250401",  # Period that identifies the cluster file
        date_start="2025-03-01",      # Analysis start date
        date_end="2025-05-15"         # Analysis end date
    )