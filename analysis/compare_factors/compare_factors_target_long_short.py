# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:15:30 2025

@author: [Your Name]

Enhanced Factor Performance Comparison Tool with Target Factor Support
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from pathlib import Path
import seaborn as sns
from functools import partial
import toml
import sys
import json

# Add project directory to path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[2]
sys.path.append(str(project_dir))

from test_and_eval.scores import get_general_return_metrics, calc_sharpe
from test_and_eval.evalutils import extend_metrics
from utils.timeutils import period_shortcut
from utils.dirutils import load_path_config

class FactorPerformanceComparison:
    """Tool for comparing performance metrics of multiple factors with target factor support."""
    
    def __init__(self, result_dir=None, param_dir=None, fee=4e-4, y_label_offset=0.02):
        """
        Initialize the comparison tool.
        
        Parameters:
        -----------
        result_dir : str or Path, optional
            Path to the directory containing test results
        param_dir : str or Path, optional
            Path to the directory containing test parameters
        fee : float, optional
            Trading fee used for net return calculation
        y_label_offset : float, optional
            Offset for positioning y-labels in plots
        """
        self.fee = fee
        self.y_label_offset = y_label_offset
        
        # Load paths from config
        file_path = Path(__file__).resolve()
        project_dir = file_path.parents[2]
        self.path_config = load_path_config(project_dir)
        
        if result_dir is None:
            result_dir = Path(self.path_config['result'])
        if param_dir is None:
            param_dir = Path(self.path_config['param'])
        
        self.result_dir = Path(result_dir)
        self.param_dir = Path(param_dir)
        
        # Set color palette for consistent visualization
        self.palette = sns.color_palette("husl", 10)
        
        # Special colors for target factor
        self.target_color = '#FF0000'  # Red for target factor
        self.target_linewidth = 3
        
    def save_comparison_config(self, factor_info_list, compare_name, target_factor_index, output_path):
        """
        Save comparison configuration to JSON file.
        
        Parameters:
        -----------
        factor_info_list : list of tuple or list of dict
            List of tuples with (tag_name, process_name, factor_name, test_name) 
            or list of dicts with keys: tag_name, process_name, factor_name, test_name, custom_path (optional), shortcut (optional)
        compare_name : str
            Name for this comparison
        target_factor_index : int or None
            Index of target factor in factor_info_list
        output_path : Path
            Directory to save configuration
        """
        config = {
            "compare_name": compare_name,
            "target_factor_index": target_factor_index,
            "factor_list": [],
            "fee": self.fee,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        for i, info in enumerate(factor_info_list):
            if isinstance(info, dict):
                # New dict format
                factor_config = {
                    "index": i,
                    "tag_name": info.get("tag_name"),
                    "process_name": info.get("process_name"),
                    "factor_name": info.get("factor_name"),
                    "test_name": info.get("test_name"),
                    "custom_path": info.get("custom_path"),
                    "shortcut": info.get("shortcut"),
                    "is_target": i == target_factor_index
                }
            else:
                # Original tuple format
                factor_config = {
                    "index": i,
                    "tag_name": info[0],
                    "process_name": info[1],
                    "factor_name": info[2],
                    "test_name": info[3],
                    "custom_path": None,
                    "shortcut": None,
                    "is_target": i == target_factor_index
                }
            config["factor_list"].append(factor_config)
        
        config_file = output_path / "comparison_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"Comparison configuration saved to {config_file}")
        
    def load_factor_data(self, factor_info_list, date_start=None, date_end=None):
        """
        Load factor data based on provided factor information.
        
        Parameters:
        -----------
        factor_info_list : list of tuple or list of dict
            List of tuples with (tag_name, process_name, factor_name, test_name) 
            or list of dicts with keys: tag_name, process_name, factor_name, test_name, custom_path (optional), shortcut (optional)
        date_start : str or datetime, optional
            Start date for filtering data
        date_end : str or datetime, optional
            End date for filtering data
            
        Returns:
        --------
        dict
            Dictionary containing loaded data and calculated metrics
        """
        factors_data = {}
        
        for idx, factor_info in enumerate(factor_info_list):
            # Support both dict and tuple formats
            if isinstance(factor_info, dict):
                tag_name = factor_info.get("tag_name")
                process_name = factor_info.get("process_name")
                factor_name = factor_info.get("factor_name")
                test_name = factor_info.get("test_name")
                custom_path = factor_info.get("custom_path")
                shortcut = factor_info.get("shortcut")
            else:
                # Original tuple format
                tag_name, process_name, factor_name, test_name = factor_info
                custom_path = None
                shortcut = None
            
            # Construct path to data
            if custom_path:
                # Use custom path if provided
                data_path = Path(custom_path) / 'data'
            else:
                # Use standard path structure
                if tag_name:
                    data_path = self.result_dir / 'test' / test_name / tag_name / process_name / 'data'
                else:
                    data_path = self.result_dir / 'test' / test_name / process_name / 'data'
            
            # Check if data exists
            gpd_path = data_path / f'gpd_{factor_name}.pkl'
            hsr_path = data_path / f'hsr_{factor_name}.pkl'
            
            if not (gpd_path.exists() and hsr_path.exists()):
                print(f"Warning: Data not found for {factor_name} in {data_path}")
                continue
            
            # Load data
            try:
                with open(gpd_path, 'rb') as f:
                    gpd = pickle.load(f)
                with open(hsr_path, 'rb') as f:
                    hsr = pickle.load(f)
                
                # Process data for all, long_only, and short_only directions
                directions = ['all', 'pos', 'neg']  # pos for long_only, neg for short_only
                direction_names = ['all', 'long_only', 'short_only']
                
                # Determine direction multiplier based on 'all' direction for consistency
                base_direction_mul = 1
                if 'all' in gpd:
                    df_gp_all = gpd['all']
                    if date_start:
                        df_gp_all = df_gp_all[df_gp_all.index >= pd.Timestamp(date_start)]
                    if date_end:
                        df_gp_all = df_gp_all[df_gp_all.index <= pd.Timestamp(date_end)]
                    cumrtn = df_gp_all['return'].sum()
                    base_direction_mul = 1 if cumrtn > 0 else -1
                
                for direction, direction_name in zip(directions, direction_names):
                    if direction not in gpd or direction not in hsr:
                        continue
                    
                    df_gp = gpd[direction]
                    df_hsr = hsr[direction]
                    
                    # Filter by date if provided
                    if date_start:
                        df_gp = df_gp[df_gp.index >= pd.Timestamp(date_start)]
                        df_hsr = df_hsr[df_hsr.index >= pd.Timestamp(date_start)]
                    if date_end:
                        df_gp = df_gp[df_gp.index <= pd.Timestamp(date_end)]
                        df_hsr = df_hsr[df_hsr.index <= pd.Timestamp(date_end)]
                    
                    # Use base direction multiplier for consistency
                    direction_mul = base_direction_mul
                    
                    # Calculate net return
                    net = (df_gp['return'] * direction_mul - self.fee * df_hsr['avg']).fillna(0)
                    
                    # Compute metrics
                    metrics = get_general_return_metrics(net)
                    hsr_avg = df_hsr["avg"].mean()
                    
                    # Calculate profit per trade
                    profit_per_trade = df_gp["return"].sum() * direction_mul / df_hsr["avg"].sum()
                    
                    # Store data
                    key = f"{factor_name}_{direction_name}"
                    if key not in factors_data:
                        factors_data[key] = {}
                    
                    display_name = shortcut if shortcut else factor_name
                    
                    factors_data[key].update({
                        'gp': df_gp,
                        'hsr': df_hsr,
                        'net': net,
                        'metrics': metrics,
                        'hsr_avg': hsr_avg,
                        'direction': direction_mul,
                        'profit_per_trade': profit_per_trade,
                        'display_name': f"{display_name} ({direction_name})",
                        'test_name': test_name,
                        'process_name': process_name,
                        'tag_name': tag_name,
                        'factor_index': idx,  # Store original index
                        'factor_name': factor_name,  # Store factor name for identification
                        'shortcut': shortcut,  # Store shortcut
                        'custom_path': custom_path,  # Store custom path info
                        'direction_type': direction_name  # Store direction type
                    })
                    
                    # Extend with yearly metrics
                    years = net.index.year.unique()
                    yearly_metrics = {}
                    for year in years:
                        net_year = net[net.index.year == year]
                        yearly_metrics[year] = get_general_return_metrics(net_year)
                    factors_data[key]['yearly_metrics'] = yearly_metrics
                    
            except Exception as e:
                print(f"Error loading {factor_name}: {str(e)}")
        
        return factors_data

    def plot_cumulative_returns(self, factors_data, target_factor_index=None, 
                              output_dir=None, filename="factor_comparison.png"):
        """
        Plot cumulative returns with 3x2 layout: all/long_only/short_only (rows) vs cumsum/diff (columns)
        
        Parameters:
        -----------
        factors_data : dict
            Dictionary containing factor performance data
        target_factor_index : int, optional
            Index of the target factor for comparison
        output_dir : str or Path, optional
            Directory to save the plot
        filename : str, optional
            Filename for the saved plot
        """
        if not factors_data:
            print("No data to plot")
            return
        
        # Find target factor data for all directions
        target_factors = {}
        if target_factor_index is not None:
            for direction_type in ['all', 'long_only', 'short_only']:
                for key, data in factors_data.items():
                    if (data.get('factor_index') == target_factor_index and 
                        data.get('direction_type') == direction_type):
                        target_factors[direction_type] = data
                        break
        
        # Create figure with 3 rows, 2 columns
        fig = plt.figure(figsize=(20, 18))
        gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.15)
        
        direction_types = ['all', 'long_only', 'short_only']
        
        for row, direction_type in enumerate(direction_types):
            print(row, direction_type)
            # Filter data for current direction
            direction_data = {k: v for k, v in factors_data.items() 
                            if v.get('direction_type') == direction_type}
            
            if not direction_data:
                continue
            
            # Left column: Cumulative returns
            ax_cum = fig.add_subplot(gs[row, 0])
            
            for i, (key, data) in enumerate(direction_data.items()):
                if 'net' not in data or 'metrics' not in data:
                    continue
                    
                net = data['net']
                sharpe = data['metrics']['sharpe_ratio']
                ppt = data['profit_per_trade']
                hsr = data['hsr_avg']
                
                # Use shortcut if available, otherwise factor name
                display_name = data.get('shortcut') or data['factor_name']
                label = f"{display_name} - SR: {sharpe:.2f}, PPT: {ppt*1000:.1f}‰, HSR: {hsr:.2f}"
                
                # Use special formatting for target factor
                if (target_factor_index is not None and 
                    data.get('factor_index') == target_factor_index):
                    color = self.target_color
                    linewidth = self.target_linewidth
                    alpha = 1.0
                else:
                    color = self.palette[i % len(self.palette)]
                    linewidth = 2
                    alpha = 0.8
                
                ax_cum.plot(net.index, net.cumsum(), label=label, 
                          linewidth=linewidth, color=color, alpha=alpha)
            
            # Format cumulative returns plot
            ax_cum.set_title(f"Cumulative Returns - {direction_type.replace('_', ' ').title()}", 
                           fontsize=14)
            ax_cum.set_ylabel("Cumulative Return", fontsize=12)
            ax_cum.grid(True, linestyle='--', alpha=0.7)
            ax_cum.legend(loc='best', fontsize=9)
            
            # Format date axis
            ax_cum.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax_cum.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax_cum.xaxis.get_majorticklabels(), rotation=45)
            
            # Right column: Difference from target
            ax_diff = fig.add_subplot(gs[row, 1])
            
            if direction_type in target_factors:
                target_data = target_factors[direction_type]
                target_net = target_data['net']
                target_cum = target_net.cumsum()
                
                for i, (key, data) in enumerate(direction_data.items()):
                    if (data.get('factor_index') == target_factor_index or 
                        'net' not in data):
                        continue
                    
                    other_net = data['net']
                    other_cum = other_net.cumsum()
                    
                    # Calculate difference (align indices)
                    common_index = target_cum.index.intersection(other_cum.index)
                    if len(common_index) == 0:
                        continue
                    
                    diff = other_cum.loc[common_index] - target_cum.loc[common_index]
                    
                    # Prepare label with metrics difference
                    sharpe_diff = (data['metrics']['sharpe_ratio'] - 
                                 target_data['metrics']['sharpe_ratio'])
                    ppt_diff = ((data['profit_per_trade'] - 
                               target_data['profit_per_trade']) * 1000)
                    
                    display_name = data.get('shortcut') or data['factor_name']
                    label = f"{display_name} - Target | ΔSR: {sharpe_diff:+.2f}, ΔPPT: {ppt_diff:+.1f}‰"
                    
                    color = self.palette[i % len(self.palette)]
                    ax_diff.plot(diff.index, diff, label=label, linewidth=2, color=color)
                
                # Add zero line
                ax_diff.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Format difference plot
                target_display_name = target_data.get('shortcut') or target_data['factor_name']
                ax_diff.set_title(f"Difference vs Target ({target_display_name}) - {direction_type.replace('_', ' ').title()}", 
                                fontsize=14)
                ax_diff.set_ylabel("Cumulative Return Difference", fontsize=12)
                ax_diff.grid(True, linestyle='--', alpha=0.7)
                ax_diff.legend(loc='best', fontsize=9)
                
                # Format date axis
                ax_diff.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax_diff.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                plt.setp(ax_diff.xaxis.get_majorticklabels(), rotation=45)
            else:
                ax_diff.text(0.5, 0.5, f'No target factor\nfor {direction_type}', 
                           transform=ax_diff.transAxes, ha='center', va='center',
                           fontsize=12)
                ax_diff.set_title(f"No Target - {direction_type.replace('_', ' ').title()}", 
                                fontsize=14)
        
        # Set x-label only for bottom row
        # for col in range(2):
        #     ax = fig.add_subplot(gs[2, col])
        #     ax.set_xlabel("Date", fontsize=12)
        
        # Save the figure
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
        plt.close()
    
    def plot_yearly_performance(self, factors_data, output_dir=None, filename="yearly_comparison.png"):
        """
        Plot yearly performance of multiple factors.
        
        Parameters:
        -----------
        factors_data : dict
            Dictionary containing factor performance data
        output_dir : str or Path, optional
            Directory to save the plot
        filename : str, optional
            Filename for the saved plot
        """
        if not factors_data:
            print("No data to plot")
            return
        
        # Collect all unique years across all factors
        all_years = set()
        for key, data in factors_data.items():
            if 'yearly_metrics' in data:
                all_years.update(data['yearly_metrics'].keys())
        
        all_years = sorted(all_years)
        if not all_years:
            print("No yearly data to plot")
            return
        
        # Calculate number of subplots needed
        n_years = len(all_years)
        n_cols = min(3, n_years)
        n_rows = (n_years + n_cols - 1) // n_cols
        
        # Create figure
        fig = plt.figure(figsize=(16, 5 * n_rows))
        
        # Plot each year
        for i, year in enumerate(all_years):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Plot data for each factor for this year
            data_to_plot = []
            labels = []
            colors = []
            for j, (key, data) in enumerate(factors_data.items()):
                if 'yearly_metrics' not in data or year not in data['yearly_metrics']:
                    continue
                
                # Extract yearly metrics
                yearly_metrics = data['yearly_metrics'][year]
                annual_return = yearly_metrics['return_annualized']
                sharpe = yearly_metrics['sharpe_ratio']
                
                # Add to plotting data
                data_to_plot.append(annual_return)
                labels.append(f"{data['display_name']} - SR: {sharpe:.2f}")
                colors.append(self.palette[j % len(self.palette)])
            
            # Plot as bar chart
            ax.bar(range(len(data_to_plot)), data_to_plot, color=colors)
            ax.set_xticks(range(len(data_to_plot)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            
            # Add year and metrics
            ax.set_title(f"Performance in {year}", fontsize=14)
            ax.set_ylabel("Annualized Return", fontsize=12)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # Add values on top of bars
            for j, v in enumerate(data_to_plot):
                ax.text(j, v + (0.01 if v >= 0 else -0.03), 
                       f"{v:.1%}", ha='center', fontsize=9)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        
        plt.show()
        plt.close()
    
    def plot_yearly_cumulative(self, factors_data, output_dir=None, filename="yearly_cumulative.png"):
        """
        Plot yearly cumulative returns of multiple factors with separate plots for long_only and short_only.
        
        Parameters:
        -----------
        factors_data : dict
            Dictionary containing factor performance data
        output_dir : str or Path, optional
            Directory to save the plot
        filename : str, optional
            Filename for the saved plot
        """
        if not factors_data:
            print("No data to plot")
            return
        
        # Process each direction separately: all, long_only, short_only
        direction_types = ['all', 'long_only', 'short_only']
        
        for direction_type in direction_types:
            # Filter data for current direction
            direction_data = {k: v for k, v in factors_data.items() 
                            if v.get('direction_type') == direction_type}
            
            if not direction_data:
                continue
            
            # Collect all unique years across all factors for this direction
            all_years = set()
            for key, data in direction_data.items():
                if 'net' in data:
                    all_years.update(data['net'].index.year.unique())
            
            all_years = sorted(all_years)
            if not all_years:
                continue
            
            # Calculate number of subplots needed
            n_years = len(all_years)
            n_cols = min(2, n_years)
            n_rows = (n_years + n_cols - 1) // n_cols
            
            # Create figure
            fig = plt.figure(figsize=(14, 6 * n_rows))
            
            # Plot each year
            for i, year in enumerate(all_years):
                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                
                # Set max_value and min_value to track y-axis limits
                max_value = float('-inf')
                min_value = float('inf')
                
                # Plot data for each factor for this year and direction
                for j, (key, data) in enumerate(direction_data.items()):
                    if 'net' not in data:
                        continue
                    
                    # Extract yearly data
                    net_year = data['net'][data['net'].index.year == year]
                    if len(net_year) == 0:
                        continue
                    
                    # Calculate metrics for this year
                    metrics = get_general_return_metrics(net_year)
                    sharpe = metrics['sharpe_ratio']
                    
                    # Prepare for plotting
                    cum_return = net_year.cumsum()
                    display_name = data.get('shortcut') or data['factor_name']
                    label = f"{display_name} - SR: {sharpe:.2f}"
                    
                    # Plot
                    color = self.palette[j % len(self.palette)]
                    ax.plot(cum_return.index, cum_return, label=label, linewidth=2, color=color)
                    
                    # Update max and min values
                    max_value = max(max_value, cum_return.max())
                    min_value = min(min_value, cum_return.min())
                
                # Set title and labels
                ax.set_title(f"Cumulative Returns in {year} - {direction_type.replace('_', ' ').title()}", 
                           fontsize=14)
                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("Cumulative Return", fontsize=12)
                
                # Format date axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                
                # Add grid and legend
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend(loc='best', fontsize=9)
                
                # Add some padding to y-axis limits
                if max_value > min_value:
                    y_padding = (max_value - min_value) * 0.1
                    ax.set_ylim(min_value - y_padding, max_value + y_padding)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the figure
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                safe_direction = direction_type.replace('_', '')
                plt.savefig(output_path / f"yearly_cumulative_{safe_direction}.png", 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
            plt.close()
    
    def plot_monthly_heatmap(self, factors_data, output_dir=None, filename_prefix="monthly_heatmap"):
        """
        Plot monthly return heatmap for each factor.
        
        Parameters:
        -----------
        factors_data : dict
            Dictionary containing factor performance data
        output_dir : str or Path, optional
            Directory to save the plots
        filename_prefix : str, optional
            Prefix for the saved plot filenames
        """
        if not factors_data:
            print("No data to plot")
            return
        
        # Create output directory if needed
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # For each factor, create a monthly heatmap
        for key, data in factors_data.items():
            if 'net' not in data:
                continue
            
            # Resample to monthly returns
            net = data['net']
            monthly_returns = net.resample('M').sum()
            
            # Create a DataFrame with Year and Month for the heatmap
            monthly_df = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Return': monthly_returns.values
            })
            
            # Convert to pivot table
            pivot_table = monthly_df.pivot_table(index='Year', columns='Month', values='Return')
            
            # Create figure
            fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
            
            # Plot heatmap
            sns.heatmap(pivot_table, annot=True, fmt='.1%', cmap='RdYlGn', 
                       center=0, ax=ax, cbar_kws={'label': 'Monthly Return'})
            
            # Set title and labels
            ax.set_title(f"Monthly Returns: {data['display_name']}", fontsize=16)
            ax.set_xlabel("Month", fontsize=12)
            ax.set_ylabel("Year", fontsize=12)
            
            # Set month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticklabels(month_names, rotation=0)
            
            # Add metrics to the plot
            metrics_text = (
                f"Sharpe Ratio: {data['metrics']['sharpe_ratio']:.2f}\n"
                f"Profit Per Trade: {data['profit_per_trade']*1000:.1f}‰\n"
                f"Max Drawdown: {data['metrics']['max_dd']*100:.2f}%\n"
                f"Annualized Return: {data['metrics']['return_annualized']*100:.2f}%"
            )
            plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save the figure
            if output_dir:
                safe_key = key.replace('/', '_').replace('\\', '_')
                plt.savefig(output_path / f"{filename_prefix}_{safe_key}.png", 
                           dpi=300, bbox_inches='tight')
            
            plt.show()
            plt.close()
    
    def plot_drawdown_analysis(self, factors_data, output_dir=None, filename="drawdown_comparison.png"):
        """
        Plot drawdown analysis of multiple factors.
        
        Parameters:
        -----------
        factors_data : dict
            Dictionary containing factor performance data
        output_dir : str or Path, optional
            Directory to save the plot
        filename : str, optional
            Filename for the saved plot
        """
        if not factors_data:
            print("No data to plot")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])
        
        # For each factor, calculate and plot drawdowns
        for i, (key, data) in enumerate(factors_data.items()):
            if 'net' not in data:
                continue
            
            net = data['net']
            
            # Calculate drawdowns
            cum_returns = net.cumsum()
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - running_max) / running_max
            
            # Plot cumulative returns on the top subplot
            color = self.palette[i % len(self.palette)]
            ax1.plot(cum_returns.index, cum_returns, 
                    label=f"{data['display_name']}", linewidth=2, color=color)
            
            # Plot drawdowns on the bottom subplot
            ax2.plot(drawdowns.index, drawdowns, linewidth=2, color=color)
            
            # Add horizontal line at max drawdown
            max_dd = data['metrics']['max_dd']
            ax2.axhline(y=-max_dd, linestyle='--', alpha=0.5, color=color)
        
        # Format top subplot
        ax1.set_title("Cumulative Returns", fontsize=16)
        ax1.set_ylabel("Cumulative Return", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='best', fontsize=10)
        
        # Format bottom subplot
        ax2.set_title("Drawdowns", fontsize=16)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Drawdown", fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.fill_between(ax2.get_xlim(), [0, 0], [-1, -1], alpha=0.1, color='red')
        
        # Format date axes
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save the figure
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        
        plt.show()
        plt.close()
    
    def generate_comparison_table(self, factors_data):
        """
        Generate a summary table comparing all factors across different directions.
        
        Parameters:
        -----------
        factors_data : dict
            Dictionary containing factor performance data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing comparison metrics
        """
        # Prepare data for table
        table_data = []
        
        for key, data in factors_data.items():
            if 'metrics' not in data:
                continue
            
            metrics = data['metrics']
            
            display_name = data.get('shortcut') or data['factor_name']
            
            row = {
                'Factor': display_name,
                'Direction': data['direction_type'],
                'Annualized Return': f"{metrics['return_annualized']:.2%}",
                'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
                'Max Drawdown': f"{metrics['max_dd']:.2%}",
                'Profit Per Trade': f"{data['profit_per_trade']*1000:.1f}‰",
                'Test Name': data.get('test_name', ''),
                'Process Name': data.get('process_name', ''),
                'Tag Name': data.get('tag_name', '')
            }
            table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Sort by Factor and Direction for better organization
        if not df.empty:
            df = df.sort_values(['Factor', 'Direction'])
        
        return df
    
    def save_results(self, factors_data, output_dir, compare_name="factor_comparison"):
        """
        Save comparison results including plots and summary table.
        
        Parameters:
        -----------
        factors_data : dict
            Dictionary containing factor performance data
        output_dir : str or Path
            Directory to save results
        compare_name : str, optional
            Name prefix for saved files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving comparison results to {output_path}")
        
        # Generate and save summary table
        summary_table = self.generate_comparison_table(factors_data)
        if not summary_table.empty:
            table_path = output_path / f"{compare_name}_summary.csv"
            summary_table.to_csv(table_path, index=False)
            print(f"Summary table saved to {table_path}")
            
            # Also save as Excel for better formatting
            excel_path = output_path / f"{compare_name}_summary.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                summary_table.to_excel(writer, sheet_name='Summary', index=False)
                
                # Format the Excel sheet
                worksheet = writer.sheets['Summary']
                for column in worksheet.columns:
                    max_length = 0
                    column = [cell for cell in column]
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = (max_length + 2)
                    worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
            
            print(f"Excel summary saved to {excel_path}")
        
        # Save raw data as pickle for future analysis
        data_path = output_path / f"{compare_name}_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump(factors_data, f)
        print(f"Raw data saved to {data_path}")
        
        # Print summary to console
        print("\n" + "="*80)
        print(f"FACTOR COMPARISON SUMMARY: {compare_name}")
        print("="*80)
        if not summary_table.empty:
            print(summary_table.to_string(index=False))
        print("="*80)
    
    def run_full_comparison(self, factor_info_list, compare_name="factor_comparison",
                          target_factor_index=None, date_start=None, date_end=None,
                          output_dir=None, create_all_plots=True):
        """
        Run a complete factor comparison analysis.
        
        Parameters:
        -----------
        factor_info_list : list of tuple or list of dict
            List of tuples with (tag_name, process_name, factor_name, test_name) 
            or list of dicts with keys: tag_name, process_name, factor_name, test_name, custom_path (optional), shortcut (optional)
        compare_name : str, optional
            Name for this comparison
        target_factor_index : int, optional
            Index of target factor for comparison
        date_start : str or datetime, optional
            Start date for filtering data
        date_end : str or datetime, optional
            End date for filtering data
        output_dir : str or Path, optional
            Directory to save results
        create_all_plots : bool, optional
            Whether to create all available plots
            
        Returns:        
        --------
        dict
            Dictionary containing loaded factor data and analysis results
        """
        print(f"Starting factor comparison: {compare_name}")
        print(f"Number of factors: {len(factor_info_list)}")
        if target_factor_index is not None:
            print(f"Target factor index: {target_factor_index}")
        
        # Set default output directory
        if output_dir is None:
            output_dir = self.result_dir / 'analysis' / 'compare_factors' / compare_name
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save comparison configuration
        self.save_comparison_config(factor_info_list, compare_name, target_factor_index, output_path)
        
        # Load factor data
        print("\nLoading factor data...")
        factors_data = self.load_factor_data(factor_info_list, date_start, date_end)
        
        if not factors_data:
            print("No valid factor data loaded. Exiting.")
            return None
        
        print(f"Loaded {len(factors_data)} factor-direction combinations")
        
        # Create plots if requested
        if create_all_plots:
            print("\nGenerating plots...")
            
            # Main cumulative returns comparison with 3x2 layout
            self.plot_cumulative_returns(factors_data, target_factor_index, 
                                       output_path, "factor_comparison.png")
            
            # Yearly cumulative returns for each direction (separate plots)
            self.plot_yearly_cumulative(factors_data, output_path, "yearly_cumulative.png")
            
            # Yearly performance comparison
            self.plot_yearly_performance(factors_data, output_path, "yearly_performance.png")
            
            # Monthly heatmaps for each factor
            self.plot_monthly_heatmap(factors_data, output_path, "monthly_heatmap")
            
            # Drawdown analysis
            self.plot_drawdown_analysis(factors_data, output_path, "drawdown_analysis.png")
        
        # Save results
        # self.save_results(factors_data, output_path, compare_name)
        
        print(f"\nComparison completed successfully!")
        print(f"Results saved to: {output_path}")
        
        return factors_data


# 使用示例的更新
if __name__ == "__main__":
    # # 方式1: 使用原来的tuple格式
    # factor_info_list_tuple = [
    #     ("zxt", "Batch10_fix_best_241218_selected_f64/v1.2_longer_wd", "LargeOrderAmountByValue_p1.0_v40000-avg_imb01_dp2-rollingAggMinuteMinMaxScale_w245d_q0.02_i5", "trade_ver3_intraday_futtwap_sp1min_s240d_icim"),
    #     ("zxt", "Batch21_250512/normal_trans_for_test_250414", "LargeOARmAuc_p1.0_v40000_f25-avg_imb01_dp2-rollingAggMinuteMinMaxScale_w245d_q0.02_i5", "trade_ver3_intraday_futtwap_sp1min_s240d_icim"),
    # ]
    
    # 方式2: 使用新的dict格式，支持自定义路径
    compare_name = "compare_dp"
    factor_info_list_dict = [
        # {
        #     "tag_name": None,
        #     "process_name": None,
        #     "factor_name": "avg_predict_150101_160301",
        #     "test_name": None,
        #     "custom_path": "/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/merge_selected_factors/batch_till20_batch_test_v2_icim_s1_m0/150101_160301/test/icim_intraday_noscale_around_op05cl0"  # 自定义路径，数据在 /path/to/custom/results/data/ 下
        # },
        # {
        #   "index": 0,
        #   "tag_name": "zxt_select_250509",
        #   "process_name": "Batch18_250425/org_trans_v1_TS_dod_all_v1_TS_final_scale_v0",
        #   "factor_name": "TimeRangeValueOrderAmount_p1.0_v40000_t30-avg_imb01_dp2-org-aggMinmax_w245d_q0.02_i5-minmax_w245d_q0.02",
        #   "test_name": "icim_intraday_noscale_around_op05cl0",
        #   "is_target": True
        # },
        # {
        #   "index": 1,
        #   "tag_name": "zxt",
        #   "process_name": "Batch18_250425/tamtratio_lt3_avg",
        #   "factor_name": "TimeRangeValueOrderAmount_p1.0_v40000_t30-selfavg_tamt_v0_lt3_imb01-rollingAggMinuteMinMaxScale_w245d_q0.02_i5",
        #   "test_name": "icim_intraday_scale_around_op05cl0",
        #   "is_target": False
        # },
        {
          "index": 0,
          "tag_name": "zxt_select_250509",
          "process_name": "Batch18_250425/org_trans_v1_TS_dod_all_v1_TS_final_scale_v0",
          "factor_name": "TimeRangeValueOrderAmount_p1.0_v40000_t30-avg_imb01_dp2-org-aggMinmax_w245d_q0.02_i5-minmax_w245d_q0.02",
          "test_name": "icim_intraday_noscale_around_op05cl0",
          "is_target": True,
          "shortcut": 'avg_imb01_dp2',
        },
        {
          "index": 1,
          "tag_name": "zxt_select_250509",
          "process_name": "Batch18_250425/org_trans_v1_TS_dod_all_v1_TS_final_scale_v0",
          "factor_name": "TimeRangeOANet_p1.0_v40000_t30-avg_imb01_dp2-org-aggMinmax_w245d_q0.02_i5-minmax_w245d_q0.02",
          "test_name": "icim_intraday_noscale_around_op05cl0",
          "is_target": False,
          "shortcut": 'order-cancel',
        },
        {
          "index": 2,
          "tag_name": "zxt_select_250509",
          "process_name": "Batch18_250425/org_trans_v1_TS_dod_all_v1_TS_final_scale_v0",
          "factor_name": "TimeRangeOTNet_p1.0_v40000_t30-avg_imb01_dp2-org-aggMinmax_w245d_q0.02_i5-minmax_w245d_q0.02",
          "test_name": "icim_intraday_noscale_around_op05cl0",
          "is_target": False,
          "shortcut": 'order-trade',
        },
      
    ]
    
    # 初始化比较工具
    comparator = FactorPerformanceComparison(fee=2.4e-4)
    
    # 使用dict格式运行比较
    results = comparator.run_full_comparison(
        factor_info_list=factor_info_list_dict,  # 可以是tuple list或dict list
        compare_name=compare_name,
        target_factor_index=0,
        date_start="2017-01-01",
        date_end="2025-04-01",
        create_all_plots=True,
    )
