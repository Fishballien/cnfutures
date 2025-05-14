# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:15:30 2025

@author: [Your Name]

Factor Performance Comparison Tool
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
    """Tool for comparing performance metrics of multiple factors."""
    
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
        
    def load_factor_data(self, factor_info_list, date_start=None, date_end=None):
        """
        Load factor data based on provided factor information.
        
        Parameters:
        -----------
        factor_info_list : list of tuple
            List of tuples with (tag_name, process_name, factor_name, test_name)
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
        
        for factor_info in factor_info_list:
            tag_name, process_name, factor_name, test_name = factor_info
            
            # Construct path to data
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
                for direction in ['all']:
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
                    
                    # Determine direction
                    cumrtn = df_gp['return'].sum()
                    direction_mul = 1 if cumrtn > 0 else -1
                    
                    # Calculate net return
                    net = (df_gp['return'] * direction_mul - self.fee * df_hsr['avg']).fillna(0)
                    
                    # Compute metrics
                    metrics = get_general_return_metrics(net)
                    
                    # Calculate profit per trade
                    profit_per_trade = df_gp["return"].sum() * direction_mul / df_hsr["avg"].sum()
                    
                    # Store data
                    key = f"{factor_name}_{direction}"
                    if key not in factors_data:
                        factors_data[key] = {}
                    
                    factors_data[key].update({
                        'gp': df_gp,
                        'hsr': df_hsr,
                        'net': net,
                        'metrics': metrics,
                        'direction': direction_mul,
                        'profit_per_trade': profit_per_trade,
                        'display_name': f"{factor_name} ({direction})",
                        'test_name': test_name,
                        'process_name': process_name,
                        'tag_name': tag_name
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
    
    def plot_cumulative_returns(self, factors_data, output_dir=None, filename="factor_comparison.png"):
        """
        Plot cumulative returns of multiple factors.
        
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
        
        # Create figure
        fig, ax = plt.figure(figsize=(16, 9)), plt.gca()
        
        # Plot cumulative returns for each factor
        for i, (key, data) in enumerate(factors_data.items()):
            if 'net' not in data or 'metrics' not in data:
                continue
                
            net = data['net']
            sharpe = data['metrics']['sharpe_ratio']
            ppt = data['profit_per_trade']
            label = f"{data['display_name']} - SR: {sharpe:.2f}, PPT: {ppt*1000:.1f}‰"
            
            color = self.palette[i % len(self.palette)]
            ax.plot(net.index, net.cumsum(), label=label, linewidth=2, color=color)
        
        # Set labels and title
        ax.set_title("Cumulative Net Returns Comparison", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Cumulative Return", fontsize=12)
        
        # Format date axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best', fontsize=10)
        
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
            
            # Plot as bar chart
            colors = self.palette[:len(data_to_plot)]
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
        Plot yearly cumulative returns of multiple factors.
        
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
            if 'net' in data:
                all_years.update(data['net'].index.year.unique())
        
        all_years = sorted(all_years)
        if not all_years:
            print("No yearly data to plot")
            return
        
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
            
            # Plot data for each factor for this year
            for j, (key, data) in enumerate(factors_data.items()):
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
                label = f"{data['display_name']} - SR: {sharpe:.2f}"
                
                # Plot
                color = self.palette[j % len(self.palette)]
                ax.plot(cum_return.index, cum_return, label=label, linewidth=2, color=color)
                
                # Update max and min values
                max_value = max(max_value, cum_return.max())
                min_value = min(min_value, cum_return.min())
            
            # Set title and labels
            ax.set_title(f"Cumulative Returns in {year}", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Cumulative Return", fontsize=12)
            
            # Format date axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='best', fontsize=9)
            
            # Add some padding to y-axis limits
            y_padding = (max_value - min_value) * 0.1
            ax.set_ylim(min_value - y_padding, max_value + y_padding)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        
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
            ax2.axhline(y=-max_dd, color=color, linestyle='--', alpha=0.5)
            
            # Add annotated max_dd
            max_dd_idx = drawdowns.idxmin()
            if max_dd_idx is not None:  # Check if drawdown exists
                ax2.annotate(f"{max_dd*100:.1f}%", 
                           xy=(max_dd_idx, -max_dd),
                           xytext=(-20, -20),
                           textcoords="offset points",
                           color=color,
                           arrowprops=dict(arrowstyle="->", color=color))
        
        # Set labels and title for the top subplot (cumulative returns)
        ax1.set_title("Cumulative Returns and Drawdowns", fontsize=16)
        ax1.set_ylabel("Cumulative Return", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='best', fontsize=10)
        
        # Format date axis for top subplot
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Set labels for the bottom subplot (drawdowns)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylabel("Drawdown", fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Format y-axis as percentage
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Format date axis for bottom subplot
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Save the figure
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
        plt.close()
    
    def generate_performance_report(self, factors_data, output_dir=None, filename="performance_report.html"):
        """
        Generate an HTML performance report for all factors.
        
        Parameters:
        -----------
        factors_data : dict
            Dictionary containing factor performance data
        output_dir : str or Path, optional
            Directory to save the report
        filename : str, optional
            Filename for the saved report
        """
        if not factors_data:
            print("No data to generate report")
            return
        
        # Create a DataFrame to hold the performance metrics
        metrics_rows = []
        
        for key, data in factors_data.items():
            if 'metrics' not in data:
                continue
            
            # Prepare metrics row
            row = {
                'Factor': data['display_name'],
                'Test': data['test_name'],
                'Process': data['process_name'],
                'Sharpe Ratio': data['metrics']['sharpe_ratio'],
                'Annualized Return': data['metrics']['return_annualized'],
                'Max Drawdown': data['metrics']['max_dd'],
                'Profit Per Trade': data['profit_per_trade'],
                'Calmar Ratio': data['metrics']['calmar_ratio'],
                'Sortino Ratio': data['metrics']['sortino_ratio'],
                'Sterling Ratio': data['metrics']['sterling_ratio'],
                'Burke Ratio': data['metrics']['burke_ratio'],
                'Ulcer Index': data['metrics']['ulcer_index'],
                'Drawdown Recovery Ratio': data['metrics']['drawdown_recovery_ratio'],
            }
            
            metrics_rows.append(row)
        
        # Create DataFrame from rows
        metrics_df = pd.DataFrame(metrics_rows)
        
        # Sort by Sharpe Ratio descending
        metrics_df = metrics_df.sort_values('Sharpe Ratio', ascending=False)
        
        # Format the metrics
        formatted_df = metrics_df.copy()
        formatted_df['Annualized Return'] = formatted_df['Annualized Return'].map('{:.2%}'.format)
        formatted_df['Max Drawdown'] = formatted_df['Max Drawdown'].map('{:.2%}'.format)
        formatted_df['Profit Per Trade'] = formatted_df['Profit Per Trade'].map('{:.4%}'.format)
        for col in ['Sharpe Ratio', 'Calmar Ratio', 'Sortino Ratio', 
                    'Sterling Ratio', 'Burke Ratio', 'Drawdown Recovery Ratio']:
            formatted_df[col] = formatted_df[col].map('{:.2f}'.format)
        formatted_df['Ulcer Index'] = formatted_df['Ulcer Index'].map('{:.2f}'.format)
        
        # Generate HTML
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Factor Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                tr:hover { background-color: #f5f5f5; }
                .positive { color: green; }
                .negative { color: red; }
            </style>
        </head>
        <body>
            <h1>Factor Performance Report</h1>
            <table>
                <tr>
        """
        
        # Add table headers
        for col in formatted_df.columns:
            html += f"<th>{col}</th>"
        html += "</tr>"
        
        # Add table rows
        for _, row in formatted_df.iterrows():
            html += "<tr>"
            for i, col in enumerate(formatted_df.columns):
                value = row[col]
                if i >= 3:  # Skip the first three columns (Factor, Test, Process)
                    # Check if the original value is positive or negative
                    original_value = metrics_df.iloc[_].iloc[i]
                    if pd.notna(original_value):
                        css_class = "positive" if original_value > 0 else "negative"
                        html += f'<td class="{css_class}">{value}</td>'
                    else:
                        html += f"<td>{value}</td>"
                else:
                    html += f"<td>{value}</td>"
            html += "</tr>"
        
        html += """
            </table>
            <p><small>Generated by FactorPerformanceComparison tool</small></p>
        </body>
        </html>
        """
        
        # Save the HTML report
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(output_path / filename, 'w') as f:
                f.write(html)
            
            print(f"Performance report saved to {output_path / filename}")
        
        return html
    
    def run_comparison(self, factor_info_list, compare_name, date_start=None, date_end=None, include_plots=True):
        """
        Run a comprehensive factor performance comparison.
        
        Parameters:
        -----------
        factor_info_list : list of tuple
            List of tuples with (tag_name, process_name, factor_name, test_name)
        compare_name : str
            Name for this comparison (used for folder naming)
        date_start : str or datetime, optional
            Start date for filtering data
        date_end : str or datetime, optional
            End date for filtering data
        include_plots : bool, optional
            Whether to generate and save plots
            
        Returns:
        --------
        dict
            Dictionary containing performance data and metrics
        """
        # Load factor data
        factors_data = self.load_factor_data(factor_info_list, date_start, date_end)
        
        if not factors_data:
            print("No valid factor data found")
            return {}
        
        # Create output directory based on path_config and compare_name
        output_path = Path(self.path_config['result']) / 'analysis' / 'compare_factors' / compare_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate performance report
        self.generate_performance_report(factors_data, output_path)
        
        if include_plots:
            # Generate plots
            self.plot_cumulative_returns(factors_data, output_path)
            self.plot_yearly_performance(factors_data, output_path)
            self.plot_yearly_cumulative(factors_data, output_path)
            self.plot_monthly_heatmap(factors_data, output_path)
            self.plot_drawdown_analysis(factors_data, output_path)
        
        return factors_data


# Example usage
if __name__ == "__main__":
    # Define the factors to compare
    compare_name = 'LargeOA_rm_bf_open'
    factor_info_list = [
        # (tag_name, process_name, factor_name, test_name)
        ("zxt", "Batch10_fix_best_241218_selected_f64/v1.2_longer_wd", "LargeOrderAmountByValue_p1.0_v40000-avg_imb01_dp2-rollingAggMinuteMinMaxScale_w245d_q0.02_i5", "trade_ver3_intraday_futtwap_sp1min_s240d_icim"),
        ("zxt", "Batch21_250512/normal_trans_for_test_250414", "LargeOARmAuc_p1.0_v40000_f25-avg_imb01_dp2-rollingAggMinuteMinMaxScale_w245d_q0.02_i5", "trade_ver3_intraday_futtwap_sp1min_s240d_icim"),
        ("zxt", "Batch21_250512/normal_trans_for_test_250414", "LargeOARmAuc_p1.0_v40000_f30-avg_imb01_dp2-rollingAggMinuteMinMaxScale_w245d_q0.02_i5", "trade_ver3_intraday_futtwap_sp1min_s240d_icim"),
        # Add more factors as needed
    ]
    
    # Initialize the comparison tool
    comparator = FactorPerformanceComparison(fee=2.4e-4)
    
    # Run the comparison
    results = comparator.run_comparison(
        factor_info_list=factor_info_list,
        compare_name=compare_name,  # This will be used for the output folder name
        date_start="2026-01-01",  # Optional date filter
        date_end="2025-02-01",    # Optional date filter
    )