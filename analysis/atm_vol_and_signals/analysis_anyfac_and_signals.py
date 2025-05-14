# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:36:37 2025

@author: Based on code by Xintang Zheng

Analysis module for examining relationships between arbitrary factors and trading signals,
with separate analysis for long and short positions.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os
from matplotlib.gridspec import GridSpec

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_trade_dfs(load_dir, factor_name):
    """
    Load trade DataFrames from the specified directory
    
    Parameters:
    load_dir (Path): Directory containing trade data
    factor_name (str): Name of the factor directory
    
    Returns:
    dict: Dictionary of trade DataFrames by instrument
    """
    factor_load_dir = load_dir / factor_name
    
    trade_dfs = {}
    instruments = ['IC', 'IF', 'IM']
    
    for instrument in instruments:
        load_path = factor_load_dir / f"{instrument}_trades.parquet"
        if load_path.exists():
            df = pd.read_parquet(load_path)
            
            # Ensure numeric data types for columns that should be numeric
            numeric_columns = ['net_return']
            for col in df.columns:
                if col in numeric_columns or col.startswith('factor_') or col.endswith('_vol'):
                    if df[col].dtype == 'object':
                        # Try to convert to float
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except Exception as e:
                            logger.warning(f"Failed to convert column {col} to numeric: {e}")
            
            trade_dfs[instrument] = df
            logger.info(f"Loaded trade records for {instrument} from {load_path}")
        else:
            logger.warning(f"No trade records found for {instrument} at {load_path}")
            trade_dfs[instrument] = pd.DataFrame()  # Empty DataFrame
    
    return trade_dfs

def plot_factor_return_correlation(trade_dfs, save_dir, factor_short_name, factor_full_name, feature_name, feature_col_name):
    """
    Plot the relationship between a factor and net return for each instrument, split by trade direction
    
    Parameters:
    trade_dfs (dict): Dictionary of trade DataFrames by instrument
    save_dir (Path): Directory to save visualizations to
    factor_short_name (str): Short name (key) of the factor for file naming
    factor_full_name (str): Full name (value) of the factor for plot titles
    feature_name (str): Name of the feature being analyzed
    feature_col_name (str): Column name for the feature
    """
    # Create output directory
    output_dir = save_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the style
    sns.set(style="whitegrid")
    
    # Create a figure with 3 columns (one per instrument) and 2 rows (long/short)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Define the instruments
    instruments = ['IC', 'IF', 'IM']
    trade_types = ['long', 'short']
    
    # Plot each instrument and trade type
    for i, trade_type in enumerate(trade_types):
        for j, instrument in enumerate(instruments):
            # Get the dataframe for the current instrument
            df = trade_dfs[instrument]
            
            if df.empty:
                axes[i, j].set_title(f'{instrument} {trade_type} (No data available)', fontsize=14)
                axes[i, j].set_xlabel(f'{feature_name}', fontsize=12)
                if j == 0:
                    axes[i, j].set_ylabel('Net Return', fontsize=12)
                continue
            
            # Filter by trade type
            df_filtered = df[df['trade_type'] == trade_type]
            
            if df_filtered.empty:
                axes[i, j].set_title(f'{instrument} {trade_type} (No {trade_type} trades)', fontsize=14)
                axes[i, j].set_xlabel(f'{feature_name}', fontsize=12)
                if j == 0:
                    axes[i, j].set_ylabel('Net Return', fontsize=12)
                continue
            
            # Drop rows with NaN values in either feature or net_return
            df_clean = df_filtered.dropna(subset=[feature_name, 'net_return'])
            
            # Create scatter plot
            sns.scatterplot(
                x=feature_name, 
                y='net_return',
                data=df_clean,
                alpha=0.6,
                s=40,
                ax=axes[i, j]
            )
            
            # Add a trend line
            sns.regplot(
                x=feature_name,
                y='net_return',
                data=df_clean,
                scatter=False,
                ax=axes[i, j],
                line_kws={'color': 'red', 'linewidth': 2}
            )
            
            # Calculate correlation
            corr = df_clean[feature_name].corr(df_clean['net_return'])
            
            # Set title and labels
            axes[i, j].set_title(f'{instrument} {trade_type} (n={len(df_clean)}, corr={corr:.4f})', fontsize=14)
            axes[i, j].set_xlabel(f'{feature_name}', fontsize=12)
            
            # Only set y-label for the first subplot in each row
            if j == 0:
                axes[i, j].set_ylabel('Net Return', fontsize=12)
            else:
                axes[i, j].set_ylabel('')
    
    # Adjust layout
    plt.tight_layout()
    plot_title = f'Factor: {factor_full_name} - {feature_name} {feature_col_name}\nRelationship between {feature_name} and Net Return by Instrument and Direction'
    plt.suptitle(plot_title, fontsize=16, y=1.05)
    
    # Save the plot using short factor name
    file_name = f"{factor_short_name}_{feature_name}_return_corr.png"
    
    # Create full save path
    save_path = output_dir / file_name
    
    # Try saving
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved factor-return correlation plot to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
    
    # Close the figure to free memory
    plt.close(fig)

def plot_factor_metrics_by_direction(trade_dfs, save_dir, factor_short_name, factor_full_name, feature_name, feature_col_name, n_bins=10):
    """
    Plot trade count, win rate, and sum return across factor bins for each instrument, split by trade direction
    
    Parameters:
    trade_dfs (dict): Dictionary of trade DataFrames by instrument
    save_dir (Path): Directory to save visualizations to
    factor_short_name (str): Short name (key) of the factor for file naming
    factor_full_name (str): Full name (value) of the factor for plot titles
    feature_name (str): Name of the feature being analyzed
    feature_col_name (str): Column name for the feature
    n_bins (int): Number of bins to use for the factor
    """
    # Create output directory
    output_dir = save_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelweight': 'bold'
    })
    
    instruments = ['IC', 'IF', 'IM']
    trade_types = ['long', 'short']
    metrics = ['Trade Count', 'Win Rate', 'Sum Return', 'Avg Return']
    
    # Create a larger figure with GridSpec for more control
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1])
    
    # Color palettes for long and short trades
    palette_long = {
        'Trade Count': '#6FA8DC',
        'Win Rate': '#F6A623',
        'Sum Return': '#7AC29A',
        'Avg Return': '#4CAF50'
    }
    
    palette_short = {
        'Trade Count': '#E06666',
        'Win Rate': '#8E7CC3',
        'Sum Return': '#FFD966',
        'Avg Return': '#D5A6BD'
    }
    
    # Collect all avg_return values to determine global min/max for consistent y-axis scaling
    all_avg_returns = []
    all_win_rates = []
    
    # First pass - collect all data to calculate appropriate y-axis limits
    for trade_type in trade_types:
        for instrument in instruments:
            df = trade_dfs[instrument]
            if df.empty:
                continue
                
            df_filtered = df[df['trade_type'] == trade_type]
            if df_filtered.empty:
                continue
                
            df_clean = df_filtered.dropna(subset=[feature_name, 'net_return'])
            if len(df_clean) == 0:
                continue
                
            # Calculate avg returns for each bin
            bin_edges = np.linspace(df_clean[feature_name].min(), df_clean[feature_name].max(), n_bins + 1)
            bin_labels = [f'{edge:.4f}' for edge in bin_edges[:-1]]
            df_clean['factor_bin'] = pd.cut(df_clean[feature_name], bins=bin_edges, labels=bin_labels, include_lowest=True)
            
            for bin_label in bin_labels:
                bin_data = df_clean[df_clean['factor_bin'] == bin_label]
                if len(bin_data) > 0:
                    avg_return = bin_data['net_return'].mean() * 100
                    win_rate = (bin_data['net_return'] > 0).mean() * 100
                    all_avg_returns.append(avg_return)
                    all_win_rates.append(win_rate)
    
    # Calculate global y-axis limits with a 20% margin
    if all_avg_returns:
        avg_return_min = min(all_avg_returns)
        avg_return_max = max(all_avg_returns)
        win_rate_min = min(all_win_rates)
        win_rate_max = max(all_win_rates)
        
        # Ensure there's always space around the values
        y_margin = max(0.2 * (avg_return_max - avg_return_min), 0.2 * (win_rate_max - win_rate_min), 1.0)
        
        # Ensure we always include 0 and 50% (for win rate) in the range
        global_y2_min = min(0, avg_return_min, win_rate_min) - y_margin
        global_y2_max = max(50, avg_return_max, win_rate_max) + y_margin
    else:
        # Default range if no data
        global_y2_min = -5
        global_y2_max = 55
    
    # Second pass - create the plots with consistent y-axis scaling
    for row_idx, trade_type in enumerate(trade_types):
        palette = palette_long if trade_type == 'long' else palette_short
        
        for i, instrument in enumerate(instruments):
            df = trade_dfs[instrument]
            
            if df.empty:
                continue
                
            # Filter by trade type
            df_filtered = df[df['trade_type'] == trade_type]
            
            if df_filtered.empty:
                continue
                
            df_clean = df_filtered.dropna(subset=[feature_name, 'net_return'])
            
            if len(df_clean) == 0:
                continue
                
            # Calculate bins based on the factor's distribution
            bin_edges = np.linspace(df_clean[feature_name].min(), df_clean[feature_name].max(), n_bins + 1)
            bin_labels = [f'{edge:.4f}' for edge in bin_edges[:-1]]
            
            df_clean['factor_bin'] = pd.cut(df_clean[feature_name], bins=bin_edges, labels=bin_labels, include_lowest=True)
            
            bin_stats = []
            for bin_label in bin_labels:
                bin_data = df_clean[df_clean['factor_bin'] == bin_label]
                if len(bin_data) > 0:
                    trade_count = len(bin_data)
                    win_count = len(bin_data[bin_data['net_return'] > 0])
                    win_rate = win_count / trade_count if trade_count > 0 else 0
                    sum_return = bin_data['net_return'].sum() * 100
                    avg_return = bin_data['net_return'].mean() * 100
                    bin_stats.append({
                        'Factor Bin': bin_label,
                        'Trade Count': trade_count,
                        'Win Rate': win_rate * 100,
                        'Sum Return': sum_return,
                        'Avg Return': avg_return
                    })
            
            bin_stats_df = pd.DataFrame(bin_stats)
            
            # Plot all metrics for this instrument and trade type in a single subplot
            ax = fig.add_subplot(gs[row_idx, i])
            
            # Create a twin axis for different scales
            ax2 = ax.twinx()
            
            # Plot Trade Count on the primary y-axis
            sns.barplot(
                x='Factor Bin', 
                y='Trade Count', 
                data=bin_stats_df, 
                color=palette['Trade Count'],
                ax=ax, 
                alpha=0.7,
                width=0.8
            )
            
            # Plot Win Rate on the secondary y-axis
            win_rate_line = ax2.plot(
                bin_stats_df.index, 
                bin_stats_df['Win Rate'], 
                marker='o', 
                color=palette['Win Rate'], 
                linewidth=2, 
                label='Win Rate (%)'
            )
            
            # Plot Avg Return on the secondary y-axis
            avg_return_line = ax2.plot(
                bin_stats_df.index, 
                bin_stats_df['Avg Return'], 
                marker='s', 
                color=palette['Avg Return'], 
                linewidth=2, 
                label='Avg Return (%)'
            )
            
            # Use the consistent y-axis limits calculated earlier
            ax2.set_ylim(global_y2_min, global_y2_max)
            
            # Add a horizontal line at y=0 for the secondary axis
            ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax2.axhline(y=50, color='gray', linestyle=':', linewidth=1)  # 50% win rate line
            
            # Highlight the specific avg_return values with labels
            for idx, value in enumerate(bin_stats_df['Avg Return']):
                ax2.annotate(
                    f'{value:.2f}%', 
                    xy=(idx, value),
                    xytext=(0, 10),  # 10 points vertical offset
                    textcoords='offset points',
                    ha='center', 
                    va='bottom',
                    color=palette['Avg Return'],
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
                )
            
            # Set titles and labels
            ax.set_title(f'{instrument} - {trade_type.capitalize()} Trades', fontsize=14, fontweight='bold')
            ax.set_xlabel(f'{feature_name} Bins', fontsize=11)
            ax.set_ylabel('Trade Count', fontsize=11, color=palette['Trade Count'])
            ax2.set_ylabel('Percentage (%)', fontsize=11)
            
            # Set x-ticks every other bin to avoid crowding
            tick_positions = list(range(0, len(bin_labels), 2))
            tick_labels = [bin_labels[i] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45)
            
            # Create a legend for the lines
            lines = win_rate_line + avg_return_line
            labels = [line.get_label() for line in lines]
            ax2.legend(lines, labels, loc='upper right')
    
    # Add a shared subplot for bin ranges
    ax_info = fig.add_subplot(gs[:, 3])
    ax_info.axis('off')
    
    # Create a summary of the bins
    bin_info = []
    for bin_label, (lower, upper) in zip(bin_labels, zip(bin_edges[:-1], bin_edges[1:])):
        bin_info.append(f"Bin {bin_label}: [{lower:.4f}, {upper:.4f})")
    
    bin_text = "\n".join(bin_info)
    bin_title = f"{feature_name} Bin Ranges"
    
    # Create the detail section with bin ranges
    ax_info.text(0.1, 0.9, bin_title, fontsize=14, fontweight='bold')
    ax_info.text(0.1, 0.8, bin_text, fontsize=10, fontfamily='monospace', verticalalignment='top')
    
    # Add a section for summary statistics
    summary_title = "Summary Statistics"
    ax_info.text(0.1, 0.4, summary_title, fontsize=14, fontweight='bold')
    
    # Calculate and display summary stats for each instrument and direction
    summary_texts = []
    for trade_type in trade_types:
        for instrument in instruments:
            df = trade_dfs[instrument]
            if df.empty:
                continue
            
            df_filtered = df[df['trade_type'] == trade_type]
            if df_filtered.empty:
                continue
                
            total_trades = len(df_filtered)
            win_rate = (df_filtered['net_return'] > 0).mean() * 100
            total_return = df_filtered['net_return'].sum() * 100
            avg_return = df_filtered['net_return'].mean() * 100
            
            summary_texts.append(
                f"{instrument} {trade_type.capitalize()}: {total_trades} trades, "
                f"Win Rate: {win_rate:.2f}%, "
                f"Total Return: {total_return:.2f}%, "
                f"Avg Return: {avg_return:.2f}%"
            )
    
    summary_text = "\n".join(summary_texts)
    ax_info.text(0.1, 0.35, summary_text, fontsize=10, verticalalignment='top')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_title = f'Factor: {factor_full_name} - {feature_name} {feature_col_name}\nTrade Metrics by Factor Bins'
    plt.suptitle(plot_title, fontsize=16, fontweight='bold', y=0.98)
    
    # Save the plot using short factor name
    file_name = f"{factor_short_name}_{feature_name}_metrics_by_direction.png"
    
    # Create full save path
    save_path = output_dir / file_name
    
    # Try saving
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved factor metrics plot to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
    
    # Close the figure to free memory
    plt.close(fig)

def plot_factor_performance_heatmap(trade_dfs, save_dir, factor_short_name, factor_full_name, feature_name, feature_col_name, n_bins=8):
    """
    Create heatmaps showing performance metrics across factor bins for each instrument and trade direction
    
    Parameters:
    trade_dfs (dict): Dictionary of trade DataFrames by instrument
    save_dir (Path): Directory to save visualizations to
    factor_short_name (str): Short name (key) of the factor for file naming
    factor_full_name (str): Full name (value) of the factor for plot titles
    feature_name (str): Name of the feature being analyzed
    feature_col_name (str): Column name for the feature
    n_bins (int): Number of bins to use for the factor
    """
    # Create output directory
    output_dir = save_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    instruments = ['IC', 'IF', 'IM']
    trade_types = ['long', 'short']
    metrics = ['Trade Count', 'Win Rate (%)', 'Avg Return (%)', 'Sum Return (%)']
    
    # Create a figure
    fig, axes = plt.subplots(len(instruments), len(trade_types), figsize=(16, 12))
    
    for i, instrument in enumerate(instruments):
        df = trade_dfs[instrument]
        
        if df.empty:
            for j, trade_type in enumerate(trade_types):
                axes[i, j].text(0.5, 0.5, f"No data for {instrument}", 
                               ha='center', va='center', fontsize=14)
                axes[i, j].set_title(f"{instrument} - {trade_type.capitalize()}")
                axes[i, j].axis('off')
            continue
        
        for j, trade_type in enumerate(trade_types):
            df_filtered = df[df['trade_type'] == trade_type]
            
            if df_filtered.empty:
                axes[i, j].text(0.5, 0.5, f"No {trade_type} trades for {instrument}", 
                               ha='center', va='center', fontsize=14)
                axes[i, j].set_title(f"{instrument} - {trade_type.capitalize()}")
                axes[i, j].axis('off')
                continue
            
            df_clean = df_filtered.dropna(subset=[feature_name, 'net_return'])
            
            if len(df_clean) == 0:
                axes[i, j].text(0.5, 0.5, f"No valid data for {instrument} {trade_type}", 
                               ha='center', va='center', fontsize=14)
                axes[i, j].set_title(f"{instrument} - {trade_type.capitalize()}")
                axes[i, j].axis('off')
                continue
            
            # Calculate bins based on the factor's distribution
            bin_edges = np.linspace(df_clean[feature_name].min(), df_clean[feature_name].max(), n_bins + 1)
            bin_labels = [f'{i+1}' for i in range(n_bins)]  # Simple numeric labels
            
            df_clean['factor_bin'] = pd.cut(df_clean[feature_name], bins=bin_edges, labels=bin_labels, include_lowest=True)
            
            # Create a DataFrame for the heatmap
            heatmap_data = []
            
            for bin_label in bin_labels:
                bin_data = df_clean[df_clean['factor_bin'] == bin_label]
                if len(bin_data) > 0:
                    trade_count = len(bin_data)
                    win_rate = (bin_data['net_return'] > 0).mean() * 100
                    avg_return = bin_data['net_return'].mean() * 100
                    sum_return = bin_data['net_return'].sum() * 100
                    
                    bin_min = bin_edges[int(bin_label) - 1]
                    bin_max = bin_edges[int(bin_label)]
                    bin_range = f"[{bin_min:.4f}, {bin_max:.4f})"
                    
                    heatmap_data.append({
                        'Bin': bin_label,
                        'Range': bin_range,
                        'Trade Count': trade_count,
                        'Win Rate (%)': win_rate,
                        'Avg Return (%)': avg_return,
                        'Sum Return (%)': sum_return
                    })
            
            heatmap_df = pd.DataFrame(heatmap_data)
            
            if heatmap_df.empty:
                axes[i, j].text(0.5, 0.5, f"Insufficient data for {instrument} {trade_type}", 
                               ha='center', va='center', fontsize=14)
                axes[i, j].set_title(f"{instrument} - {trade_type.capitalize()}")
                axes[i, j].axis('off')
                continue
            
            # Create a table-like visualization
            axes[i, j].axis('tight')
            axes[i, j].axis('off')
            
            # Define colors for the metrics
            cell_colors = []
            for _, row in heatmap_df.iterrows():
                row_colors = []
                for metric in metrics:
                    if metric == 'Trade Count':
                        # Blue scale for trade count
                        intensity = min(1.0, row[metric] / heatmap_df['Trade Count'].max())
                        row_colors.append((0.7 - 0.5 * intensity, 0.7 - 0.3 * intensity, 1.0))
                    elif metric == 'Win Rate (%)':
                        # Yellow to green scale centered at 50%
                        if row[metric] >= 50:
                            # Green for >50%
                            intensity = min(1.0, (row[metric] - 50) / 50)
                            row_colors.append((0.7 - 0.7 * intensity, 0.9, 0.7 - 0.7 * intensity))
                        else:
                            # Yellow to red for <50%
                            intensity = min(1.0, (50 - row[metric]) / 50)
                            row_colors.append((1.0, 1.0 - 0.7 * intensity, 0.7 - 0.7 * intensity))
                    elif metric == 'Avg Return (%)' or metric == 'Sum Return (%)':
                        # Red to green scale centered at 0
                        if row[metric] >= 0:
                            # Green for positive returns
                            intensity = min(1.0, row[metric] / max(0.01, heatmap_df[metric].max()))
                            row_colors.append((0.9 - 0.6 * intensity, 0.9, 0.9 - 0.6 * intensity))
                        else:
                            # Red for negative returns
                            intensity = min(1.0, abs(row[metric]) / max(0.01, abs(heatmap_df[metric].min())))
                            row_colors.append((1.0, 1.0 - 0.6 * intensity, 1.0 - 0.6 * intensity))
                cell_colors.append(row_colors)
            
            # Create the table
            table = axes[i, j].table(
                cellText=heatmap_df[metrics].values,
                rowLabels=heatmap_df['Range'].values,
                colLabels=metrics,
                cellColours=cell_colors,
                loc='center',
                cellLoc='center'
            )
            
            # Adjust table style
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            # Add title
            axes[i, j].set_title(f"{instrument} - {trade_type.capitalize()}", pad=20)
    
    # Adjust layout
    plt.tight_layout()
    plot_title = f'Factor: {factor_full_name} - {feature_name} {feature_col_name}\nPerformance Metrics by Factor Bins'
    plt.suptitle(plot_title, fontsize=16, y=1.02)
    
    # Save the plot using short factor name
    file_name = f"{factor_short_name}_{feature_name}_heatmap.png"
    
    # Create full save path
    save_path = output_dir / file_name
    
    # Try saving
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance heatmap to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
    
    # Close the figure to free memory
    plt.close(fig)

def visualize_factor_signals(load_dir, save_dir, factor_dict, feature_name, feature_col_name, n_bins=10):
    """
    Main function to load trade data, generate visualizations, and save them
    
    Parameters:
    load_dir (Path): Directory to load trade DataFrames from
    save_dir (Path): Directory to save visualizations to
    factor_dict (dict): Dictionary mapping short names (keys) to full factor names (values)
    feature_name (str): Name of the feature being analyzed
    feature_col_name (str): Column name for the feature
    n_bins (int): Number of bins to use for analysis
    """
    for factor_short_name, factor_full_name in factor_dict.items():
        # Create the factor-specific save directory using the short name
        factor_save_dir = save_dir / factor_short_name
        factor_save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing factor: '{factor_short_name}' (full name: '{factor_full_name}')")
        
        # Load trade DataFrames using the full name
        trade_dfs = load_trade_dfs(load_dir, factor_full_name)
        
        # Generate and save visualizations using the short name for file names and full name for titles
        plot_factor_return_correlation(trade_dfs, factor_save_dir, factor_short_name, factor_full_name, feature_name, feature_col_name)
        plot_factor_metrics_by_direction(trade_dfs, factor_save_dir, factor_short_name, factor_full_name, feature_name, feature_col_name, n_bins)
        plot_factor_performance_heatmap(trade_dfs, factor_save_dir, factor_short_name, factor_full_name, feature_name, feature_col_name, n_bins)
        
        logger.info(f"Completed visualization for factor '{factor_short_name}'")

# Example usage
if __name__ == "__main__":
    # Example parameters
    load_dir = Path(r'/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/analysis/trades')
    save_dir = Path(r'/mnt/Data/xintang/CNIndexFutures/timeseries/factor_test/results/analysis/analysis_basis_and_signals')
    
    # feature_name = "atm_vol"
    # feature_col_name = "IO"

    feature_name = "IC_z1-rollingAggMinuteMinMaxScale_w245d_q0.02_i5"
    feature_col_name = "z1"
    
    # Dictionary mapping short names (keys) to full factor names (values)
    factor_dict = {
        # 'ValueTimeDecay': 'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb04_dp2-org',
        # 'LargeOrderAmount': 'LargeOrderAmountByValue_p1.0_v40000-avg_imb01_dp2-org',
        # 'Jump': 'IntraRm_m30_IntraRelQtl_d20_q002_dp2_wstr_jump5mVwap',
        # 'OrderAmt_Dollar_LX': 'order_amount_Dollar_LX_R3_dp2_SumIntraRm5_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15',
        # 'OrderAmt_Dollar': 'order_amount_Dollar_R3_dp2_SumIntraRm5_LXPct_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15',
        # 'TradeAmt_Dollar_LX': 'trade_amount_Dollar_LX_R3_dp2_SumIntraRm5_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15',
        # 'TradeAmt_Dollar': 'trade_amount_Dollar_R3_dp2_SumIntraRm5_LXPct_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15'
        'TimeRangeValueOrderAmount_p1.0_v200000_t30': 'TimeRangeValueOrderAmount_p1.0_v200000_t30-wavg_imb01_dp2-rollingAggMinutePctl_w245d_i5',
    }

    # Run the analysis
    visualize_factor_signals(load_dir, save_dir, factor_dict, feature_name, feature_col_name, n_bins=10)