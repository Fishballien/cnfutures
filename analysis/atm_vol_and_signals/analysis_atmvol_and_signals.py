# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 15:36:37 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_trade_dfs(load_dir, factor_name):
    """Load trade DataFrames from the specified directory"""
    factor_load_dir = load_dir / factor_name
    
    trade_dfs = {}
    instruments = ['IC', 'IF', 'IM']
    
    for instrument in instruments:
        load_path = factor_load_dir / f"{instrument}_trades.parquet"
        if load_path.exists():
            df = pd.read_parquet(load_path)
            
            # Ensure numeric data types for columns that should be numeric
            numeric_columns = ['atm_vol', 'net_return']
            for col in numeric_columns:
                if col in df.columns and df[col].dtype == 'object':
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

def plot_volatility_return_correlation(trade_dfs, save_dir, factor_short_name, factor_full_name, feature_name, feature_col_name):
    """
    Plot the relationship between ATM volatility and net return for each instrument
    
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
    
    # Create a figure with 3 subplots horizontally
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define the instruments
    instruments = ['IC', 'IF', 'IM']
    
    # Plot each instrument
    for i, instrument in enumerate(instruments):
        # Get the dataframe for the current instrument
        df = trade_dfs[instrument]
        
        if df.empty:
            axes[i].set_title(f'{instrument} (No data available)', fontsize=14)
            axes[i].set_xlabel('ATM Volatility', fontsize=12)
            if i == 0:
                axes[i].set_ylabel('Net Return', fontsize=12)
            continue
        
        # Drop rows with NaN values in either atm_vol or net_return
        df_clean = df.dropna(subset=['atm_vol', 'net_return'])
        
        # Create scatter plot
        sns.scatterplot(
            x='atm_vol', 
            y='net_return',
            hue='trade_type',  # Color by trade type (long/short)
            data=df_clean,
            alpha=0.6,
            s=40,
            ax=axes[i]
        )
        
        # Add a trend line
        sns.regplot(
            x='atm_vol',
            y='net_return',
            data=df_clean,
            scatter=False,
            ax=axes[i],
            line_kws={'color': 'red', 'linewidth': 2}
        )
        
        # Calculate correlation
        corr = df_clean['atm_vol'].corr(df_clean['net_return'])
        
        # Set title and labels
        axes[i].set_title(f'{instrument} (n={len(df_clean)}, corr={corr:.4f})', fontsize=14)
        axes[i].set_xlabel('ATM Volatility', fontsize=12)
        
        # Only set y-label for the first subplot
        if i == 0:
            axes[i].set_ylabel('Net Return', fontsize=12)
        else:
            axes[i].set_ylabel('')
    
    # Adjust layout
    plt.tight_layout()
    plot_title = f'Factor: {factor_full_name} - {feature_name} {feature_col_name}\nRelationship between ATM Volatility and Net Return by Instrument'
    plt.suptitle(plot_title, fontsize=16, y=1.05)
    
    # Save the plot using short factor name
    file_name = f"{factor_short_name}_vol_return_corr.png"
    
    # Create full save path
    save_path = output_dir / file_name
    
    # Try saving
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved volatility-return correlation plot to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
    
    # Close the figure to free memory
    plt.close(fig)

def plot_volatility_metrics(trade_dfs, save_dir, factor_short_name, factor_full_name, feature_name, feature_col_name):
    """
    Plot trade count, win rate, and sum return across volatility bins for each instrument
    
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
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelweight': 'bold'
    })
    
    instruments = ['IC', 'IF', 'IM']
    metrics = ['Trade Count', 'Win Rate', 'Sum Return']
    
    bin_edges = np.arange(0.005, 0.07, 0.0025)
    bin_labels = [f'{edge*100:.2f}' for edge in bin_edges[:-1]]
    
    palette_dict = {
        'Trade Count': '#6FA8DC',
        'Win Rate': '#F6A623',
        'Sum Return': '#7AC29A'
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex='col')
    
    for i, instrument in enumerate(instruments):
        df = trade_dfs[instrument]
        
        if df.empty:
            for j in range(3):
                axes[i, j].set_title(f"No data for {instrument}", fontsize=14)
            continue
            
        df_clean = df.dropna(subset=['atm_vol', 'net_return'])
        df_clean['vol_bin'] = pd.cut(df_clean['atm_vol'], bins=bin_edges, labels=bin_labels)
        
        bin_stats = []
        for bin_label in bin_labels:
            bin_data = df_clean[df_clean['vol_bin'] == bin_label]
            if len(bin_data) > 0:
                trade_count = len(bin_data)
                win_count = len(bin_data[bin_data['net_return'] > 0])
                win_rate = win_count / trade_count if trade_count > 0 else 0
                sum_return = bin_data['net_return'].sum() * 100
                bin_stats.append({
                    'Vol Bin': bin_label,
                    'Trade Count': trade_count,
                    'Win Rate': win_rate * 100,
                    'Sum Return': sum_return
                })
        
        bin_stats_df = pd.DataFrame(bin_stats)
        if len(bin_stats) == 0:
            bin_stats_df = pd.DataFrame(columns=['Vol Bin', 'Trade Count', 'Win Rate', 'Sum Return'])
            
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            if not bin_stats_df.empty:
                sns.barplot(
                    x='Vol Bin', 
                    y=metric, 
                    data=bin_stats_df, 
                    color=palette_dict[metric],
                    ax=ax, 
                    width=0.7
                )
                for bar in ax.patches:
                    if not np.isnan(bar.get_height()):
                        label_value = f"{int(bar.get_height())}" if metric == 'Trade Count' else f"{int(bar.get_height())}%"
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3,
                            label_value,
                            ha='center', va='bottom',
                            fontsize=9
                        )
            
            # Adjust title and axes
            if i == 0:
                ax.set_title(metric, fontsize=14, fontweight='bold', pad=20)
            if j == 0:
                ax.set_ylabel(f'{instrument}', fontsize=13, fontweight='bold')
            else:
                ax.set_ylabel('')
            if i == 2:
                ax.set_xlabel('Volatility Bin', fontsize=11)
            else:
                ax.set_xlabel('')
            
            ax.tick_params(axis='x', rotation=45)
            
            if metric == 'Win Rate':
                ax.set_ylim(0, 100)
            elif metric == 'Sum Return':
                min_return = min([-1, bin_stats_df['Sum Return'].min() if not bin_stats_df.empty else 0]) - 0.5
                max_return = max([1, bin_stats_df['Sum Return'].max() if not bin_stats_df.empty else 0]) + 0.5
                ax.set_ylim(min_return, max_return)
                ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    # Adjust layout
    plt.tight_layout(pad=3.5)
    plt.subplots_adjust(top=0.90, bottom=0.1, hspace=0.1, wspace=0.1)
    plot_title = f'Factor: {factor_full_name}\nby feature: {feature_name} {feature_col_name} bin'
    plt.suptitle(plot_title, fontsize=18, fontweight='bold', y=0.99)
    
    # Save the plot using short factor name
    file_name = f"{factor_short_name}_vol_metrics.png"
    
    # Create full save path
    save_path = output_dir / file_name
    
    # Try saving
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved volatility metrics plot to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
    
    # Close the figure to free memory
    plt.close(fig)

def visualize_trades(load_dir, save_dir, factor_dict, feature_name, feature_col_name):
    """
    Main function to load trade data, generate visualizations, and save them
    
    Parameters:
    load_dir (Path): Directory to load trade DataFrames from
    save_dir (Path): Directory to save visualizations to
    factor_dict (dict): Dictionary mapping short names (keys) to full factor names (values)
    feature_name (str): Name of the feature being analyzed
    feature_col_name (str): Column name for the feature
    """
    for factor_short_name, factor_full_name in factor_dict.items():
        # Create the factor-specific save directory using the short name
        factor_save_dir = save_dir / factor_short_name
        factor_save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing factor: '{factor_short_name}' (full name: '{factor_full_name}')")
        
        # Load trade DataFrames using the full name
        trade_dfs = load_trade_dfs(load_dir, factor_full_name)
        
        # Generate and save visualizations using the short name for file names and full name for titles
        plot_volatility_return_correlation(trade_dfs, factor_save_dir, factor_short_name, factor_full_name, feature_name, feature_col_name)
        plot_volatility_metrics(trade_dfs, factor_save_dir, factor_short_name, factor_full_name, feature_name, feature_col_name)
        
        logger.info(f"Completed visualization for factor '{factor_short_name}'")

# Example usage
if __name__ == "__main__":
    # Example parameters
    load_dir = Path(r'D:/mnt/CNIndexFutures/timeseries/factor_test/results/analysis/trades')
    save_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\filter_signals\options\analysis_atmvol_and_signals')
    
    feature_name = "atm_vol"
    feature_col_name = "IO"
    
    # Dictionary mapping short names (keys) to full factor names (values)
    factor_dict = {
        # 'ValueTimeDecay': 'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb04_dp2-org',
        # 'LargeOrderAmount': 'LargeOrderAmountByValue_p1.0_v40000-avg_imb01_dp2-org',
        'Jump': 'IntraRm_m30_IntraRelQtl_d20_q002_dp2_wstr_jump5mVwap',
        # 'OrderAmt_Dollar_LX': 'order_amount_Dollar_LX_R3_dp2_SumIntraRm5_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15',
        # 'OrderAmt_Dollar': 'order_amount_Dollar_R3_dp2_SumIntraRm5_LXPct_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15',
        # 'TradeAmt_Dollar_LX': 'trade_amount_Dollar_LX_R3_dp2_SumIntraRm5_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15',
        # 'TradeAmt_Dollar': 'trade_amount_Dollar_R3_dp2_SumIntraRm5_LXPct_R4_Imb1_R5_IntraQtl_120_R6_IntraRm_15'
    }

    visualize_trades(load_dir, save_dir, factor_dict, feature_name, feature_col_name)