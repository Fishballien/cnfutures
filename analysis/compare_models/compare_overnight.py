# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:48:46 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


from test_and_eval.scores import get_general_return_metrics


# %%
# =============================================================================
# compare_name = 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18'
# model_list = {
#     '1_2_hold_overnight': {
#         'model_name': 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18',
#         'test_name': 'trade_ver3_futtwap_sp1min_s240d_icim_v6',
#         },
#     '1_2_intraday_only': {
#         'model_name': 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18',
#         'test_name': 'trade_ver3_3_futtwap_sp1min_s240d_icim_v6',
#         },
#     }
# =============================================================================

compare_name = 'avg_agg_250402_by_trade_net_v18'
model_list = {
    '1_2_hold_overnight': {
        'model_name': 'avg_agg_250402_by_trade_net_v18',
        'test_name': 'trade_ver3_new_futtwap_sp1min_s240d_icim_v6',
        },
    '1_2_intraday_only': {
        'model_name': 'avg_agg_250402_by_trade_net_v18',
        'test_name': 'trade_ver3_new_2_futtwap_sp1min_s240d_icim_v6',
        },
    }

model_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model')
analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\compare_overnight')
summary_dir = analysis_dir / compare_name
summary_dir.mkdir(parents=True, exist_ok=True)
fee = 0.00024
target_start_date = '20250101'

# %%
# Dict to store net returns for each model
model_nets = {}

# Dict to store yearly metrics for each model
yearly_metrics = {}

# Metrics we want to analyze
metric_names = [
    'return', 'return_annualized', 'max_dd', 'sharpe_ratio',
    'calmar_ratio', 'sortino_ratio', 'sterling_ratio',
    'burke_ratio', 'ulcer_index', 'drawdown_recovery_ratio',
    'profit_per_trade'  # Added profit_per_trade to metrics list
]

# Store model data for profit_per_trade calculation
model_data = {}

# Process each model
for model_test_name, model_info in model_list.items():
    model_name = model_info['model_name']
    test_name = model_info['test_name']
    
    # Load paths
    gp_path = model_dir / f'{model_name}/test/{test_name}/data/gpd_predict_{model_name}.pkl'
    hsr_path = model_dir / f'{model_name}/test/{test_name}/data/hsr_predict_{model_name}.pkl'
    
    # Load data
    with open(gp_path, 'rb') as f:
        gp = pickle.load(f)
    with open(hsr_path, 'rb') as f:
        hsr = pickle.load(f)
    
    # Store model data for profit_per_trade calculation
    model_data[model_test_name] = {'gp': gp, 'hsr': hsr}
    
    # Calculate net return
    net = gp['all']['return'] - hsr['all']['avg'] * fee
    
    # Store net return for this model
    model_nets[model_test_name] = net
    
    # Dictionary to store metrics for each year
    model_yearly_metrics = {}
    
    # Calculate metrics by year
    years = sorted(net.index.year.unique())
    for year in years:
        year_data = net[net.index.year == year]
        year_gp = gp['all']['return'][gp['all']['return'].index.year == year]
        year_hsr = hsr['all']['avg'][hsr['all']['avg'].index.year == year]
        
        if not year_data.empty:
            try:
                metrics = get_general_return_metrics(year_data.values)
                
                # Calculate profit_per_trade for this year
                if year_hsr.sum() > 0:  # To avoid division by zero
                    profit_per_trade = year_gp.sum() / year_hsr.sum()
                else:
                    profit_per_trade = 0
                    
                metrics['profit_per_trade'] = profit_per_trade
                model_yearly_metrics[year] = metrics
            except Exception as e:
                print(f"Error processing {model_test_name} for year {year}: {e}")
    
    # Store the yearly metrics for this model
    yearly_metrics[model_test_name] = model_yearly_metrics
    
    # Calculate overall profit_per_trade for the model
    overall_profit_per_trade = gp['all']['return'].sum() / hsr['all']['avg'].sum() if hsr['all']['avg'].sum() > 0 else 0
    print(f"{model_test_name} overall profit_per_trade: {overall_profit_per_trade:.6f}")

# 1) Plot net return curves and their difference if there are only 2 models
def plot_net_returns():
    # Create DataFrame from model_nets
    net_df = pd.DataFrame(model_nets)
    
    # Calculate cumulative returns using cumsum() as requested
    cum_returns = net_df.cumsum()
    
    # Plot cumulative returns for each model and difference on the same chart if there are 2 models
    if len(model_nets) == 2:
        model_names = list(model_nets.keys())
        diff = model_nets[model_names[0]] - model_nets[model_names[1]]
        diff_cum = diff.cumsum()
        
        # Create a figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot cumulative returns on the primary y-axis
        for model in cum_returns.columns:
            ax1.plot(cum_returns.index, cum_returns[model], label=model)
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return')
        ax1.tick_params(axis='y')
        
        # Create a secondary y-axis for the difference
        ax2 = ax1.twinx()
        ax2.plot(diff_cum.index, diff_cum, color='purple', linestyle='--', label=f'Diff: {model_names[0]} - {model_names[1]}')
        ax2.set_ylabel('Cumulative Difference')
        ax2.tick_params(axis='y')
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.title('Cumulative Returns and Difference Comparison')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(summary_dir / 'cumulative_returns_and_diff.png', dpi=300)
        plt.close()
    else:
        # If more than 2 models, just plot the cumulative returns
        plt.figure(figsize=(14, 8))
        for model in cum_returns.columns:
            plt.plot(cum_returns.index, cum_returns[model], label=model)
        
        plt.title('Cumulative Returns Comparison')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(summary_dir / 'cumulative_returns.png', dpi=300)
        plt.close()

# 2) Create heatmaps for each metric
def create_metric_heatmaps():
    # Initialize a dictionary to store DataFrames for each metric
    metric_dfs = {metric: pd.DataFrame() for metric in metric_names}
    
    # Prepare data for heatmaps
    all_years = sorted(set().union(*[set(yearly_metrics[model].keys()) for model in yearly_metrics]))
    
    for metric in metric_names:
        # Create DataFrame for this metric
        data = {model: [yearly_metrics[model].get(year, {}).get(metric, np.nan) for year in all_years] 
                for model in yearly_metrics}
        metric_df = pd.DataFrame(data, index=all_years)
        
        # Save to our collection
        metric_dfs[metric] = metric_df
        
        # Create heatmap
        plt.figure(figsize=(10, len(all_years) * 0.6))
        sns.heatmap(metric_df, annot=True, cmap='RdYlGn', fmt='.3g', cbar=True, 
                   linewidths=0.5, center=(0 if metric == 'max_dd' else None))
        plt.title(f'{metric} by Year and Model')
        plt.tight_layout()
        plt.savefig(summary_dir / f'heatmap_{metric}.png', dpi=300)
        plt.close()
    
    return metric_dfs

# 3) Additional analysis: Yearly performance comparison chart
def plot_yearly_performance_comparison():
    # For key metrics, create bar charts comparing models by year
    key_metrics = ['return_annualized', 'sharpe_ratio', 'max_dd', 'profit_per_trade']
    
    for metric in key_metrics:
        yearly_data = {model: [] for model in yearly_metrics}
        years = []
        
        for year in sorted(set().union(*[set(yearly_metrics[model].keys()) for model in yearly_metrics])):
            years.append(year)
            for model in yearly_metrics:
                yearly_data[model].append(yearly_metrics[model].get(year, {}).get(metric, np.nan))
        
        # Set up the bar chart
        x = np.arange(len(years))
        width = 0.35 if len(yearly_metrics) == 2 else 0.25
        
        fig, ax = plt.figure(figsize=(14, 8)), plt.gca()
        
        # Plot bars for each model
        for i, (model, values) in enumerate(yearly_data.items()):
            ax.bar(x + (i - len(yearly_metrics)/2 + 0.5) * width, values, width, label=model)
        
        # Set chart properties
        ax.set_xlabel('Year')
        ax.set_ylabel(metric)
        ax.set_title(f'Yearly {metric} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(years)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(summary_dir / f'yearly_{metric}_comparison.png', dpi=300)
        plt.close()

# 4) Performance distribution analysis
def plot_performance_distributions():
    # Create box plots for key metrics
    key_metrics = ['return', 'sharpe_ratio', 'max_dd', 'profit_per_trade']
    
    for metric in key_metrics:
        data = []
        labels = []
        
        for model in yearly_metrics:
            metric_values = [yearly_metrics[model][year][metric] for year in yearly_metrics[model]]
            data.append(metric_values)
            labels.append(model)
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(data, labels=labels)
        plt.title(f'Distribution of Yearly {metric}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(summary_dir / f'boxplot_{metric}.png', dpi=300)
        plt.close()
        
        # Also create violin plots for more detailed distribution view
        plt.figure(figsize=(10, 6))
        plt.violinplot(data)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.title(f'Distribution Density of Yearly {metric}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(summary_dir / f'violinplot_{metric}.png', dpi=300)
        plt.close()

# 5) Rolling performance analysis
def plot_rolling_metrics():
    # Calculate rolling metrics (e.g., rolling Sharpe ratio) for each model
    window = 252  # Approximately 1 year of trading days
    
    # Convert individual series to DataFrame
    net_df = pd.DataFrame(model_nets)
    
    # Calculate rolling returns and metrics
    rolling_returns = net_df.rolling(window).sum()
    rolling_volatility = net_df.rolling(window).std() * np.sqrt(window)
    rolling_sharpe = rolling_returns / rolling_volatility
    
    # Plot rolling Sharpe ratio
    plt.figure(figsize=(14, 8))
    for model in model_nets:
        plt.plot(rolling_sharpe.index, rolling_sharpe[model], label=f'{model} Rolling Sharpe')
    
    plt.title(f'Rolling {window}-day Sharpe Ratio')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(summary_dir / 'rolling_sharpe.png', dpi=300)
    plt.close()
    
    # Plot rolling cumulative returns
    rolling_cum_returns = (1 + net_df.rolling(window).apply(lambda x: (1 + x).prod() - 1)) - 1
    
    plt.figure(figsize=(14, 8))
    for model in model_nets:
        plt.plot(rolling_cum_returns.index, rolling_cum_returns[model], label=f'{model} Rolling Return')
    
    plt.title(f'Rolling {window}-day Cumulative Return')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(summary_dir / 'rolling_returns.png', dpi=300)
    plt.close()

# Add a new function to plot returns from a target date
def plot_returns_from_target_date(target_start_date):
    # Create DataFrame from model_nets
    net_df = pd.DataFrame(model_nets)
    
    # Filter data from the target start date
    filtered_net_df = net_df[net_df.index >= target_start_date]
    
    # If no data after the target date, return with a message
    if filtered_net_df.empty:
        print(f"No data available after {target_start_date}")
        return
    
    # Calculate cumulative returns from target date using cumsum()
    cum_returns = filtered_net_df.cumsum()
    
    # Plot cumulative returns and difference if there are 2 models
    if len(model_nets) == 2:
        model_names = list(model_nets.keys())
        diff = filtered_net_df[model_names[0]] - filtered_net_df[model_names[1]]
        diff_cum = diff.cumsum()
        
        # Create a figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot cumulative returns on the primary y-axis
        for model in cum_returns.columns:
            ax1.plot(cum_returns.index, cum_returns[model], label=model)
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return')
        ax1.tick_params(axis='y')
        
        # Create a secondary y-axis for the difference
        ax2 = ax1.twinx()
        ax2.plot(diff_cum.index, diff_cum, color='purple', linestyle='--', label=f'Diff: {model_names[0]} - {model_names[1]}')
        ax2.set_ylabel('Cumulative Difference')
        ax2.tick_params(axis='y')
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.title(f'Cumulative Returns and Difference from {target_start_date}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(summary_dir / f'cumulative_returns_from_{target_start_date.strftime("%Y%m%d")}.png', dpi=300)
        plt.close()
    else:
        # If more than 2 models, just plot the cumulative returns
        plt.figure(figsize=(14, 8))
        for model in cum_returns.columns:
            plt.plot(cum_returns.index, cum_returns[model], label=model)
        
        plt.title(f'Cumulative Returns from {target_start_date}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(summary_dir / f'cumulative_returns_from_{target_start_date.strftime("%Y%m%d")}.png', dpi=300)
        plt.close()

# Execute all analyses
plot_net_returns()
metric_dfs = create_metric_heatmaps()
plot_yearly_performance_comparison()
plot_performance_distributions()
plot_rolling_metrics()

# Generate additional chart from target date if defined
if 'target_start_date' in globals():
    plot_returns_from_target_date(target_start_date)
    print(f"Generated additional chart from target date: {target_start_date}")

# Generate a summary table of overall performance
def create_overall_summary():
    # Calculate overall metrics for the full period
    overall_metrics = {}
    
    for model in model_nets:
        try:
            metrics = get_general_return_metrics(model_nets[model].values)
            
            # Add profit_per_trade to overall metrics
            model_gp = model_data[model]['gp']
            model_hsr = model_data[model]['hsr']
            overall_profit_per_trade = model_gp['all']['return'].sum() / model_hsr['all']['avg'].sum() if model_hsr['all']['avg'].sum() > 0 else 0
            metrics['profit_per_trade'] = overall_profit_per_trade
            
            overall_metrics[model] = metrics
        except Exception as e:
            print(f"Error calculating overall metrics for {model}: {e}")
    
    # Create a DataFrame for the overall metrics
    overall_df = pd.DataFrame(overall_metrics)
    
    # Save to CSV
    overall_df.to_csv(summary_dir / 'overall_performance_summary.csv')
    
    # Also create a bar chart comparing key overall metrics
    key_metrics = ['return_annualized', 'sharpe_ratio', 'max_dd', 'sortino_ratio', 'profit_per_trade']
    
    for metric in key_metrics:
        values = [overall_metrics[model][metric] for model in overall_metrics]
        models = list(overall_metrics.keys())
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, values)
        plt.title(f'Overall {metric} Comparison')
        plt.ylabel(metric)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, v in enumerate(values):
            plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(summary_dir / f'overall_{metric}.png', dpi=300)
        plt.close()
    
    return overall_df

overall_summary = create_overall_summary()

print("Analysis complete. All visualizations have been saved to the summary_dir folder.")