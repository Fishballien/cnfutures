# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:45:55 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
from pathlib import Path
import pandas as pd


# %%
price_name = 't1min_fq1min_dl1min'


# %%
fut_dir = Path('/mnt/data1/futuretwap')
analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\overnight_statistics')
summary_dir = analysis_dir
summary_dir.mkdir(parents=True, exist_ok=True)


# %%
price_data = pd.read_parquet(fut_dir / f'{price_name}.parquet')


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# Load the price data
price_name = 't1min_fq1min_dl1min'
fut_dir = Path('/mnt/data1/futuretwap')
analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\overnight_statistics')
summary_dir = analysis_dir
summary_dir.mkdir(parents=True, exist_ok=True)

# Load the data - assume this is already done and price_data is available
# price_data = pd.read_parquet(fut_dir / f'{price_name}.parquet')

# Extract the dates from the index for filtering
price_data = price_data.copy()
price_data['date'] = pd.to_datetime(price_data.index.date)
price_data['time'] = price_data.index.time

# Function to calculate overnight returns
def calculate_overnight_returns(df):
    # Create dataframes for closing and opening prices
    closing_prices = df[df['time'] == pd.Timestamp('14:55:00').time()].copy()
    opening_prices = df[df['time'] == pd.Timestamp('09:30:00').time()].copy()
    
    # Set the date as index for easier joining
    closing_prices.set_index('date', inplace=True)
    opening_prices.set_index('date', inplace=True)
    
    # Calculate overnight returns for each asset
    overnight_returns = {}
    
    for asset in ['IC', 'IF', 'IH', 'IM']:
        # Skip assets with no data
        if asset not in df.columns:
            continue
            
        # Extract close and open prices
        close_series = closing_prices[asset].dropna()
        
        # For each closing price, we need the next day's opening price
        # Create a series of dates that are one day after the closing dates
        next_day_dates = pd.Series(pd.DatetimeIndex(close_series.index) + pd.Timedelta(days=1), 
                                   index=close_series.index)
        
        # Filter only business days where we have both closing and opening prices
        valid_close_dates = []
        valid_next_open_dates = []
        valid_close_prices = []
        valid_open_prices = []
        
        for close_date, next_date in zip(close_series.index, next_day_dates):
            # Try to find the next trading day's opening price
            try:
                # Convert next_date to Timestamp to ensure proper comparison
                next_date_ts = pd.Timestamp(next_date)
                # Find the next available trading day
                available_open_dates = opening_prices.index[opening_prices.index >= next_date_ts]
                if len(available_open_dates) > 0:
                    next_trading_date = available_open_dates[0]
                    next_open_price = opening_prices.loc[next_trading_date, asset]
                    
                    # If we have valid data, add to our lists
                    if not pd.isna(next_open_price) and not pd.isna(close_series[close_date]):
                        valid_close_dates.append(close_date)
                        valid_next_open_dates.append(next_trading_date)
                        valid_close_prices.append(close_series[close_date])
                        valid_open_prices.append(next_open_price)
            except Exception as e:
                continue
        
        # Create a dataframe with the results
        if valid_close_dates:
            results_df = pd.DataFrame({
                'close_date': valid_close_dates,
                'next_open_date': valid_next_open_dates,
                'close_price': valid_close_prices,
                'next_open_price': valid_open_prices
            })
            
            # Calculate returns
            results_df['overnight_return'] = (results_df['next_open_price'] / results_df['close_price'] - 1) * 100
            results_df['year'] = pd.DatetimeIndex(results_df['close_date']).year
            
            overnight_returns[asset] = results_df
    
    return overnight_returns

# Calculate overnight returns
overnight_returns = calculate_overnight_returns(price_data)

# Plot histograms of overnight returns by year for each asset
def plot_overnight_returns_by_year(overnight_returns, save_dir=None):
    for asset, returns_df in overnight_returns.items():
        years = sorted(returns_df['year'].unique())
        num_years = len(years)
        
        # Calculate number of rows needed (3 plots per row)
        num_rows = math.ceil(num_years / 3)
        
        # Create the figure with subplots
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
        fig.suptitle(f'Overnight Return Distribution by Year for {asset}', fontsize=16)
        
        # Flatten axes if multiple rows
        if num_rows > 1:
            axes = axes.flatten()
        
        # Loop through each year and create a histogram
        for i, year in enumerate(years):
            year_data = returns_df[returns_df['year'] == year]['overnight_return']
            
            # Skip years with no data
            if len(year_data) == 0:
                continue
            
            # Get the correct axis
            if num_rows == 1 and len(years) == 1:  # Special case for only one year
                ax = axes
            elif num_rows == 1:  # Special case for one row
                ax = axes[i]
            else:
                ax = axes[i]
            
            # Plot histogram
            ax.hist(year_data, bins=30, alpha=0.75, color='skyblue', edgecolor='black')
            ax.set_title(f'{year} (n={len(year_data)})')
            ax.set_xlabel('Overnight Return (%)')
            ax.set_ylabel('Frequency')
            
            # Add statistics
            mean_return = year_data.mean()
            std_return = year_data.std()
            ax.axvline(mean_return, color='red', linestyle='dashed', linewidth=1)
            ax.text(0.05, 0.95, f'Mean: {mean_return:.3f}%\nStd: {std_return:.3f}%', 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Hide unused subplots
        for j in range(i + 1, num_rows * 3):
            if num_rows == 1:
                if j < len(axes):
                    axes[j].set_visible(False)
            else:
                if j < len(axes):
                    axes[j].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to make space for the suptitle
        
        # Save the figure if a directory is provided
        if save_dir:
            filename = f"{asset}_overnight_returns_by_year.png"
            plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Saved {filename}")
        
        plt.show()

# Create summary statistics
def create_summary_statistics(overnight_returns, save_dir=None):
    for asset, returns_df in overnight_returns.items():
        # Group by year and calculate statistics
        yearly_stats = returns_df.groupby('year')['overnight_return'].agg([
            'count', 'mean', 'std', 'min', 'max',
            lambda x: np.percentile(x, 25),
            lambda x: np.percentile(x, 50),
            lambda x: np.percentile(x, 75)
        ]).reset_index()
        
        # Rename columns for clarity
        yearly_stats.columns = ['year', 'count', 'mean', 'std', 'min', 'max', '25%', '50%', '75%']
        
        # Calculate overall statistics
        overall_stats = returns_df['overnight_return'].agg([
            'count', 'mean', 'std', 'min', 'max',
            lambda x: np.percentile(x, 25),
            lambda x: np.percentile(x, 50),
            lambda x: np.percentile(x, 75)
        ]).reset_index()
        overall_stats.columns = ['statistic', 'overall']
        overall_stats['statistic'] = ['count', 'mean', 'std', 'min', 'max', '25%', '50%', '75%']
        
        # Save the statistics if a directory is provided
        if save_dir:
            yearly_filename = f"{asset}_yearly_overnight_return_statistics.csv"
            overall_filename = f"{asset}_overall_overnight_return_statistics.csv"
            
            yearly_stats.to_csv(save_dir / yearly_filename, index=False)
            overall_stats.to_csv(save_dir / overall_filename, index=False)
            
            print(f"Saved {yearly_filename} and {overall_filename}")
        
        # Return the statistics
        return yearly_stats, overall_stats

# Plot the overnight returns by year
plot_overnight_returns_by_year(overnight_returns, summary_dir)

# Create and save summary statistics
for asset in overnight_returns.keys():
    yearly_stats, overall_stats = create_summary_statistics({asset: overnight_returns[asset]}, summary_dir)
    print(f"\nYearly Statistics for {asset}:")
    print(yearly_stats)
    print(f"\nOverall Statistics for {asset}:")
    print(overall_stats)