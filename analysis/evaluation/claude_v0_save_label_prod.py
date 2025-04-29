# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 22:27:19 2025

@author: Xintang Zheng

Memory-optimized version with chunked processing to prevent OOM errors.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import ruptures as rpt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import gc  # For garbage collection

def global_classification(close, vol_threshold=None, slope_threshold=0.001, pen=2000):
    """
    Classify price series from a global perspective.
    
    Parameters:
    - close: Minute closing price series (np.array)
    - vol_threshold: Annualized volatility threshold (default uses global median)
    - slope_threshold: Trend slope threshold for log returns
    - pen: Penalty parameter for change point detection (higher = fewer segments)
    
    Returns:
    - labels: Classification label for each minute (np.array)
    - breakpoints: Detected change points
    """
    
    # Factor to annualize volatility (sqrt of trading minutes per year)
    # Assuming 245 trading days per year
    ANNUALIZATION_FACTOR = np.sqrt(60 * 4)  # For minute data, adjust as needed
    
    # 1. Change point detection
    algo = rpt.Pelt(model="l2").fit(close)
    # Penalty parameter (pen) - lower values create more segments, higher values create fewer
    breakpoints = algo.predict(pen=pen)  # Use the pen parameter passed to the function
    
    # Ensure 0 is included at the beginning
    if breakpoints[0] != 0:
        breakpoints = np.concatenate(([0], breakpoints))
    
    # 2. Divide into segments
    segments = [(breakpoints[i], breakpoints[i+1]) for i in range(len(breakpoints)-1)]
    
    # 3. Calculate global volatility threshold using realized volatility
    log_returns = np.diff(np.log(close))
    if vol_threshold is None:
        # Calculate realized volatility for each segment (annualized)
        segment_vols = []
        for start, end in segments:
            if end > start + 1:
                # Get log returns for the segment
                segment_log_returns = np.diff(np.log(close[start:end]))
                
                # Calculate realized volatility: sqrt(sum(returns^2)) * annualization factor
                realized_vol = np.sqrt(np.sum(segment_log_returns**2) / len(segment_log_returns)) * ANNUALIZATION_FACTOR
                segment_vols.append(realized_vol)
        
        vol_threshold = np.median(segment_vols) if segment_vols else 0.01
    
    # 4. Classify each segment
    labels = np.full(len(close), "Unknown", dtype=object)
    for start, end in segments:
        segment_data = close[start:end]
        
        # Skip segments that are too short for meaningful analysis
        if end <= start + 1:
            labels[start:end] = "Unknown"
            continue
            
        segment_log_returns = np.diff(np.log(segment_data))
        
        # Calculate realized volatility (annualized)
        realized_vol = np.sqrt(np.sum(segment_log_returns**2) / len(segment_log_returns)) * ANNUALIZATION_FACTOR
        vol_label = "high" if realized_vol > vol_threshold else "low"
        
        # Calculate trend using cumulative log returns
        cum_log_returns = np.cumsum(np.diff(np.log(segment_data)))
        if len(cum_log_returns) > 1:  # Ensure we have enough data points
            x = np.arange(len(cum_log_returns)).reshape(-1, 1)
            y = cum_log_returns.reshape(-1, 1)
            reg = LinearRegression().fit(x, y)
            slope = reg.coef_[0][0]
            trend_label = "trend" if abs(slope) > slope_threshold else "no_trend"
        else:
            trend_label = "no_trend"
        
        # Assign labels
        if vol_label == "low" and trend_label == "no_trend":
            label = "Low_Vol_Range"
        elif vol_label == "low" and trend_label == "trend":
            label = "Low_Vol_Trend"
        elif vol_label == "high" and trend_label == "no_trend":
            label = "High_Vol_Range"
        else:
            label = "High_Vol_Trend"
        
        labels[start:end] = label
    
    return labels, breakpoints

def classify_single_asset_chunked(asset_prices, vol_threshold=0.001, slope_threshold=5e-5, 
                                 pen=2000, chunk_size=10000):
    """
    Apply global classification to a single asset in chunks to avoid memory issues.
    
    Parameters:
    - asset_prices: Series with price data for a single asset
    - vol_threshold: Volatility threshold for classification
    - slope_threshold: Slope threshold for classification
    - pen: Penalty parameter for segmentation
    - chunk_size: Number of data points to process in each chunk
    
    Returns:
    - labels: Array of classification labels
    - breakpoints: Dictionary of breakpoints for each chunk
    """
    # Get price values and handle NaN
    prices = asset_prices.values
    is_nan = pd.isna(prices)
    
    # Create output labels array (same size as input, filled with NaN for now)
    labels = np.full(len(prices), np.nan, dtype=object)
    
    # Process data in chunks
    chunk_breakpoints = {}
    
    # Determine valid chunks that contain non-NaN data
    valid_indices = np.where(~is_nan)[0]
    if len(valid_indices) == 0:
        return labels, {}
    
    # Find start and end indices for chunks
    start_idx = 0
    while start_idx < len(prices):
        end_idx = min(start_idx + chunk_size, len(prices))
        
        # Extract chunk data
        chunk_prices = prices[start_idx:end_idx]
        chunk_is_nan = is_nan[start_idx:end_idx]
        
        # Skip chunks that are all NaN
        if np.all(chunk_is_nan):
            start_idx = end_idx
            continue
        
        # Replace NaN with interpolated values for processing
        valid_chunk_indices = np.where(~chunk_is_nan)[0]
        if len(valid_chunk_indices) < 2:  # Need at least 2 valid points
            start_idx = end_idx
            continue
            
        # Create a clean version of data for processing
        clean_chunk = np.copy(chunk_prices)
        clean_chunk[chunk_is_nan] = np.interp(
            np.where(chunk_is_nan)[0],
            valid_chunk_indices,
            chunk_prices[valid_chunk_indices]
        )
        
        # Classify this chunk
        try:
            chunk_labels, local_breakpoints = global_classification(
                clean_chunk,
                vol_threshold=vol_threshold,
                slope_threshold=slope_threshold,
                pen=pen
            )
            
            # Convert local breakpoints to global indices
            global_breakpoints = [bp + start_idx for bp in local_breakpoints]
            chunk_breakpoints[start_idx] = global_breakpoints
            
            # Copy non-NaN labels to final result
            labels[start_idx:end_idx] = chunk_labels
            labels[start_idx + np.where(chunk_is_nan)[0]] = np.nan  # Reset NaN positions
            
        except Exception as e:
            print(f"Error processing chunk from {start_idx} to {end_idx}: {e}")
        
        # Move to next chunk
        start_idx = end_idx
        
        # Force garbage collection to free memory
        gc.collect()
    
    return labels, chunk_breakpoints

def classify_all_assets_chunked(price_data, vol_threshold=0.001, slope_threshold=5e-5, 
                               pen=2000, chunk_size=10000, time_chunks=None):
    """
    Apply global classification to all assets in the price DataFrame in chunks.
    
    Parameters:
    - price_data: DataFrame with datetime index and assets as columns
    - vol_threshold: Volatility threshold for classification
    - slope_threshold: Slope threshold for classification
    - pen: Penalty parameter for segmentation
    - chunk_size: Number of data points to process in each chunk
    - time_chunks: Optional list of date ranges to process as separate chunks
    
    Returns:
    - labels_df: DataFrame with same structure as price_data but containing labels
    """
    # Initialize result DataFrame with same structure as input
    labels_df = pd.DataFrame(index=price_data.index, columns=price_data.columns)
    
    if time_chunks:
        # Process by time chunks
        print(f"Processing {len(price_data.columns)} assets in {len(time_chunks)-1} time chunks")
        for i in range(len(time_chunks)-1):
            start_date = time_chunks[i]
            end_date = time_chunks[i+1]
            print(f"Processing time chunk: {start_date} to {end_date}")
            
            # Get subset of data for this time period
            chunk_data = price_data.loc[start_date:end_date]
            
            # Process each asset in this time period
            for column in tqdm(chunk_data.columns, desc=f"Chunk {i+1}/{len(time_chunks)-1}"):
                asset_prices = chunk_data[column]
                chunk_labels, _ = classify_single_asset_chunked(
                    asset_prices,
                    vol_threshold=vol_threshold,
                    slope_threshold=slope_threshold,
                    pen=pen,
                    chunk_size=chunk_size
                )
                
                # Update the full labels DataFrame
                labels_df.loc[chunk_data.index, column] = chunk_labels
            
            # Force garbage collection after each time chunk
            gc.collect()
    else:
        # Process each asset entirely, but in data chunks
        for column in tqdm(price_data.columns, desc="Classifying assets"):
            asset_prices = price_data[column]
            asset_labels, _ = classify_single_asset_chunked(
                asset_prices,
                vol_threshold=vol_threshold,
                slope_threshold=slope_threshold,
                pen=pen,
                chunk_size=chunk_size
            )
            
            labels_df[column] = asset_labels
            
            # Force garbage collection after each asset
            gc.collect()
    
    return labels_df

def generate_time_chunks_by_month(price_data):
    """
    Generate time-based chunks for processing by month.
    
    Parameters:
    - price_data: DataFrame with datetime index
    
    Returns:
    - chunks: List of dates defining chunk boundaries by month
    """
    # Convert index to pandas DatetimeIndex if not already
    if not isinstance(price_data.index, pd.DatetimeIndex):
        price_data.index = pd.DatetimeIndex(price_data.index)
    
    # Extract years and months from the index
    year_months = [(d.year, d.month) for d in price_data.index]
    unique_year_months = sorted(set(year_months))
    
    # Create a list of the first day of each month from the data
    chunks = []
    for year, month in unique_year_months:
        # Find the first date in the dataset for this month
        month_data = price_data[
            (price_data.index.year == year) & 
            (price_data.index.month == month)
        ]
        if not month_data.empty:
            chunks.append(month_data.index.min().date())
    
    # Add the end date (first day of next month after the last date)
    last_date = price_data.index.max()
    # If we're not already at the end of the month, add the last date
    if chunks[-1] != last_date.date():
        chunks.append(last_date.date())
    
    return chunks

def plot_asset_classification(price_data, labels_df, asset_name, start_date=None, end_date=None):
    """
    Plot the classification results for a specific asset.
    
    Parameters:
    - price_data: Original price DataFrame
    - labels_df: Labels DataFrame
    - asset_name: Name of the asset to plot
    - start_date, end_date: Optional date range to plot
    """
    # Filter data to date range if specified
    if start_date is not None or end_date is not None:
        plot_price = price_data.loc[start_date:end_date, asset_name]
        plot_labels = labels_df.loc[start_date:end_date, asset_name]
    else:
        plot_price = price_data[asset_name]
        plot_labels = labels_df[asset_name]
    
    # Get data arrays
    close = plot_price.values
    labels = plot_labels.values
    x_values = np.arange(len(close))
    
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    
    # Plot price
    plt.plot(x_values, close, label='Price', color='black', linewidth=1)
    
    # Set colors for different classification labels
    colors = {
        "Low_Vol_Range": "blue",
        "Low_Vol_Trend": "green",
        "High_Vol_Range": "red",
        "High_Vol_Trend": "orange",
        "Unknown": "gray"
    }
    
    # Find segment boundaries (where labels change)
    change_points = [0]
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1] or pd.isna(labels[i]) != pd.isna(labels[i-1]):
            change_points.append(i)
    change_points.append(len(labels))
    
    # Fill each segment with appropriate color
    for i in range(len(change_points)-1):
        start = change_points[i]
        end = change_points[i+1]
        
        # Skip if segment is too small
        if end <= start:
            continue
            
        # Get label for this segment
        if start < len(labels) and not pd.isna(labels[start]):
            label = labels[start]
            color = colors.get(label, "gray")
            
            # Add colored background for segment
            rect = patches.Rectangle(
                (start, np.nanmin(close)), 
                end-start, 
                np.nanmax(close)-np.nanmin(close),
                alpha=0.2, 
                color=color
            )
            ax.add_patch(rect)
    
    # Add a legend for segment types
    legend_elements = [
        patches.Patch(color=colors["Low_Vol_Range"], alpha=0.3, label="Low Volatility Range"),
        patches.Patch(color=colors["Low_Vol_Trend"], alpha=0.3, label="Low Volatility Trend"),
        patches.Patch(color=colors["High_Vol_Range"], alpha=0.3, label="High Volatility Range"),
        patches.Patch(color=colors["High_Vol_Trend"], alpha=0.3, label="High Volatility Trend")
    ]
    
    # Add date range to title if provided
    date_range = ""
    if start_date is not None and end_date is not None:
        date_range = f" ({start_date} to {end_date})"
    
    plt.title(f"Price Series Classification - {asset_name}{date_range}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend(handles=legend_elements, loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_all_classification_stats(labels_df):
    """
    Plot statistics of classifications across all assets.
    
    Parameters:
    - labels_df: DataFrame with classifications for all assets
    """
    # Get overall statistics
    all_labels = labels_df.values.flatten()
    all_labels = all_labels[~pd.isna(all_labels)]
    
    # Count each label type
    label_counts = {}
    for label in ["Low_Vol_Range", "Low_Vol_Trend", "High_Vol_Range", "High_Vol_Trend", "Unknown"]:
        count = np.sum(all_labels == label)
        label_counts[label] = count
    
    # Create a bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(label_counts.keys(), label_counts.values())
    
    # Color the bars
    colors = {
        "Low_Vol_Range": "blue",
        "Low_Vol_Trend": "green",
        "High_Vol_Range": "red",
        "High_Vol_Trend": "orange",
        "Unknown": "gray"
    }
    
    for bar, label in zip(bars, label_counts.keys()):
        bar.set_color(colors.get(label, "gray"))
        bar.set_alpha(0.7)
    
    plt.title("Distribution of Market Regimes Across All Assets")
    plt.xlabel("Market Regime")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Configuration
    vol_threshold = 0.012
    slope_threshold = 2e-5
    pen = 30000
    chunk_size = 20000  # Data points per processing chunk
    
    label_name = f'rv{vol_threshold}_slp{slope_threshold}_pen{pen}'
    label_dir = Path('/mnt/data1/labels')
    label_dir.mkdir(exist_ok=True)
    
    # Load data
    price_name = 't1min_fq1min_dl1min'
    fut_dir = Path('/mnt/data1/future_twap')
    price_data = pd.read_parquet(fut_dir / f'{price_name}.parquet').dropna(how='all')
    
    print(f"Loaded price data with shape: {price_data.shape}")
    print(f"Date range: {price_data.index.min()} to {price_data.index.max()}")

    # 使用新的按月份生成时间块的函数，而不是原来的固定天数
    time_chunks = generate_time_chunks_by_month(price_data)
    print(f"Processing in {len(time_chunks)-1} monthly chunks from {time_chunks[0]} to {time_chunks[-1]}")
    
    # Classify all assets in chunks
    labels_df = classify_all_assets_chunked(
        price_data, 
        vol_threshold=vol_threshold, 
        slope_threshold=slope_threshold,
        pen=pen,
        chunk_size=chunk_size,
        time_chunks=time_chunks
    )
    
    # Save results to parquet
    labels_df.to_parquet(label_dir / f'{label_name}.parquet')
    print(f"Labels saved to: {label_dir / f'{label_name}.parquet'}")
    
    # Plot results for a specific asset (e.g., 'IC') - using just the last month for visualization
    if 'IC' in price_data.columns:
        last_chunk_start = time_chunks[-2]
        plot_asset_classification(price_data, labels_df, 'IC', 
                                start_date=last_chunk_start, 
                                end_date=None)
    
    # Plot overall statistics
    plot_all_classification_stats(labels_df)
    
    # Print detailed statistics for specific assets
    for column in ['IC', 'IF']:  # Add or remove assets as needed
        if column in price_data.columns:
            asset_labels = labels_df[column].dropna()
            print(f"\nClassification Statistics for {column}:")
            label_counts = asset_labels.value_counts()
            for label, count in label_counts.items():
                print(f"{label}: {count} minutes ({count/len(asset_labels)*100:.2f}%)")