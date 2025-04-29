# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 22:27:19 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
from pathlib import Path
import pandas as pd
import numpy as np
import ruptures as rpt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
# Example usage
# =============================================================================
# np.random.seed(42)  # For reproducibility
# close = np.cumsum(np.random.randn(3000)) + 100  # Simulated price data
# =============================================================================

price_name = 't1min_fq1min_dl1min'
fut_dir = Path('/mnt/data1/future_twap')
price_data = pd.read_parquet(fut_dir / f'{price_name}.parquet').dropna(how='all')
# price_data = price_data.loc['20240901':'20241101']
# price_data = price_data.loc['20220101':'20230101']
price_data = price_data.loc['20250101':'20250301']
close = price_data['IC'].values

labels, breakpoints = global_classification(close, vol_threshold=0.012, slope_threshold=2e-5, pen=30000)

# Plot
plt.figure(figsize=(15, 8))
ax = plt.gca()

# Plot price
plt.plot(close, label='Price', color='black', linewidth=1)

# Set colors for different classification labels
colors = {
    "Low_Vol_Range": "blue",
    "Low_Vol_Trend": "green",
    "High_Vol_Range": "red",
    "High_Vol_Trend": "orange",
    "Unknown": "gray"
}

# Fill each segment with appropriate color
for i in range(len(breakpoints)-1):
    start = breakpoints[i]
    end = breakpoints[i+1]
    label = labels[start]
    color = colors.get(label, "gray")
    
    # Add colored background for segment
    rect = patches.Rectangle((start, min(close)), end-start, max(close)-min(close), 
                             alpha=0.2, color=color)
    ax.add_patch(rect)

# Draw all rupture segment positions
for point in breakpoints:
    if point < len(close):  # Ensure point is within range
        plt.axvline(x=point, color='purple', linestyle='--', alpha=0.7)

# Add a legend for segment types
legend_elements = [
    patches.Patch(color=colors["Low_Vol_Range"], alpha=0.3, label="Low Volatility Range"),
    patches.Patch(color=colors["Low_Vol_Trend"], alpha=0.3, label="Low Volatility Trend"),
    patches.Patch(color=colors["High_Vol_Range"], alpha=0.3, label="High Volatility Range"),
    patches.Patch(color=colors["High_Vol_Trend"], alpha=0.3, label="High Volatility Trend"),
    plt.Line2D([0], [0], color='purple', linestyle='--', label='Segment Boundary')
]

plt.title("Price Series Classification")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend(handles=legend_elements, loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print statistics about the segments
segment_counts = {label: np.sum(labels == label) for label in set(labels)}
print("Segment Statistics:")
for label, count in segment_counts.items():
    print(f"{label}: {count} minutes ({count/len(labels)*100:.2f}%)")