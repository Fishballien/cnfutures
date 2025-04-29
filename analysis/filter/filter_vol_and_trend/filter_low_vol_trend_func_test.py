# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:42:15 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import numpy as np


# %%
def vol_filter_sigmoid(vol, T=0.01125, k=600):
    """Sigmoid function transformation"""
    return 1 / (1 + np.exp(-k * (vol - T)))


def trend_consistency_filter(trend_score, k=2000, trend_th=0.0005):
    """
    计算趋势一致门控值
    
    参数:
    trend_score : float 或 numpy.ndarray
        趋势一致性得分
    k : float, 可选
        sigmoid函数的陡度系数，默认为1000
    trend_th : float, 可选
        趋势阈值，默认为0.001
        
    返回:
    float 或 numpy.ndarray
        趋势一致门控值，范围在0到1之间
    """
    # 计算sigmoid
    trend_filter = 1 / (1 + np.exp(k * (trend_score - trend_th)))
    
    return trend_filter


# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Define the filter functions
def vol_filter_sigmoid(vol, T=0.01, k=800):
    """Sigmoid function transformation for volatility"""
    return 1 / (1 + np.exp(-k * (vol - T)))

def trend_consistency_filter(trend_score, k=2000, trend_th=0.0005):
    """Sigmoid function transformation for trend consistency"""
    return 1 / (1 + np.exp(k * (trend_score - trend_th)))

# Define the soft_and function
def soft_and(vol_filter, trend_filter):
    """Soft AND operation: 1 - (1 - vol_filter) * (1 - trend_filter)"""
    return 1 - (1 - vol_filter) * (1 - trend_filter)

# Create a grid of values for vol and trend_score
vol_range = np.linspace(0.005, 0.03, 100)
trend_score_range = np.linspace(-0.01, 0.01, 100)

# Create meshgrid for heatmap
vol_grid, trend_grid = np.meshgrid(vol_range, trend_score_range)

# Calculate the filters
vol_filter_values = vol_filter_sigmoid(vol_grid)
trend_filter_values = trend_consistency_filter(trend_grid)

# Calculate soft_and values
soft_and_values = soft_and(vol_filter_values, trend_filter_values)

# Set up the figure
plt.figure(figsize=(10, 8))

# Create a custom colormap that goes from white to dark blue
cmap = plt.cm.Blues
norm = colors.Normalize(vmin=0, vmax=1)

# Create the heatmap
heatmap = plt.pcolormesh(vol_grid, trend_grid, soft_and_values, 
                        cmap=cmap, norm=norm, shading='auto')
colorbar = plt.colorbar(heatmap, label='Soft AND Value')

# Add lines at threshold values
plt.axvline(x=0.01, color='red', linestyle='--', alpha=0.7, label='Vol Threshold (T=0.01125)')
plt.axhline(y=0.0005, color='red', linestyle='--', alpha=0.7, label='Trend Threshold (th=0.0005)')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.3)

# Set labels and title
plt.xlabel('Volatility (vol)', fontsize=12)
plt.ylabel('Trend Score', fontsize=12)
plt.title('Heatmap of Soft AND Function\nsoft_and = 1 - (1 - vol_filter) * (1 - trend_filter)', fontsize=14)

# Add a legend
plt.legend(loc='lower right')

# Print some key values for reference
print(f"Vol filter at threshold (0.01125): {vol_filter_sigmoid(0.01125):.4f}")
print(f"Trend filter at threshold (0.0005): {trend_consistency_filter(0.0005):.4f}")
print(f"Soft AND at both thresholds: {soft_and(vol_filter_sigmoid(0.01125), trend_consistency_filter(0.0005)):.4f}")

# Save the figure
plt.tight_layout()
plt.savefig('soft_and_heatmap.png', dpi=300)
plt.show()