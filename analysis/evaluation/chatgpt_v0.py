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
import numpy as np
import ruptures as rpt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def global_classification(close, vol_threshold=None, slope_threshold=0.001, n_bkps=10):
    """
    从全局视角对价格序列进行分类。
    
    参数:
    - close: 分钟收盘价序列 (np.array)
    - vol_threshold: 波动性阈值 (默认使用全局中位数)
    - slope_threshold: 趋势斜率阈值
    - n_bkps: 预期的变化点数量
    
    返回:
    - labels: 每分钟的分类标签 (np.array)
    """
    # 1. 变化点检测
    algo = rpt.Pelt(model="l2").fit(close)
    breakpoints = algo.predict(pen=500)  # 惩罚参数可调整
    
    # 2. 划分区段
    segments = [(breakpoints[i], breakpoints[i+1]) for i in range(len(breakpoints)-1)]
    
    # 3. 计算全局波动性阈值
    log_returns = np.diff(np.log(close))
    if vol_threshold is None:
        segment_vols = [np.std(np.diff(np.log(close[start:end]))) for start, end in segments]
        vol_threshold = np.median(segment_vols)
    
    # 4. 分类每个区段
    labels = np.full(len(close), "未知", dtype=object)
    for start, end in segments:
        segment_data = close[start:end]
        segment_log_returns = np.diff(np.log(segment_data))
        
        # 计算波动性
        segment_vol = np.std(segment_log_returns)
        vol_label = "high" if segment_vol > vol_threshold else "low"
        
        # 计算趋势
        x = np.arange(len(segment_data)).reshape(-1, 1)
        y = segment_data.reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        slope = reg.coef_[0][0]
        trend_label = "trend" if abs(slope) > slope_threshold else "no_trend"
        
        # 分配标签
        if vol_label == "low" and trend_label == "no_trend":
            label = "低波震荡"
        elif vol_label == "low" and trend_label == "trend":
            label = "低波趋势"
        elif vol_label == "high" and trend_label == "no_trend":
            label = "高波震荡"
        else:
            label = "高波趋势"
        
        labels[start:end] = label
    
    return labels

# 示例使用
close = np.cumsum(np.random.randn(3000)) + 100  # 模拟价格数据
labels = global_classification(close, vol_threshold=0.01, slope_threshold=0.01)

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(close, label='价格', color='black', linewidth=1)

# 设定不同分类标签的颜色
colors = {
    "低波震荡": "blue",
    "低波趋势": "green",
    "高波震荡": "red",
    "高波趋势": "orange",
    "未知": "gray"
}

# 绘制每个区段的不同颜色
for i in range(1, len(close)):
    if labels[i] != labels[i-1]:
        plt.axvline(x=i, color=colors.get(labels[i], "gray"), linestyle='--', alpha=0.5)

plt.title("价格序列分类")
plt.xlabel("时间")
plt.ylabel("价格")
plt.legend()
plt.grid(True)
plt.show()
