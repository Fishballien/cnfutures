# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:56:47 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
import numpy as np
import matplotlib.pyplot as plt

# 假设trend_score在-0.01到0.01之间
trend_score_range = np.linspace(-0.01, 0.01, 1000)
trend_th = 0.0005  # 设置阈值
k = 2000  # sigmoid陡度系数

# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 计算趋势一致门控
trend_filter = sigmoid(k * (trend_score_range - trend_th))

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(trend_score_range, trend_filter, 'b-', linewidth=2)
plt.axvline(x=trend_th, color='r', linestyle='--', label=f'Threshold (trend_th={trend_th})')
plt.grid(True, alpha=0.3)
plt.xlabel('Trend Score')
plt.ylabel('Trend Filter (Sigmoid)')
plt.title(f'Trend Consistency Gate using Sigmoid (k={k}, trend_th={trend_th})')
plt.legend()

# 标注阈值处的点
plt.scatter([trend_th], [0.5], color='red', zorder=5)
plt.annotate('(0.001, 0.5)', xy=(trend_th, 0.5), xytext=(trend_th+0.002, 0.6),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# 设置适当的轴范围
plt.xlim(-0.01, 0.01)
plt.ylim(-0.05, 1.05)

# 保存图像
plt.savefig('trend_consistency_gate.png')
plt.show()

# 打印一些关键点的值
print("\n趋势分数(trend_score) | 趋势门控(trend_filter)")
print("-" * 40)
for ts in [-0.01, -0.005, 0, trend_th, 0.002, 0.005, 0.01]:
    tf = sigmoid(k * (ts - trend_th))
    print(f"{ts:15.4f} | {tf:15.6f}")
