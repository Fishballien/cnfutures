# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:33:14 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
from statsmodels.tsa.stattools import adfuller


# %%
def check_stationarity(timeseries, significance_level=0.05):
    """
    检验时间序列的平稳性（使用ADF检验）。
    
    参数：
    timeseries (pd.Series): 时间序列数据
    significance_level (float): 显著性水平，默认0.05
    
    返回：
    dict: 包含检验统计量、p值、滞后数、ADF检验结论等信息的字典
    """
    try:
        result = adfuller(timeseries)
    except:
        return {
            "Is Stationary": False,
        }
    test_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    # 判断平稳性
    is_stationary = p_value < significance_level
    
    return {
        "Test Statistic": test_statistic,
        "P-Value": p_value,
        "Lags Used": result[2],
        "Number of Observations": result[3],
        "Critical Values": critical_values,
        "Is Stationary": is_stationary
    }


# %%
def calculate_positive_ratio(df):
    """
    计算 DataFrame 每列大于 0 的比例。
    
    参数：
    df (pd.DataFrame): 输入的 DataFrame
    
    返回：
    pd.Series: 每列大于 0 的比例
    """
    return (df > 0).mean()