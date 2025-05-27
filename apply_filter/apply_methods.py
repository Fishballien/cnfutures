# -*- coding: utf-8 -*-
"""
Created on Tue May 27 16:41:22 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
import pandas as pd
import numpy as np

def mul_filter(target_alpha, signal_df):
    """
    通过重新索引signal来匹配目标alpha，然后相乘应用过滤器
    
    Parameters:
    -----------
    target_alpha : pd.DataFrame
        目标alpha数据，作为基准的索引和列
    signal_df : pd.DataFrame  
        信号数据，需要重新索引以匹配target_alpha
        
    Returns:
    --------
    pd.DataFrame
        过滤后的alpha结果
    """
    # 获取目标alpha的索引和列
    alpha_index = target_alpha.index
    alpha_columns = target_alpha.columns
    
    # 重新索引signal以匹配目标alpha的索引和列
    signal_aligned = signal_df.reindex(index=alpha_index, columns=alpha_columns)
    
    # 与目标alpha相乘
    result = target_alpha * signal_aligned
    
    return result


def conditional_mul_filter(alpha, pos_filter=None, neg_filter=None):
    """
    根据alpha值的正负性有条件地应用不同的过滤器
    
    Parameters:
    -----------
    alpha : pd.DataFrame or pd.Series
        原始alpha数据
    pos_filter : pd.DataFrame or pd.Series, optional
        应用于正值的过滤器，如果为None则正值保持不变
    neg_filter : pd.DataFrame or pd.Series, optional  
        应用于负值的过滤器，如果为None则负值保持不变
        
    Returns:
    --------
    pd.DataFrame or pd.Series
        过滤后的alpha结果
    """
    # 创建结果副本
    filtered_alpha = alpha.copy()
    
    # 如果提供了正值过滤器，应用到正值
    if pos_filter is not None:
        positive_mask = alpha > 0
        filtered_alpha[positive_mask] = alpha[positive_mask] * pos_filter[positive_mask]
    
    # 如果提供了负值过滤器，应用到负值
    if neg_filter is not None:
        negative_mask = alpha < 0
        filtered_alpha[negative_mask] = alpha[negative_mask] * neg_filter[negative_mask]
    
    return filtered_alpha


# 使用示例
if __name__ == "__main__":
    # 示例数据
    dates = pd.date_range('2023-01-01', periods=5)
    stocks = ['AAPL', 'GOOGL', 'MSFT']
    
    # 创建示例alpha数据
    alpha_data = pd.DataFrame(
        np.random.randn(5, 3), 
        index=dates, 
        columns=stocks
    )
    
    # 创建示例signal数据（可能有不同的索引/列）
    signal_data = pd.DataFrame(
        np.random.uniform(0.5, 1.0, (5, 3)), 
        index=dates, 
        columns=stocks
    )
    
    # 方法1：使用signal过滤
    result1 = apply_signal_filter(alpha_data, signal_data)
    print("Signal过滤结果:")
    print(result1)
    print()
    
    # 方法2：使用条件过滤
    pos_filter_data = pd.DataFrame(
        np.random.uniform(0.8, 1.2, (5, 3)), 
        index=dates, 
        columns=stocks
    )
    neg_filter_data = pd.DataFrame(
        np.random.uniform(0.6, 1.0, (5, 3)), 
        index=dates, 
        columns=stocks
    )
    
    result2 = apply_conditional_filter(alpha_data, pos_filter_data, neg_filter_data)
    print("条件过滤结果:")
    print(result2)
