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

def mul_filter(target_alpha, filter_signal):
    """
    通过重新索引signal来匹配目标alpha，然后相乘应用过滤器
    
    Parameters:
    -----------
    target_alpha : pd.DataFrame
        目标alpha数据，作为基准的索引和列
    filter_signal : pd.DataFrame  
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
    signal_aligned = filter_signal.reindex(index=alpha_index, columns=alpha_columns)
    
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
    
    # 如果提供了正值过滤器，先重新索引然后应用到正值
    if pos_filter is not None:
        # 重新索引pos_filter以匹配alpha的索引和列
        pos_filter_aligned = pos_filter.reindex(index=alpha.index, columns=alpha.columns)
        positive_mask = alpha > 0
        filtered_alpha[positive_mask] = alpha[positive_mask] * pos_filter_aligned[positive_mask]
    
    # 如果提供了负值过滤器，先重新索引然后应用到负值
    if neg_filter is not None:
        # 重新索引neg_filter以匹配alpha的索引和列
        neg_filter_aligned = neg_filter.reindex(index=alpha.index, columns=alpha.columns)
        negative_mask = alpha < 0
        filtered_alpha[negative_mask] = alpha[negative_mask] * neg_filter_aligned[negative_mask]
    
    return filtered_alpha

