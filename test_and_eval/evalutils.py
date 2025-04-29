# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 14:12:18 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
def extend_metrics(eval_res):
    """
    Extend evaluation metrics with additional calculated values.
    
    Parameters:
    -----------
    eval_res : DataFrame
        DataFrame containing the base evaluation results with metrics like
        'net_return_annualized', 'hsr', and correlation metrics.
        
    Returns:
    -----------
    eval_res : DataFrame
        The same DataFrame with additional calculated metrics.
    """
    # Calculate net_ppt for different directions
    for direction_suffix in ('', '_long_only', '_short_only'):
        eval_res[f'net_ppt{direction_suffix}'] = (eval_res[f'net_return_annualized{direction_suffix}'] 
                                                  / eval_res[f'hsr{direction_suffix}'] / 245)
    
    # Calculate average correlations for different time windows
    for corr_type in ('cont', 'dist'):
        lt720_cols = [f'corr_{corr_type}_wd30', f'corr_{corr_type}_wd60', 
                      f'corr_{corr_type}_wd240', f'corr_{corr_type}_wd720']
        eval_res[f'corr_{corr_type}_lt720_avg'] = eval_res[lt720_cols].mean(axis=1)
    
    return eval_res