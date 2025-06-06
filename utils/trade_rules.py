# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:03:02 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import pandas as pd
import numpy as np
from numba import jit, prange


# %%
def trade_rule_by_trigger_v0(signal, openthres=0.8, closethres=0):
    positions = np.full_like(signal, np.nan)  # Initialize positions as NaN
    current_position = 0
    has_valid_signal = False  # Flag to track if we've seen any valid signal
    
    for i in range(len(signal)):
        if np.isnan(signal[i]):
            if has_valid_signal:
                # If we've seen valid signals before, maintain the previous position
                positions[i] = current_position
            # else: positions[i] remains NaN (initialized value)
            continue  # Skip the rest of this iteration
        
        # We've encountered a valid signal
        has_valid_signal = True
        
        # Apply trading logic
        if current_position == 0:
            if signal[i] > openthres:
                current_position = 1
            elif signal[i] < -openthres:
                current_position = -1
        elif current_position == 1:
            if signal[i] < closethres:
                current_position = 0
        elif current_position == -1:
            if signal[i] > -closethres:
                current_position = 0
        
        positions[i] = current_position
    
    return positions


def trade_rule_by_trigger_v0_1(signal, openthres=0.8, closethres=0):
    """
    改进版本的交易规则，允许在同一个时间片内平仓后立即开仓
    
    Parameters:
    signal (array-like): 输入信号数组
    openthres (float): 开仓阈值
    closethres (float): 平仓阈值
    
    Returns:
    array: 头寸数组
    """
    positions = np.full_like(signal, np.nan)  # Initialize positions as NaN
    current_position = 0
    has_valid_signal = False  # Flag to track if we've seen any valid signal
    
    for i in range(len(signal)):
        if np.isnan(signal[i]):
            if has_valid_signal:
                # If we've seen valid signals before, maintain the previous position
                positions[i] = current_position
            # else: positions[i] remains NaN (initialized value)
            continue  # Skip the rest of this iteration
        
        # We've encountered a valid signal
        has_valid_signal = True
        
        # 应用交易逻辑 - 先检查平仓条件
        if current_position == 1:  # 多头
            if signal[i] < closethres:
                current_position = 0  # 平多
        elif current_position == -1:  # 空头
            if signal[i] > -closethres:
                current_position = 0  # 平空
        
        # 如果当前无头寸(原本就无头寸或刚刚平仓)，检查是否需要开新仓
        if current_position == 0:
            if signal[i] > openthres:
                current_position = 1  # 开多
            elif signal[i] < -openthres:
                current_position = -1  # 开空
        
        positions[i] = current_position
    
    return positions


def binary_directional_single_position(signal, trade_direction, openthres=0.8, closethres=0):
    """
    根据指定交易方向的交易规则，当不持有主动交易方向仓位时默认持有反向仓位
    
    Parameters:
    signal (array-like): 输入信号数组
    trade_direction (int): 交易方向，1代表交易多头，-1代表交易空头
    openthres (float): 开仓阈值
    closethres (float): 平仓阈值
    
    Returns:
    array: 头寸数组
    """
    positions = np.full_like(signal, np.nan)  # Initialize positions as NaN
    # 初始仓位设为反向仓位
    current_position = -trade_direction
    has_valid_signal = False  # Flag to track if we've seen any valid signal
    
    for i in range(len(signal)):
        if np.isnan(signal[i]):
            if has_valid_signal:
                # If we've seen valid signals before, maintain the previous position
                positions[i] = current_position
            # else: positions[i] remains NaN (initialized value)
            continue  # Skip the rest of this iteration
        
        # We've encountered a valid signal
        has_valid_signal = True
        
        # 根据交易方向应用不同的逻辑
        if trade_direction == 1:  # 交易多头
            if current_position == 1:  # 当前持有多头
                if signal[i] < closethres:
                    current_position = -1  # 平多并切换到空头
            else:  # 当前持有空头
                if signal[i] > openthres:
                    current_position = 1  # 开多
        elif trade_direction == -1:  # 交易空头
            if current_position == -1:  # 当前持有空头
                if signal[i] > -closethres:
                    current_position = 1  # 平空并切换到多头
            else:  # 当前持有多头
                if signal[i] < -openthres:
                    current_position = -1  # 开空
        
        positions[i] = current_position
    
    return positions

# =============================================================================
# def trade_rule_by_trigger_v0(signal, openthres=0.8, closethres=0):
#     positions = np.full_like(signal, np.nan)  # Initialize positions as NaN
#     current_position = 0
#     
#     for i in range(len(signal)):
#         if np.isnan(signal[i]):
#             positions[i] = np.nan  # Keep position as NaN if signal is NaN
#             continue  # Skip this iteration
# 
#         if current_position == 0:
#             if signal[i] > openthres:
#                 current_position = 1
#             elif signal[i] < -openthres:
#                 current_position = -1
#         elif current_position == 1:
#             if signal[i] < closethres:
#                 current_position = 0
#         elif current_position == -1:
#             if signal[i] > -closethres:
#                 current_position = 0
#         
#         positions[i] = current_position
# 
#     return positions
# =============================================================================


# =============================================================================
# def trade_rule_by_trigger_v0(signal, openthres=0.8, closethres=0):
#     positions = np.zeros_like(signal)
#     current_position = 0
#     for i in range(len(signal)):
#         if current_position == 0:
#             if signal[i] > openthres:
#                 current_position = 1
#             elif signal[i] < -openthres:  
#                 current_position = -1
#         elif current_position == 1:
#             if signal[i] < closethres:
#                 current_position = 0
#         elif current_position == -1:
#             if signal[i] > -closethres:
#                 current_position = 0
#         positions[i] = current_position
#     return positions
# =============================================================================

# # 对 DataFrame 的每列应用函数
# def apply_to_dataframe(df, openthres, closethres):
#     result = df.apply(lambda col: generate_positions(col.values, openthres, closethres), axis=0)
#     return result

# # 示例 DataFrame
# data = {
#     "signal1": [0.1, 0.5, 0.7, 0.8, 0.6, 0.2, -0.1, -0.4, -0.7, -0.8, -0.5, 0.3],
#     "signal2": [-0.1, -0.5, -0.7, -0.8, -0.6, -0.2, 0.1, 0.4, 0.7, 0.8, 0.5, -0.3],
# }
# df = pd.DataFrame(data)

# # 设置阈值
# openthres = 0.6
# closethres = 0.4

# # 生成新的 DataFrame
# positions_df = apply_to_dataframe(df, openthres, closethres)
# print(positions_df)


def trade_rule_by_trigger_v1(signal, openthres=0.8, closethres=0, trigger_thres=3):
    positions = np.zeros_like(signal)
    current_position = 0
    open_count = 0
    close_count = 0
    
    for i in range(len(signal)):
        if current_position == 0:
            if signal[i] > openthres:
                open_count += 1
                if open_count >= trigger_thres:
                    current_position = 1
                    open_count = 0  # Reset counter after opening a position
            elif signal[i] < -openthres:
                open_count += 1
                if open_count >= trigger_thres:
                    current_position = -1
                    open_count = 0  # Reset counter after opening a position
            else:
                open_count = 0  # Reset if the streak is broken
        
        elif current_position == 1:
            if signal[i] < closethres:
                close_count += 1
                if close_count >= trigger_thres:
                    current_position = 0
                    close_count = 0  # Reset counter after closing a position
            else:
                close_count = 0  # Reset if the streak is broken
        
        elif current_position == -1:
            if signal[i] > -closethres:
                close_count += 1
                if close_count >= trigger_thres:
                    current_position = 0
                    close_count = 0  # Reset counter after closing a position
            else:
                close_count = 0  # Reset if the streak is broken
        
        positions[i] = current_position
    
    return positions


def trade_rule_by_trigger_v2(signal, openthres=0.8, closethres=0, window_size=5, trigger_count=3):
    positions = np.zeros_like(signal)
    current_position = 0
    open_signals = []
    close_signals = []
    
    for i in range(len(signal)):
        if current_position == 0:
            open_signals.append(signal[i] > openthres)
            open_signals.append(signal[i] < -openthres)
            
            if len(open_signals) > window_size:
                open_signals.pop(0)
            
            if sum(open_signals) >= trigger_count:
                current_position = 1 if signal[i] > openthres else -1
                open_signals.clear()
        
        elif current_position == 1:
            close_signals.append(signal[i] < closethres)
            
            if len(close_signals) > window_size:
                close_signals.pop(0)
            
            if sum(close_signals) >= trigger_count:
                current_position = 0
                close_signals.clear()
        
        elif current_position == -1:
            close_signals.append(signal[i] > -closethres)
            
            if len(close_signals) > window_size:
                close_signals.pop(0)
            
            if sum(close_signals) >= trigger_count:
                current_position = 0
                close_signals.clear()
        
        positions[i] = current_position
    
    return positions


# =============================================================================
# def trade_rule_by_trigger_v3(signal, threshold_combinations):
#     # Create an empty list to store positions for each threshold combination
#     all_positions = []
#     
#     for openthres, closethres in threshold_combinations:
#         positions = np.zeros_like(signal)
#         current_position = 0
#         for i in range(len(signal)):
#             if current_position == 0:
#                 if signal[i] > openthres:
#                     current_position = 1
#                 elif signal[i] < -openthres:
#                     current_position = -1
#             elif current_position == 1:
#                 if signal[i] < closethres:
#                     current_position = 0
#             elif current_position == -1:
#                 if signal[i] > -closethres:
#                     current_position = 0
#             positions[i] = current_position
#         
#         all_positions.append(positions)
#     
#     # Convert the list of positions into a numpy array for easy averaging
#     all_positions = np.array(all_positions)
#     
#     # Compute the equal-weighted average of positions across all threshold combinations
#     avg_positions = np.mean(all_positions, axis=0)
#     
#     return avg_positions
# =============================================================================


# =============================================================================
# def trade_rule_by_trigger_v3(signal, threshold_combinations):
#     # Create an empty list to store positions for each threshold combination
#     all_positions = []
#     
#     # Iterate over each threshold combination
#     for openthres, closethres in threshold_combinations:
#         positions = np.full_like(signal, np.nan)  # Initialize positions as NaN
#         current_position = 0
#         
#         for i in range(len(signal)):
#             if np.isnan(signal[i]):
#                 positions[i] = np.nan  # Keep position as NaN if signal is NaN
#                 continue  # Skip this iteration
#             
#             if current_position == 0:
#                 if signal[i] > openthres:
#                     current_position = 1
#                 elif signal[i] < -openthres:
#                     current_position = -1
#             elif current_position == 1:
#                 if signal[i] < closethres:
#                     current_position = 0
#             elif current_position == -1:
#                 if signal[i] > -closethres:
#                     current_position = 0
#             positions[i] = current_position
#         
#         # Append the positions for this threshold combination
#         all_positions.append(positions)
#     
#     # Convert the list of positions into a numpy array for easy averaging
#     all_positions = np.array(all_positions)
#     
#     # Compute the equal-weighted average of positions across all threshold combinations
#     avg_positions = np.mean(all_positions, axis=0)
#     
#     # Handle NaN values in the final averaged positions: keep NaN where any position is NaN
#     avg_positions = np.where(np.isnan(avg_positions), np.nan, avg_positions)
#     
#     return avg_positions
# 
# =============================================================================


# %%
def trade_rule_by_trigger_v3(signal, threshold_combinations):
    # Create an empty list to store positions for each threshold combination
    all_positions = []
    
    # Iterate over each threshold combination
    for openthres, closethres in threshold_combinations:
        positions = np.full_like(signal, np.nan)  # Initialize positions as NaN
        current_position = 0
        has_valid_signal = False  # Flag to track if we've seen any valid signal
        
        for i in range(len(signal)):
            if np.isnan(signal[i]):
                if has_valid_signal:
                    # If we've seen valid signals before, maintain the previous position
                    positions[i] = current_position
                # else: positions[i] remains NaN (initialized value)
                continue  # Skip this iteration
            
            # We've encountered a valid signal
            has_valid_signal = True
            
            # Apply trading logic
            if current_position == 0:
                if signal[i] > openthres:
                    current_position = 1
                elif signal[i] < -openthres:
                    current_position = -1
            elif current_position == 1:
                if signal[i] < closethres:
                    current_position = 0
            elif current_position == -1:
                if signal[i] > -closethres:
                    current_position = 0
                    
            positions[i] = current_position
        
        # Append the positions for this threshold combination
        all_positions.append(positions)
    
    # Convert the list of positions into a numpy array for easy averaging
    all_positions = np.array(all_positions)
    
    # Compute the equal-weighted average of positions across all threshold combinations
    avg_positions = np.mean(all_positions, axis=0)
    
    return avg_positions


def trade_rule_by_trigger_v3_1(signal, threshold_combinations, time_threshold_minutes, close_long=True, close_short=True):
    """
    优化版本：通过向量化操作和减少循环提高性能
    
    Parameters:
    signal (pd.Series): 带有datetime索引的输入信号
    threshold_combinations (list of tuples): 每个元组包含(open_threshold, close_threshold)
    time_threshold_minutes (int or float): 超过该时间阈值（分钟）时将关闭头寸
    close_long (bool): 超过时间阈值时是否关闭多头头寸(> 0)
    close_short (bool): 超过时间阈值时是否关闭空头头寸(< 0)
    
    Returns:
    pd.Series: 所有阈值组合的平均头寸
    """
    # 将分钟转换为timedelta
    time_threshold = pd.Timedelta(minutes=time_threshold_minutes)
    
    # 检查输入类型
    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series with datetime index")
    
    # 预先计算时间差（用于快速检查时间间隔）
    time_diffs = signal.index.to_series().diff().shift(-1)
    time_gaps = time_diffs > time_threshold
    
    # 创建存储所有阈值组合结果的DataFrame
    all_positions = pd.DataFrame(index=signal.index, columns=range(len(threshold_combinations)))
    
    # 为向量化操作预处理信号数据
    signal_values = signal.values
    signal_valid = ~np.isnan(signal_values)
    signal_indices = np.arange(len(signal.index))
    
    # 对每个阈值组合进行处理
    for col_idx, (openthres, closethres) in enumerate(threshold_combinations):
        # 初始化头寸数组
        positions = np.full(len(signal.index), np.nan)
        current_position = 0
        has_valid_signal = False
        
        # 通过单次遍历更新头寸
        for i in signal_indices:
            # 检查是否需要因时间间隔而关闭头寸
            if i < len(signal.index) - 1 and time_gaps.iloc[i]:
                if (current_position < 0 and close_short) or (current_position > 0 and close_long):
                    positions[i] = 0
                    current_position = 0
                    continue
            
            # 处理无效信号
            if not signal_valid[i]:
                if has_valid_signal:
                    positions[i] = current_position
                continue
            
            # 标记遇到有效信号
            has_valid_signal = True
            
            # 应用交易逻辑
            if current_position == 0:  # 无头寸
                if signal_values[i] > openthres:
                    current_position = 1  # 开多
                elif signal_values[i] < -openthres:
                    current_position = -1  # 开空
            elif current_position == 1:  # 多头
                if signal_values[i] < closethres:
                    current_position = 0  # 平多
            elif current_position == -1:  # 空头
                if signal_values[i] > -closethres:
                    current_position = 0  # 平空
            
            positions[i] = current_position
        
        # 将此阈值组合的结果添加到DataFrame
        all_positions.iloc[:, col_idx] = positions
    
    # 计算所有阈值组合头寸的均值
    avg_positions = all_positions.mean(axis=1)
    
    return avg_positions


def trade_rule_by_trigger_v3_2(signal, threshold_combinations, time_threshold_minutes, close_long=True, close_short=True):
    """
    Second version: When the time gap between current index and next index exceeds a threshold,
    close positions based on specified conditions, but reopen the same positions at the next index.
    
    Parameters:
    signal (pd.Series): Input signal with datetime index
    threshold_combinations (list of tuples): Each tuple contains (open_threshold, close_threshold)
    time_threshold_minutes (int or float): Time threshold in minutes beyond which positions will be closed
    close_long (bool): Whether to close long positions (> 0) when time threshold is exceeded
    close_short (bool): Whether to close short positions (< 0) when time threshold is exceeded
    
    Returns:
    pd.Series: Average positions across all threshold combinations
    """
    # 将分钟转换为timedelta
    time_threshold = pd.Timedelta(minutes=time_threshold_minutes)
    
    # 检查输入类型
    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series with datetime index")
    
    # 预先计算时间差（前向和后向）
    forward_time_diffs = signal.index.to_series().diff().shift(-1)  # 当前到下一个的时间差
    backward_time_diffs = signal.index.to_series().diff()  # 前一个到当前的时间差
    
    # 创建存储所有阈值组合结果的DataFrame
    all_positions = pd.DataFrame(index=signal.index, columns=range(len(threshold_combinations)))
    
    # 为向量化操作预处理信号数据
    signal_values = signal.values
    signal_valid = ~np.isnan(signal_values)
    signal_indices = np.arange(len(signal.index))
    
    # 对每个阈值组合进行处理
    for col_idx, (openthres, closethres) in enumerate(threshold_combinations):
        # 初始化头寸数组
        positions = np.full(len(signal.index), np.nan)
        current_position = 0
        has_valid_signal = False
        position_to_reopen = 0  # 记录时间间隔后要重新开仓的头寸
        
        # 通过单次遍历更新头寸
        for i in signal_indices:
            # 检查此索引是否在时间间隔之后，需要重新开仓
            if i > 0:
                time_gap = backward_time_diffs.iloc[i]
                if time_gap > time_threshold and position_to_reopen != 0:
                    # 重新开仓
                    current_position = position_to_reopen
                    position_to_reopen = 0  # 重置
            
            # 检查是否需要记录在时间间隔后要重新开仓的头寸
            if i < len(signal.index) - 1:
                time_gap = forward_time_diffs.iloc[i]
                
                # 如果时间间隔超过阈值且满足平仓条件
                if time_gap > time_threshold:
                    if (current_position < 0 and close_short) or (current_position > 0 and close_long):
                        # 记录要重新开仓的头寸
                        position_to_reopen = current_position
                        
                        # 平仓
                        positions[i] = 0
                        current_position = 0
                        
                        # 继续下一个迭代
                        continue
            
            # 处理无效信号
            if not signal_valid[i]:
                if has_valid_signal:
                    positions[i] = current_position
                continue
            
            # 标记遇到有效信号
            has_valid_signal = True
            
            # 应用交易逻辑
            if current_position == 0:  # 无头寸
                if signal_values[i] > openthres:
                    current_position = 1  # 开多
                elif signal_values[i] < -openthres:
                    current_position = -1  # 开空
            elif current_position == 1:  # 多头
                if signal_values[i] < closethres:
                    current_position = 0  # 平多
            elif current_position == -1:  # 空头
                if signal_values[i] > -closethres:
                    current_position = 0  # 平空
            
            positions[i] = current_position
        
        # 将此阈值组合的结果添加到DataFrame
        all_positions.iloc[:, col_idx] = positions
    
    # 计算所有阈值组合头寸的均值
    avg_positions = all_positions.mean(axis=1)
    
    return avg_positions


def trade_rule_by_trigger_v3_3(signal, threshold_combinations, time_threshold_minutes, close_long=True, close_short=True):
    """
    V4.1版本：允许在同一个时间切片内，平仓后立即判断是否开新仓位
    
    Parameters:
    signal (pd.Series): 带有datetime索引的输入信号
    threshold_combinations (list of tuples): 每个元组包含(open_threshold, close_threshold)
    time_threshold_minutes (int or float): 超过该时间阈值（分钟）时将关闭头寸
    close_long (bool): 超过时间阈值时是否关闭多头头寸(> 0)
    close_short (bool): 超过时间阈值时是否关闭空头头寸(< 0)
    
    Returns:
    pd.Series: 所有阈值组合的平均头寸
    """
    # 将分钟转换为timedelta
    time_threshold = pd.Timedelta(minutes=time_threshold_minutes)
    
    # 检查输入类型
    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series with datetime index")
    
    # 预先计算时间差（用于快速检查时间间隔）
    time_diffs = signal.index.to_series().diff().shift(-1)
    time_gaps = time_diffs > time_threshold
    
    # 创建存储所有阈值组合结果的DataFrame
    all_positions = pd.DataFrame(index=signal.index, columns=range(len(threshold_combinations)))
    
    # 为向量化操作预处理信号数据
    signal_values = signal.values
    signal_valid = ~np.isnan(signal_values)
    signal_indices = np.arange(len(signal.index))
    
    # 对每个阈值组合进行处理
    for col_idx, (openthres, closethres) in enumerate(threshold_combinations):
        # 初始化头寸数组
        positions = np.full(len(signal.index), np.nan)
        current_position = 0
        has_valid_signal = False
        
        # 通过单次遍历更新头寸
        for i in signal_indices:
            # 检查是否需要因时间间隔而关闭头寸
            if i < len(signal.index) - 1 and time_gaps.iloc[i]:
                if (current_position < 0 and close_short) or (current_position > 0 and close_long):
                    current_position = 0
                    # 注意：这里不继续处理，因为我们要先记录平仓状态
            
            # 处理无效信号
            if not signal_valid[i]:
                if has_valid_signal:
                    positions[i] = current_position
                continue
            
            # 标记遇到有效信号
            has_valid_signal = True
            
            # 应用交易逻辑 - 先检查平仓条件
            if current_position == 1:  # 多头
                if signal_values[i] < closethres:
                    current_position = 0  # 平多
            elif current_position == -1:  # 空头
                if signal_values[i] > -closethres:
                    current_position = 0  # 平空
            
            # 如果当前无头寸(原本就无头寸或刚刚平仓)，检查是否需要开新仓
            if current_position == 0:
                if signal_values[i] > openthres:
                    current_position = 1  # 开多
                elif signal_values[i] < -openthres:
                    current_position = -1  # 开空
            
            positions[i] = current_position
        
        # 将此阈值组合的结果添加到DataFrame
        all_positions.iloc[:, col_idx] = positions
    
    # 计算所有阈值组合头寸的均值
    avg_positions = all_positions.mean(axis=1)
    
    return avg_positions


@jit(nopython=True)
def _compute_positions_with_time_gaps_3_4(signal_values, time_gap_flags, day_end_flags, openthres, closethres, 
                                     close_long, close_short, time_threshold_minutes):
    """
    Numba-accelerated core function to compute positions for a single threshold combination.
    添加了day_end_flags参数用于识别每天的最后一个交易时间点（如14:55）
    修改了逻辑允许在平仓后立即开新仓位
    修正：无论14:55前是否有仓位，都强制设置为平仓状态，不允许开新仓
    """
    positions = np.full(len(signal_values), np.nan)
    current_position = 0
    has_valid_signal = False
    
    for i in range(len(signal_values)):
        # 检查是否需要因时间间隔或日终而关闭头寸
        force_close = False
        
        # 如果是日终时间点（如14:55）或时间间隔间断点，强制平仓且不允许开新仓
        if day_end_flags[i] or (i < len(signal_values) - 1 and time_gap_flags[i]):
            if current_position != 0:
                if (current_position < 0 and close_short) or (current_position > 0 and close_long):
                    current_position = 0
            force_close = True  # 无论之前是否有仓位，都设置force_close为True，不允许开新仓
        
        # 处理无效信号
        if np.isnan(signal_values[i]):
            if has_valid_signal:
                positions[i] = current_position
            continue
        
        # 标记遇到有效信号
        has_valid_signal = True
        
        # 应用交易逻辑 - 先检查平仓条件
        if current_position == 1:  # 多头
            if signal_values[i] < closethres:
                current_position = 0  # 平多
        elif current_position == -1:  # 空头
            if signal_values[i] > -closethres:
                current_position = 0  # 平空
        
        # 如果当前无头寸(原本就无头寸或刚刚平仓)，检查是否需要开新仓
        # 关键修改：只有在不是强制平仓的情况下才允许开新仓
        if current_position == 0 and not force_close:
            if signal_values[i] > openthres:
                current_position = 1  # 开多
            elif signal_values[i] < -openthres:
                current_position = -1  # 开空
        
        positions[i] = current_position
    
    return positions

@jit(nopython=True, parallel=True)
def _compute_all_positions_3_4(signal_values, time_gap_flags, day_end_flags, threshold_combinations, 
                          close_long, close_short, time_threshold_minutes):
    """
    Numba-accelerated function to compute positions for all threshold combinations.
    """
    n_thresholds = len(threshold_combinations)
    n_signals = len(signal_values)
    
    # Initialize output array
    all_positions = np.full((n_thresholds, n_signals), np.nan)
    
    # Compute positions for each threshold combination in parallel
    for i in prange(n_thresholds):  # Using prange for parallel execution
        openthres = threshold_combinations[i, 0]
        closethres = threshold_combinations[i, 1]
        all_positions[i] = _compute_positions_with_time_gaps_3_4(
            signal_values, time_gap_flags, day_end_flags, openthres, closethres, 
            close_long, close_short, time_threshold_minutes
        )
    
    return all_positions

def trade_rule_by_trigger_v3_4(signal, threshold_combinations, time_threshold_minutes=None, 
                              close_long=True, close_short=True, end_time="14:55"):
    """
    Numba-accelerated version of trade rule that supports both time gap and end-time position closing
    并允许在同一时间切片内平仓后立即开仓
    
    Parameters:
    signal (pd.Series): 带有datetime索引的输入信号
    threshold_combinations (list of tuples): 每个元组包含(open_threshold, close_threshold)
    time_threshold_minutes (int or float, optional): 超过该时间阈值（分钟）时将关闭头寸，如果为None则不启用此功能
    close_long (bool): 在触发条件时是否关闭多头头寸(> 0)
    close_short (bool): 在触发条件时是否关闭空头头寸(< 0)
    end_time (str): 每日平仓的时间点，格式为"HH:MM"，如果为None则不启用此功能
    
    Returns:
    pd.Series: 所有阈值组合的平均头寸
    """
    # Check input type
    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series with datetime index")
    
    # Convert threshold_combinations to numpy array for Numba
    threshold_combinations_array = np.array(threshold_combinations, dtype=np.float64)
    
    # Pre-compute time gaps
    time_gap_flags = np.zeros(len(signal), dtype=np.bool_)
    if time_threshold_minutes is not None:
        time_threshold = pd.Timedelta(minutes=time_threshold_minutes)
        time_diffs = signal.index.to_series().diff().shift(-1)
        time_gap_flags = (time_diffs > time_threshold).values
    
    # Pre-compute day-end flags
    day_end_flags = np.zeros(len(signal), dtype=np.bool_)
    if end_time is not None:
        for i, timestamp in enumerate(signal.index):
            if timestamp.strftime("%H:%M") == end_time:
                day_end_flags[i] = True
    
    # Get signal values as numpy array
    signal_values = signal.values
    
    # Compute positions using Numba-accelerated function
    all_positions = _compute_all_positions_3_4(
        signal_values, time_gap_flags, day_end_flags, threshold_combinations_array,
        close_long, close_short, time_threshold_minutes or 0  # Use 0 as default if None
    )
    
    # Compute average positions
    avg_positions = np.nanmean(all_positions, axis=0)
    
    # Convert back to pandas Series
    return pd.Series(avg_positions, index=signal.index)


@jit(nopython=True)
def _compute_positions_with_time_gaps_3_5(signal_values, time_gap_flags, day_end_flags, no_new_position_flags, 
                                     openthres, closethres, close_long, close_short, time_threshold_minutes):
    """
    Numba-accelerated core function to compute positions for a single threshold combination.
    添加了day_end_flags参数用于识别每天的最后一个交易时间点（如14:55）
    添加了no_new_position_flags参数用于识别禁止开新仓的时间点（如14:30）
    修改了逻辑允许在平仓后立即开新仓位
    
    逻辑说明：
    - day_end_flags: 强制平仓且不允许开新仓
    - no_new_position_flags: 不允许开新仓，但老仓位根据信号正常平仓
    - time_gap_flags: 强制平仓且不允许开新仓
    """
    positions = np.full(len(signal_values), np.nan)
    current_position = 0
    has_valid_signal = False
    
    for i in range(len(signal_values)):
        # 检查是否需要因时间间隔或日终而关闭头寸
        force_close = False
        no_new_position = False
        
        # 如果是日终时间点（如14:55）或时间间隔间断点，强制平仓且不允许开新仓
        if day_end_flags[i] or (i < len(signal_values) - 1 and time_gap_flags[i]):
            if current_position != 0:
                if (current_position < 0 and close_short) or (current_position > 0 and close_long):
                    current_position = 0
            force_close = True  # 强制平仓且不允许开新仓
        
        # 如果是禁止开新仓时间点（如14:30），不允许开新仓但不强制平老仓位
        elif no_new_position_flags[i]:
            no_new_position = True  # 只是不允许开新仓，老仓位正常处理
        
        # 处理无效信号
        if np.isnan(signal_values[i]):
            if has_valid_signal:
                positions[i] = current_position
            continue
        
        # 标记遇到有效信号
        has_valid_signal = True
        
        # 应用交易逻辑 - 先检查平仓条件
        if current_position == 1:  # 多头
            if signal_values[i] < closethres:
                current_position = 0  # 平多
        elif current_position == -1:  # 空头
            if signal_values[i] > -closethres:
                current_position = 0  # 平空
        
        # 如果当前无头寸(原本就无头寸或刚刚平仓)，检查是否需要开新仓
        # 关键：只有在不是强制平仓且不是禁止开仓的情况下才允许开新仓
        if current_position == 0 and not force_close and not no_new_position:
            if signal_values[i] > openthres:
                current_position = 1  # 开多
            elif signal_values[i] < -openthres:
                current_position = -1  # 开空
        
        positions[i] = current_position
    
    return positions

@jit(nopython=True, parallel=True)
def _compute_all_positions_3_5(signal_values, time_gap_flags, day_end_flags, no_new_position_flags, 
                          threshold_combinations, close_long, close_short, time_threshold_minutes):
    """
    Numba-accelerated function to compute positions for all threshold combinations.
    """
    n_thresholds = len(threshold_combinations)
    n_signals = len(signal_values)
    
    # Initialize output array
    all_positions = np.full((n_thresholds, n_signals), np.nan)
    
    # Compute positions for each threshold combination in parallel
    for i in prange(n_thresholds):  # Using prange for parallel execution
        openthres = threshold_combinations[i, 0]
        closethres = threshold_combinations[i, 1]
        all_positions[i] = _compute_positions_with_time_gaps_3_5(
            signal_values, time_gap_flags, day_end_flags, no_new_position_flags, 
            openthres, closethres, close_long, close_short, time_threshold_minutes
        )
    
    return all_positions

def trade_rule_by_trigger_v3_5(signal, threshold_combinations, time_threshold_minutes=None, 
                              close_long=True, close_short=True, end_time="14:55", 
                              no_new_position_time="14:30"):
    """
    Numba-accelerated version of trade rule that supports both time gap and end-time position closing
    并允许在同一时间切片内平仓后立即开仓
    新增：支持禁止开新仓时间点
    
    Parameters:
    signal (pd.Series): 带有datetime索引的输入信号
    threshold_combinations (list of tuples): 每个元组包含(open_threshold, close_threshold)
    time_threshold_minutes (int or float, optional): 超过该时间阈值（分钟）时将关闭头寸，如果为None则不启用此功能
    close_long (bool): 在触发条件时是否关闭多头头寸(> 0)
    close_short (bool): 在触发条件时是否关闭空头头寸(< 0)
    end_time (str): 每日强制平仓的时间点，格式为"HH:MM"，如果为None则不启用此功能
    no_new_position_time (str): 每日禁止开新仓的时间点，格式为"HH:MM"，如果为None则不启用此功能
    
    Returns:
    pd.Series: 所有阈值组合的平均头寸
    """
    # Check input type
    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series with datetime index")
    
    # Convert threshold_combinations to numpy array for Numba
    threshold_combinations_array = np.array(threshold_combinations, dtype=np.float64)
    
    # Pre-compute time gaps
    time_gap_flags = np.zeros(len(signal), dtype=np.bool_)
    if time_threshold_minutes is not None:
        time_threshold = pd.Timedelta(minutes=time_threshold_minutes)
        time_diffs = signal.index.to_series().diff().shift(-1)
        time_gap_flags = (time_diffs > time_threshold).values
    
    # Pre-compute day-end flags (强制平仓时间)
    day_end_flags = np.zeros(len(signal), dtype=np.bool_)
    if end_time is not None:
        for i, timestamp in enumerate(signal.index):
            if timestamp.strftime("%H:%M") == end_time:
                day_end_flags[i] = True
    
    # Pre-compute no-new-position flags (禁止开新仓时间)
    no_new_position_flags = np.zeros(len(signal), dtype=np.bool_)
    if no_new_position_time is not None:
        for i, timestamp in enumerate(signal.index):
            if timestamp.strftime("%H:%M") >= no_new_position_time and timestamp.strftime("%H:%M") < end_time:
                no_new_position_flags[i] = True
    
    # Get signal values as numpy array
    signal_values = signal.values
    
    # Compute positions using Numba-accelerated function
    all_positions = _compute_all_positions_3_5(
        signal_values, time_gap_flags, day_end_flags, no_new_position_flags,
        threshold_combinations_array, close_long, close_short, time_threshold_minutes or 0
    )
    
    # Compute average positions
    avg_positions = np.nanmean(all_positions, axis=0)
    
    # Convert back to pandas Series
    return pd.Series(avg_positions, index=signal.index)


# %%
@jit(nopython=True)
def _compute_positions_with_time_gaps_3_4_t1(signal_values, time_index_values, openthres, closethres):
    """
    Numba-accelerated core function to compute positions for a single threshold combination with T+1 trading rule.
    - 当天有信号就开多头
    - 如果当天触发平仓信号，则第二天开盘时平仓
    - 如果触发平仓但第二天第一个信号又符合开仓条件，则继续持仓
    """
    positions = np.full(len(signal_values), np.nan)
    current_position = 0
    has_valid_signal = False
    close_next_day = False
    current_day = -1
    first_bar_of_day = np.zeros(len(signal_values), dtype=np.bool_)
    
    # 预处理: 识别每天的第一个交易时间点(9:31)
    for i in range(len(time_index_values)):
        day = time_index_values[i] // 10000  # 提取日期部分 (YYYYMMDD)
        time = time_index_values[i] % 10000  # 提取时间部分 (HHMM)
        
        if day != current_day:
            current_day = day
            # 标记每天的第一个交易点
            first_bar_of_day[i] = True
    
    # 重置, 用于主循环
    current_day = -1
    
    for i in range(len(signal_values)):
        day = time_index_values[i] // 10000  # 提取日期部分
        
        # 检测新的交易日
        if day != current_day:
            current_day = day
            
            # 如果新的一天开始且前一天触发了平仓信号
            if first_bar_of_day[i] and close_next_day:
                # 第二天开盘时检查是否需要平仓或继续持仓
                if np.isnan(signal_values[i]) or signal_values[i] <= openthres:
                    # 如果新的一天第一个信号不足以开仓，则执行平仓
                    current_position = 0
                # 否则，如果信号足够强，保持持仓
                close_next_day = False  # 重置平仓标志
        
        # 处理无效信号
        if np.isnan(signal_values[i]):
            if has_valid_signal:
                positions[i] = current_position
            continue
        
        # 标记遇到有效信号
        has_valid_signal = True
        
        # 应用交易逻辑 - 先检查平仓条件 (但实际平仓在次日)
        if current_position == 1:  # 多头
            if signal_values[i] < closethres:
                # 标记需要在下一个交易日开盘时平仓
                close_next_day = True
        
        # 如果当前无头寸，检查是否需要开多头
        if current_position == 0:
            if signal_values[i] > openthres:
                current_position = 1  # 只开多头
                close_next_day = False  # 取消可能的平仓标记
        
        positions[i] = current_position
    
    return positions

@jit(nopython=True, parallel=True)
def _compute_all_positions_3_4_t1(signal_values, time_index_values, threshold_combinations):
    """
    Numba-accelerated function to compute positions for all threshold combinations with T+1 rule.
    """
    n_thresholds = len(threshold_combinations)
    n_signals = len(signal_values)
    
    # Initialize output array
    all_positions = np.full((n_thresholds, n_signals), np.nan)
    
    # Compute positions for each threshold combination in parallel
    for i in prange(n_thresholds):  # Using prange for parallel execution
        openthres = threshold_combinations[i, 0]
        closethres = threshold_combinations[i, 1]
        all_positions[i] = _compute_positions_with_time_gaps_3_4_t1(
            signal_values, time_index_values, openthres, closethres
        )
    
    return all_positions

def trade_rule_by_trigger_v3_4_t1(signal, threshold_combinations):
    """
    适用于T+1交易规则的版本：
    - 当天有信号就开多头
    - 如果当天触发平仓信号，则第二天开盘时(9:31)平仓
    - 如果触发平仓但第二天第一个信号又符合开仓条件，则继续持仓
    - 支持多组开平仓参数，并计算平均仓位
    
    Parameters:
    signal (pd.Series): 带有datetime索引的输入信号
    threshold_combinations (list of tuples): 每个元组包含(open_threshold, close_threshold)
    
    Returns:
    pd.Series: 所有阈值组合的平均头寸
    """
    # Check input type
    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series with datetime index")
    
    # Convert threshold_combinations to numpy array for Numba
    threshold_combinations_array = np.array(threshold_combinations, dtype=np.float64)
    
    # 创建数值化的时间索引 (格式: YYYYMMDDHHMM)
    time_index_values = np.array([
        int(ts.strftime('%Y%m%d%H%M')) for ts in signal.index
    ], dtype=np.int64)
    
    # Get signal values as numpy array
    signal_values = signal.values
    
    # Compute positions using Numba-accelerated function
    all_positions = _compute_all_positions_3_4_t1(
        signal_values, time_index_values, threshold_combinations_array
    )
    
    # Compute average positions
    avg_positions = np.nanmean(all_positions, axis=0)
    
    # Convert back to pandas Series
    return pd.Series(avg_positions, index=signal.index)

# 使用示例:
# import pandas as pd
# import numpy as np
# from numba import jit, prange
# 
# # 创建测试数据
# dates = pd.date_range('2023-01-01 09:31:00', '2023-01-05 15:00:00', freq='1min')
# # 过滤掉非交易时间
# mask = ((dates.hour >= 9) & (dates.minute >= 31) | (dates.hour >= 10)) & (dates.hour < 15)
# dates = dates[mask]
# 
# # 创建随机信号数据
# np.random.seed(42)
# signal_data = np.random.randn(len(dates))
# signal = pd.Series(signal_data, index=dates)
# 
# # 定义多组开平仓阈值
# threshold_combinations = [(0.5, 0.2), (0.6, 0.3), (0.7, 0.4)]
# 
# # 运行T+1交易策略
# positions = trade_rule_by_trigger_v3_4_t1(signal, threshold_combinations)
# 
# # 查看结果
# print(positions.head(20))


# %%
def trade_rule_by_trigger_v4(signal, price, openthres=0.8, closethres=0, stoploss_pct=0.05, takeprofit_drawdown_pct=0.03):
    positions = np.full_like(signal, np.nan)  # Initialize positions as NaN
    current_position = 0
    has_valid_signal = False  # Flag to track if we've seen any valid signal
    
    entry_price = None  # Price at which position was opened
    highest_profit_price = None  # Highest price reached during profitable long position
    lowest_profit_price = None  # Lowest price reached during profitable short position
    
    for i in range(len(signal)):
        if np.isnan(signal[i]) or np.isnan(price[i]):
            if has_valid_signal:
                # If we've seen valid signals before, maintain the previous position
                positions[i] = current_position
            # else: positions[i] remains NaN (initialized value)
            continue  # Skip the rest of this iteration
        
        # We've encountered a valid signal
        has_valid_signal = True
        
        # Store previous position before applying logic
        prev_position = current_position
        
        # Apply standard trading logic first
        if current_position == 0:
            if signal[i] > openthres:
                current_position = 1
                entry_price = price[i]
                highest_profit_price = price[i]
            elif signal[i] < -openthres:
                current_position = -1
                entry_price = price[i]
                lowest_profit_price = price[i]
        elif current_position == 1:
            # Update highest price for take-profit tracking in long position
            if price[i] > highest_profit_price:
                highest_profit_price = price[i]
                
            # Check for stop-loss (price falls below entry by stoploss_pct)
            if price[i] <= entry_price * (1 - stoploss_pct):
                current_position = 0
            # Check for take-profit (price falls from highest by takeprofit_drawdown_pct)
            elif price[i] <= highest_profit_price * (1 - takeprofit_drawdown_pct):
                current_position = 0
            # Check for standard close signal
            elif signal[i] < closethres:
                current_position = 0
        elif current_position == -1:
            # Update lowest price for take-profit tracking in short position
            if price[i] < lowest_profit_price:
                lowest_profit_price = price[i]
                
            # Check for stop-loss (price rises above entry by stoploss_pct)
            if price[i] >= entry_price * (1 + stoploss_pct):
                current_position = 0
            # Check for take-profit (price rises from lowest by takeprofit_drawdown_pct)
            elif price[i] >= lowest_profit_price * (1 + takeprofit_drawdown_pct):
                current_position = 0
            # Check for standard close signal
            elif signal[i] > -closethres:
                current_position = 0
        
        # Reset tracking variables if position closed
        if prev_position != 0 and current_position == 0:
            entry_price = None
            highest_profit_price = None
            lowest_profit_price = None
        
        positions[i] = current_position
    
    return positions


def trade_rule_by_trigger_v5(signal, price, openthres=0.8, closethres=0, stoploss_pct=0.05, takeprofit_drawdown_pct=0.03):
    positions = np.full_like(signal, np.nan)  # Initialize positions as NaN
    current_position = 0
    has_valid_signal = False  # Flag to track if we've seen any valid signal
    
    entry_price = None  # Price at which position was opened
    highest_profit_price = None  # Highest price reached during profitable long position
    lowest_profit_price = None  # Lowest price reached during profitable short position
    
    for i in range(len(signal)):
        if np.isnan(signal[i]) or np.isnan(price[i]):
            if has_valid_signal:
                # If we've seen valid signals before, maintain the previous position
                positions[i] = current_position
            # else: positions[i] remains NaN (initialized value)
            continue  # Skip the rest of this iteration
        
        # We've encountered a valid signal
        has_valid_signal = True
        
        # Store previous position before applying logic
        prev_position = current_position
        
        # Apply standard trading logic first
        if current_position == 0:
            if signal[i] > openthres:
                current_position = 1
                entry_price = price[i]
                highest_profit_price = price[i]
            elif signal[i] < -openthres:
                current_position = -1
                entry_price = price[i]
                lowest_profit_price = price[i]
        elif current_position == 1:
            # If a new long signal comes in (signal[i] > openthres), reset the stop-loss and take-profit points
            if signal[i] > openthres:
                entry_price = price[i]
                highest_profit_price = price[i]
            # Update highest price for take-profit tracking in long position
            if price[i] > highest_profit_price:
                highest_profit_price = price[i]
                
            # Check for stop-loss (price falls below entry by stoploss_pct)
            if price[i] <= entry_price * (1 - stoploss_pct):
                current_position = 0
            # Check for take-profit (price falls from highest by takeprofit_drawdown_pct)
            elif price[i] <= highest_profit_price * (1 - takeprofit_drawdown_pct):
                current_position = 0
            # Check for standard close signal
            elif signal[i] < closethres:
                current_position = 0
        elif current_position == -1:
            # If a new short signal comes in (signal[i] < -openthres), reset the stop-loss and take-profit points
            if signal[i] < -openthres:
                entry_price = price[i]
                lowest_profit_price = price[i]
            # Update lowest price for take-profit tracking in short position
            if price[i] < lowest_profit_price:
                lowest_profit_price = price[i]
                
            # Check for stop-loss (price rises above entry by stoploss_pct)
            if price[i] >= entry_price * (1 + stoploss_pct):
                current_position = 0
            # Check for take-profit (price rises from lowest by takeprofit_drawdown_pct)
            elif price[i] >= lowest_profit_price * (1 + takeprofit_drawdown_pct):
                current_position = 0
            # Check for standard close signal
            elif signal[i] > -closethres:
                current_position = 0
        
        # Reset tracking variables if position closed
        if prev_position != 0 and current_position == 0:
            entry_price = None
            highest_profit_price = None
            lowest_profit_price = None
        
        positions[i] = current_position
    
    return positions


def trade_rule_by_trigger_v6(signal, price, openthres=0.8, closethres=0, stoploss_pct=0.05, takeprofit_pct=0.1):
    positions = np.full_like(signal, np.nan)  # Initialize positions as NaN
    current_position = 0
    has_valid_signal = False  # Flag to track if we've seen any valid signal
    
    entry_price = None  # Price at which position was opened
    
    for i in range(len(signal)):
        if np.isnan(signal[i]) or np.isnan(price[i]):
            if has_valid_signal:
                # If we've seen valid signals before, maintain the previous position
                positions[i] = current_position
            # else: positions[i] remains NaN (initialized value)
            continue  # Skip the rest of this iteration
        
        # We've encountered a valid signal
        has_valid_signal = True
        
        # Store previous position before applying logic
        prev_position = current_position
        
        # Apply standard trading logic first
        if current_position == 0:
            if signal[i] > openthres:
                current_position = 1
                entry_price = price[i]
            elif signal[i] < -openthres:
                current_position = -1
                entry_price = price[i]
        elif current_position == 1:
            # If a new long signal comes in (signal[i] > openthres), reset the stop-loss and take-profit points
            if signal[i] > openthres:
                entry_price = price[i]
                
            # Check for stop-loss (price falls below entry by stoploss_pct)
            if price[i] <= entry_price * (1 - stoploss_pct):
                current_position = 0
            # Check for take-profit (price rises above entry by takeprofit_pct)
            elif price[i] >= entry_price * (1 + takeprofit_pct):
                current_position = 0
            # Check for standard close signal
            elif signal[i] < closethres:
                current_position = 0
        elif current_position == -1:
            # If a new short signal comes in (signal[i] < -openthres), reset the stop-loss and take-profit points
            if signal[i] < -openthres:
                entry_price = price[i]
                
            # Check for stop-loss (price rises above entry by stoploss_pct)
            if price[i] >= entry_price * (1 + stoploss_pct):
                current_position = 0
            # Check for take-profit (price falls below entry by takeprofit_pct)
            elif price[i] <= entry_price * (1 - takeprofit_pct):
                current_position = 0
            # Check for standard close signal
            elif signal[i] > -closethres:
                current_position = 0
        
        # Reset tracking variables if position closed
        if prev_position != 0 and current_position == 0:
            entry_price = None
        
        positions[i] = current_position
    
    return positions


# %%
def trade_rule_by_reversal(signal, 
                          threshold=0.8, 
                          observation_period=15, 
                          min_observation_periods=3,
                          slope_threshold=0.05, 
                          holding_period=30, 
                          close_on_threshold_retrigger=False):
    """
    基于信号反转的交易规则:
    1. 在信号突破阈值后观察一段时间(observation_period)
    2. 如果在观察期内信号反转且满足斜率条件，开反向仓位
    3. 持仓固定时间后平仓
    4. 可选择是否在再次触发阈值时平仓
    
    参数:
    - signal: 输入信号数组
    - threshold: 触发反转观察的阈值
    - observation_period: 突破阈值后的观察期 (多少个切片)
    - min_observation_periods: 开仓前至少需要观察的最小切片数
    - slope_threshold: 反转斜率阈值，每个周期应下降/上升的最小幅度
    - holding_period: 开仓后的固定持仓时间 (多少个切片)
    - close_on_threshold_retrigger: 若为True，则在再次触发阈值时平仓
    
    返回:
    - positions: 仓位数组，值为 1(多), -1(空), 0(不持仓), np.nan(无有效信号)
    """
    positions = np.full_like(signal, np.nan)  # 初始化仓位为NaN
    current_position = 0
    has_valid_signal = False  # 标记是否已有有效信号
    
    # 跟踪反转观察状态和持仓状态
    watching_reversal = False  # 是否正在观察反转
    reversal_start_idx = 0  # 开始观察反转的索引
    trigger_value = 0  # 触发观察的信号值
    position_start_idx = 0  # 开仓的索引
    
    for i in range(len(signal)):
        if np.isnan(signal[i]):
            if has_valid_signal:
                # 如果之前有有效信号，保持前一个仓位
                positions[i] = current_position
            continue  # 跳过本次迭代
        
        # 已有有效信号
        has_valid_signal = True
        
        # 检查是否应该平仓 (基于固定持仓时间)
        if current_position != 0 and i - position_start_idx >= holding_period:
            current_position = 0
            watching_reversal = False  # 平仓后重置观察状态
        
        # 检查是否在阈值再触发时平仓
        if close_on_threshold_retrigger and current_position != 0:
            # 持有反转空仓时，如果信号再次突破正阈值则平仓
            if current_position == -1 and signal[i] > threshold:
                current_position = 0
                watching_reversal = False  # 平仓后重置观察状态
            # 持有反转多仓时，如果信号再次突破负阈值则平仓
            elif current_position == 1 and signal[i] < -threshold:
                current_position = 0
                watching_reversal = False  # 平仓后重置观察状态
        
        # 检查是否应该开始观察反转
        if not watching_reversal and current_position == 0:
            if signal[i] > threshold or signal[i] < -threshold:
                watching_reversal = True
                reversal_start_idx = i
                trigger_value = signal[i]
        
        # 如果正在观察反转，检查是否满足反转条件
        elif watching_reversal and current_position == 0:
            periods_passed = i - reversal_start_idx
            
            # 只在观察期内检查反转条件
            if periods_passed <= observation_period:
                # 计算期望的反转信号值（基于线性斜率）
                expected_reversal = trigger_value - (periods_passed * slope_threshold * np.sign(trigger_value))
                
                # 只有在经过了最小观察期后才考虑开仓
                if periods_passed >= min_observation_periods:
                    # 信号从正阈值反转向下
                    if trigger_value > threshold and signal[i] < expected_reversal:
                        current_position = -1  # 开空仓
                        position_start_idx = i
                        watching_reversal = False
                    
                    # 信号从负阈值反转向上
                    elif trigger_value < -threshold and signal[i] > expected_reversal:
                        current_position = 1  # 开多仓
                        position_start_idx = i
                        watching_reversal = False
            else:
                # 超过观察期，停止观察
                watching_reversal = False
        
        positions[i] = current_position
    
    return positions


# 变种v1: 每次出现新的原始开仓信号，刷新开始观察点
# 效果：变差很多，有可能会出现持续有信号后信号消失的情况，不属于要捕捉的反转
def trade_rule_by_reversal_v1(signal, 
                          threshold=0.8, 
                          observation_period=15, 
                          min_observation_periods=3,
                          slope_threshold=0.05, 
                          holding_period=30, 
                          close_on_threshold_retrigger=False):
    """
    基于信号反转的交易规则:
    1. 在信号突破阈值后观察一段时间(observation_period)
    2. 如果在观察期内信号反转且满足斜率条件，开反向仓位
    3. 持仓固定时间后平仓
    4. 可选择是否在再次触发阈值时平仓
    5. 每次触发阈值时重置观察状态
    
    参数:
    - signal: 输入信号数组
    - threshold: 触发反转观察的阈值
    - observation_period: 突破阈值后的观察期 (多少个切片)
    - min_observation_periods: 开仓前至少需要观察的最小切片数
    - slope_threshold: 反转斜率阈值，每个周期应下降/上升的最小幅度
    - holding_period: 开仓后的固定持仓时间 (多少个切片)
    - close_on_threshold_retrigger: 若为True，则在再次触发阈值时平仓
    
    返回:
    - positions: 仓位数组，值为 1(多), -1(空), 0(不持仓), np.nan(无有效信号)
    """
    positions = np.full_like(signal, np.nan)  # 初始化仓位为NaN
    current_position = 0
    has_valid_signal = False  # 标记是否已有有效信号
    
    # 跟踪反转观察状态和持仓状态
    watching_reversal = False  # 是否正在观察反转
    reversal_start_idx = 0  # 开始观察反转的索引
    trigger_value = 0  # 触发观察的信号值
    position_start_idx = 0  # 开仓的索引
    
    for i in range(len(signal)):
        if np.isnan(signal[i]):
            if has_valid_signal:
                # 如果之前有有效信号，保持前一个仓位
                positions[i] = current_position
            continue  # 跳过本次迭代
        
        # 已有有效信号
        has_valid_signal = True
        
        # 检查是否应该平仓 (基于固定持仓时间)
        if current_position != 0 and i - position_start_idx >= holding_period:
            current_position = 0
            watching_reversal = False  # 平仓后重置观察状态
        
        # 检查是否在阈值再触发时平仓
        if close_on_threshold_retrigger and current_position != 0:
            # 持有反转空仓时，如果信号再次突破正阈值则平仓
            if current_position == -1 and signal[i] > threshold:
                current_position = 0
                watching_reversal = False  # 平仓后重置观察状态
            # 持有反转多仓时，如果信号再次突破负阈值则平仓
            elif current_position == 1 and signal[i] < -threshold:
                current_position = 0
                watching_reversal = False  # 平仓后重置观察状态
        
        # 每次信号突破阈值时都重置观察状态
        if signal[i] > threshold or signal[i] < -threshold:
            watching_reversal = True
            reversal_start_idx = i
            trigger_value = signal[i]
        
        # 如果正在观察反转，检查是否满足反转条件
        if watching_reversal and current_position == 0:
            periods_passed = i - reversal_start_idx
            
            # 只在观察期内检查反转条件
            if periods_passed <= observation_period:
                # 计算期望的反转信号值（基于线性斜率）
                expected_reversal = trigger_value - (periods_passed * slope_threshold * np.sign(trigger_value))
                
                # 只有在经过了最小观察期后才考虑开仓
                if periods_passed >= min_observation_periods:
                    # 信号从正阈值反转向下
                    if trigger_value > threshold and signal[i] < expected_reversal:
                        current_position = -1  # 开空仓
                        position_start_idx = i
                        watching_reversal = False
                    
                    # 信号从负阈值反转向上
                    elif trigger_value < -threshold and signal[i] > expected_reversal:
                        current_position = 1  # 开多仓
                        position_start_idx = i
                        watching_reversal = False
            else:
                # 超过观察期，停止观察
                watching_reversal = False
        
        positions[i] = current_position
    
    return positions


# 变种v2: 计算从最高点的反弹，而非初始触发点
# 效果：略微变差，较好的参数范围移动
def trade_rule_by_reversal_v2(signal, 
                          threshold=0.8, 
                          observation_period=15, 
                          min_observation_periods=3,
                          slope_threshold=0.05, 
                          holding_period=30, 
                          close_on_threshold_retrigger=False):
    """
    基于信号反转的交易规则:
    1. 在信号突破阈值后观察一段时间(observation_period)
    2. 如果在观察期内信号从极值点反转且满足斜率条件，开反向仓位
    3. 持仓固定时间后平仓
    4. 可选择是否在再次触发阈值时平仓
    
    参数:
    - signal: 输入信号数组
    - threshold: 触发反转观察的阈值
    - observation_period: 突破阈值后的观察期 (多少个切片)
    - min_observation_periods: 开仓前至少需要观察的最小切片数
    - slope_threshold: 反转斜率阈值，每个周期应下降/上升的最小幅度
    - holding_period: 开仓后的固定持仓时间 (多少个切片)
    - close_on_threshold_retrigger: 若为True，则在再次触发阈值时平仓
    
    返回:
    - positions: 仓位数组，值为 1(多), -1(空), 0(不持仓), np.nan(无有效信号)
    """
    import numpy as np
    
    positions = np.full_like(signal, np.nan)  # 初始化仓位为NaN
    current_position = 0
    has_valid_signal = False  # 标记是否已有有效信号
    
    # 跟踪反转观察状态和持仓状态
    watching_reversal = False  # 是否正在观察反转
    reversal_start_idx = 0  # 开始观察反转的索引
    position_start_idx = 0  # 开仓的索引
    extreme_value = 0  # 观察期内的极值点
    extreme_idx = 0  # 极值点的索引
    
    for i in range(len(signal)):
        if np.isnan(signal[i]):
            if has_valid_signal:
                # 如果之前有有效信号，保持前一个仓位
                positions[i] = current_position
            continue  # 跳过本次迭代
        
        # 已有有效信号
        has_valid_signal = True
        
        # 检查是否应该平仓 (基于固定持仓时间)
        if current_position != 0 and i - position_start_idx >= holding_period:
            current_position = 0
            watching_reversal = False  # 平仓后重置观察状态
        
        # 检查是否在阈值再触发时平仓
        if close_on_threshold_retrigger and current_position != 0:
            # 持有反转空仓时，如果信号再次突破正阈值则平仓
            if current_position == -1 and signal[i] > threshold:
                current_position = 0
                watching_reversal = False  # 平仓后重置观察状态
            # 持有反转多仓时，如果信号再次突破负阈值则平仓
            elif current_position == 1 and signal[i] < -threshold:
                current_position = 0
                watching_reversal = False  # 平仓后重置观察状态
        
        # 检查是否应该开始观察反转
        if not watching_reversal and current_position == 0:
            if signal[i] > threshold or signal[i] < -threshold:
                watching_reversal = True
                reversal_start_idx = i
                extreme_value = signal[i]  # 初始化极值为触发值
                extreme_idx = i  # 初始化极值索引
        
        # 如果正在观察反转，检查是否满足反转条件
        elif watching_reversal and current_position == 0:
            periods_passed = i - reversal_start_idx
            
            # 更新极值
            if signal[i] * np.sign(extreme_value) > abs(extreme_value):
                extreme_value = signal[i]
                extreme_idx = i
            
            # 只在观察期内检查反转条件
            if periods_passed <= observation_period:
                # 从极值点开始计算
                periods_from_extreme = i - extreme_idx
                
                # 计算期望的反转信号值（基于从极值点开始的线性斜率）
                expected_reversal = extreme_value - (periods_from_extreme * slope_threshold * np.sign(extreme_value))
                
                # 只有在经过了最小观察期后才考虑开仓，同时确保从极值点已经经过至少1个周期
                if periods_passed >= min_observation_periods and periods_from_extreme >= 1:
                    # 信号从正极值反转向下
                    if extreme_value > threshold and signal[i] < expected_reversal:
                        current_position = -1  # 开空仓
                        position_start_idx = i
                        watching_reversal = False
                    
                    # 信号从负极值反转向上
                    elif extreme_value < -threshold and signal[i] > expected_reversal:
                        current_position = 1  # 开多仓
                        position_start_idx = i
                        watching_reversal = False
            else:
                # 超过观察期，停止观察
                watching_reversal = False
        
        positions[i] = current_position
    
    return positions


# %%
def trade_rule_by_reversal_v3(signal, 
                              threshold=0.8, 
                              observation_period=15, 
                              min_observation_periods=3,
                              slope_threshold=0.05, 
                              max_slope_periods=5,  # 新增参数：斜率计算的最大周期数
                              holding_period=30, 
                              close_on_opposite_threshold=True,
                              time_gap_minutes=240,  # 4小时
                              cooldown_minutes=30,
                              lookback_periods=5):
    """
    基于信号反转的交易规则(修改版):
    1. 在信号突破阈值后观察一段时间(observation_period)
    2. 如果在观察期内信号反转且满足斜率条件，开反向仓位
    3. 持仓固定时间后平仓
    4. 当触发对侧阈值时平仓
    5. 考虑时间间隔，处理隔夜和午休情况
    6. 添加冷却期和前n个周期检查
    
    参数:
    - signal: 输入信号Series，index为timestamp
    - threshold: 触发反转观察的阈值
    - observation_period: 突破阈值后的观察期 (多少个切片)
    - min_observation_periods: 开仓前至少需要观察的最小切片数
    - slope_threshold: 反转斜率阈值，每个周期应下降/上升的最小幅度
    - holding_period: 开仓后的固定持仓时间 (多少个切片)
    - close_on_opposite_threshold: 若为True，则在触发对侧阈值时平仓
    - time_gap_minutes: 时间间隔阈值(分钟)，大于此值视为隔夜或午休
    - cooldown_minutes: 隔夜或午休后的冷却期(分钟)
    - lookback_periods: 触发观察前需要检查的前n个周期
    - max_slope_periods: 斜率计算的最大周期数，默认为5
    
    返回:
    - positions: 仓位Series，值为 1(多), -1(空), 0(不持仓), np.nan(无有效信号)
    """

    # 确保输入是Series
    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series with timestamp index")
    
    # 初始化结果Series
    positions = pd.Series(np.nan, index=signal.index)
    
    # 初始化状态变量
    current_position = 0
    watching_reversal = False
    reversal_start_idx = None
    trigger_value = 0
    position_start_idx = None
    last_timestamp = None
    in_cooldown = False
    cooldown_end_time = None
    
    # 遍历每个时间点
    for timestamp, value in signal.items():
        if np.isnan(value):
            if current_position != 0:
                positions[timestamp] = current_position
            continue
        
        # 检查时间间隔
        if last_timestamp is not None:
            time_diff = (timestamp - last_timestamp).total_seconds() / 60
            
            # 如果超过时间间隔阈值，认为是隔夜或午休
            if time_diff > time_gap_minutes:
                # 重置信号观察状态
                watching_reversal = False
                # 设置冷却期
                in_cooldown = True
                cooldown_end_time = timestamp + pd.Timedelta(minutes=cooldown_minutes)
        
        # 更新最后一个时间戳
        last_timestamp = timestamp
        
        # 检查是否在冷却期内
        if in_cooldown:
            if timestamp >= cooldown_end_time:
                in_cooldown = False
            else:
                # 在冷却期内保持原有仓位，不进行任何信号观察
                positions[timestamp] = current_position
                continue
        
        # 检查是否应该平仓 (基于固定持仓时间)
        if current_position != 0 and position_start_idx is not None:
            periods_held = len(signal.loc[position_start_idx:timestamp])
            if periods_held > holding_period:
                current_position = 0
                watching_reversal = False
        
        # 检查是否在触发对侧阈值时平仓
        if close_on_opposite_threshold and current_position != 0:
            # 持有空仓时，如果信号大于正阈值或小于负阈值则平仓
            if current_position == -1 and (value > threshold or value < -threshold):
                current_position = 0
                watching_reversal = False
            # 持有多仓时，如果信号大于正阈值或小于负阈值则平仓
            elif current_position == 1 and (value > threshold or value < -threshold):
                current_position = 0
                watching_reversal = False
        
        # 检查是否应该开始观察反转
        if not watching_reversal and current_position == 0:
            if (value > threshold or value < -threshold):
                # 检查lookback_periods分钟内的数据点是否都未触发阈值
                valid_start = True
                if lookback_periods > 0:
                    # 计算lookback时间窗口的起始时间
                    lookback_start_time = timestamp - pd.Timedelta(minutes=lookback_periods)
                    
                    # 获取lookback时间窗口内的所有数据点
                    lookback_data = signal.loc[lookback_start_time:timestamp].iloc[:-1]  # 不包括当前点
                    
                    # 检查这些点是否有触发阈值的
                    for prev_ts, prev_val in lookback_data.items():
                        if not np.isnan(prev_val) and (prev_val > threshold or prev_val < -threshold):
                            valid_start = False
                            break
                
                if valid_start:
                    watching_reversal = True
                    reversal_start_idx = timestamp
                    trigger_value = value
        
        # 如果正在观察反转，检查是否满足反转条件
        elif watching_reversal and current_position == 0:
            periods_passed = len(signal.loc[reversal_start_idx:timestamp])
            
            # 只在观察期内检查反转条件
            if periods_passed <= observation_period:
                # 计算期望的反转信号值（基于线性斜率，但有上限）
                max_reversal = slope_threshold * min(periods_passed, max_slope_periods)
                expected_reversal = trigger_value - (max_reversal * np.sign(trigger_value))
                
                # 只有在经过了最小观察期后才考虑开仓
                if periods_passed >= min_observation_periods:
                    # 信号从正阈值反转向下
                    if trigger_value > threshold and value < expected_reversal:
                        current_position = -1  # 开空仓
                        position_start_idx = timestamp
                        watching_reversal = False
                    
                    # 信号从负阈值反转向上
                    elif trigger_value < -threshold and value > expected_reversal:
                        current_position = 1  # 开多仓
                        position_start_idx = timestamp
                        watching_reversal = False
            else:
                # 超过观察期，停止观察
                watching_reversal = False
        
        positions[timestamp] = current_position
    
    return positions


# 变种v4：止损止盈
def trade_rule_by_reversal_v4(signal, price, 
                              threshold=0.8, 
                              observation_period=15, 
                              min_observation_periods=3,
                              slope_threshold=0.05, 
                              max_slope_periods=5,
                              holding_period=30, 
                              close_on_opposite_threshold=True,
                              time_gap_minutes=240,  # 4小时
                              cooldown_minutes=30,
                              lookback_periods=5,
                              stop_loss_minutes=15,  # 止损判断的最小持仓时间
                              stop_loss_threshold=-0.001):  # 止损阈值，负值表示亏损比例
    """
    基于信号反转的交易规则(修改版V4):
    1. 在信号突破阈值后观察一段时间(observation_period)
    2. 如果在观察期内信号反转且满足斜率条件，开反向仓位
    3. 持仓固定时间后平仓
    4. 当触发对侧阈值时平仓
    5. 考虑时间间隔，处理隔夜和午休情况
    6. 添加冷却期和前n个周期检查
    7. 新增：当持仓超过stop_loss_minutes且收益率低于stop_loss_threshold时平仓(止损)
    
    参数:
    - signal: 输入信号Series，index为timestamp
    - price: 价格Series，与signal等长且index一致
    - threshold: 触发反转观察的阈值
    - observation_period: 突破阈值后的观察期 (多少个切片)
    - min_observation_periods: 开仓前至少需要观察的最小切片数
    - slope_threshold: 反转斜率阈值，每个周期应下降/上升的最小幅度
    - max_slope_periods: 斜率计算的最大周期数
    - holding_period: 开仓后的固定持仓时间 (多少个切片)
    - close_on_opposite_threshold: 若为True，则在触发对侧阈值时平仓
    - time_gap_minutes: 时间间隔阈值(分钟)，大于此值视为隔夜或午休
    - cooldown_minutes: 隔夜或午休后的冷却期(分钟)
    - lookback_periods: 触发观察前需要检查的前n个周期
    - stop_loss_minutes: 止损判断的最小持仓时间
    - stop_loss_threshold: 止损阈值，低于此收益率时平仓
    
    返回:
    - positions: 仓位Series，值为 1(多), -1(空), 0(不持仓), np.nan(无有效信号)
    """
    import pandas as pd
    import numpy as np
    
    # 确保输入是Series
    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series with timestamp index")
    if not isinstance(price, pd.Series):
        raise TypeError("price must be a pandas Series with timestamp index")
    
    # 确保signal和price有相同的index
    if not signal.index.equals(price.index):
        raise ValueError("signal and price must have the same index")
    
    # 初始化结果Series
    positions = pd.Series(np.nan, index=signal.index)
    
    # 初始化状态变量
    current_position = 0
    watching_reversal = False
    reversal_start_idx = None
    trigger_value = 0
    position_start_idx = None
    entry_price = None  # 开仓价格
    last_timestamp = None
    in_cooldown = False
    cooldown_end_time = None
    
    # 遍历每个时间点
    for timestamp, value in signal.items():
        current_price = price[timestamp]
        
        if np.isnan(value) or np.isnan(current_price):
            if current_position != 0:
                positions[timestamp] = current_position
            continue
        
        # 检查时间间隔
        if last_timestamp is not None:
            time_diff = (timestamp - last_timestamp).total_seconds() / 60
            
            # 如果超过时间间隔阈值，认为是隔夜或午休
            if time_diff > time_gap_minutes:
                # 重置信号观察状态
                watching_reversal = False
                # 设置冷却期
                in_cooldown = True
                cooldown_end_time = timestamp + pd.Timedelta(minutes=cooldown_minutes)
        
        # 更新最后一个时间戳
        last_timestamp = timestamp
        
        # 检查是否在冷却期内
        if in_cooldown:
            if timestamp >= cooldown_end_time:
                in_cooldown = False
            else:
                # 在冷却期内保持原有仓位，不进行任何信号观察
                positions[timestamp] = current_position
                continue
        
        # 止损检查：如果持有仓位且超过最小止损时间
        if current_position != 0 and position_start_idx is not None and entry_price is not None:
            periods_held = len(signal.loc[position_start_idx:timestamp])
            
            # 计算当前收益率
            if periods_held >= stop_loss_minutes:
                if current_position == 1:  # 多仓
                    returns = (current_price / entry_price) - 1
                else:  # 空仓
                    returns = 1 - (current_price / entry_price)
                
                # 如果收益率低于止损阈值，平仓
                if returns < stop_loss_threshold:
                    current_position = 0
                    watching_reversal = False
                    entry_price = None
                    positions[timestamp] = current_position
                    continue
        
        # 检查是否应该平仓 (基于固定持仓时间)
        if current_position != 0 and position_start_idx is not None:
            periods_held = len(signal.loc[position_start_idx:timestamp])
            if periods_held > holding_period:
                current_position = 0
                watching_reversal = False
                entry_price = None
        
        # 检查是否在触发对侧阈值时平仓
        if close_on_opposite_threshold and current_position != 0:
            # 持有空仓时，如果信号大于正阈值或小于负阈值则平仓
            if current_position == -1 and (value > threshold or value < -threshold):
                current_position = 0
                watching_reversal = False
                entry_price = None
            # 持有多仓时，如果信号大于正阈值或小于负阈值则平仓
            elif current_position == 1 and (value > threshold or value < -threshold):
                current_position = 0
                watching_reversal = False
                entry_price = None
        
        # 检查是否应该开始观察反转
        if not watching_reversal and current_position == 0:
            if (value > threshold or value < -threshold):
                # 检查lookback_periods分钟内的数据点是否都未触发阈值
                valid_start = True
                if lookback_periods > 0:
                    # 计算lookback时间窗口的起始时间
                    lookback_start_time = timestamp - pd.Timedelta(minutes=lookback_periods)
                    
                    # 获取lookback时间窗口内的所有数据点
                    lookback_data = signal.loc[lookback_start_time:timestamp].iloc[:-1]  # 不包括当前点
                    
                    # 检查这些点是否有触发阈值的
                    for prev_ts, prev_val in lookback_data.items():
                        if not np.isnan(prev_val) and (prev_val > threshold or prev_val < -threshold):
                            valid_start = False
                            break
                
                if valid_start:
                    watching_reversal = True
                    reversal_start_idx = timestamp
                    trigger_value = value
        
        # 如果正在观察反转，检查是否满足反转条件
        elif watching_reversal and current_position == 0:
            periods_passed = len(signal.loc[reversal_start_idx:timestamp])
            
            # 只在观察期内检查反转条件
            if periods_passed <= observation_period:
                # 计算期望的反转信号值（基于线性斜率，但有上限）
                max_reversal = slope_threshold * min(periods_passed, max_slope_periods)
                expected_reversal = trigger_value - (max_reversal * np.sign(trigger_value))
                
                # 只有在经过了最小观察期后才考虑开仓
                if periods_passed >= min_observation_periods:
                    # 信号从正阈值反转向下
                    if trigger_value > threshold and value < expected_reversal:
                        current_position = -1  # 开空仓
                        position_start_idx = timestamp
                        entry_price = current_price  # 记录开仓价格
                        watching_reversal = False
                    
                    # 信号从负阈值反转向上
                    elif trigger_value < -threshold and value > expected_reversal:
                        current_position = 1  # 开多仓
                        position_start_idx = timestamp
                        entry_price = current_price  # 记录开仓价格
                        watching_reversal = False
            else:
                # 超过观察期，停止观察
                watching_reversal = False
        
        positions[timestamp] = current_position
    
    return positions


# 变种5：从最高点回落给定数值
def trade_rule_by_reversal_v5(signal, 
                              threshold=0.8, 
                              observation_period=15, 
                              reversal_value=0.2,  # 新参数：从最高/最低点回落/上升的数值
                              holding_period=30, 
                              close_on_opposite_threshold=True,
                              time_gap_minutes=240,  # 4小时
                              cooldown_minutes=30,
                              lookback_periods=5):
    """
    基于信号反转的交易规则(修改版V4):
    1. 在信号突破阈值后观察一段时间(observation_period)
    2. 记录观察期内的最高/最低点，当信号从最高/最低点回落/上升指定数值时开反向仓位
    3. 持仓固定时间后平仓
    4. 当触发对侧阈值时平仓
    5. 考虑时间间隔，处理隔夜和午休情况
    6. 添加冷却期和前n个周期检查
    
    参数:
    - signal: 输入信号Series，index为timestamp
    - threshold: 触发反转观察的阈值
    - observation_period: 突破阈值后的观察期 (多少个切片)

    - reversal_value: 从极值点回落/上升的数值，达到此数值时开仓
    - holding_period: 开仓后的固定持仓时间 (多少个切片)
    - close_on_opposite_threshold: 若为True，则在触发对侧阈值时平仓
    - time_gap_minutes: 时间间隔阈值(分钟)，大于此值视为隔夜或午休
    - cooldown_minutes: 隔夜或午休后的冷却期(分钟)
    - lookback_periods: 触发观察前需要检查的前n个周期
    
    返回:
    - positions: 仓位Series，值为 1(多), -1(空), 0(不持仓), np.nan(无有效信号)
    """
    import pandas as pd
    import numpy as np
    
    # 确保输入是Series
    if not isinstance(signal, pd.Series):
        raise TypeError("signal must be a pandas Series with timestamp index")
    
    # 初始化结果Series
    positions = pd.Series(np.nan, index=signal.index)
    
    # 初始化状态变量
    current_position = 0
    watching_reversal = False
    reversal_start_idx = None
    trigger_value = 0
    position_start_idx = None
    last_timestamp = None
    in_cooldown = False
    cooldown_end_time = None
    
    # 新增变量
    extreme_value = None  # 记录观察期内的极值
    extreme_timestamp = None  # 记录极值对应的时间戳
    
    # 遍历每个时间点
    for timestamp, value in signal.items():
        if np.isnan(value):
            if current_position != 0:
                positions[timestamp] = current_position
            continue
        
        # 检查时间间隔
        if last_timestamp is not None:
            time_diff = (timestamp - last_timestamp).total_seconds() / 60
            
            # 如果超过时间间隔阈值，认为是隔夜或午休
            if time_diff > time_gap_minutes:
                # 重置信号观察状态
                watching_reversal = False
                extreme_value = None
                extreme_timestamp = None
                # 设置冷却期
                in_cooldown = True
                cooldown_end_time = timestamp + pd.Timedelta(minutes=cooldown_minutes)
        
        # 更新最后一个时间戳
        last_timestamp = timestamp
        
        # 检查是否在冷却期内
        if in_cooldown:
            if timestamp >= cooldown_end_time:
                in_cooldown = False
            else:
                # 在冷却期内保持原有仓位，不进行任何信号观察
                positions[timestamp] = current_position
                continue
        
        # 检查是否应该平仓 (基于固定持仓时间)
        if current_position != 0 and position_start_idx is not None:
            periods_held = len(signal.loc[position_start_idx:timestamp])
            if periods_held > holding_period:
                current_position = 0
                watching_reversal = False
                extreme_value = None
                extreme_timestamp = None
        
        # 检查是否在触发对侧阈值时平仓
        if close_on_opposite_threshold and current_position != 0:
            # 持有空仓时，如果信号大于正阈值或小于负阈值则平仓
            if current_position == -1 and (value > threshold or value < -threshold):
                current_position = 0
                watching_reversal = False
                extreme_value = None
                extreme_timestamp = None
            # 持有多仓时，如果信号大于正阈值或小于负阈值则平仓
            elif current_position == 1 and (value > threshold or value < -threshold):
                current_position = 0
                watching_reversal = False
                extreme_value = None
                extreme_timestamp = None
        
        # 检查是否应该开始观察反转
        if not watching_reversal and current_position == 0:
            if (value > threshold or value < -threshold):
                # 检查lookback_periods分钟内的数据点是否都未触发阈值
                valid_start = True
                if lookback_periods > 0:
                    # 计算lookback时间窗口的起始时间
                    lookback_start_time = timestamp - pd.Timedelta(minutes=lookback_periods)
                    
                    # 获取lookback时间窗口内的所有数据点
                    lookback_data = signal.loc[lookback_start_time:timestamp].iloc[:-1]  # 不包括当前点
                    
                    # 检查这些点是否有触发阈值的
                    for prev_ts, prev_val in lookback_data.items():
                        if not np.isnan(prev_val) and (prev_val > threshold or prev_val < -threshold):
                            valid_start = False
                            break
                
                if valid_start:
                    watching_reversal = True
                    reversal_start_idx = timestamp
                    trigger_value = value
                    extreme_value = value  # 初始化极值为触发值
                    extreme_timestamp = timestamp  # 初始化极值时间戳
        
        # 如果正在观察反转，检查是否满足反转条件
        elif watching_reversal and current_position == 0:
            periods_passed = len(signal.loc[reversal_start_idx:timestamp])
            
            # 只在观察期内检查反转条件
            if periods_passed <= observation_period:
                # 更新极值
                if trigger_value > threshold and value > extreme_value:  # 上突破情况
                    extreme_value = value
                    extreme_timestamp = timestamp
                elif trigger_value < -threshold and value < extreme_value:  # 下突破情况
                    extreme_value = value
                    extreme_timestamp = timestamp
                
                # 只要存在极值记录就考虑开仓
                if extreme_value is not None and extreme_timestamp is not None:
                    # 计算从极值点开始已经过了多少周期
                    periods_from_extreme = len(signal.loc[extreme_timestamp:timestamp])
                    
                    # 从极值点开始计算回落/上升
                    if periods_from_extreme > 0:  # 确保不是当前点就是极值点
                        if trigger_value > threshold and (extreme_value - value) >= reversal_value:
                            # 信号从上突破高点回落了指定数值，开空仓
                            current_position = -1
                            position_start_idx = timestamp
                            watching_reversal = False
                            extreme_value = None
                            extreme_timestamp = None
                        elif trigger_value < -threshold and (value - extreme_value) >= reversal_value:
                            # 信号从下突破低点上升了指定数值，开多仓
                            current_position = 1
                            position_start_idx = timestamp
                            watching_reversal = False
                            extreme_value = None
                            extreme_timestamp = None
            else:
                # 超过观察期，停止观察
                watching_reversal = False
                extreme_value = None
                extreme_timestamp = None
        
        positions[timestamp] = current_position
    
    return positions