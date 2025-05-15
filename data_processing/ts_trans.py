# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:13:48 2024

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


from utils.timeutils import parse_time_string


# %%
def ts_normalize(factor, param):
    scale_window = param.get('scale_window')
    if scale_window is None:
        return (factor - 0.5) * 2
    scale_quantile = param['scale_quantile']
    sp = param['sp']
    scale_method = param.get('scale_method', 'minmax_scale')
    
    
    scale_func = globals()[scale_method]
    scale_step = int(parse_time_string(scale_window) / parse_time_string(sp))
    if scale_method in ['minmax_scale', 'minmax_scale_separate']:
        factor_scaled = scale_func(factor, window=scale_step, quantile=scale_quantile)
    elif scale_method in ['rolling_percentile']:
        factor_scaled = scale_func(factor, window=scale_step)
    else:
        raise NotImplementedError()
    factor_scaled = (factor_scaled - 0.5) * 2
    return factor_scaled


# %%
def minmax_scale(df: pd.DataFrame, step=1, window=20, min_periods=10, quantile=0.05) -> pd.DataFrame:
    """
    滚动窗口分位数归一化。

    参数:
        df (pd.DataFrame): 用于计算滚动分位数归一化的变量
        step (int, optional): 计算归一化时的步长。默认值为1。
        window (int, optional): 滚动窗口大小。默认值为20。
        min_periods (int, optional): 窗口中要求的最少观察数。默认值为10。
        quantile (float, optional): 用于归一化的分位数。默认值为0.05，即5%。

    返回:
        pd.DataFrame: 归一化后的DataFrame
    """
    
    # 定义5%和95%分位数
    lower_quantile = quantile
    upper_quantile = 1 - quantile
    
    # 计算滚动窗口的分位数
    df_lower = df.rolling(window=window, min_periods=min_periods, step=step).quantile(lower_quantile)
    df_upper = df.rolling(window=window, min_periods=min_periods, step=step).quantile(upper_quantile)
    
    # 进行分位数归一化
    df_quantile_scaled = (df.iloc[::step] - df_lower) / (df_upper - df_lower).replace(0, np.nan)
    
    # 裁剪结果使其在 0 和 1 之间
    df_quantile_scaled = df_quantile_scaled.clip(lower=0, upper=1)
    
    return df_quantile_scaled


def minmax_scale_separate(df: pd.DataFrame, step=1, window=20, min_periods=10, quantile=0.05) -> pd.DataFrame:
    """
    滚动窗口分位数归一化，正值和负值分别归一化。

    参数:
        df (pd.DataFrame): 用于计算滚动分位数归一化的变量
        step (int, optional): 计算归一化时的步长。默认值为1。
        window (int, optional): 滚动窗口大小。默认值为20。
        min_periods (int, optional): 窗口中要求的最少观察数。默认值为10。
        quantile (float, optional): 用于归一化的分位数。默认值为0.05，即5%。

    返回:
        pd.DataFrame: 归一化后的DataFrame
    """
    # 定义5%和95%分位数
    lower_quantile = quantile
    upper_quantile = 1 - quantile
    
    # 创建正值和负值的掩码
    positive_mask = df > 0
    negative_mask = df < 0
    
    # 计算正值部分的滚动窗口分位数
    df_positive = df[positive_mask]
    df_positive_lower = df_positive.rolling(window=window, min_periods=min_periods, step=step).quantile(lower_quantile)
    df_positive_upper = df_positive.rolling(window=window, min_periods=min_periods, step=step).quantile(upper_quantile)
    
    # 计算负值部分的滚动窗口分位数
    df_negative = df[negative_mask]
    df_negative_lower = df_negative.rolling(window=window, min_periods=min_periods, step=step).quantile(lower_quantile)
    df_negative_upper = df_negative.rolling(window=window, min_periods=min_periods, step=step).quantile(upper_quantile)
    
    # 初始化归一化后的DataFrame
    df_quantile_scaled = pd.DataFrame(index=df.index, columns=df.columns)
    
    # 对正值部分进行归一化，使其在 0.5 到 1 之间
    df_quantile_scaled[positive_mask] = 0.5 + 0.5 * ((df_positive - df_positive_lower) / (df_positive_upper - df_positive_lower).replace(0, np.nan))
    
    # 对负值部分进行归一化，使其在 0 到 0.5 之间
    df_quantile_scaled[negative_mask] = 0.5 * ((df_negative - df_negative_upper) / (df_negative_lower - df_negative_upper).replace(0, np.nan))
    
    # 裁剪结果使其在 0 和 1 之间
    df_quantile_scaled = df_quantile_scaled.clip(lower=0, upper=1)
    
    return df_quantile_scaled


def clip_values(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    将 DataFrame 中大于 threshold 且小于 1 - threshold 的部分掩盖为 0。

    参数:
        df (pd.DataFrame): 输入的 DataFrame。
        threshold (float): 阈值，范围 (0, 0.5)。

    返回:
        pd.DataFrame: 掩盖后的 DataFrame。
    """
    clipped_df = df.copy()
    mask = (clipped_df > threshold) & (clipped_df < (1 - threshold))
    clipped_df[mask] = 0.5
    return clipped_df


# %%
def rolling_zero_percentile(df, window):
    """
    计算给定 DataFrame 的滚动窗口中0所在的百分位。

    参数：
    df (pd.DataFrame): 输入的 Pandas DataFrame。
    window (int): 滚动窗口的大小。

    返回：
    pd.DataFrame: 每个滚动窗口中0所在的百分位。
    """
    def percentile_in_window(window):
        if len(window) == 0:
            return np.nan
        return (np.sum(window < 0) + np.sum(window == 0) / 2) / len(window)

    return df.rolling(window, min_periods=1).apply(lambda x: percentile_in_window(x), raw=True)


def rolling_quantile_value(data_df, quantile_df, window):
    """
    在数据 DataFrame 每个位置上回看一定窗口，找到 quantile DataFrame 中对应位置的 quantile 数所对应的值。

    参数：
    data_df (pd.DataFrame): 输入的数据 DataFrame。
    quantile_df (pd.DataFrame): 每个位置上的分位数 DataFrame。
    window (int): 滚动窗口的大小。

    返回：
    pd.DataFrame: 每个滚动窗口中 quantile 所对应的值。
    """
    def quantile_in_window(window, quantile):
        if len(window) == 0 or np.isnan(quantile):
            return np.nan
        return np.nanquantile(window, quantile)

    result = pd.DataFrame(index=data_df.index, columns=data_df.columns)
    for col in data_df.columns:
        result[col] = data_df[col].rolling(window, min_periods=1).apply(
            lambda x: quantile_in_window(x, quantile_df[col].loc[x.index[-1]]), raw=False
        )
    return result


# =============================================================================
# from numba import jit
# 
# @jit(nopython=True, parallel=True)
# def quantile_in_window_optimized(window, quantile):
#     window = window[~np.isnan(window)]  # Remove NaN values
#     if len(window) == 0 or np.isnan(quantile):
#         return np.nan
#     return np.quantile(window, quantile)
# 
# def rolling_quantile_value(data_df, quantile_df, window):
#     """
#     在数据 DataFrame 每个位置上回看一定窗口，找到 quantile DataFrame 中对应位置的 quantile 数所对应的值。
# 
#     参数：
#     data_df (pd.DataFrame): 输入的数据 DataFrame。
#     quantile_df (pd.DataFrame): 每个位置上的分位数 DataFrame。
#     window (int): 滚动窗口的大小。
# 
#     返回：
#     pd.DataFrame: 每个滚动窗口中 quantile 所对应的值。
#     """
#     result = pd.DataFrame(index=data_df.index, columns=data_df.columns)
#     for col in data_df.columns:
#         rolling_values = data_df[col].rolling(window, min_periods=1).apply(lambda x: quantile_in_window_optimized(x, quantile_df[col].loc[x.index[-1]]), raw=True)
#         result[col] = rolling_values
#     return result
# =============================================================================


def minmax_scale_adj_by_his_rtn(data_df, rtn_df, window, rtn_window, quantile=0.02):
    """
    综合使用 rolling_zero_percentile 和 rolling_quantile_value 的函数。

    参数：
    data_df (pd.DataFrame): 输入的数据 DataFrame。
    rtn_df (pd.DataFrame): 用于计算百分位的 DataFrame。
    window (int): 数据的滚动窗口大小。
    rtn_window (int): 用于计算百分位的滚动窗口大小。
    quantile (float): 用于掩码的分位数，默认值为 0.02。

    返回：
    pd.DataFrame: 归一化后的 DataFrame。
    """
    # breakpoint()
    # 第一步：对 rtn_df 计算 rolling_zero_percentile，窗口为 (window - rtn_window)，然后 shift(rtn_window)
    zero_percentile = rolling_zero_percentile(rtn_df, window - rtn_window).shift(rtn_window)

    # 第二步：使用 rolling_quantile_value 计算 data_df 和 zero_percentile 的值
    quantile_values = rolling_quantile_value(data_df, zero_percentile, window)

    # 第三步：计算滚动窗口的分位数
    df_lower = data_df.rolling(window=window, min_periods=1).quantile(quantile)
    df_upper = data_df.rolling(window=window, min_periods=1).quantile(1 - quantile)

    # 对大于 quantile_values 部分进行归一化
    upper_mask = data_df > quantile_values
    df_quantile_scaled = pd.DataFrame(index=data_df.index, columns=data_df.columns)
    df_quantile_scaled[upper_mask] = 0.5 + 0.5 * (data_df[upper_mask] - quantile_values[upper_mask]) / (df_upper[upper_mask] - quantile_values[upper_mask]).replace(0, np.nan)

    # 对小于 quantile_values 部分进行归一化
    lower_mask = data_df < quantile_values
    df_quantile_scaled[lower_mask] = 0.5 * (data_df[lower_mask] - quantile_values[lower_mask]) / (quantile_values[lower_mask] - df_lower[lower_mask]).replace(0, np.nan)

    # 裁剪结果使其在 0 和 1 之间
    df_quantile_scaled = df_quantile_scaled.clip(lower=0, upper=1)

    return df_quantile_scaled


def zscore_adj_by_his_rtn_and_minmax(data_df, rtn_df, window, rtn_window, quantile=0.02):
    """
    综合使用 rolling_zero_percentile 和 rolling_quantile_value 的函数。

    参数：
    data_df (pd.DataFrame): 输入的数据 DataFrame。
    rtn_df (pd.DataFrame): 用于计算百分位的 DataFrame。
    window (int): 数据的滚动窗口大小。
    rtn_window (int): 用于计算百分位的滚动窗口大小。
    quantile (float): 用于掩码的分位数，默认值为 0.02。

    返回：
    pd.DataFrame: 归一化后的 DataFrame。
    """
    # breakpoint()
    # 第一步：对 rtn_df 计算 rolling_zero_percentile，窗口为 (window - rtn_window)，然后 shift(rtn_window)
    zero_percentile = rolling_zero_percentile(rtn_df, window - rtn_window).shift(rtn_window)

    # 第二步：使用 rolling_quantile_value 计算 data_df 和 zero_percentile 的值
    quantile_values = rolling_quantile_value(data_df, zero_percentile, window)

    df_std = data_df.rolling(window=window, min_periods=1).std()
    df_zscore = (data_df - quantile_values) / df_std
    
    return minmax_scale(df_zscore, window=window, quantile=quantile)


# %%
def rolling_percentile(data_df, window):
    """
    计算每个位置的当前值在滚动窗口历史值中的分位数。

    参数：
    data_df (pd.DataFrame): 输入的数据 DataFrame。
    window (int): 滚动窗口的大小。

    返回：
    pd.DataFrame: 每个位置的当前值在滚动窗口中的分位数。
    """
    def calc_percentile(series):
        current_value = series.iloc[-1]
        rolling_window = series.iloc[:-1].dropna()
        if len(rolling_window) == 0:
            return np.nan
        return (rolling_window <= current_value).mean()

    return data_df.rolling(window, min_periods=1).apply(calc_percentile, raw=False)


def adj_by_his_rtn_percentile(data_df, rtn_df, window, rtn_window):
    """
    综合使用 rolling_zero_percentile 和 rolling_quantile_value 的函数。

    参数：
    data_df (pd.DataFrame): 输入的数据 DataFrame。
    rtn_df (pd.DataFrame): 用于计算百分位的 DataFrame。
    window (int): 数据的滚动窗口大小。
    rtn_window (int): 用于计算百分位的滚动窗口大小。
    quantile (float): 用于掩码的分位数，默认值为 0.02。

    返回：
    pd.DataFrame: 归一化后的 DataFrame。
    """
    # breakpoint()
    # 第一步：对 rtn_df 计算 rolling_zero_percentile，窗口为 (window - rtn_window)，然后 shift(rtn_window)
    zero_percentile = rolling_zero_percentile(rtn_df, window - rtn_window).shift(rtn_window)

    # 第二步：使用 rolling_quantile_value 计算 data_df 和 zero_percentile 的值
    quantile_values = rolling_quantile_value(data_df, zero_percentile, window)

    df_quantile_scaled = pd.DataFrame(index=data_df.index, columns=data_df.columns)
    
    # 对大于 quantile_values 部分进行归一化
    upper_mask = data_df > quantile_values
    df_quantile_scaled[upper_mask] = 0.5 + 0.5 * rolling_percentile(data_df.mask(~upper_mask), window)

    # 对小于 quantile_values 部分进行归一化
    lower_mask = data_df < quantile_values
    df_quantile_scaled[lower_mask] = 0.5 * rolling_percentile(data_df.mask(~lower_mask), window)

    # 裁剪结果使其在 0 和 1 之间
    df_quantile_scaled = df_quantile_scaled.clip(lower=0, upper=1)

    return df_quantile_scaled


def percentile_adj_by_his_rtn(data_df, rtn_df, window, rtn_window):
    """
    综合使用 rolling_zero_percentile 和 rolling_quantile_value 的函数。

    参数：
    data_df (pd.DataFrame): 输入的数据 DataFrame。
    rtn_df (pd.DataFrame): 用于计算百分位的 DataFrame。
    window (int): 数据的滚动窗口大小。
    rtn_window (int): 用于计算百分位的滚动窗口大小。
    quantile (float): 用于掩码的分位数，默认值为 0.02。

    返回：
    pd.DataFrame: 归一化后的 DataFrame。
    """
    # breakpoint()
    # 第一步：对 rtn_df 计算 rolling_zero_percentile，窗口为 (window - rtn_window)，然后 shift(rtn_window)
    zero_percentile = rolling_zero_percentile(rtn_df, window - rtn_window).shift(rtn_window)
    
    data_pct = rolling_percentile(data_df, window)

    df_quantile_scaled = pd.DataFrame(index=data_df.index, columns=data_df.columns)
    
    # 对大于 quantile_values 部分进行归一化
    upper_mask = data_pct > zero_percentile
    df_upper = pd.DataFrame(1, index=data_df.index, columns=data_df.columns)
    df_quantile_scaled[upper_mask] = 0.5 + 0.5 * (data_pct[upper_mask] - zero_percentile[upper_mask]) / (df_upper[upper_mask] - zero_percentile[upper_mask]).replace(0, np.nan)

    # 对小于 quantile_values 部分进行归一化
    lower_mask = data_pct < zero_percentile
    df_lower = pd.DataFrame(0, index=data_df.index, columns=data_df.columns)
    df_quantile_scaled[lower_mask] = 0.5 * (data_pct[lower_mask] - zero_percentile[lower_mask]) / (zero_percentile[lower_mask] - df_lower[lower_mask]).replace(0, np.nan)

    # 裁剪结果使其在 0 和 1 之间
    df_quantile_scaled = df_quantile_scaled.clip(lower=0, upper=1)

    return df_quantile_scaled
