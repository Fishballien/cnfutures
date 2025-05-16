# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 15:27:33 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import adfuller
import matplotlib.dates as mdates
import json
from pathlib import Path


# %% overnight
def analyze_returns_by_interval(factor_pos, gp):
    """
    按照时间间隔类型和多空仓位分析收益率 (向量化处理版本)
    
    参数:
    factor_pos (DataFrame): 含有因子仓位的DataFrame，索引为时间戳
    gp (DataFrame): 含有收益率的DataFrame，索引为时间戳，对应列为因子收益率
    
    返回:
    dict: 包含各因子在不同时间间隔和多空仓位下的收益率统计
    """
    # 复制数据框
    df = factor_pos.copy()
    mul = 1 if gp["return"].sum() > 0 else -1
    gp = gp * mul
    
    # 添加下一行的时间信息
    df['next_timestamp'] = df.index.to_series().shift(-1)
    
    # 提取当前和下一个时间点的日期和星期几（向量化操作）
    df['current_date'] = df.index.date
    df['next_date'] = df['next_timestamp'].dt.date
    df['current_dayofweek'] = df.index.dayofweek
    df['next_dayofweek'] = df['next_timestamp'].dt.dayofweek
    
    # 定义一个函数来确定间隔类型
    def determine_interval_type(row):
        if pd.isna(row['next_timestamp']):
            return None
        # 同一天 - 日内
        elif row['current_date'] == row['next_date']:
            return 'intraday'
        # 下一天 - 隔夜
        elif (row['next_date'] - row['current_date']).days == 1:
            return 'overnight'
        # 周五到周一 - 周末
        elif row['current_dayofweek'] == 4 and row['next_dayofweek'] == 0:
            return 'weekend'
        # 其他情况 - 节假日
        else:
            return 'holiday'
    
    # 应用函数确定间隔类型
    df['interval_type'] = df.apply(determine_interval_type, axis=1)
    
    # 获取所有因子列（排除辅助列）
    helper_columns = ['next_timestamp', 'current_date', 'next_date', 
                     'current_dayofweek', 'next_dayofweek', 'interval_type']
    factor_columns = [col for col in df.columns if col not in helper_columns]
    
    # 为每个因子添加对应的收益率列和多空方向列
    for col in factor_columns:
        # 只有当gp中存在相同列名时才添加对应的收益率
        if col in gp.columns:
            df[f'{col}_return'] = gp[col]
        else:
            # 如果gp中不存在相同列名，则尝试使用'return'列
            if 'return' in gp.columns:
                df[f'{col}_return'] = gp['return']
            else:
                raise ValueError(f"找不到'{col}'因子的收益率列")
        
        # 添加多空方向
        df[f'{col}_direction'] = np.where(df[col] > 0, 1, np.where(df[col] < 0, -1, 0))
    
    # 初始化结果字典
    results = {}
    
    # 对每个因子进行分析
    for factor in factor_columns:
        results[factor] = {
            'long': {},
            'short': {}
        }
        
        # 分析多头和空头
        for position, direction in [('long', 1), ('short', -1)]:
            # 对每种间隔类型进行分析
            for interval in ['intraday', 'overnight', 'weekend', 'holiday']:
                # 筛选条件
                mask = (df['interval_type'] == interval) & (df[f'{factor}_direction'] == direction)
                
                # 获取相应的收益率
                filtered_returns = df.loc[mask, f'{factor}_return']
                
                if len(filtered_returns) > 0 and not filtered_returns.isna().all():
                    results[factor][position][interval] = {
                        'mean': filtered_returns.mean(),
                        'sum': filtered_returns.sum(),
                        'count': len(filtered_returns.dropna())
                    }
                else:
                    results[factor][position][interval] = {
                        'mean': np.nan,
                        'sum': 0,
                        'count': 0
                    }
    
    return results


def print_results(results):
    """
    打印分析结果
    
    参数:
    results (dict): analyze_returns_by_interval 函数返回的结果
    """
    for factor, data in results.items():
        print(f"\n===== {factor} 因子结果 =====")
        
        print("\n多头仓位:")
        for interval in ['intraday', 'overnight', 'weekend', 'holiday']:
            stats = data['long'][interval]
            print(f"{interval}: 平均值={stats['mean']:.6f}, 数量={stats['count']}, 总和={stats['sum']:.6f}")
        
        print("\n空头仓位:")
        for interval in ['intraday', 'overnight', 'weekend', 'holiday']:
            stats = data['short'][interval]
            print(f"{interval}: 平均值={stats['mean']:.6f}, 数量={stats['count']}, 总和={stats['sum']:.6f}")


# 使用示例:
# results = analyze_returns_by_interval(factor_pos, gp)
# print_results(results)


# %% get factor basic info
def calculate_distribution_stats(factor_series):
    """
    Calculate distribution statistics for a factor series
    
    Parameters:
    -----------
    factor_series : pandas.Series
        The factor series to analyze
        
    Returns:
    --------
    dict
        Dictionary containing distribution statistics
    """
    stats = {
        'count': factor_series.count(),
        'mean': factor_series.mean(),
        'median': factor_series.median(),
        'std': factor_series.std(),
        'skew': factor_series.skew(),
        'kurtosis': factor_series.kurtosis(),
        'min': factor_series.min(),
        'max': factor_series.max()
    }
    
    return stats

def calculate_stationarity(factor_series):
    """
    Perform stationarity test (ADF test) on the factor series
    
    Parameters:
    -----------
    factor_series : pandas.Series
        The factor series to analyze
        
    Returns:
    --------
    dict
        Dictionary containing stationarity test results
    """
    # Perform stationarity test (ADF test) on the original series
    try:
        adf_result = adfuller(factor_series.dropna().iloc[::100].values)
    except:
        return {
            'adf_statistic': None,
            'p_value': None,
            'critical_values': None,
            'is_stationary': None
        }
    adf_stat, p_value, _, _, critical_values, _ = adf_result
    
    is_stationary = p_value < 0.05
    
    stationarity = {
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_stationary': is_stationary
    }
    
    return stationarity

def calculate_grouped_returns(factor_series, fut_rtn_list):
    """
    Calculate mean returns by factor quantiles for multiple prediction periods
    
    Parameters:
    -----------
    factor_series : pandas.Series
        The factor series to analyze
    fut_rtn_list : list of (pandas.DataFrame, str)
        List of tuples, each containing:
        - Future returns DataFrame
        - Period name (e.g., '30min', '2h', '1d', '3d')
        
    Returns:
    --------
    dict
        Dictionary containing grouped returns for each period
    """
    # Create decile groups based on factor
    temp_factor = pd.DataFrame(factor_series)
    temp_factor.columns = ['factor']
    # 使用 rank 打破重复值，再分位数切分
    ranked_factor = temp_factor['factor'].rank(method='first')
    temp_factor['factor_group'] = pd.qcut(ranked_factor, q=10, labels=False)
    
    all_grouped_returns = {}
    
    # Process each future return period
    for fut_rtn, period_name in fut_rtn_list:
        # Merge with fut_rtn
        merged = pd.merge(fut_rtn, temp_factor[['factor_group']], 
                        left_index=True, right_index=True)
        
        # Calculate mean returns by group
        grouped_means = merged.groupby('factor_group').mean()
        
        # Convert to dictionary and store with period name
        all_grouped_returns[period_name] = grouped_means.iloc[:, 0].to_dict()

    return all_grouped_returns

def analyze_factor(factor_series, fut_rtn_list):
    """
    Analyze factor data and calculate statistics
    
    Parameters:
    -----------
    factor_series : pandas.Series
        The factor series to analyze, with datetime index
    fut_rtn_list : list of (pandas.DataFrame, str)
        List of tuples, each containing:
        - Future returns DataFrame with columns like 'IC', 'IF', 'IH', 'IM'
        - Period name (e.g., '30min', '2h', '1d', '3d')
        Each DataFrame must have the same index as factor_series
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    
    # Calculate statistics
    distribution_stats = calculate_distribution_stats(factor_series)
    stationarity = calculate_stationarity(factor_series)
    grouped_returns = calculate_grouped_returns(factor_series, fut_rtn_list)
    
    # Calculate intraday pattern
    temp_df = pd.DataFrame(factor_series)
    temp_df.columns = ['factor']
    temp_df['intraday_time'] = temp_df.index.strftime('%H:%M')
    intraday_pattern = temp_df.groupby('intraday_time')['factor'].mean()
    intraday_data = {time: value for time, value in zip(intraday_pattern.index, intraday_pattern.values)}
    
    # Compile results
    results = {
        'distribution_stats': distribution_stats,
        'stationarity': stationarity,
        'grouped_returns': grouped_returns,
        'intraday_pattern': intraday_data
    }
    
    return results


def plot_factor_analysis(factor_name, factor_series, fut_rtn_list, results=None, save_path=None):
    """
    Create a comprehensive 2x2 subplot visualization for factor analysis
    
    Parameters:
    -----------
    factor_name : str
        Name of the factor being analyzed (used in titles)
    factor_series : pandas.Series
        The factor series to analyze, with datetime index
    fut_rtn_list : list of (pandas.DataFrame, str)
        List of tuples, each containing:
        - Future returns DataFrame with columns like 'IC', 'IF', 'IH', 'IM'
        - Period name (e.g., '30min', '2h', '1d', '3d')
        Each DataFrame must have the same index as factor_series
    results : dict, optional
        Pre-calculated analysis results
    save_path : str, optional
        Path to save the figure, if None, the figure is not saved
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    # Calculate results if not provided
    if results is None:
        results = analyze_factor(factor_series, fut_rtn_list)
    
    # Create figure with custom grid to allow for better control
    fig = plt.figure(figsize=(16, 14))  # 增加高度以留出更多空间
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)  # 增加子图间距
    
    # =====================================================================
    # Subplot 1: Distribution of factor values (Histogram)
    # =====================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Extract statistics
    mean_val = results['distribution_stats']['mean']
    median_val = results['distribution_stats']['median']
    std_val = results['distribution_stats']['std']
    
    # Plot histogram - High contrast professional color scheme B
    n, bins, patches = ax1.hist(factor_series.dropna(), bins=50, 
                               alpha=0.7, color='#2A93D5', density=True)
    
    # Add vertical lines for mean and median
    ax1.axvline(mean_val, color='#D64242', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.4f}')
    ax1.axvline(median_val, color='#37A76F', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.4f}')
    
    # Add curve for normal distribution with same mean and std
    x = np.linspace(min(bins), max(bins), 100)
    ax1.plot(x, 1/(std_val * np.sqrt(2 * np.pi)) * np.exp(-(x - mean_val)**2 / (2 * std_val**2)), 
             linewidth=2, color='#7D3AC1', label='Normal Dist.')
    
    # Add labels and title
    ax1.set_title(f'Distribution of Factor Values', fontsize=14, pad=15)  # 增加标题边距
    ax1.set_xlabel('Factor Value')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 修改：将统计信息移到左上角，避免与其他指标重叠
    stats_text = (f"N: {results['distribution_stats']['count']}\n"
                  f"Std Dev: {std_val:.4f}\n"
                  f"Skew: {results['distribution_stats']['skew']:.4f}\n"
                  f"Kurtosis: {results['distribution_stats']['kurtosis']:.4f}")
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =====================================================================
    # Subplot 2: Long-term trend with stationarity test
    # =====================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Ensure data is available for daily resampling
    if len(factor_series) > 0:
        # Resample to daily frequency
        daily_series = factor_series.resample('1d').mean().dropna()
        
        # Plot the resampled series - High contrast professional color scheme B
        ax2.plot(daily_series.index, daily_series.values, color='#304DC9', linewidth=1)
        
        # Add a rolling mean
        rolling_mean = daily_series.rolling(window=30).mean()
        ax2.plot(rolling_mean.index, rolling_mean.values, color='#FFB400', linewidth=2, 
                label='30-day MA')
    
    # Format the ADF test results
    adf_stat = results['stationarity']['adf_statistic']
    p_value = results['stationarity']['p_value']
    critical_values = results['stationarity']['critical_values']
    is_stationary = results['stationarity']['is_stationary']
    
    stationarity_text = f"ADF Test Results:\nTest Statistic: {adf_stat:.4f}\np-value: {p_value:.4f}\n"
    stationarity_text += f"Critical Values:\n"
    for key, value in critical_values.items():
        stationarity_text += f"  {key}: {value:.4f}\n"
    stationarity_text += f"\nConclusion: {'Stationary' if is_stationary else 'Non-stationary'}"
    
    # Add the results as text box
    ax2.text(0.02, 0.02, stationarity_text, transform=ax2.transAxes,
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add labels and title
    ax2.set_title(f'Long-term Trend of Factor (Daily Resampled, ADF on Original Series)', 
                 fontsize=14, pad=15)  # 增加标题边距
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Factor Value')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis date labels
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # =====================================================================
    # Subplot 3: Intraday pattern
    # =====================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Extract intraday pattern
    intraday_pattern = pd.Series(results['intraday_pattern'])
    
    # Plot the intraday pattern - High contrast professional color scheme B
    ax3.plot(range(len(intraday_pattern)), intraday_pattern.values, 
             color='#19A5B1', linewidth=2, marker='o', markersize=3)
    
    # Set x-ticks to time values (every 30 minutes)
    time_ticks = []
    tick_positions = []
    for i, time_str in enumerate(intraday_pattern.index):
        if i % 30 == 0:  # Every 30 minutes
            time_ticks.append(time_str)
            tick_positions.append(i)
    
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(time_ticks, rotation=45)
    
    # Add labels and title
    ax3.set_title(f'Intraday Pattern of Factor', fontsize=14, pad=15)  # 增加标题边距
    ax3.set_xlabel('Time of Day (HH:MM)')
    ax3.set_ylabel('Average Factor Value')
    ax3.grid(True, alpha=0.3)
    
    # Add vertical lines for market open/close if applicable
    # Assuming trading hours are from 9:30 to 15:00 (adjust as needed)
    try:
        open_idx = list(intraday_pattern.index).index('09:30')
        close_idx = list(intraday_pattern.index).index('15:00')
        ax3.axvline(open_idx, color='#2176FF', linestyle='--', alpha=0.5, label='Market Open')
        ax3.axvline(close_idx, color='#D40E52', linestyle='--', alpha=0.5, label='Market Close')
        ax3.legend()
    except (ValueError, IndexError):
        # Times not in index, skip adding these lines
        pass
    
    # =====================================================================
    # Subplot 4: Group returns based on factor quantiles - MODIFIED FOR MULTIPLE PERIODS
    # =====================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Get the number of deciles and periods
    num_deciles = 10  # Default is 10 deciles
    periods = list(results['grouped_returns'].keys())
    num_periods = len(periods)
    
    # Create a consistent color palette for periods
    period_colors = plt.cm.tab10(np.linspace(0, 1, num_periods))
    
    # Set up grouped bar chart
    bar_width = 0.8 / num_periods  # Width of each bar within a group
    
    # For each period, plot bars for each decile
    for period_idx, period in enumerate(periods):
        period_data = []
        
        # Find the first contract in the period data (usually there's only one)
        first_decile_data = results['grouped_returns'][period].get(0, {})
        # breakpoint()
        if first_decile_data:  # If we have data for this period
            # Extract data for all deciles for this period and contract
            for decile in range(10):  # Assume 10 deciles (0-9)
                if decile in results['grouped_returns'][period]:
                    period_data.append(results['grouped_returns'][period][decile])
                else:
                    period_data.append(0)  # Fill with zeros if data not available
            
            # Calculate x positions
            x_positions = np.arange(len(period_data))
            
            # Offset position based on period index
            offset = (period_idx - num_periods/2 + 0.5) * bar_width
            
            # Plot the bars for this period
            bars = ax4.bar(x_positions + offset, period_data, 
                          width=bar_width, 
                          color=period_colors[period_idx], 
                          label=period)
    
    # Add labels and title
    ax4.set_title(f'Average Returns by Factor Decile Across Time Periods', fontsize=14, pad=15)
    ax4.set_xlabel(f'Factor Decile (0=Low, 9=High)')
    ax4.set_ylabel('Average Return')
    ax4.set_xticks(range(10))
    ax4.grid(True, axis='y', alpha=0.3)
    
    # Add horizontal line at zero
    ax4.axhline(y=0, color='#212121', linestyle='-', alpha=0.3)
    
    # Add legend for periods
    if len(periods) > 0:
        ax4.legend(loc='best', title='Periods')
    
    
    # 调整主标题位置，给长因子名预留更多空间
    plt.subplots_adjust(top=0.9)  # 调整顶部边距
    plt.suptitle(f'Comprehensive Analysis of {factor_name} Factor', fontsize=16, y=0.98)
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def get_factor_basic_info(factor_name, factor_data, price_data, pp_by_sp_list, data_dir, plot_dir):
    """
    Process factor data and save analysis results for multiple prediction periods
    
    Parameters:
    -----------
    factor_name : str
        Name of the factor being analyzed
    factor_data : pandas.Series or pandas.DataFrame
        Factor data to analyze, with datetime index
    price_data : pandas.DataFrame
        Price data DataFrame with columns for different instruments
    pp_by_sp_list : list of (int, str)
        List of tuples, each containing:
        - Number of periods to use for calculating future returns
        - Period label (e.g., '30min', '2h', '1d', '3d')
    data_dir : pathlib.Path
        Directory to save analysis results data
    plot_dir : pathlib.Path
        Directory to save plots
        
    Returns:
    --------
    dict
        Dictionary mapping factor names to analysis results
    """
    # Create directories if they don't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate future returns for each period
    fut_rtn_list = []
    for periods, period_label in pp_by_sp_list:
        # Calculate future returns for this period
        fut_rtn = price_data.pct_change(periods, fill_method=None).shift(-periods) / periods
        fut_rtn_list.append((fut_rtn, period_label))
    
    results = {}
    # Handle both Series and DataFrame inputs
    if isinstance(factor_data, pd.Series):
        # Analyze factor
        factor_results = analyze_factor(factor_data, fut_rtn_list)
        results[factor_name] = factor_results
        
        # Save results
        with open(data_dir / f"{factor_name}_analysis.json", 'w') as f:
            json.dump(factor_results, f, indent=4, default=str)
        
        # Plot factor analysis
        plot_factor_analysis(factor_name, factor_data, fut_rtn_list, factor_results, 
                           plot_dir / f"{factor_name}_analysis.png")
    
    else:
        # Case 2: factor_data is a DataFrame
        for column in factor_data.columns:
            # Process by column and check if it exists in price data
            if column in price_data.columns:
                factor_series = factor_data[column]
                
                # Create a list of future returns for this specific column
                column_fut_rtn_list = []
                for fut_rtn, period_label in fut_rtn_list:
                    if column in fut_rtn.columns:
                        column_fut_rtn_list.append((fut_rtn[[column]], period_label))
                
                # Only proceed if we have future returns data for this column
                if column_fut_rtn_list:
                    try:
                        # Analyze factor using only the corresponding fut_rtn column
                        factor_results = analyze_factor(factor_series, column_fut_rtn_list)
                        results[f"{factor_name}_{column}"] = factor_results
                        
                        # Save results with both factor name and column name
                        result_filename = f"{factor_name}_{column}_analysis.json"
                        with open(data_dir / result_filename, 'w') as f:
                            json.dump(factor_results, f, indent=4, default=str)
                        
                        # Plot factor analysis with both factor name and column name
                        plot_filename = f"{factor_name}_{column}_analysis.png"
                        plot_factor_analysis(f"{factor_name} - {column}", factor_series, 
                                           column_fut_rtn_list, factor_results, 
                                           plot_dir / plot_filename)
                    except:
                        pass
    
    return results
# Example usage:
# import pandas as pd
# import numpy as np
# from pathlib import Path
# 
# # Create example data
# dates = pd.date_range('2020-01-01', '2021-01-01', freq='1h')
# 
# # Price data
# price_data = pd.DataFrame({
#     'IC': 100 + np.cumsum(np.random.randn(len(dates)) * 0.1),
#     'IF': 200 + np.cumsum(np.random.randn(len(dates)) * 0.2),
#     'IH': 300 + np.cumsum(np.random.randn(len(dates)) * 0.15),
#     'IM': 400 + np.cumsum(np.random.randn(len(dates)) * 0.25)
# }, index=dates)
#
# # Factor data
# factor_data = pd.Series(np.random.randn(len(dates)), index=dates, name='PutCallOIDiff')
# 
# # Process data
# pp_by_sp = 30  # periods for future returns
# data_dir = Path('data_output')
# plot_dir = Path('plot_output')
# results = process_factor_data(factor_data, price_data, pp_by_sp, data_dir, plot_dir)
# 
# # Example with DataFrame factor
# factor_df = pd.DataFrame({
#     'IC': np.random.randn(len(dates)),
#     'IF': np.random.randn(len(dates)),
#     'IH': np.random.randn(len(dates))
# }, index=dates)
# factor_df.name = 'CustomFactor'  # Name for the DataFrame to use in filenames
#
# results_df = process_factor_data(factor_df, price_data, pp_by_sp, data_dir, plot_dir)