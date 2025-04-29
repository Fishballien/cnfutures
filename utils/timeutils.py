# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:51:38 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import numpy as np
import pandas as pd
from dateutil import rrule, relativedelta
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import re


# %%
def parse_time_string(time_string):
    """
    解析格式为 "xxdxxhxxminxxs" 的时间字符串并转换为总秒数。

    参数:
    time_string (str): 表示时间间隔的字符串，如 "1d2h30min45s"。

    返回:
    int: 转换后的总秒数。

    异常:
    ValueError: 如果时间字符串格式无效。
    """
    pattern = re.compile(r'(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)min)?(?:(\d+)s)?')
    match = pattern.fullmatch(time_string)
    
    if not match:
        raise ValueError("Invalid time string format")
    
    days = int(match.group(1)) if match.group(1) else 0
    hours = int(match.group(2)) if match.group(2) else 0
    mins = int(match.group(3)) if match.group(3) else 0
    secs = int(match.group(4)) if match.group(4) else 0
    
    total_seconds = days * 4 * 60 * 60 + hours * 60 * 60 + mins * 60 + secs
    return total_seconds


# %%
def timestr_to_seconds(time_str):
    td = pd.to_timedelta(time_str)
    return td.total_seconds()


def timestr_to_minutes(time_str):
    td = pd.to_timedelta(time_str)
    return int(td.total_seconds() / 60)


def timedelta_to_seconds(time_delta_info):
    time_delta = timedelta(**time_delta_info)
    return time_delta / timedelta(seconds=1)


def datetime_to_shortcut(dt):
    return dt.strftime('%y%m%d')


# %%
def get_period(start_date, end_date):
    return [dt.strftime("%Y-%m-%d") for dt in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date)]


# %%
@dataclass
class RollingPeriods(object):
    
    fstart: datetime
    pstart: datetime
    puntil: datetime
    window_kwargs: dict = field(default_factory=dict)
    rrule_kwargs: dict = field(default_factory=dict)  # HINT💡 like: {'freq': 'M', 'bymonthday': -1}
    end_by: str = field(default_factory='')
    
    def __post_init__(self):
        FREQNAMES = rrule.FREQNAMES[:5]
        # FREQNAMES = ['YEARLY', 'MONTHLY', 'WEEKLY', 'DAILY', 'HOURLY']
        freq_mapping = {FREQNAMES[i][0]: i for i in range(len(FREQNAMES))}
        # {'Y': 0, 'M': 1, 'W': 2, 'D': 3, 'H': 4}
        if (freq := freq_mapping.get(self.rrule_kwargs['freq'], None)) is not None:
            self.rrule_kwargs['freq'] = freq
        cut_points = list(rrule.rrule(
            **self.rrule_kwargs, 
            dtstart=self.pstart, until=self.puntil))
        # breakpoint()
        if self.pstart == cut_points[0]:
            cut_points = cut_points[1:] if cut_points else []
        if cut_points and self.puntil == cut_points[-1]:
            cut_points = cut_points if cut_points else []
        pred_period_start = [self.pstart] + cut_points
        pred_period_end = [cut_point - timedelta(days=1) if self.end_by == 'date' else cut_point
                           for cut_point in cut_points] + [self.puntil if self.end_by == 'date' 
                                                           else self.puntil]
        windows = relativedelta.relativedelta(**self.window_kwargs)
        fit_period_start = [max(dt - windows, self.fstart) for dt in pred_period_start]
        assert self.end_by in ['date', 'time']
        fit_period_end = [dt - relativedelta.relativedelta(days=1) if self.end_by == 'date' else dt 
                          for dt in pred_period_start]
        self.predict_periods = list(zip(pred_period_start, pred_period_end))
        self.fit_periods = list(zip(fit_period_start, fit_period_end))
        
        
def generate_timeline_params(start_date, end_date):
    return {
        'start_date': start_date,
        'end_date': end_date,
        'data_start_date': start_date,
        'data_end_date': end_date + timedelta(days=1),
        }


def period_shortcut(start_date, end_date):
    return f'{datetime_to_shortcut(start_date)}_{datetime_to_shortcut(end_date)}'


def translate_rolling_params(rolling_params):
    for pr in ['fstart', 'pstart', 'puntil']:
        rolling_params[pr] = (datetime.strptime(rolling_params[pr], '%Y%m%d') 
                              if isinstance(rolling_params[pr], str)
                              else rolling_params[pr])
    return rolling_params


def get_rolling_dates(fstart, pstart, puntil):
    """
    Convert date strings to datetime objects.
    
    Parameters:
    -----------
    fstart : str
        Start date in 'YYYYMMDD' format
    pstart : str
        Period start date in 'YYYYMMDD' format
    puntil : str
        Period end date in 'YYYYMMDD' format
        
    Returns:
    --------
    dict
        Dictionary with datetime objects for each date
    """
    dates = {
        'fstart': fstart,
        'pstart': pstart,
        'puntil': puntil,
    }
    
    rolling_dates = {date_name: datetime.strptime(dates[date_name], '%Y%m%d')
                     for date_name in dates}
    
    return rolling_dates


def get_lb_fit_periods(rolling_dates, rolling_params, lb):
    """
    Generate fit periods for a single lookback window.

    Parameters:
    -----------
    rolling_dates : dict
        Dictionary with datetime objects from get_rolling_dates
    rolling_params : dict
        Dictionary with rolling parameters
    lb : int
        Lookback window size in months

    Returns:
    --------
    list
        Fit periods for the given lookback window
    """
    rolling_params_copy = rolling_params.copy()
    rolling_params_copy.update(rolling_dates)
    rolling_params_copy['window_kwargs'] = {'months': lb}

    rolling = RollingPeriods(**rolling_params_copy)
    return rolling.fit_periods


def get_rolling_periods(rolling_dates, rolling_params, lb):
    """
    Generate fit periods for a single lookback window.

    Parameters:
    -----------
    rolling_dates : dict
        Dictionary with datetime objects from get_rolling_dates
    rolling_params : dict
        Dictionary with rolling parameters
    lb : int
        Lookback window size in months

    Returns:
    --------
    list
        Rolling periods for the given lookback window
    """
    rolling_params_copy = rolling_params.copy()
    rolling_params_copy.update(rolling_dates)
    rolling_params_copy['window_kwargs'] = {'months': lb}

    rolling = RollingPeriods(**rolling_params_copy)
    return rolling


def get_lb_list_fit_periods(rolling_dates, rolling_params, lb_list):
    """
    Generate fit periods for different lookback windows.
    
    Parameters:
    -----------
    rolling_dates : dict
        Dictionary with datetime objects from get_rolling_dates
    rolling_params : dict
        Dictionary with rolling parameters
    lb_list : list
        List of lookback window sizes in months
        
    Returns:
    --------
    list
        List of fit periods for each lookback window
    """
    rolling_params_copy = rolling_params.copy()
    rolling_params_copy.update(rolling_dates)
    
    lb_rolling_pr_list = [
        {**rolling_params_copy, **{'window_kwargs': {'months': lb}}}
        for lb in lb_list
    ]
    
    lb_rolling_list = [RollingPeriods(**lb_rolling_pr) for lb_rolling_pr in lb_rolling_pr_list]
    lb_fit_periods_list = [rolling.fit_periods for rolling in lb_rolling_list]
    
    return lb_fit_periods_list


def find_matching_select_period(eval_period_end, select_fit_periods):
    """
    Find the matching select period for an eval period end date.
    
    Parameters:
    -----------
    eval_period_end : datetime
        The end date of the evaluation period
    select_fit_periods : list of tuples
        List of (start, end) datetime tuples representing select periods
        
    Returns:
    --------
    tuple or None
        The matching select period (start, end) or None if no match found
    """
    matching_period = None
    
    # Find the latest select period with an end date <= eval_period_end
    for period in select_fit_periods:
        _, select_end = period
        if select_end <= eval_period_end:
            # If we don't have a match yet, or this one is more recent
            if matching_period is None or select_end > matching_period[1]:
                matching_period = period
    
    return matching_period


# %%
def get_eq_spaced_intraday_time_series(date: datetime, params, mode='r'):
    if mode == 'r':
        start_time = date + timedelta(**params)
        end_time = date + timedelta(days=1) + timedelta(microseconds=1)
    elif mode == 'l':
        start_time = date
        end_time = date + timedelta(days=1)
    interval = timedelta(**params)
    time_series = np.arange(start_time, end_time, interval).astype('i8') // 1e3
    # time_series[-1] -= 100 # 500ms
    return time_series


# %%
def get_wd_name(wd_pr):
    key, value = next(iter(wd_pr.items()))
    return f"{key}_{value}"


# %% relativedelta
def parse_relativedelta(time_string):
    """
    解析时间间隔字符串并返回 relativedelta 对象。

    参数:
    time_string : str - 表示时间间隔的字符串，比如 "2 years, 3 months"。

    返回值:
    relativedelta - 对应时间间隔的 relativedelta 对象。
    """
    time_string = time_string.lower()
    units = {
        'years': 'years',
        'year': 'years',
        'months': 'months',
        'month': 'months',
        'weeks': 'weeks',
        'week': 'weeks',
        'days': 'days',
        'day': 'days',
        'hours': 'hours',
        'hour': 'hours',
        'minutes': 'minutes',
        'minute': 'minutes',
        'seconds': 'seconds',
        'second': 'seconds',
    }
    
    # 使用正则表达式提取所有的时间单位和数值
    matches = re.findall(r"(\d+)\s*(years?|months?|weeks?|days?|hours?|minutes?|seconds?)", time_string)
    
    # 构建关键字参数字典
    kwargs = {}
    for value, unit in matches:
        if unit in units:
            kwargs[units[unit]] = int(value)
    
    return relativedelta.relativedelta(**kwargs)