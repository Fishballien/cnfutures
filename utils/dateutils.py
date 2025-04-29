# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:13:26 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import pandas as pd
from datetime import date, timedelta, datetime


# %%
path = 'D:/CNIndexFutures/timeseries/future_price/sample_data/TradingDays.csv'
# path = r'/mnt/30.131_Raid1_data/TradingDays.csv'
trading_days = pd.read_csv(path)
trading_days['TradingDay'] = pd.to_datetime(trading_days['TradingDay'])
trading_days_set = set(trading_days['TradingDay'].dt.date)


# %%
def get_cffex_index_futures_expiration_date(year, month):
    """
    获取中金所股指期货指定年份和月份的交割日。
    遇到非交易日顺延。

    参数:
        year (int): 年份
        month (int): 月份

    返回:
        date: 交割日（避开非交易日）
    """
    # 找到该月的第三个星期五
    first_day = date(year, month, 1)
    fridays = [first_day + timedelta(days=i) for i in range(31) if (first_day + timedelta(days=i)).weekday() == 4]
    fridays = [f for f in fridays if f.month == month]

    if len(fridays) >= 3:
        third_friday = fridays[2]
    else:
        raise ValueError("该月没有足够的星期五")

    # 如果第三个星期五不是交易日，则顺延到下一个交易日
    while third_friday not in trading_days_set:
        third_friday += timedelta(days=1)

    return third_friday


# =============================================================================
# year = 2024
# month = 2
# expiration_date = get_cffex_index_futures_expiration_date(year, month)
# print(f"{year}年{month}月的股指期货交割日是：{expiration_date}")
# =============================================================================


def get_next_curr_month(date: str) -> str:
    """
    计算下一日的当月或下月合约编号。
    如果下一日日期大于当月交割日，则为下月合约，否则为当月合约。

    参数:
        date (str): 输入日期，格式为 'YYYYMMDD'。
    
    返回:
        str: 合约编号，格式为 'YYMM'。
    """
    # if date == '20150619':
    #     breakpoint()
    # 将字符串转为日期对象
    current_date = datetime.strptime(date, '%Y%m%d')
    next_date = (current_date + timedelta(days=1)).date()  # 计算下一日

    # 获取当月交割日
    expiration_date = get_cffex_index_futures_expiration_date(next_date.year, next_date.month)

    # 判断是否超过交割日
    if next_date > expiration_date:
        # 如果超过交割日，则为下月合约
        next_month = next_date.month + 1
        next_year = next_date.year
        if next_month > 12:  # 如果是12月，则切换到下一年
            next_month = 1
            next_year += 1
        year_suffix = str(next_year)[-2:]  # 获取年份后两位
        contract_code = f"{year_suffix}{next_month:02d}"
    else:
        # 如果未超过交割日，则为当月合约
        year_suffix = str(next_date.year)[-2:]
        contract_code = f"{year_suffix}{next_date.month:02d}"

    return contract_code


# =============================================================================
# input_date = '20150722'
# contract_code = get_next_curr_month(input_date)
# print(f"输入日期为 {input_date}，下一日的当月合约编号为：{contract_code}")
# =============================================================================


def get_cffex_trading_days_by_date_range(start_date: date, end_date: date):
    """
    获取指定日期范围内的中金所全部交易日。

    参数:
        start_date (date): 起始日期，datetime.date 类型
        end_date (date): 结束日期，datetime.date 类型

    返回:
        list: 包含所有交易日的日期对象列表
    """
    # 使用全局变量 trading_days_set 筛选交易日
    trading_days = [
        current_date
        for current_date in (start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1))
        if current_date in trading_days_set
    ]
    
    return trading_days


# =============================================================================
# # 示例调用
# start_date = date(2025, 1, 1)  # 起始日期
# end_date = date(2025, 1, 31)  # 结束日期
# trading_days = get_cffex_trading_days_by_date_range(start_date, end_date)
# 
# # 打印结果
# print(f"从 {start_date} 到 {end_date} 的交易日共有 {len(trading_days)} 天：")
# print(trading_days)
# =============================================================================


def get_previous_trading_day(input_date: int) -> int:
    """
    获取指定日期的前一个交易日。

    参数:
        input_date (int): 输入日期，格式为 YYYYMMDD（如 20200101）
    
    返回:
        int: 前一个交易日，格式为 YYYYMMDD（如 20191231）
    """
    # 将输入日期转换为 datetime.date 类型
    input_date_dt = datetime.strptime(str(input_date), "%Y%m%d").date()

    # 从前一天开始往回查找
    previous_day = input_date_dt - timedelta(days=1)

    # 判断是否是交易日
    while previous_day not in trading_days_set:
        previous_day -= timedelta(days=1)

    # 返回结果转换为整数格式 YYYYMMDD
    return int(previous_day.strftime("%Y%m%d"))


# =============================================================================
# # 示例调用
# input_date = 20250105  # 示例输入日期
# previous_trading_day = get_previous_trading_day(input_date)
# print(f"输入日期为 {input_date}，前一个交易日是：{previous_trading_day}")
# =============================================================================


def get_previous_n_trading_day(input_date: str, n: int) -> str:
    """
    获取指定日期的第 n 天前的最近一个交易日。

    参数:
        input_date (str): 输入日期，格式为 YYYYMMDD（如 "20200101"）
        n (int): 向前数第 n 天，从该日期找到最近的交易日
    
    返回:
        str: 第 n 天前的最近一个交易日，格式为 YYYYMMDD（如 "20191231"）
    """
    # 将输入日期转换为 datetime.date 类型
    input_date_dt = datetime.strptime(input_date, "%Y%m%d").date()

    # 计算第 n 天前的日期
    target_date = input_date_dt - timedelta(days=n)

    # 如果第 n 天前的日期不是交易日，找到最近的前一个交易日
    while target_date not in trading_days_set:
        target_date -= timedelta(days=1)

    # 返回结果转换为字符串格式 YYYYMMDD
    return target_date.strftime("%Y%m%d")


# %%
if __name__=='__main__':
    pass