# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 17:12:30 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import time
import pandas as pd


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[0]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config


# %% from meta
def load_and_process_stat_file(file_path):
    """
    读取 .stat 文件并清洗数据。
    
    参数:
        file_path (str): .stat 文件的路径。
        
    返回:
        pd.DataFrame: 处理后的 DataFrame，包含最新日期的 Symbol 数据。
    """
    # 定义列名
    column_names = ["Date", "Symbol", "Trade Complete Rate", "Inc Missing Num"]
    
    # 读取文本数据
    df = pd.read_csv(file_path, sep=",", names=column_names, skipinitialspace=True)
    
    # 转换日期列为 datetime 格式
    df["Date"] = pd.to_datetime(df["Date"])
    
    # 提取数字（整数或浮点数）并转换类型
    df["Trade Complete Rate"] = df["Trade Complete Rate"].str.extract(r"(\d+\.\d+|\d+)$").astype(float)
    df["Inc Missing Num"] = df["Inc Missing Num"].str.extract(r"(\d+\.\d+|\d+)$").astype(float)
    
    # 按日期降序排序，并根据 Symbol 去重，保留最新日期
    df = df.sort_values(by="Date", ascending=False).drop_duplicates(subset="Symbol", keep="first")
    
    return df


def calculate_statistics(df):
    """
    计算统计信息，包括 Trade Complete Rate < 1 和 Inc Missing Num > 0 的情况。
    
    参数:
        df (pd.DataFrame): 处理后的 DataFrame。
        
    返回:
        dict: 包含统计信息的字典。
    """
    # 1) 计算 Trade Complete Rate < 1 的数量、最小值和均值
    trade_incomplete = df[df["Trade Complete Rate"] < 1]
    trade_incomplete_count = trade_incomplete.shape[0]
    trade_min = trade_incomplete["Trade Complete Rate"].min()
    trade_mean = trade_incomplete["Trade Complete Rate"].mean()
    
    # 2) 计算 Inc Missing Num > 0 的数量、最大值和均值
    inc_missing = df[df["Inc Missing Num"] > 0]
    inc_missing_count = inc_missing.shape[0]
    inc_max = inc_missing["Inc Missing Num"].max()
    inc_mean = inc_missing["Inc Missing Num"].mean()
    
    return {
        "trade_incomplete_count": trade_incomplete_count,
        "trade_min": trade_min,
        "trade_mean": trade_mean,
        "inc_missing_count": inc_missing_count,
        "inc_max": inc_max,
        "inc_mean": inc_mean
    }


class DailyDataChecker:
    
    def __init__(self, project_dir, config, daily_update_sender, msg_sender, log):
        self.daily_update_sender = daily_update_sender
        self.msg_sender = msg_sender
        self.log = log
        
        path_config = load_path_config(project_dir)
        self.yl_meta_dir = Path(path_config['yl_meta'])
        self.tradable_dir = Path(path_config['tradable'])

        # 加载配置
        self.error_repo_thres = timedelta(**config['error_repo_thres'])
        self.error_report_interval = timedelta(**config['error_repo_interval'])
        self.time_interval = timedelta(**config['time_interval'])

        # 日期和状态初始化
        self.today = datetime.utcnow().date()
        self.start_time = datetime.now()
        self.last_error_report = self.start_time
        self.tradable_success = False
        self.data_success = False

    def check_tradable(self):
        today_in_str = self.today.strftime('%Y-%m-%d')
        try:
            with open(self.tradable_dir / 'exchange_info_update_meta.json', 'r') as f:
                data = json.load(f)
            if data['date'] >= today_in_str:
                self.daily_update_sender.insert('tradable', data['date'])
                self.tradable_success = True
        except Exception as e:
            self.log.exception(f"检查tradable时发生异常: {e}")

    def check_data(self):
        today_in_shortstr = self.today.strftime('%y%m%d')
        today_in_fullstr = self.today.strftime('%Y-%m-%d')
        meta_file_name = f'{today_in_shortstr}_meta'
        meta_file_path = self.yl_meta_dir / meta_file_name
        stat_dir = self.yl_meta_dir / 'stat_daily'
        stat_file_name = f'{today_in_fullstr}.stat'
        stat_file_path = stat_dir / stat_file_name
        
        try:
            if not os.path.exists(meta_file_path):
                return
            with open(meta_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            if data['success']:
                self.daily_update_sender.insert('yl_source_data', self.today)
                self.data_success = True
                if os.path.exists(stat_file_path):
                    stat_df = load_and_process_stat_file(stat_file_path)
                    stats = calculate_statistics(stat_df)
                    self.msg_sender.insert('warning', 'yl_source完整性检查结果', str(stats))
            else:
                error = data['error']
                self.msg_sender.insert('error', '日更yl_source错误', str(error))
        except Exception as e:
            self.log.exception(f"检查data时发生异常: {e}")

    def run_check_loop(self, date_today=None):
        self.today = datetime.utcnow().date() if date_today is None else datetime.strptime(date_today, '%Y%m%d').date()
        today_in_str = self.today.strftime('%Y-%m-%d')
            
        while True:
            try:
                # 执行检查
                if not self.tradable_success:
                    self.check_tradable()

                if not self.data_success:
                    self.check_data()

                # 成功后退出循环
                if self.tradable_success and self.data_success:
                    msg = f'[{today_in_str}] tradable和yl_source已更新'
                    self.log.success(msg)
                    self.msg_sender.insert('success', msg, '')
                    return 0

                # 超过5小时每小时播报
                elapsed_time = datetime.now() - self.start_time
                if elapsed_time > self.error_repo_thres:
                    if datetime.now() - self.last_error_report >= self.error_report_interval:
                        msg = f'[{today_in_str}] tradable更新状态: {self.tradable_success}, yl_source更新状态：{self.data_success}'
                        self.log.warning(msg)
                        self.msg_sender.insert('warning', f'[{today_in_str}] tradable和yl_source还未更新', msg)
                        self.last_error_report = datetime.now()

            except Exception as e:
                self.log.exception(f"发生异常: {e}")
                e_format = traceback.format_exc()
                self.msg_sender.insert('error', '日更tradable和yl_source错误', e_format)

            time.sleep(self.time_interval.total_seconds())
                
        return 1
                

# %% from db
def check_status(fetch_res, today):
    """
    检查 fetch_res 列表中的每个元素，判断是否符合条件
    :param fetch_res: [(data_ts, status), ...]
    :return: True 如果满足条件的项存在，否则 False
    """
    for data_ts, obj_status in fetch_res:
        if data_ts.date() >= today and obj_status == 1:
            return True  # 一旦找到满足条件的项，返回 True
    return False  # 如果没有满足条件的项，返回 False


class CheckDb:
    
    def __init__(self, daily_update_reader, msg_sender, log, config={}):
        self.daily_update_reader = daily_update_reader
        self.msg_sender = msg_sender
        self.log = log
        self.time_interval = timedelta(**config.get('time_interval', {'minutes': 1}))
        self.error_repo_interval = timedelta(**config.get('error_repo_interval', {'minutes': 0}))
        self.error_repo_thres = timedelta(**config.get('error_repo_thres', {'minutes': 1}))
        self.max_retry = config.get('max_retry', 1e13)

    def loop_check(self, theme, objs, date_today=None, max_retry=None):
        today = datetime.utcnow().date() if date_today is None else datetime.strptime(date_today, '%Y%m%d').date()
        today_in_str = today.strftime('%Y-%m-%d')
        objs = [tuple(obj) for obj in objs]
        
        start_time = datetime.now()
        last_error_report = start_time
        retry_count = 0
        max_retry = max_retry or self.max_retry
        status = {obj: False for obj in objs}
        
        while True:
            retry_count += 1
            for obj in objs:
                if status[obj]:
                    continue
                fetch_res = self.daily_update_reader.fetch(*obj, today)
                if fetch_res is None:
                    continue
                status[obj] = check_status(fetch_res, today)

            if all(status.values()):
                self.log.success(f'[{today_in_str}] {theme}对象已更新')
                return 0
            
            if retry_count >= max_retry:
                return 1

            elapsed_time = datetime.now() - start_time
            if elapsed_time > self.error_repo_thres:
                if datetime.now() - last_error_report >= self.error_repo_interval:
                    msg = f"以下{theme}对象尚未更新: {[obj for obj, stat in status.items() if not stat]}"
                    self.log.warning(msg)
                    self.msg_sender.insert('warning', f'[{today_in_str}] 部分{theme}对象未更新', msg)
                    last_error_report = datetime.now()

            time.sleep(self.time_interval.total_seconds())
            
        return 1
    
    
# %% ProcessUpdateCoordinator
class EmptyContext:
    def __enter__(self):
        # Do nothing, this context is a "skip" placeholder
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        pass  # No operation, as we are skipping the task


class ProcessUpdateCoordinator:
    
    def __init__(self, check_db, daily_update_sender, msg_sender, log):
        self.check_db = check_db
        self.daily_update_sender = daily_update_sender
        self.msg_sender = msg_sender
        self.log = log
        self.target_date = None
        self.output = None
        self.dependency = None

    def set_target_date(self, target_date):
        self.target_date = target_date

    def check_before_task(self, output, dependency):
        output_theme = output['theme']
        output_objs = output['objs']
        deps_theme = dependency['theme']
        deps_objs = dependency['objs']

        # Check if the task is already finished
        if self.check_db.loop_check(output_theme, output_objs, self.target_date, max_retry=1) == 0:
            self.log.info(f"Skip task: {output_theme}")
            return -1  # Return -1 to indicate the task should be skipped

        # Check if the dependency has been updated
        deps_status = self.check_db.loop_check(deps_theme, deps_objs, self.target_date)
        return deps_status

    def insert_status(self, output):
        insert_item = {
            'obj': output['objs'][0][1],
            'data_ts': self.target_date,
        }
        self.daily_update_sender.insert(**insert_item)

    def __enter__(self):
        if self.output is None or self.dependency is None:
            raise ValueError("Output and dependency must be passed directly into the context.")

        # 判断任务是否跳过
        self.skip_task = False
        check_status = self.check_before_task(self.output, self.dependency)
        if check_status == -1:
            self.skip_task = True  # 设置跳过任务的标志
            self.log.info(f"Task skipped: {self.output['theme']}")
            return self  # 直接返回当前实例，跳过任务

        elif check_status != 0:
            raise RuntimeError(f"Dependency not updated for task: {self.output['theme']}")

        # 继续执行任务
        self.log.info(f"Start task: {self.output['theme']}")
        return self  # 返回自身，继续任务逻辑

    def __exit__(self, exc_type, exc_value, tb):
        """
        上下文管理器退出时的处理
        
        参数:
            exc_type: 异常类型，如果没有异常则为None
            exc_value: 异常值，如果没有异常则为None
            tb: 异常的traceback对象，如果没有异常则为None
        """
        if self.skip_task:
            self.log.info(f"Skipped task: {self.output['theme']}")
            return  # 跳过任务时，不进行任何处理
        
        if exc_type is None:  # 只有没有异常时才插入状态
            self.log.info(f"Task completed successfully: {self.output['theme']}")
            self.insert_status(self.output)
            self.msg_sender.insert('success', self.output['theme'], '')
        else:
            # 导入traceback模块来处理异常堆栈
            import traceback as tb_module
            
            # 使用传入的tb参数格式化异常堆栈
            if tb:
                # 使用format_exception获取完整的异常信息，包括类型、值和堆栈
                error = "".join(tb_module.format_exception(exc_type, exc_value, tb))
            else:
                # 如果tb为None（不太可能，但以防万一），尝试使用format_exc
                error = tb_module.format_exc()
            
            self.log.error(f"Task failed with exception: {self.output['theme']}")
            self.log.error(error)
            
            # 发送错误消息
            self.msg_sender.insert('error', self.output['theme'], error)
            
            # 不抑制异常传播，除非你想要这样做
            # 返回True会抑制异常，让上下文管理器外部代码继续执行
            # 返回None或False会让异常继续传播
            return False  # 或者返回None，让异常继续传播

    def __call__(self, output, dependency):
        self.output = output  # Store the output as input data
        self.dependency = dependency  # Store the dependency
        return self
