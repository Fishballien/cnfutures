# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 15:09:42 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
from pathlib import Path
import sys
from datetime import datetime
import yaml
import argparse


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.logutils import FishStyleLogger
from update.database_handler import DailyUpdateSender, DailyUpdateReader, DailyUpdateMsgSender
from update.loop_check import CheckDb, ProcessUpdateCoordinator
from utils.dateutils import get_previous_n_trading_day
from update.update_tecm import UpdateTestEvalClusterModel


# %%
def update_tecm(update_name=None, delay=1):
    
    # 更新至 ————
    date_today = datetime.today().strftime('%Y%m%d')
    target_date = get_previous_n_trading_day(date_today, delay)
    
    # 读取路径配置
    path_config = load_path_config(project_dir)
    param_dir = Path(path_config['workflow_param'])
    
    # 读取参数
    with open(param_dir / 'update_tecm' / f'{update_name}.yaml', "r") as file:
        params = yaml.safe_load(file)
    
    # 数据库交互
    # Initialize logger and senders
    mysql_name = params['mysql_name']
    author = params['author']
    log = FishStyleLogger()
    daily_update_sender = DailyUpdateSender(mysql_name, author, log=log)
    daily_update_reader = DailyUpdateReader(mysql_name, log=log)
    msg_sender = DailyUpdateMsgSender(mysql_name, author, log=log)
    
    # Initialize check database and coordinator
    check_db_params = params['check_db_params']
    check_db = CheckDb(daily_update_reader, msg_sender, log, config=check_db_params)
    coordinator = ProcessUpdateCoordinator(check_db, daily_update_sender, msg_sender, log)
    coordinator.set_target_date(target_date)
    
    ## update
    
    # 更新近10-20天股票ind
    tecm_params = params['tecm']
    update_name = tecm_params['update_name']
    mode = tecm_params['mode']
    output = tecm_params['output']
    dependency = tecm_params['dependency']
    
    with coordinator(output, dependency):
        if not coordinator.skip_task:
            # raise ValueError
            instance = UpdateTestEvalClusterModel(update_name, delay, mode)
            instance.run()


def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-un', '--update_name', type=str, help='update_name')
    parser.add_argument('-dl', '--delay', type=int, help='delay')

    args = parser.parse_args()
    update_name, delay = args.update_name, args.delay
    
    
    update_tecm(update_name, delay)
    
    
# %%
if __name__=='__main__':
    main()
