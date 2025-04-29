# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:28:24 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% import public
import sys
from pathlib import Path
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from update.update_tecm import UpdateTestEvalClusterModel
from utils.dirutils import load_path_config
from utils.logutils import FishStyleLogger
from update.database_handler import DailyUpdateSender, DailyUpdateReader, DailyUpdateMsgSender
from update.loop_check import CheckDb, ProcessUpdateCoordinator
from update.dateutils import get_previous_n_trading_day


# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-un', '--update_name', type=str, help='update_name')
    parser.add_argument('-dl', '--delay', type=str, help='delay')
    parser.add_argument('-m', '--mode', type=str, help='mode')

    args = parser.parse_args()
    update_name, delay, mode = args.update_name, args.delay, args.mode
    
    # 更新至 ————
    date_today = datetime.today().strftime('%Y%m%d')
    target_date = get_previous_n_trading_day(date_today, delay)
    
    # 读取路径配置
    path_config = load_path_config(project_dir)
    param_dir = Path(path_config['param'])
    
    # 读取参数
    with open(param_dir / 'update_factors' / f'{update_name}.yaml', "r") as file:
        params = yaml.safe_load(file)
    # params = toml.load(param_dir / 'update_factors' / f'{update_name}.toml')
    
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
    lob_ind_params = params['lob_indicators']
    lob_ind_pj_name = lob_ind_params['lob_ind_pj_name']
    ind_ver_name = lob_ind_params['ind_ver_name']
    output = lob_ind_params['output']
    dependency = lob_ind_params['dependency']
    
    with coordinator(output, dependency):
        if not coordinator.skip_task:
            with temporary_sys_path(path_config[lob_ind_pj_name]):
                from core.updater import IncrementalUpdate as IndicatorInc
                instance = IndicatorInc(ind_ver_name)
                instance.run(target_date)
    
    updater = UpdateTestEvalClusterModel(update_name, delay, mode=mode)
    updater.run()
    

# %% main
if __name__ == "__main__":
    main()