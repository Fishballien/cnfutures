# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:59:39 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import sys
from pathlib import Path
import yaml
import toml
import argparse
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import DirectoryProcessor
from utils.logutils import FishStyleLogger
from test_and_eval.factor_tester import FactorTesterByContinuous, FactorTesterByDiscrete
from test_and_eval.rolling_eval import RollingEval
from synthesis.rolling_cluster import RollingCluster
from scripts.rolling_fit_pred_backtest import main as rolling_fit_and_predict
from utils.dateutils import get_previous_n_trading_day
from update.database_handler import PythonTradeBacktestHandler


# %% class
class UpdateTestEvalClusterModel:
    
    def __init__(self, update_name, delay, mode='rolling'):
        self.update_name = update_name
        self.delay = delay
        self.mode = mode
        
        self.log = FishStyleLogger()
        self._load_paths()
        self._load_agg_params()
        self._get_target_date()

    def _load_paths(self):
        file_path = Path(__file__).resolve()
        file_dir = file_path.parents[1]
        path_config_path = file_dir / '.path_config.yaml'
        with path_config_path.open('r') as file:
            path_config = yaml.safe_load(file)

        self.param_dir = Path(path_config['param'])
        self.result_dir = Path(path_config['result'])
        
    def _load_agg_params(self):
        self.params = toml.load(self.param_dir / 'update' / f'{self.update_name}.toml')

    def _init_director_processor(self):
        root_dir_dict = self.params['root_dir_dict']
        
        self.dirp = DirectoryProcessor(root_dir_dict)
        
    def _get_target_date(self):
        date_today = datetime.today().strftime('%Y%m%d')
        target_date = get_previous_n_trading_day(date_today, self.delay)
        self.target_date = (datetime.strptime(target_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
        # self.target_date = (datetime.now() - timedelta(days=self.delay)).strftime('%Y%m%d')

    def multi_test(self):
        mapping = self.dirp.mapping
        test_pr = self.params['test']
        mode = test_pr.get('mode', 'test')
        test_name = test_pr['test_name']
        skip_plot = test_pr['skip_plot']
        test_wkr = test_pr['n_workers']
        check_consistency = test_pr.get('check_consistency', True)
        
        if mode == 'test':
            test_class = FactorTesterByContinuous
        elif mode == 'trade':
            test_class = FactorTesterByDiscrete
        else:
            NotImplementedError()
        
        for root_dir in mapping:
            root_info = mapping[root_dir]
            tag_name = root_info['tag_name']
            process_name_list = root_info['leaf_dirs']
            kwargs = {
                'tag_name': tag_name,
                'factor_data_dir': Path(root_dir), 
                'test_name': test_name, 
                'skip_plot': skip_plot,
                'n_workers': test_wkr,
                'check_consistency': check_consistency,
            }
    
            for process_name in process_name_list:
                self.log.info(f'Test Started: {process_name}')
                tester = test_class(process_name, **kwargs)
                tester.test_multi_factors()
                self.log.success(f'Test Finished: {process_name}')
            self.log.success(f'Root Finished: {root_dir}')

    def evaluate(self):
        eval_prs = self.params['eval']
        if isinstance(eval_prs, dict):
            eval_prs = [eval_prs]
        
        for eval_pr in eval_prs:
            eval_pr['puntil'] = self.target_date
            eval_pr['eval_type'] = self.mode
            eval_pr['check_consistency'] = eval_pr.get('check_consistency', True)
            
            rolling_eval = RollingEval(**eval_pr)
            rolling_eval.run()

    def cluster(self):
        cluster_prs = self.params['cluster']
        if isinstance(cluster_prs, dict):
            cluster_prs = [cluster_prs]
            
        for cluster_pr in cluster_prs:
            cluster_pr['puntil'] = self.target_date
            cluster_pr['cluster_type'] = self.mode
            cluster_pr['check_consistency'] = cluster_pr.get('check_consistency', True)
            
            rolling_cluster = RollingCluster(**cluster_pr)
            rolling_cluster.run()
        
    def fit_and_predict(self):
        model_prs = self.params['model']
        if isinstance(model_prs, dict):
            model_prs = [model_prs]
        
        for model_pr in model_prs:
            model_pr['puntil'] = self.target_date
            model_pr['mode'] = self.mode
            model_pr['check_consistency'] = model_pr.get('check_consistency', True)
            
            rolling_fit_and_predict(**model_pr)
            
    def send_pos_to_db(self):
        db_pr = self.params['send_pos_to_db']
        
        if isinstance(db_pr, dict):
            db_pr_list = [db_pr]
        elif isinstance(db_pr, list):
            db_pr_list = db_pr
            
        for db_pr in db_pr_list:
            mysql_name = db_pr['mysql_name']
            acname = db_pr['acname']
            start_date = db_pr.get('start_date')
            backtest_sender = PythonTradeBacktestHandler(mysql_name, log=self.log)
            
            model_name = db_pr.get('model_name') or self.params['model']['test_name']
            test_name = db_pr['test_name']
            model_test_dir = self.result_dir / 'model' / model_name / 'test' / test_name / 'data'
            pos_filename = f'pos_predict_{model_name}'
            pos_path = model_test_dir / f'{pos_filename}.parquet'
            
            pos = pd.read_parquet(pos_path)
            pos_to_send = pos.loc[start_date:].dropna(how='all')
            backtest_sender.insert(pos_to_send, acname)
            
    def run(self):
        self._init_director_processor()
        self.multi_test()
        self.evaluate()
        self.cluster()
        self.fit_and_predict()
        self.send_pos_to_db()


# %% main
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-un', '--update_name', type=str, help='update_name')
    parser.add_argument('-dl', '--delay', type=int, help='delay')
    parser.add_argument('-m', '--mode', type=str, help='mode')

    args = parser.parse_args()
    update_name, delay, mode = args.update_name, args.delay, args.mode
    
    updater = UpdateTestEvalClusterModel(update_name, delay, mode=mode)
    updater.run()
    

# %% main
if __name__ == "__main__":
    main()
