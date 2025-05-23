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
from synthesis.trade_rules_walk_forward import TradeRulesWalkForward


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
    
    def walk_forward_optimize(self):
        """
        Run walk-forward optimization for trade rules after fit_and_predict.
        Uses the same target date and mode as other components.
        """
        if 'trade_rules' not in self.params:
            self.log.info("No trade rules configurations found, skipping walk forward optimization")
            return
            
        trade_rules_prs = self.params['trade_rules']
        if isinstance(trade_rules_prs, dict):
            trade_rules_prs = [trade_rules_prs]
            
        for trade_rules_pr in trade_rules_prs:
            # Pass the target date as puntil
            if 'puntil' not in trade_rules_pr:
                trade_rules_pr['puntil'] = self.target_date
                
            # Set up common parameters
            config_file = trade_rules_pr.get('config_file')
            test_name = trade_rules_pr.get('test_name')
            pstart = trade_rules_pr.get('pstart')
            fstart = trade_rules_pr.get('fstart')
            n_workers = trade_rules_pr.get('n_workers', 1)
            base_model_name = trade_rules_pr.get('model_name')
            skip_test_eval = trade_rules_pr.get('skip_test_eval', False)
            check_consistency = trade_rules_pr.get('check_consistency', True)
            
            # Initialize and run walk forward optimization
            self.log.info(f"Starting walk forward optimization with config: {config_file}")
            walk_forward = TradeRulesWalkForward(
                config_file=config_file,
                test_name=test_name,
                pstart=pstart,
                puntil=self.target_date,  # Use the same target date
                fstart=fstart,
                n_workers=n_workers,
                base_model_name=base_model_name,
                skip_test_eval=skip_test_eval,
                check_consistency=check_consistency
            )
            
            walk_forward.run_walk_forward()
            self.log.success(f"Completed walk forward optimization with config: {config_file}")
            
    def send_pos_to_db(self):
        if 'send_pos_to_db' not in self.params:
            self.log.info("No database configurations found, skipping database upload")
            return
            
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
            
            # Determine position source and path
            pos_source = db_pr.get('pos_source', 'test')  # Default to 'test' for backward compatibility
            model_name = db_pr.get('model_name') or self.params['model']['test_name']
            
            if pos_source == 'test':
                # Original logic for positions from test directory
                test_name = db_pr['test_name']
                model_test_dir = self.result_dir / 'model' / model_name / 'test' / test_name / 'data'
                pos_filename = f'pos_predict_{model_name}'
                pos_path = model_test_dir / f'{pos_filename}.parquet'
            elif pos_source == 'rolling_model':
                # New logic for positions from rolling_model directory
                config_file = db_pr['config_file']
                version_name = db_pr.get('version_name', f"{model_name}_{config_file}")
                rolling_model_dir = self.result_dir / 'model' / model_name / 'rolling_model' / config_file / 'pos'
                pos_path = rolling_model_dir / f"pos_{version_name}.parquet"
            else:
                self.log.error(f"Unknown position source: {pos_source}")
                continue
                
            self.log.info(f"Reading positions from {pos_path}")
            try:
                pos = pd.read_parquet(pos_path)
                pos_to_send = pos.loc[start_date:].dropna(how='all')
                backtest_sender.insert(pos_to_send, acname)
                self.log.success(f"Successfully sent positions to database: {mysql_name}, account: {acname}")
            except Exception as e:
                self.log.error(f"Failed to send positions to database: {str(e)}")
            
    def run(self):
        self._init_director_processor()
        self.multi_test()
        self.evaluate()
        self.cluster()
        self.fit_and_predict()
        self.walk_forward_optimize()  # Add walk forward optimization after fit_and_predict
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