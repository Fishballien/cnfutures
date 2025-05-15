# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:15:44 2025

@author: Xintang Zheng

A module for testing and optimizing trade rules with walk-forward analysis.
Processes multiple threshold combinations in parallel and selects optimal parameters.

"""
# %%
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import toml
import copy
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from functools import partial


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from test_and_eval.factor_tester import FactorTesterByDiscrete, FactorTesterByContinuous
from test_and_eval.scores import get_general_return_metrics
from test_and_eval.factor_evaluation import eval_one_factor_one_period_net_public, eval_one_factor_one_period_by_net
from utils.timeutils import RollingPeriods, period_shortcut
from utils.datautils import add_dataframe_to_dataframe_reindex
from utils.logutils import FishStyleLogger


# Define multiprocessing functions outside of the class
def test_single_threshold(factor_data_dir, process_name, tag_name, test_name, test_param, result_dir, openthres, closethres):
    """
    Test a single combination of open and close thresholds.
    
    Parameters:
    -----------
    factor_data_dir : Path
        Directory containing factor data
    process_name : str
        Process name
    tag_name : str
        Tag name
    test_name : str
        Base test name
    test_param : dict
        Test parameters
    result_dir : Path
        Result directory
    openthres : float
        Open threshold value
    closethres : float
        Close threshold value
        
    Returns:
    --------
    str : Test name for the combination
    """
    test_pr = copy.deepcopy(test_param)
    test_pr['trade_rule_param']['threshold_combinations'] = [[openthres, closethres]]
    
    new_test_name = f"{test_name}_op{openthres}_cl{closethres}"
    
    tester = FactorTesterByDiscrete(
        process_name, 
        tag_name, 
        factor_data_dir, 
        test_name=new_test_name, 
        result_dir=result_dir, 
        params=test_pr
    )
    tester.test_multi_factors()
    
    return new_test_name


def eval_single_threshold(factor_data_dir, model_name, test_name, tag_name, process_name, fee, openthres, closethres, date_start, date_end):
    """
    Evaluate a single threshold combination for a specific period.
    
    Parameters:
    -----------
    factor_data_dir : Path
        Directory containing factor data
    model_name : str
        Model name
    test_name : str
        Base test name
    tag_name : str
        Tag name
    process_name : str
        Process name
    fee : float
        Trading fee
    openthres : float
        Open threshold value
    closethres : float
        Close threshold value
    date_start : datetime
        Start date for evaluation
    date_end : datetime
        End date for evaluation
        
    Returns:
    --------
    dict : Evaluation metrics for this threshold combination
    """
    res_info = {
        'openthres': openthres, 
        'closethres': closethres, 
    }
    
    pred_name = f"predict_{model_name}"
    new_test_name = f"{test_name}_op{openthres}_cl{closethres}"
    test_data_dir = factor_data_dir / 'test' / new_test_name / tag_name / process_name / 'data'
    
    res_dict = eval_one_factor_one_period_net_public(
        pred_name, res_info, test_data_dir, date_start, date_end, fee)
    
    return res_dict


class TradeRulesWalkForward:
    """
    A class to perform walk-forward testing and optimization of trading rules.
    Tests multiple open/close threshold combinations and selects optimal parameters.
    """

    def __init__(self, config_file=None, test_name=None, pstart=None, puntil=None, n_workers=1,
                fstart=None, check_consistency=True, base_model_name=None):
        """
        Initialize the TradeRulesWalkForward object.
        
        Parameters:
        -----------
        config_file : str
            Name of the configuration file (without .toml extension) located in param/test_trade_rules/ directory.
        test_name : str, optional
            Name of the test to run. If provided, overrides the config file value.
        pstart : str, optional
            Start date for prediction in 'YYYYMMDD' format. If provided, overrides the config file value.
        puntil : str, optional
            End date for prediction in 'YYYYMMDD' format. If provided, overrides the config file value.
        n_workers : int, optional
            Number of workers for parallel processing.
        fstart : str, optional
            Start date for filtering in 'YYYYMMDD' format. If provided, overrides the config file value.
        check_consistency : bool, optional
            Whether to check consistency of results with previous runs.
        base_model_name : str, optional
            Base model name for directory structure.
        """
        self.config_file = config_file
        self.test_name_override = test_name
        self.pstart_override = pstart
        self.puntil_override = puntil
        self.fstart_override = fstart
        self.n_workers = n_workers
        self.check_consistency = check_consistency
        self.base_model_name = base_model_name
        
        self.logger = FishStyleLogger()
        
        self._load_config()
        self._load_paths()
        self._init_dirs()
        self._init_parameters()
    
    def _load_config(self):
        """
        Load configuration from a YAML file.
        Looks for the config file in the parameter directory under 'test_trade_rules/{config_name}.yaml'
        """
        # Load path configuration first to find parameter directory
        path_config = load_path_config(project_dir)
        param_dir = Path(path_config['param'])
        
        if self.config_file:
            # Build the complete path to the config file
            config_path = param_dir / 'test_trade_rules' / f"{self.config_file}.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
                
            import yaml
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            self.logger.info(f"Loaded configuration from {config_path}")
        else:
            # Use minimal default configuration
            self.config = {}
            self.logger.warning("No configuration file specified. Using command line arguments only.")
            
        # Override config with command-line arguments if provided
        if self.test_name_override:
            self.config['test_name'] = self.test_name_override
        if self.pstart_override:
            self.config['pstart'] = self.pstart_override
        if self.puntil_override:
            self.config['puntil'] = self.puntil_override
        if self.fstart_override:
            self.config['fstart'] = self.fstart_override
            
        # Ensure required configurations are present
        required_keys = [
            'model_name', 'suffix', 'process_name', 'tag_name', 'test_name',
            'multi_test_name', 'multi_test_func_name', 'fee', 'fstart',
            'pstart', 'puntil', 'window_kwargs', 'rrule_kwargs', 'end_by',
            'version_name', 'version_params', 'openthres_list', 'closethres_list'
        ]
        
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ValueError(f"Missing required configuration parameters: {', '.join(missing_keys)}\n"
                           f"Please provide a configuration file with all required parameters.")

            
    def _load_paths(self):
        """Load path configuration."""
        path_config = load_path_config(project_dir)
        self.param_dir = Path(path_config['param'])
        self.result_dir = Path(path_config['result'])
        self.model_dir = self.result_dir / 'model'
        
    def _init_dirs(self):
        """Initialize directories for outputs."""
        # Extract parameters from config
        model_name = self.config['model_name']
        version_name = self.config['version_name']
        
        # Setup directories
        self.factor_data_dir = self.model_dir / model_name
        self.rolling_model_dir = self.result_dir / 'rolling_model' / version_name
        self.pos_dir = self.rolling_model_dir / 'pos'
        self.pos_dir.mkdir(parents=True, exist_ok=True)
        
        self.summary_dir = self.factor_data_dir / 'trade_rule_summary'
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        
    def _init_parameters(self):
        """Initialize parameters from config."""
        # Get test parameters
        test_name = self.config['test_name']
        test_param_path = self.param_dir / 'test' / f'{test_name}.toml'
        self.test_param = toml.load(test_param_path)
        
        # Get threshold lists
        self.openthres_list = list(reversed(self.config.get('openthres_list', 
                                            [-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])))
        self.closethres_list = self.config.get('closethres_list',
                                            [-0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        
        # Setup rolling periods
        self.rolling = RollingPeriods(
            fstart=datetime.strptime(self.config['fstart'], '%Y%m%d'),
            pstart=datetime.strptime(self.config['pstart'], '%Y%m%d'),
            puntil=datetime.strptime(self.config['puntil'], '%Y%m%d'),
            window_kwargs=self.config['window_kwargs'],
            rrule_kwargs=self.config['rrule_kwargs'],
            end_by=self.config['end_by'],
        )
            
    def test_thresholds(self):
        """
        Test all combinations of open and close thresholds.
        Uses parallel processing for efficiency.
        """
        self.logger.info(f"Testing {len(self.openthres_list)} open thresholds and {len(self.closethres_list)} close thresholds")
        
        # Create test combinations and tasks
        test_tasks = []
        for openthres in self.openthres_list:
            for closethres in self.closethres_list:
                test_tasks.append((openthres, closethres))
        
        # Process in parallel
        if self.n_workers > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [
                    executor.submit(
                        test_single_threshold,
                        self.factor_data_dir,
                        self.config['process_name'],
                        self.config['tag_name'],
                        self.config['test_name'],
                        self.test_param,
                        self.result_dir,
                        openthres,
                        closethres
                    ) for openthres, closethres in test_tasks
                ]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc='Testing thresholds'):
                    # No need to collect results as they are saved directly
                    result = future.result()
        else:
            # Sequential processing
            for openthres, closethres in tqdm(test_tasks, desc='Testing thresholds'):
                test_single_threshold(
                    self.factor_data_dir,
                    self.config['process_name'],
                    self.config['tag_name'],
                    self.config['test_name'],
                    self.test_param,
                    self.result_dir,
                    openthres,
                    closethres
                )
    
    def eval_all_thresholds_for_period(self, date_start, date_end):
        """
        Evaluate all threshold combinations for a specific period.
        
        Parameters:
        -----------
        date_start : datetime
            Start date for evaluation
        date_end : datetime
            End date for evaluation
            
        Returns:
        --------
        pd.DataFrame : Evaluation results for all thresholds
        """
        self.logger.info(f"Evaluating thresholds for period {date_start} to {date_end}")
        
        # Create evaluation tasks
        eval_tasks = []
        for openthres in self.openthres_list:
            for closethres in self.closethres_list:
                eval_tasks.append((openthres, closethres))
                
        # Process tasks
        if self.n_workers > 1:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [
                    executor.submit(
                        eval_single_threshold,
                        self.factor_data_dir,
                        self.config['model_name'],
                        self.config['test_name'],
                        self.config['tag_name'],
                        self.config['process_name'],
                        self.config['fee'],
                        openthres,
                        closethres,
                        date_start,
                        date_end
                    ) for openthres, closethres in eval_tasks
                ]
                
                results = []
                for future in tqdm(as_completed(futures), total=len(futures), desc='Evaluating thresholds'):
                    results.append(future.result())
        else:
            # Sequential processing
            results = []
            for openthres, closethres in tqdm(eval_tasks, desc='Evaluating thresholds'):
                results.append(
                    eval_single_threshold(
                        self.factor_data_dir,
                        self.config['model_name'],
                        self.config['test_name'],
                        self.config['tag_name'],
                        self.config['process_name'],
                        self.config['fee'],
                        openthres,
                        closethres,
                        date_start,
                        date_end
                    )
                )
        
        # Combine results
        res_df = pd.DataFrame(results)
        return res_df
    
    def generate_condition(self, data, params):
        """
        Generate condition based on the evaluation metrics and threshold parameters.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Evaluation data for threshold combinations
        params : dict
            Filtering parameters
            
        Returns:
        --------
        pd.Series : Boolean series indicating which combinations pass the filters
        """
        condition = (
            (data['net_sharpe_ratio'] > params['net_sharpe_ratio']) & 
            (data['profit_per_trade'] > params['profit_per_trade']) & 
            (data['net_max_dd'] < params['net_max_dd']) & 
            (data['net_burke_ratio'] > params['net_burke_ratio'])
        )
        return condition
    
    def generate_condition_with_neighbors(self, data, params):
        """
        Generate condition based on thresholds and their neighboring values.
        Requires all neighboring threshold combinations to also pass the filter.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Evaluation data for threshold combinations
        params : dict
            Filtering parameters
            
        Returns:
        --------
        pd.Series : Boolean series indicating which combinations pass the filters
        """
        condition = pd.Series(True, index=data.index)  # Initialize condition as True
        
        # Get indices of openthres and closethres values
        openthres_idx = {v: i for i, v in enumerate(self.openthres_list)}
        closethres_idx = {v: i for i, v in enumerate(self.closethres_list)}

        for idx, row in data.iterrows():
            openthres, closethres = row.name
            
            # Get neighboring threshold positions
            # Define the five cases directly
            neighbors = [
                data.loc[(openthres, closethres)],  # Current value
                data.loc[(self.openthres_list[openthres_idx[openthres] + 1] if openthres_idx[openthres] + 1 < len(self.openthres_list) else openthres, closethres)],  # open + 1
                data.loc[(self.openthres_list[openthres_idx[openthres] - 1] if openthres_idx[openthres] - 1 >= 0 else openthres, closethres)],  # open - 1
                data.loc[(openthres, self.closethres_list[closethres_idx[closethres] + 1] if closethres_idx[closethres] + 1 < len(self.closethres_list) else closethres)],  # close + 1
                data.loc[(openthres, self.closethres_list[closethres_idx[closethres] - 1] if closethres_idx[closethres] - 1 >= 0 else closethres)]  # close - 1
            ]
            
            # Check neighbors
            for neighbor in neighbors:
                # If any neighbor doesn't meet the conditions, mark current position as False
                if not (
                    neighbor['net_sharpe_ratio'] > params['net_sharpe_ratio'] and
                    neighbor['profit_per_trade'] > params['profit_per_trade'] and
                    neighbor['net_max_dd'] < params['net_max_dd'] and
                    neighbor['net_burke_ratio'] > params['net_burke_ratio']
                ):
                    condition[idx] = False
                    break
            
            # Double-check that current position meets conditions
            if not (
                row['net_sharpe_ratio'] > params['net_sharpe_ratio'] and
                row['profit_per_trade'] > params['profit_per_trade'] and
                row['net_max_dd'] < params['net_max_dd'] and
                row['net_burke_ratio'] > params['net_burke_ratio']
            ):
                condition[idx] = False

        return condition
    
    def generate_conditions_text(self, params):
        """
        Generate text description of the filtering conditions.
        
        Parameters:
        -----------
        params : dict
            Filtering parameters
            
        Returns:
        --------
        str : Description of filtering conditions
        """
        conditions_text = "Filter conditions:\n"
        for key, value in params.items():
            if key == 'net_max_dd':
                conditions_text += f"- {key} < {value}\n"
            else:
                conditions_text += f"- {key} > {value}\n"
        return conditions_text
    
    def filter_conditions(self, res_df, date_start, date_end, to_plot=None):
        """
        Filter threshold combinations based on evaluation metrics and plot results.
        
        Parameters:
        -----------
        res_df : pd.DataFrame
            Evaluation results for threshold combinations
        date_start : datetime
            Start date of the evaluation period
        date_end : datetime
            End date of the evaluation period
        to_plot : list, optional
            Metrics to plot
            
        Returns:
        --------
        list : Valid threshold combinations
        """
        period = period_shortcut(date_start, date_end)
        
        # Set 'openthres' and 'closethres' as index
        heatmap_data = res_df.set_index(['openthres', 'closethres'])
        
        # Generate condition based on parameters
        if self.config['version_params'].get('neighbor'):
            condition = self.generate_condition_with_neighbors(heatmap_data, self.config['version_params'])
        else:
            condition = self.generate_condition(heatmap_data, self.config['version_params'])
            
        conditions_text = self.generate_conditions_text(self.config['version_params'])
        
        # Extract valid combinations
        valid_pairs = condition[condition].index.tolist()
        self.logger.info(f"Version {self.config['version_name']} {period} valid pairs: {valid_pairs}")
        
        # Create mask for visualization
        mask = ~condition.unstack()  # Convert to unstack and invert
        
        # Plot heatmaps for selected metrics
        to_plot = to_plot or heatmap_data.columns
        for column in to_plot:
            fig, ax = plt.subplots(figsize=(12, 8))  # Increased width for text area
            
            # Convert data to matrix format for heatmap
            heatmap_matrix = heatmap_data[column].unstack()
            
            # Create heatmap with mask
            sns.heatmap(heatmap_matrix, annot=True, cmap='coolwarm', fmt='.2f', mask=mask, ax=ax)
        
            # Adjust labels and title
            plt.title(f'Masked Heatmap for {column} {period} (filtered by conditions) - {self.config["version_name"]}')
            plt.xlabel('closethres')
            plt.ylabel('openthres')
        
            # Add filtering conditions text
            plt.figtext(1.05, 0.5, conditions_text, fontsize=12, verticalalignment="center", 
                      bbox=dict(facecolor='white', alpha=0.5))
        
            # Adjust layout
            plt.tight_layout()
        
            # Save figure
            img_filename = self.summary_dir / f'{self.config["version_name"]}_{period}_{column}_masked_heatmap.png'
            plt.savefig(img_filename, bbox_inches='tight')
            plt.close()
            
        # Create test parameter for multi-threshold test
        test_pr = copy.deepcopy(self.test_param)
        test_pr['trade_rule_name'] = self.config['multi_test_func_name']
        test_pr['trade_rule_param'] = {'threshold_combinations': valid_pairs}
        
        new_test_name = f"{self.config['multi_test_name']}_{self.config['version_name']}_{period}"
        
        # Save test parameters
        with open(self.param_dir / 'test' / f"{new_test_name}.toml", "w") as f:
            toml.dump(test_pr, f)
        
        # Test factor with multi-threshold
        tester = FactorTesterByDiscrete(
            None, None, self.factor_data_dir / 'predict', 
            test_name=new_test_name, 
            result_dir=self.factor_data_dir, 
            params=test_pr
        )
        tester.test_one_factor(f"predict_{self.config['model_name']}")
        
        return valid_pairs
    
    def run_walk_forward(self):
        """
        Run the entire walk-forward optimization process.
        Tests thresholds, evaluates performance, and selects optimal parameters for each period.
        """
        # First test all threshold combinations
        self.test_thresholds()
        
        # For each rolling period, filter and select optimal combinations
        for fp in tqdm(self.rolling.fit_periods, 'Rolling filter'):
            eval_res = self.eval_all_thresholds_for_period(*fp)
            self.filter_conditions(eval_res, *fp, to_plot=['net_sharpe_ratio'])
        
        # Collect all position data across periods
        pos_all = pd.DataFrame()
        for fp, pp in tqdm(list(zip(self.rolling.fit_periods, self.rolling.predict_periods)), desc='Concat prediction'):
            period = period_shortcut(*fp)
            new_test_name = f"{self.config['multi_test_name']}_{self.config['version_name']}_{period}"
            test_data_dir = self.factor_data_dir / 'test' / new_test_name / 'data'
            pos_filename = f"pos_predict_{self.config['model_name']}"
            pos_path = test_data_dir / f'{pos_filename}.parquet'
            pos = pd.read_parquet(pos_path)
            pos_to_predict = pos.loc[pp[0]:pp[1]]
            
            pos_all = add_dataframe_to_dataframe_reindex(pos_all, pos_to_predict)
        
        # Save combined positions
        pos_all.to_csv(self.pos_dir / f"pos_{self.config['version_name']}.csv")
        pos_all.to_parquet(self.pos_dir / f"pos_{self.config['version_name']}.parquet")
        
        # Run final tests on combined positions
        self.run_final_tests()
        
    def run_final_tests(self):
        """
        Run final tests on the combined positions across all periods.
        """
        for test_info in self.config['final_test_list']:
            mode = test_info['mode']
            test_name = test_info['test_name']
            date_start = test_info.get('date_start')

            if mode == 'test':
                test_class = FactorTesterByContinuous
            elif mode == 'trade':
                test_class = FactorTesterByDiscrete
            else:
                raise NotImplementedError(f"Unknown test mode: {mode}")

            ft = test_class(
                None, None, self.pos_dir, 
                test_name=test_name, 
                result_dir=self.rolling_model_dir
            )
            ft.test_one_factor(f"pos_{self.config['version_name']}")
            
        self.logger.success(f"Final tests completed for version {self.config['version_name']}")


def main():
    """
    Parse command line arguments and run the module.
    """
    parser = argparse.ArgumentParser(description='Walk-forward testing of trading rules')
    parser.add_argument('-c', '--config', type=str, required=True, 
                       help='Name of config file (without .toml extension) in param/test_trade_rules directory')
    parser.add_argument('-t', '--test_name', type=str, 
                       help='Test name (overrides config)')
    parser.add_argument('-p', '--pstart', type=str, 
                       help='Prediction start date (YYYYMMDD)')
    parser.add_argument('-u', '--puntil', type=str, 
                       help='Prediction until date (YYYYMMDD)')
    parser.add_argument('-f', '--fstart', type=str, 
                       help='Filter start date (YYYYMMDD)')
    parser.add_argument('-w', '--workers', type=int, default=1, 
                       help='Number of workers for parallel processing')
    parser.add_argument('-m', '--model', type=str, 
                       help='Base model name')
    
    args = parser.parse_args()
    
    # Initialize and run
    walk_forward = TradeRulesWalkForward(
        config_file=args.config,
        test_name=args.test_name,
        pstart=args.pstart,
        puntil=args.puntil,
        fstart=args.fstart,
        n_workers=args.workers,
        base_model_name=args.model
    )
    
    walk_forward.run_walk_forward()


if __name__ == '__main__':
    main()