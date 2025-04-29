# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:12:24 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import sys
from pathlib import Path
import pandas as pd
import toml


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.datautils import compute_dataframe_dict_average
from test_and_eval.factor_tester import FactorTesterByContinuous, FactorTesterByDiscrete


#%%
class MergePos:
    
    def __init__(self, merge_name):
        self.merge_name = merge_name
        self._load_path_config()  # 加载路径配置
        self._init_dirs()  # 初始化目录
        self.params = self._load_params()

    def _load_path_config(self):
        """Load the project configuration file and set the path configuration."""
        file_path = Path(__file__).resolve()
        project_dir = file_path.parents[1]  # 假设路径配置文件在父级目录的父级目录
        self.path_config = load_path_config(project_dir)

    def _init_dirs(self):
        """Initialize directories based on the loaded path configuration."""
        self.param_dir = Path(self.path_config['param'])
        self.result_dir = Path(self.path_config['result'])
        self.model_dir = self.result_dir / 'model'
        self.merged_model_dir = self.result_dir / 'merged_model' / self.merge_name
        self.pos_dir = self.merged_model_dir / 'pos'
        self.pos_dir.mkdir(parents=True, exist_ok=True)

    def _load_params(self):
        """Load parameters from the TOML file."""
        return toml.load(self.param_dir / 'merge_model' / f'{self.merge_name}.toml')

    def merge_pos(self):
        """Merge position data from different models."""
        if_seperate_long_short = self.params.get('if_seperate_long_short', False)
        model_list = self.params['model_list']
        pos_dict = {}
        weight_dict = {}

        for model_info in model_list:
            model_name = model_info['model_name']
            test_name = model_info.get('test_name')
            weight = model_info.get('weight', 1)
            direction = model_info.get('direction')
            assert direction is None or direction in [-1, 1]

            test_data_dir = (self.model_dir / model_name / 'test' / test_name / 'data' 
                             if test_name is not None
                             else self.model_dir / model_name)
            pos_filename = f'pos_predict_{model_name}' if test_name is not None else f'pos_{model_name}'
            pos_path = test_data_dir / f'{pos_filename}.parquet'

            pos = pd.read_parquet(pos_path)
            pos_dict[(model_name, test_name, direction)] = pos
            weight_dict[(model_name, test_name, direction)] = weight
            
        if not if_seperate_long_short:
            pos_average = compute_dataframe_dict_average(pos_dict, weight_dict)
        else:
            pos_seperate = {}
            clip_side = {-1: 'upper', 1: 'lower'}
            for direction in (-1, 1):
                pos_dict_direction = {k: v.clip(**{clip_side[direction]: 0}) for k, v in pos_dict.items() if k[2] == direction}
                weight_dict_direction = {k: v for k, v in weight_dict.items() if k[2] == direction}
                pos_seperate[direction] = compute_dataframe_dict_average(pos_dict_direction, weight_dict_direction)
            pos_average = pos_seperate[-1] + pos_seperate[1]
        pos_average.to_csv(self.pos_dir / f'pos_{self.merge_name}.csv')
        pos_average.to_parquet(self.pos_dir / f'pos_{self.merge_name}.parquet')

    def run_tests(self):
        """Run the tests for each model."""
        test_list = self.params['test_list']
        for test_info in test_list:
            mode = test_info['mode']
            test_name = test_info['test_name']
            date_start = test_info.get('date_start')

            if mode == 'test':
                test_class = FactorTesterByContinuous
            elif mode == 'trade':
                test_class = FactorTesterByDiscrete
            else:
                raise NotImplementedError()

            ft = test_class(None, None, self.pos_dir, test_name=test_name, result_dir=self.merged_model_dir)
            ft.test_one_factor(f'pos_{self.merge_name}', date_start=date_start)
            
    def run(self):
        self.merge_pos()
        self.run_tests()
            
            
#%%
if __name__=='__main__':
    # Example usage
    merge_name = '1.2.0'
    merge_pos = MergePos(merge_name)
    merge_pos.run()