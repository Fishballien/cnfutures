# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 10:43:47 2025

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
import argparse
import warnings
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %% import self-defined
from utils.dirutils import load_path_config
from test_and_eval.batch_test import batch_test


# %%
def main():
    # init dir
    path_config = load_path_config(project_dir)
    factor_data_dir = path_config['factor_data']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--process_name', type=str, help='process_name')
    parser.add_argument('-tag', '--tag_name', type=str, default=None, help='tag_name')
    parser.add_argument('-fdir', '--factor_data_dir', type=str, default=factor_data_dir, help='factor_data_dir')
    parser.add_argument('-bt', '--batch_test_name', type=str, help='batch_test_name')
    args = parser.parse_args()
    args_dict = vars(args)
    
    batch_test(**args_dict)
    
    
# %%
if __name__=='__main__':
    main()