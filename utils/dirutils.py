# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:00:28 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import os
import yaml
import fnmatch
from pathlib import Path
import re


# %% leaf
def find_leaf_directories(root_dir):
    leaf_directories = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not dirnames:  # 如果当前目录没有子目录
            relative_path = os.path.relpath(dirpath, root_dir).replace(os.sep, '/')  # 获取相对于根目录的相对路径
            leaf_directories.append(relative_path)
    return leaf_directories


class DirectoryProcessor:
    
    def __init__(self, root_dir_dict):
        self.root_dir_dict = root_dir_dict
        self._generate_mapping()
        self._generate_list()
    
    def _generate_mapping(self):
        self._mapping = {}
        for root_dir in self.root_dir_dict:
            root_info = self.root_dir_dict[root_dir]
            tag_name = root_info.get('tag_name')
            target_leaf = root_info.get('target_leaf', [])
            exclude = root_info.get('exclude', [])
            leaf_dirs = target_leaf if target_leaf else find_leaf_directories(root_dir)
            leaf_dirs = [leaf_dir for leaf_dir in leaf_dirs if leaf_dir not in exclude]
            self._mapping[root_dir] = {'tag_name': tag_name, 'leaf_dirs': leaf_dirs}
        return self._mapping

    def _generate_list(self):
        self._list = [
            (root_dir, mapping['tag_name'], leaf_dir)
            for root_dir, mapping in self._mapping.items()
            for leaf_dir in mapping['leaf_dirs']
        ]
        return self._list
    
    @property
    def mapping(self):
        return self._mapping
    
    @property
    def list_of_tuple(self):
        return self._list
    

# %% count
def count_files_in_directory(directory, pattern):
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if fnmatch.fnmatch(filename, pattern):
                file_count += 1
    return file_count


def compare_directories(dir1, pattern1, dir2, pattern2, tolerance_ratio):
    count1 = count_files_in_directory(dir1, pattern1)
    count2 = count_files_in_directory(dir2, pattern2)
    
    if count1 == 0:
        raise ValueError("The first directory is empty or contains no matching files, comparison is not valid.")
    
    ratio = count2 / count1
    difference = abs(ratio - 1)
    
    if difference <= tolerance_ratio:
        return True, count1, count2, ratio
    else:
        return False, count1, count2, ratio
    
    
# %% load path
def load_path_config(project_dir):
    path_config_path = project_dir / '.path_config.yaml'
    with path_config_path.open('r') as file:
        path_config = yaml.safe_load(file)
    return path_config


# %%
def get_file_names_without_extension(folder_path, suffix=".csv"):
    """
    获取文件夹下所有指定后缀的文件名（不包含后缀）。

    参数:
        folder_path (str or Path): 文件夹路径。
        suffix (str): 要筛选的文件后缀，默认是 ".csv"。

    返回:
        list: 所有符合条件的文件名（不包含后缀）。
    """
    # 转换为 Path 对象
    folder = Path(folder_path)
    # 获取文件夹下所有指定后缀的文件名
    file_names = [file.stem for file in folder.iterdir() if file.is_file() and file.suffix == suffix]
    return file_names


# %%
def list_pattern_matches(directory, pattern):
    matches = []
    
    # 将 pattern 中的 '*' 转换为正则表达式中的 '.*?'，即匹配任意字符（非贪婪）
    # 并确保 '.' 被转义为 '\.'，以匹配字面上的点
    regex_pattern = '^' + re.escape(pattern).replace(r'\*', r'(.*?)') + '$'

    # 遍历目录中的所有文件
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            # 用正则表达式匹配文件名
            match = re.match(regex_pattern, filename)
            if match:
                # 提取并添加捕获到的部分
                matches.append(match.group(1))  # 获取正则表达式中第一个捕获组

    return matches


# %%



