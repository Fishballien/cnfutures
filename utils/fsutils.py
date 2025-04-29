# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:28:13 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import os
import re
import shutil
from pathlib import Path
from typing import List, Union


# %%
def find_files_with_prefix(directory_path: Union[str, Path], target_prefix: str = None) -> List[str]:
    """
    在指定目录中查找所有以目标前缀开头，后接两个由下划线分隔字段（即总共两个额外字段）的文件名。
    如果未提供前缀，则返回所有文件。

    参数：
        directory_path (str or Path): 要查找的文件夹路径
        target_prefix (str, optional): 目标前缀，比如 "IC_xxx_yyy"，不指定则返回所有文件

    返回：
        List[str]: 匹配到的完整文件名列表
    """
    directory_path = Path(directory_path)
    matched_files = []
    
    if not directory_path.exists() or not directory_path.is_dir():
        print(f"警告: 目录 {directory_path} 不存在或不是一个目录")
        return matched_files
    
    if target_prefix:
        # 构造正则表达式：以目标前缀开头，后接两个下划线字段
        escaped_prefix = re.escape(target_prefix)
        pattern = re.compile(rf"^{escaped_prefix}_[^_]+_[^_]+$")
        
        for filename in os.listdir(directory_path):
            file_path = directory_path / filename
            if file_path.is_file() and pattern.match(filename):
                matched_files.append(filename)
    else:
        # 如果没有指定前缀，则返回所有文件
        matched_files = [filename for filename in os.listdir(directory_path) 
                        if (directory_path / filename).is_file()]
    
    return matched_files


def copy_file(source_path: Union[str, Path], target_path: Union[str, Path], overwrite: bool = True) -> bool:
    """
    复制文件从源路径到目标路径

    参数:
        source_path (str or Path): 源文件路径
        target_path (str or Path): 目标文件路径
        overwrite (bool, optional): 是否覆盖已存在的文件，默认为True

    返回:
        bool: 复制成功返回True，否则返回False
    """
    source_path = Path(source_path)
    target_path = Path(target_path)
    
    # 检查源文件是否存在
    if not source_path.exists() or not source_path.is_file():
        print(f"错误: 源文件 {source_path} 不存在或不是一个文件")
        return False
    
    # 检查目标目录是否存在，不存在则创建
    target_dir = target_path.parent
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查目标文件是否已存在
    if target_path.exists() and not overwrite:
        print(f"警告: 目标文件 {target_path} 已存在且不允许覆盖")
        return False
    
    try:
        shutil.copy2(source_path, target_path)
        return True
    except Exception as e:
        print(f"复制文件时出错: {e}")
        return False