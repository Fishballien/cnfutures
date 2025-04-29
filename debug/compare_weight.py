# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 09:53:28 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import pandas as pd


# %%
new_path = 'D:/mnt/data1/stockweights/new/000905.parquet'
old_path = 'D:/mnt/data1/stockweights/000905.parquet'


# %%
new_data = pd.read_parquet(new_path)
old_data = pd.read_parquet(old_path)
