# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:57:09 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
def check_if_valid_v0(ts_test_res):
    adf_test_res = ts_test_res['adf']
    ratio_diff = ts_test_res['ratio_diff']
    adf_valid = all([adf_test_res[fut]["Is Stationary"] for fut in adf_test_res])
    pos_ratio_valid = all(ratio_diff < 0.1)
    return adf_valid and pos_ratio_valid