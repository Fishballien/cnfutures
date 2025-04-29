# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 16:44:04 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 配置信息
smtp_server = "smtp.qiye.aliyun.com"
smtp_port = 465
sender_email = "zhengxintang@evolutionlabs.com.cn"
sender_password = "JbyrNogNHJNTmbIR"

# 创建邮件
msg = MIMEMultipart()
msg["From"] = sender_email
msg["To"] = "2332746075@qq.com"
msg["Subject"] = "Daily Model Performance Report - 2025-04-29 1234"
body = "Please find attached the daily model performance report for 2025-04-29.\n\nRegards,\nAutomated Reporting System"
msg.attach(MIMEText(body, "plain"))

# 发送邮件
with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, ["2332746075@qq.com"], msg.as_string())
