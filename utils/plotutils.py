# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:33:59 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% all
__all__ = ["test_plot_ver1", "test_plot_ver2"]


# %% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from utils.timeutils import parse_time_string


from test_and_eval.scores import get_general_return_metrics


# %% plot ver 1
def test_plot_ver1(factor_name, gp_dict, gpd_dict, hsr_dict, midprice, params={}, plot_dir=''):
    pp = params['pp']
    sp = params['sp']
    scale_window = params.get('scale_window', '')
    scale_method = params.get('scale_method', 'minmax_scale')
    direction_choices = params['direction_choices']
    
    # 创建图形并设置大小
    fig = plt.figure(figsize=(48, 24))
    gs = gridspec.GridSpec(4, 3, height_ratios=[4, 1, 1, 1], hspace=0.2, wspace=0.15)

    # 获取 gp_dict 所有列的最大和最小 Y 值以对齐 Y 轴
    max_cum_return = max(
        [
            gp_dict[direction].cumsum().max().max()  # 对所有列求累计和的最大值
            for direction in direction_choices
        ]
    )
    min_cum_return = min(
        [
            gp_dict[direction].cumsum().min().min()  # 对所有列求累计和的最小值
            for direction in direction_choices
        ]
    )
    
    direction_choices = direction_choices.copy()

    mul = 1 if gpd_dict['all']["return"].sum() > 0 else -1
    if mul == -1:
        direction_choices[1] = 'neg'
        direction_choices[2] = 'pos'
    plot_titles = ['all', 'long_only', 'short_only']

    # 遍历方向并绘制子图
    for i, direction in enumerate(direction_choices):
        col = i  # 控制左右排列的列索引
        
        # gp 图
        ax_gp = fig.add_subplot(gs[0, col])
        gpd = gpd_dict[direction]
        
        for i_c, column in enumerate(gpd.columns):
            if column != "return":
                ax_gp.plot(gpd.index, gpd[column].cumsum()*mul, alpha=0.7, label=column, color=plt.cm.tab10(i_c))
        
        ax_gp.plot(gpd.index, gpd["return"].cumsum()*mul, color="black", linewidth=4, label="Return")
        ax_gp.set_title(f"Direction: {plot_titles[i].capitalize()}", fontsize=28, pad=20)
        
        # 设置 Y 轴范围和标签
        ylim = ([min_cum_return * 1.1, max_cum_return * 1.1] if mul == 1
                else [max_cum_return * 1.1 * mul, min_cum_return * 1.1 * mul])
        ax_gp.set_ylim(ylim)  # 动态调整 Y 轴范围
        ax_gp.set_ylabel("Cumulative Return", fontsize=22, labelpad=20)
        if i == 0:
            ax_gp.yaxis.set_label_position('right')
        else:
            ax_gp.yaxis.label.set_visible(False)
        ax_gp.grid(True, linestyle="--", linewidth=0.8)
        ax_gp.tick_params(axis="y", labelsize=22)
        
        # 使用 AutoDateLocator 和 AutoDateFormatter 动态调整刻度
        locator = mdates.AutoDateLocator()  # 自动选择合适的刻度单位
        formatter = mdates.AutoDateFormatter(locator)  # 自动格式化刻度
        ax_gp.xaxis.set_major_locator(locator)  # 应用主刻度定位器
        ax_gp.xaxis.set_major_formatter(formatter)  # 应用主刻度格式化器
        
        # 调整主刻度字体和样式
        ax_gp.tick_params(axis="x", which="major", labelsize=16, pad=15)  # 主刻度字体更大
        
        if i == 2:  # 只显示一次 legend
            ax_gp.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=20, frameon=True, shadow=True)
        else:
            ax_gp.legend().remove()
            
        # midprice 图（新增的第四幅子图，位于第二行）
        ax_midprice = fig.add_subplot(gs[1, col], sharex=ax_gp)
        mid_log_return = np.log(midprice).diff().resample('1d').sum()  # 转为对数收益率
        mid_cum_log_return = mid_log_return.cumsum()  # 累计对数收益率

        # 绘制累计对数收益
        for i_c, column in enumerate(mid_cum_log_return.columns):
            ax_midprice.plot(mid_cum_log_return.index, mid_cum_log_return[column], 
                             alpha=0.7, label=column, color=plt.cm.tab10(i_c))
        if i == 2:
            ax_midprice.set_ylabel("Index CumRtn", fontsize=22, labelpad=20)
            ax_midprice.yaxis.set_label_position('right')
        ax_midprice.grid(True, linestyle="--", linewidth=0.8)
        ax_midprice.tick_params(axis="y", labelsize=22)
        
        plt.setp(ax_midprice.get_xticklabels(), visible=False)
            
        # hsr 图
        ax_hsr = fig.add_subplot(gs[2, col], sharex=ax_gp)
        hsrd = hsr_dict[direction]
        ax_hsr.bar(hsrd.index, hsrd["avg"], linewidth=2, label="HSR Avg")
        
        # 设置 HSR 图的 Y 轴范围和标签
        ax_hsr.set_ylim([0, 0.25])
        if i == 2:
            ax_hsr.set_ylabel("HSR Avg", fontsize=22, labelpad=20)
            ax_hsr.yaxis.set_label_position('right')
        ax_hsr.grid(True, linestyle="--", linewidth=0.8)
        ax_hsr.tick_params(axis="y", labelsize=22)
        
        # 移除 HSR 图的 X 轴刻度
        plt.setp(ax_hsr.get_xticklabels(), visible=False)
        
        # 每分钟收益的平均值图
        ax_minute = fig.add_subplot(gs[3, col])
        gp = gp_dict[direction]
        
        # 按时间聚合，提取每分钟的时间部分并求均值
        gp.index = pd.to_datetime(gp.index)  # 确保索引为 datetime 类型
        gp["minute"] = gp.index.time  # 提取分钟部分
        avg_per_minute = gp.groupby("minute")["return"].mean()  # 按分钟分组求平均收益
        
        # 将 `datetime.time` 转换为数值索引（例如分钟序号）
        minute_labels = [t.hour * 60 + t.minute for t in avg_per_minute.index]
        
        # 绘制条形图
        ax_minute.bar(minute_labels, avg_per_minute*mul, color=plt.cm.Set2(3), label="Avg Return")
        
        # 设置 X 轴为原始时间格式
        ax_minute.set_xticks(minute_labels[::30])  # 每隔 30 分钟显示一次刻度
        ax_minute.set_xticklabels([f"{t // 60:02}:{t % 60:02}" for t in minute_labels[::30]], rotation=45, fontsize=16)
        
        # 设置 Y 轴
        if i == 2:
            ax_minute.set_ylabel("Avg Minute Return", fontsize=22, labelpad=20)
            ax_minute.yaxis.set_label_position('right')
        ax_minute.grid(True, linestyle="--", linewidth=0.8)
        ax_minute.tick_params(axis="y", labelsize=22)

    # 设置全局标题
    fig.suptitle(f"{factor_name}  {scale_method}\nsp: {sp}  pp: {pp}  scale: {scale_window}  direction: {mul}", fontsize=40, y=0.97)

    # 生成文件名
    plot_file_path = plot_dir / f'{factor_name}.jpg'

    # 保存图表到 sample_data_dir
    plt.savefig(plot_file_path, bbox_inches='tight', dpi=300)

    # 显示图表
    plt.show()
    
    
# %% plot ver 2
# =============================================================================
# def test_plot_ver2(factor_name, test_name, gp_dict, gpd_dict, hsr_dict, midprice, factor, factor_pos, params={}, plot_dir=''):
#     pp = params['pp']
#     sp = params['sp']
#     scale_window = params.get('scale_window', '')
#     scale_method = params.get('scale_method', '')
#     direction_choices = params.get('direction_choices', '')
#     fee = params.get('fee', 4e-4)
#     
#     # 创建图形并设置大小
#     fig = plt.figure(figsize=(48, 36))  # 增大图形尺寸
#     gs = gridspec.GridSpec(5, 3, height_ratios=[4, 1, 1, 1, 2], hspace=0.2, wspace=0.15)  # 调整间距为 4 行布局
# 
#     # 获取 gpd_dict 所有列的最大和最小 Y 值以对齐 Y 轴
#     max_cum_return = max(
#         [
#             gpd_dict[direction].cumsum().max().max()  # 对所有列求累计和的最大值
#             for direction in direction_choices
#         ]
#     )
#     min_cum_return = min(
#         [
#             gpd_dict[direction].cumsum().min().min()  # 对所有列求累计和的最小值
#             for direction in direction_choices
#         ]
#     )
# 
#     direction_choices = direction_choices.copy()
# 
#     mul = 1 if gpd_dict['all']["return"].sum() > 0 else -1
#     if mul == -1:
#         direction_choices[1] = 'neg'
#         direction_choices[2] = 'pos'
#     plot_titles = ['all', 'long_only', 'short_only']
# 
#     # 遍历方向并绘制子图
#     for i, direction in enumerate(direction_choices):
#         col = i  # 控制左右排列的列索引
#         
#         gpd = gpd_dict[direction].fillna(0)
#         hsrd = hsr_dict[direction].fillna(0)
#         
#         profit_per_trade = gpd["return"].sum() * mul / hsrd["avg"].sum()
#         net = (gpd["return"] * mul - hsrd["avg"] * fee).fillna(0)
#         dwr = (net > 0).sum() / (net != 0).sum()
#         metrics = get_general_return_metrics(net.values)
#         
#         # gp 图
#         ax_gp = fig.add_subplot(gs[0, col])
#         
#         for i_c, column in enumerate(gpd.columns):
#             if column != "return":
#                 ax_gp.plot(gpd.index, gpd[column].cumsum()*mul, alpha=0.7, label=column, color=plt.cm.tab10(i_c))
#         
#         ax_gp.plot(gpd.index, gpd["return"].cumsum()*mul, color="black", linewidth=4, label="Return")
#         plot_title = f"Direction - {plot_titles[i].capitalize()}\nPPT: {profit_per_trade:.4%}    NSP: {metrics['sharpe_ratio']:.2f}    DWR: {dwr:.2%}"
#         ax_gp.set_title(plot_title, fontsize=28, pad=20)
#         
#         # 设置 Y 轴范围和标签
#         ylim = ([min_cum_return * 1.1, max_cum_return * 1.1] if mul == 1
#                 else [max_cum_return * 1.1 * mul, min_cum_return * 1.1 * mul])
#         ax_gp.set_ylim(ylim)  # 动态调整 Y 轴范围
#         ax_gp.set_ylabel("Cumulative Return", fontsize=22, labelpad=20)
#         if i == 0:
#             ax_gp.yaxis.set_label_position('right')
#         else:
#             ax_gp.yaxis.label.set_visible(False)
#         ax_gp.grid(True, linestyle="--", linewidth=0.8)
#         ax_gp.tick_params(axis="y", labelsize=22)
# 
#         # 使用 AutoDateLocator 和 AutoDateFormatter 动态调整刻度
#         locator = mdates.AutoDateLocator()  # 自动选择合适的刻度单位
#         formatter = mdates.AutoDateFormatter(locator)  # 自动格式化刻度
#         ax_gp.xaxis.set_major_locator(locator)  # 应用主刻度定位器
#         ax_gp.xaxis.set_major_formatter(formatter)  # 应用主刻度格式化器
#         
#         # 调整主刻度字体和样式
#         ax_gp.tick_params(axis="x", which="major", labelsize=16, pad=15)  # 主刻度字体更大
#         
#         if i == 2:  # 只显示一次 legend
#             ax_gp.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=20, frameon=True, shadow=True)
#         else:
#             ax_gp.legend().remove()
#             
#         # midprice 图（新增的第四幅子图，位于第二行）
#         ax_midprice = fig.add_subplot(gs[1, col], sharex=ax_gp)
#         mid_log_return = np.log(midprice).diff().resample('1d').sum()  # 转为对数收益率
#         mid_cum_log_return = mid_log_return.cumsum()  # 累计对数收益率
# 
#         # 绘制累计对数收益
#         for i_c, column in enumerate(mid_cum_log_return.columns):
#             ax_midprice.plot(mid_cum_log_return.index, mid_cum_log_return[column], 
#                              alpha=0.7, label=column, color=plt.cm.tab10(i_c))
#         if i == 2:
#             ax_midprice.set_ylabel("Index CumRtn", fontsize=22, labelpad=20)
#             ax_midprice.yaxis.set_label_position('right')
#         ax_midprice.grid(True, linestyle="--", linewidth=0.8)
#         ax_midprice.tick_params(axis="y", labelsize=22)
#         
#         plt.setp(ax_midprice.get_xticklabels(), visible=False)
# 
#             
#         # hsr 图
#         ax_hsr = fig.add_subplot(gs[2, col], sharex=ax_gp)
#         ax_hsr.bar(hsrd.index, hsrd["avg"], linewidth=2, label="HSR Avg")
#         
#         # 设置 HSR 图的 Y 轴范围和标签
#         # ax_hsr.set_ylim([0, 0.25])
#         if i == 2:
#             ax_hsr.set_ylabel("HSR Avg", fontsize=22, labelpad=20)
#             ax_hsr.yaxis.set_label_position('right')
#         ax_hsr.grid(True, linestyle="--", linewidth=0.8)
#         ax_hsr.tick_params(axis="y", labelsize=22)
#         
#         # 移除 HSR 图的 X 轴刻度
#         plt.setp(ax_hsr.get_xticklabels(), visible=False)
# 
#         # 每分钟收益的平均值图
#         ax_minute = fig.add_subplot(gs[3, col])
#         gp = gp_dict[direction]
#         
#         # 按时间聚合，提取每分钟的时间部分并求均值
#         gp.index = pd.to_datetime(gp.index)  # 确保索引为 datetime 类型
#         gp["minute"] = gp.index.time  # 提取分钟部分
#         avg_per_minute = gp.groupby("minute")["return"].mean()  # 按分钟分组求平均收益
#         
#         # 将 `datetime.time` 转换为数值索引（例如分钟序号）
#         minute_labels = [t.hour * 60 + t.minute for t in avg_per_minute.index]
#         
#         # 绘制条形图
#         ax_minute.bar(minute_labels, avg_per_minute*mul, color=plt.cm.Set2(3), label="Avg Return")
#         
#         # 设置 X 轴为原始时间格式
#         ax_minute.set_xticks(minute_labels[::30])  # 每隔 30 分钟显示一次刻度
#         ax_minute.set_xticklabels([f"{t // 60:02}:{t % 60:02}" for t in minute_labels[::30]], rotation=45, fontsize=16)
#         
#         # 设置 Y 轴
#         if i == 2:
#             ax_minute.set_ylabel("Avg Minute Return", fontsize=22, labelpad=20)
#             ax_minute.yaxis.set_label_position('right')
#         ax_minute.grid(True, linestyle="--", linewidth=0.8)
#         ax_minute.tick_params(axis="y", labelsize=22)
#     
#     # 针对 "all" 方向进行计算
#     gpd_all = gpd_dict['all'].fillna(0)
#     hsrd_all = hsr_dict['all'].fillna(0)
#     net = (gpd_all["return"] * mul - hsrd_all["avg"] * fee).fillna(0)
#     annual_net = net.resample('YE').sum()
# 
#     table_data = []
#     for year in annual_net.index:
#         try:
#             metrics = get_general_return_metrics(net[net.index.year == year.year].values)
#         except:
#             print(year)
#             print(factor_name)
#             print(net[net.index.year == year.year])
#         hsr_avg = np.mean(hsrd_all[hsrd_all.index.year == year.year]['avg'])
#         table_data.append([
#             year.strftime('%Y'),
#             f"{annual_net.loc[year] * 100:.2f}%",  # Annual Return 百分比格式
#             f"{metrics['max_dd'] * 100:.2f}%",   # Max DD 百分比格式
#             f"{metrics['sharpe_ratio']:.2f}",   # Sharpe Ratio 保留两位小数
#             f"{metrics['calmar_ratio']:.2f}",   # Calmar Ratio 保留两位小数
#             f"{hsr_avg:.2f}"    # Calmar Ratio 保留两位小数
#         ])
# 
#     # 添加表格
#     ax_table = fig.add_subplot(gs[4, 0])  # 表格占两行
#     columns = ['Year', 'Ann.Rtn', 'Max DD', 'Sharpe', 'Calmar', 'Hsr']
#     table = ax_table.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')  # 调整为居中
#     table.auto_set_font_size(False)
#     table.set_fontsize(25)  # 增大字体
#     # table.auto_set_column_width(col=list(range(len(columns))))
#     for key, cell in table.get_celld().items():
#         cell.set_width(1 / len(columns))  # 调整列宽占满第一列
#     table.scale(1, 2.5)  # 调整行间距
# 
#     ax_table.axis('off')  # 隐藏坐标轴
#     
#     # 动量or反转
#     if midprice.columns.equals(factor.columns):
#         his_rtn = midprice.pct_change(30, fill_method=None)
#         corr_cont = his_rtn.corrwith(factor, drop=True)
#         corr_dist = his_rtn.corrwith(factor_pos, drop=True)
#         fut_names = corr_cont.index
#         table_data1 = [[fut, f"{corr_cont[fut] * 100:.2f}%", f"{corr_dist[fut] * 100:.2f}%"] for fut in fut_names]
#         
#         ax_table1 = fig.add_subplot(gs[4, 1])  # 表格占两行
#         columns = ['Fut', 'CorrCont', 'CorrDist']
#         table = ax_table1.table(cellText=table_data1, colLabels=columns, loc='center', cellLoc='center')  # 调整为居中
#         table.auto_set_font_size(False)
#         table.set_fontsize(25)  # 增大字体
#         # table.auto_set_column_width(col=list(range(len(columns))))
#         for key, cell in table.get_celld().items():
#             cell.set_width(1 / len(columns))  # 调整列宽占满第一列
#         table.scale(1, 2.5)  # 调整行间距
#     
#         ax_table1.axis('off')  # 隐藏坐标轴
# 
#     # 设置全局标题
#     fig.suptitle(f"{factor_name}  {scale_method}  {test_name}\nsp: {sp}  pp: {pp}  scale: {scale_window}  direction: {mul}  fee: {fee}",
#                  fontsize=40, y=0.97)
# 
#     # 生成文件名
#     plot_file_path = plot_dir / f'{factor_name}.jpg'
# 
#     # 保存图表到 sample_data_dir
#     plt.savefig(plot_file_path, bbox_inches='tight', dpi=300)
# 
#     # 显示图表
#     plt.show()
#     plt.close()
# =============================================================================

def test_plot_ver2(factor_name, test_name, gp_dict, gpd_dict, hsr_dict, midprice, factor, factor_pos, return_category_by_interval=None, params={}, plot_dir=''):
    pp = params['pp']
    sp = params['sp']
    scale_window = params.get('scale_window', '')
    scale_method = params.get('scale_method', '')
    direction_choices = params.get('direction_choices', '')
    fee = params.get('fee', 4e-4)
    
    # 创建图形并设置大小
    fig = plt.figure(figsize=(48, 36))  # 增大图形尺寸
    gs = gridspec.GridSpec(5, 3, height_ratios=[4, 1, 1, 1, 2], hspace=0.2, wspace=0.15)  # 调整间距为 5 行布局

    # 获取 gpd_dict 所有列的最大和最小 Y 值以对齐 Y 轴
    max_cum_return = max(
        [
            gpd_dict[direction].cumsum().max().max()  # 对所有列求累计和的最大值
            for direction in direction_choices
        ]
    )
    min_cum_return = min(
        [
            gpd_dict[direction].cumsum().min().min()  # 对所有列求累计和的最小值
            for direction in direction_choices
        ]
    )

    direction_choices = direction_choices.copy()

    mul = 1 if gpd_dict['all']["return"].sum() > 0 else -1
    if mul == -1:
        direction_choices[1] = 'neg'
        direction_choices[2] = 'pos'
    plot_titles = ['all', 'long_only', 'short_only']

    # 遍历方向并绘制子图
    for i, direction in enumerate(direction_choices):
        col = i  # 控制左右排列的列索引
        
        gpd = gpd_dict[direction].fillna(0)
        hsrd = hsr_dict[direction].fillna(0)
        
        profit_per_trade = gpd["return"].sum() * mul / hsrd["avg"].sum()
        net = (gpd["return"] * mul - hsrd["avg"] * fee).fillna(0)
        dwr = (net > 0).sum() / (net != 0).sum()
        metrics = get_general_return_metrics(net.values)
        
        # gp 图
        ax_gp = fig.add_subplot(gs[0, col])
        
        for i_c, column in enumerate(gpd.columns):
            if column != "return":
                ax_gp.plot(gpd.index, gpd[column].cumsum()*mul, alpha=0.7, label=column, color=plt.cm.tab10(i_c))
        
        ax_gp.plot(gpd.index, gpd["return"].cumsum()*mul, color="black", linewidth=4, label="Return")
        plot_title = f"Direction - {plot_titles[i].capitalize()}\nPPT: {profit_per_trade:.4%}    NSP: {metrics['sharpe_ratio']:.2f}    DWR: {dwr:.2%}"
        ax_gp.set_title(plot_title, fontsize=28, pad=20)
        
        # 设置 Y 轴范围和标签
        ylim = ([min_cum_return * 1.1, max_cum_return * 1.1] if mul == 1
                else [max_cum_return * 1.1 * mul, min_cum_return * 1.1 * mul])
        ax_gp.set_ylim(ylim)  # 动态调整 Y 轴范围
        ax_gp.set_ylabel("Cumulative Return", fontsize=22, labelpad=20)
        if i == 0:
            ax_gp.yaxis.set_label_position('right')
        else:
            ax_gp.yaxis.label.set_visible(False)
        ax_gp.grid(True, linestyle="--", linewidth=0.8)
        ax_gp.tick_params(axis="y", labelsize=22)

        # 使用 AutoDateLocator 和 AutoDateFormatter 动态调整刻度
        locator = mdates.AutoDateLocator()  # 自动选择合适的刻度单位
        formatter = mdates.AutoDateFormatter(locator)  # 自动格式化刻度
        ax_gp.xaxis.set_major_locator(locator)  # 应用主刻度定位器
        ax_gp.xaxis.set_major_formatter(formatter)  # 应用主刻度格式化器
        
        # 调整主刻度字体和样式
        ax_gp.tick_params(axis="x", which="major", labelsize=16, pad=15)  # 主刻度字体更大
        
        if i == 2:  # 只显示一次 legend
            ax_gp.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=20, frameon=True, shadow=True)
        else:
            ax_gp.legend().remove()
            
        # midprice 图（新增的第四幅子图，位于第二行）
        ax_midprice = fig.add_subplot(gs[1, col], sharex=ax_gp)
        mid_log_return = np.log(midprice).diff().resample('1d').sum()  # 转为对数收益率
        mid_cum_log_return = mid_log_return.cumsum()  # 累计对数收益率

        # 绘制累计对数收益
        for i_c, column in enumerate(mid_cum_log_return.columns):
            ax_midprice.plot(mid_cum_log_return.index, mid_cum_log_return[column], 
                             alpha=0.7, label=column, color=plt.cm.tab10(i_c))
        if i == 2:
            ax_midprice.set_ylabel("Index CumRtn", fontsize=22, labelpad=20)
            ax_midprice.yaxis.set_label_position('right')
        ax_midprice.grid(True, linestyle="--", linewidth=0.8)
        ax_midprice.tick_params(axis="y", labelsize=22)
        
        plt.setp(ax_midprice.get_xticklabels(), visible=False)

            
        # hsr 图
        ax_hsr = fig.add_subplot(gs[2, col], sharex=ax_gp)
        ax_hsr.bar(hsrd.index, hsrd["avg"], linewidth=2, label="HSR Avg")
        
        # 设置 HSR 图的 Y 轴范围和标签
        # ax_hsr.set_ylim([0, 0.25])
        if i == 2:
            ax_hsr.set_ylabel("HSR Avg", fontsize=22, labelpad=20)
            ax_hsr.yaxis.set_label_position('right')
        ax_hsr.grid(True, linestyle="--", linewidth=0.8)
        ax_hsr.tick_params(axis="y", labelsize=22)
        
        # 移除 HSR 图的 X 轴刻度
        plt.setp(ax_hsr.get_xticklabels(), visible=False)

        # 每分钟收益的平均值图
        ax_minute = fig.add_subplot(gs[3, col])
        gp = gp_dict[direction]
        
        # 按时间聚合，提取每分钟的时间部分并求均值
        gp.index = pd.to_datetime(gp.index)  # 确保索引为 datetime 类型
        gp["minute"] = gp.index.time  # 提取分钟部分
        avg_per_minute = gp.groupby("minute")["return"].mean()  # 按分钟分组求平均收益
        
        # 将 `datetime.time` 转换为数值索引（例如分钟序号）
        minute_labels = [t.hour * 60 + t.minute for t in avg_per_minute.index]
        
        # 绘制条形图
        ax_minute.bar(minute_labels, avg_per_minute*mul, color=plt.cm.Set2(3), label="Avg Return")
        
        # 设置 X 轴为原始时间格式
        ax_minute.set_xticks(minute_labels[::30])  # 每隔 30 分钟显示一次刻度
        ax_minute.set_xticklabels([f"{t // 60:02}:{t % 60:02}" for t in minute_labels[::30]], rotation=45, fontsize=16)
        
        # 设置 Y 轴
        if i == 2:
            ax_minute.set_ylabel("Avg Minute Return", fontsize=22, labelpad=20)
            ax_minute.yaxis.set_label_position('right')
        ax_minute.grid(True, linestyle="--", linewidth=0.8)
        ax_minute.tick_params(axis="y", labelsize=22)
    
    # 针对 "all" 方向进行计算
    gpd_all = gpd_dict['all'].fillna(0)
    hsrd_all = hsr_dict['all'].fillna(0)
    net = (gpd_all["return"] * mul - hsrd_all["avg"] * fee).fillna(0)
    annual_net = net.resample('Y').sum()

    table_data = []
    for year in annual_net.index:
        try:
            metrics = get_general_return_metrics(net[net.index.year == year.year].values)
        except:
            print(year)
            print(factor_name)
            print(net[net.index.year == year.year])
        hsr_avg = np.mean(hsrd_all[hsrd_all.index.year == year.year]['avg'])
        table_data.append([
            year.strftime('%Y'),
            f"{annual_net.loc[year] * 100:.2f}%",  # Annual Return 百分比格式
            f"{metrics['max_dd'] * 100:.2f}%",   # Max DD 百分比格式
            f"{metrics['sharpe_ratio']:.2f}",   # Sharpe Ratio 保留两位小数
            f"{metrics['calmar_ratio']:.2f}",   # Calmar Ratio 保留两位小数
            f"{hsr_avg:.2f}"    # Calmar Ratio 保留两位小数
        ])

    # 添加表格
    ax_table = fig.add_subplot(gs[4, 0])  # 表格占一列
    columns = ['Year', 'Ann.Rtn', 'Max DD', 'Sharpe', 'Calmar', 'Hsr']
    table = ax_table.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')  # 调整为居中
    table.auto_set_font_size(False)
    table.set_fontsize(25)  # 增大字体
    # table.auto_set_column_width(col=list(range(len(columns))))
    for key, cell in table.get_celld().items():
        cell.set_width(1 / len(columns))  # 调整列宽占满第一列
    table.scale(1, 2.5)  # 调整行间距

    ax_table.axis('off')  # 隐藏坐标轴
    
    # 动量or反转
    if midprice.columns.equals(factor.columns):
        his_rtn = midprice.pct_change(30, fill_method=None)
        corr_cont = his_rtn.corrwith(factor, drop=True)
        corr_dist = his_rtn.corrwith(factor_pos, drop=True)
        fut_names = corr_cont.index
        table_data1 = [[fut, f"{corr_cont[fut] * 100:.2f}%", f"{corr_dist[fut] * 100:.2f}%"] for fut in fut_names]
        
        ax_table1 = fig.add_subplot(gs[4, 1])  # 表格占一列
        columns = ['Fut', 'CorrCont', 'CorrDist']
        table = ax_table1.table(cellText=table_data1, colLabels=columns, loc='center', cellLoc='center')  # 调整为居中
        table.auto_set_font_size(False)
        table.set_fontsize(25)  # 增大字体
        # table.auto_set_column_width(col=list(range(len(columns))))
        for key, cell in table.get_celld().items():
            cell.set_width(1 / len(columns))  # 调整列宽占满第一列
        table.scale(1, 2.5)  # 调整行间距
    
        ax_table1.axis('off')  # 隐藏坐标轴
    
    # 添加 return_category_by_interval 图表
    if return_category_by_interval is not None:
        ax_category = fig.add_subplot(gs[4, 2])  # 在第5行第3列位置
        
        # 初始化储存不同时间区间的总收益
        categories = ['intraday', 'overnight', 'weekend', 'holiday']
        
        # 计算每个类别的总收益(sum)
        category_sums = {
            'long': {cat: 0 for cat in categories},
            'short': {cat: 0 for cat in categories}
        }
        
        # 遍历所有标的计算总和
        for instrument in return_category_by_interval:
            for direction in ['long', 'short']:
                for category in categories:
                    # 如果该分类存在则累加sum值
                    if category in return_category_by_interval[instrument][direction]:
                        category_sums[direction][category] += return_category_by_interval[instrument][direction][category]['sum']
        
        # 设置条形图的位置
        x = np.arange(len(categories))
        width = 0.35  # 条形宽度
        
        # 绘制两个方向的收益
        bars1 = ax_category.bar(x - width/2, [category_sums['long'][cat] for cat in categories], 
                                width, label='Long', color='red', alpha=0.7)
        bars2 = ax_category.bar(x + width/2, [category_sums['short'][cat] for cat in categories], 
                                width, label='Short', color='green', alpha=0.7)
        
        # 添加标签和标题
        ax_category.set_ylabel('Total Return Sum', fontsize=22, labelpad=20)
        ax_category.set_title('Return by Time Interval', fontsize=20)
        ax_category.set_xticks(x)
        ax_category.set_xticklabels(categories, fontsize=20)
        ax_category.tick_params(axis="y", labelsize=20)
        ax_category.legend(fontsize=20)
        ax_category.grid(True, linestyle="--", linewidth=0.8, axis='y')
        
        # 添加数值标签
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                value = height
                if abs(height) < 0.01:  # 小数值显示科学计数法
                    label = f"{value:.2e}"
                else:
                    label = f"{value:.4f}"
                # 将标签放在底部
                y_pos = -0.03  # 标签位置微调
                ax_category.annotate(label,
                                    xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                                    textcoords="offset points",
                                    xytext=(0, -20),
                                    ha='center', va='top',
                                    fontsize=16, rotation=0)
        
        add_labels(bars1)
        add_labels(bars2)

    # 设置全局标题
    fig.suptitle(f"{factor_name}  {scale_method}  {test_name}\nsp: {sp}  pp: {pp}  scale: {scale_window}  direction: {mul}  fee: {fee}",
                 fontsize=40, y=0.97)

    # 生成文件名
    plot_file_path = plot_dir / f'{factor_name}.jpg'

    # 保存图表到 sample_data_dir
    plt.savefig(plot_file_path, bbox_inches='tight', dpi=300)

    # 显示图表
    plt.close()