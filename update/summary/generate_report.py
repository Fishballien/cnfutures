# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 18:32:49 2025

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
from datetime import datetime
import markdown


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[2]
sys.path.append(str(project_dir))


# %%
from update.summary.plot_recent_pnl import plot_daily_returns, plot_cumulative_returns
from update.summary.plot_pos import plot_price_and_positions
from update.summary.plot_bt_rt_comparison import load_and_generate_comparisons


# %%
target_date = '2025-04-02'
period_start_date='2025-04-01'
period_end_date='2025-04-25'
plot_start_date = '2025-01-01'


# 新的模型映射格式，包含自定义颜色
model_mapping = {
    '1_2_3_overnight': {
        'model_name': 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18',
        'test_name': 'trade_ver3_futtwap_sp1min_s240d_icim_v6',
        'prod_name': 'agg_1.2.0_3',
        'color': 'r',
        'rt_color': 'k',
    },
    '1_2_3_intraday': {
        'model_name': 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18',
        'test_name': 'trade_ver3_3_futtwap_sp1min_s240d_icim_v6',
        'prod_name': 'agg_1.2.0_4',
        'color': 'g',
        'rt_color': 'b',
    },
}

# 这些参数应根据你的实际环境设置
price_path = Path(r"D:\mnt\data1\futuretwap") / f"t1min_fq1min_dl1min.parquet"
model_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model')
model_dir_mapping = {
    'merged_model': Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model'),
    'model': model_dir,
}
rt_perdist_dir = Path(r'D:\CNIndexFutures\timeseries\prod\merge_to_pos\persist')
save_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\test_update_summary')

# 这些参数应根据你的实际环境设置
# =============================================================================
# price_path = Path(r"/mnt/data1/futuretwap") / f"t1min_fq1min_dl1min.parquet"
# model_dir = Path(r'/mnt/data1/xintang/CNIndexFutures/timeseries/factor_test/results/model')
# model_dir_mapping = {
#     'merged_model': Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model'),
#     'model': model_dir,
# }
# rt_perdist_dir = Path(r'/home/xintang/CNIndexFutures/timeseries/prod/merge_to_pos/persist')
# save_dir = Path(r'/mnt/data1/xintang/CNIndexFutures/timeseries/factor_test/results/daily_report') / target_date
# save_dir.mkdir(parents=True, exist_ok=True)
# =============================================================================


# # %% Recent Daily Return
# plot_daily_returns(model_mapping, model_dir, save_dir, start_date=period_start_date, end_date=period_end_date)
# # 已保存到：save_dir / f'daily_returns_{period_start_date}_to_{period_end_date}.png'

# # %% Cumulative Return Since {plot_start_date}
# plot_cumulative_returns(model_mapping, model_dir, save_dir, start_date=plot_start_date)
# # 已保存到：save_dir / f'cumulative_returns_from_{plot_start_date}.png'

# # %% Recent Positions
# plot_price_and_positions(
#     price_path=price_path,
#     model_mapping=model_mapping,
#     model_dir_mapping=model_dir_mapping,
#     save_dir=save_dir,
#     start_date=period_start_date,
#     end_date=period_end_date,
#     use_bar=False  # 使用线图
# )
# # 已保存到：save_dir / f"price_positions_line_{period_start_date}_to_{period_end_date}.png"

# # %% Backtest & Realtime Comparison
# # date_format = datetime.strftime(datetime.strptime(target_date, '%Y-%m-%d'), '%Y%m%d')
# load_and_generate_comparisons(model_mapping, rt_perdist_dir, model_dir, target_date, save_dir)
# # 已保存到：save_dir / f"{date}_{rt_name}_vs_{model_name}_pred_summary_comparison.png"
# # 以及 save_dir / f"{date}_{rt_name}_vs_{model_name}_pos_summary_comparison.png"

# %%
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import os
import numpy as np
from PIL import Image

def generate_daily_report_pdf(
    target_date,
    period_start_date,
    period_end_date,
    plot_start_date,
    model_mapping,
    save_dir
):
    """
    Generate a daily report PDF containing all analysis plots.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = save_dir / f"daily_analysis_report_{target_date}.pdf"

    # 预定义各图片路径
    daily_returns_path = save_dir / f'daily_returns_{period_start_date}_to_{period_end_date}.png'
    cumulative_returns_path = save_dir / f'cumulative_returns_from_{plot_start_date}.png'
    positions_path = save_dir / f"price_positions_line_{period_start_date}_to_{period_end_date}.png"

    pred_comparison_paths = []
    pos_comparison_paths = []
    for tag_name, model_info in model_mapping.items():
        model_name = model_info['model_name']
        prod_name = model_info['prod_name']
        pred_comparison_paths.append((tag_name, save_dir / f"{target_date}_{prod_name}_vs_{model_name}_pred_summary_comparison.png"))
        pos_comparison_paths.append((tag_name, save_dir / f"{target_date}_{prod_name}_vs_{model_name}_pos_summary_comparison.png"))

    # 通用参数
    figsize = (12, 16)
    title_fontsize = 16
    section_fontsize = 14

    with PdfPages(pdf_path) as pdf:
        # --------- 第一页：收益率 + 累计收益曲线 ---------
        fig, axes = create_figure_with_gridspec(rows=2, title=f'Daily Analysis Report {target_date} - Page 1', figsize=figsize)
        
        add_section_title(fig, '1. Recent Daily Returns', y=0.96, fontsize=section_fontsize)
        add_image_to_subplot(axes[0], daily_returns_path)

        add_section_title(fig, '2. Cumulative Return', y=0.49, fontsize=section_fontsize)
        add_image_to_subplot(axes[1], cumulative_returns_path)

        finalize_page(fig, pdf)

        # --------- 第二页：价格和仓位 ---------
        fig, axes = create_figure_with_gridspec(rows=1, title=f'Daily Analysis Report {target_date} - Page 2', figsize=figsize)

        add_section_title(fig, '3. Price and Positions', y=0.96, fontsize=section_fontsize)
        add_image_to_subplot(axes[0], positions_path)

        finalize_page(fig, pdf)

        # --------- 第三页：预测对比 ---------
        if pred_comparison_paths:
            fig, axes = create_figure_with_gridspec(rows=len(pred_comparison_paths), title=f'Daily Analysis Report {target_date} - Page 3', figsize=figsize)

            add_section_title(fig, '4. Prediction Comparisons', y=0.96, fontsize=section_fontsize)

            for ax, (tag_name, img_path) in zip(axes, pred_comparison_paths):
                add_image_to_subplot(ax, img_path)
                ax.set_title(tag_name, fontsize=12)

            finalize_page(fig, pdf)

        # --------- 第四页：仓位对比 ---------
        if pos_comparison_paths:
            fig, axes = create_figure_with_gridspec(rows=len(pos_comparison_paths), title=f'Daily Analysis Report {target_date} - Page 4', figsize=figsize)

            add_section_title(fig, '5. Position Comparisons', y=0.96, fontsize=section_fontsize)

            for ax, (tag_name, img_path) in zip(axes, pos_comparison_paths):
                add_image_to_subplot(ax, img_path)
                ax.set_title(tag_name, fontsize=12)

            finalize_page(fig, pdf)

        # 填充PDF元信息
        meta = pdf.infodict()
        meta['Title'] = f'Daily Analysis Report {target_date}'
        meta['Author'] = 'Automated System'
        meta['Subject'] = 'Model Analysis Report'
        meta['Keywords'] = 'Model Analysis, Trading Strategies'
        meta['CreationDate'] = datetime.now()

    print(f"PDF report generated at: {pdf_path}")
    return pdf_path

def create_figure_with_gridspec(rows, title, figsize=(12, 16)):
    """创建带网格布局的图形和子图列表"""
    plt.style.use('ggplot')
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16)
    gs = fig.add_gridspec(rows, 1, hspace=0.4, top=0.92)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(rows)]
    return fig, axes

def add_section_title(fig, text, y=0.96, fontsize=14):
    """在页面上方添加章节标题"""
    fig.text(0.1, y, text, fontsize=fontsize, weight='bold')

def add_image_to_subplot(ax, image_path):
    """向子图中添加图片"""
    ax.axis('off')
    if not os.path.exists(image_path):
        ax.text(0.5, 0.5, f"Image not found:\n{image_path.name}", ha='center', va='center', fontsize=12, color='red')
        return
    try:
        img = Image.open(image_path)
        ax.imshow(img)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error loading image:\n{image_path.name}", ha='center', va='center', fontsize=12, color='red')

def finalize_page(fig, pdf):
    """保存并关闭当前页"""
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    # Example usage with the provided parameters
    target_date = '2025-04-02'
    period_start_date = '2025-04-01'
    period_end_date = '2025-04-25'
    plot_start_date = '2025-01-01'
    
    # Model mapping as provided
    model_mapping = {
        '1_2_3_overnight': {
            'model_name': 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18',
            'test_name': 'trade_ver3_futtwap_sp1min_s240d_icim_v6',
            'prod_name': 'agg_1.2.0_3',
            'color': 'r',
            'rt_color': 'k',
        },
        '1_2_3_intraday': {
            'model_name': 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18',
            'test_name': 'trade_ver3_3_futtwap_sp1min_s240d_icim_v6',
            'prod_name': 'agg_1.2.0_4',
            'color': 'g',
            'rt_color': 'b',
        },
    }
    
    # Save directory
    save_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\test_update_summary')
    
    # Generate PDF report
    pdf_path = generate_daily_report_pdf(
        target_date,
        period_start_date,
        period_end_date,
        plot_start_date,
        model_mapping,
        save_dir
    )
    
    print(f"PDF report has been generated at: {pdf_path}")