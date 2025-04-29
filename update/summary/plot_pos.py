# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 12:55:57 2025

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
import matplotlib.pyplot as plt
import numpy as np


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[2]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config


# %%
def plot_price_and_positions(
    price_path, 
    model_mapping, 
    model_dir_mapping, 
    save_dir,
    start_date, 
    end_date,
    instruments=['IC', 'IM'],
    use_bar=False  # 新增参数，决定是使用条形图还是线图来表示仓位
):
    """
    绘制价格和模型仓位数据图表
    
    参数:
    price_path (str/Path): 价格数据文件的路径
    model_mapping (dict): 模型映射字典，键为标签名，值为带有model_name、test_name、prod_name和color的配置字典
    model_dir_mapping (dict): 模型类型到目录的映射
    save_dir (str/Path): 保存图表的目录
    start_date (str/datetime): 开始日期
    end_date (str/datetime): 结束日期
    instruments (list): 要绘制的品种列表，默认为['IC', 'IM']
    use_bar (bool): 是否使用条形图表示仓位，默认为False（使用线图）
    """
    # 确保save_dir是Path对象
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换日期为Timestamp对象(如果它们不是)
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    # 加载价格数据
    price_data = pd.read_parquet(price_path)
    
    # 加载所有模型仓位到字典
    model_positions = {}
    model_colors = {}  # 存储每个模型的颜色
    
    for tag_name, model_config in model_mapping.items():
        # 获取模型名称、测试名称和颜色
        model_name = model_config.get('model_name')
        test_name = model_config.get('test_name')
        model_color = model_config.get('color')  # 获取颜色
        
        # 存储模型颜色
        model_colors[tag_name] = model_color
        
        model_pos_path = model_dir_mapping['model'] / model_name / 'test' / test_name / 'data' / f'pos_predict_{model_name}.parquet'
        
        model_pos = pd.read_parquet(model_pos_path)
        model_positions[tag_name] = model_pos
    
    # 筛选仅包含所需品种的列
    price_data = price_data[instruments]
    
    # 筛选数据到所需日期范围
    price_data = price_data.loc[(price_data.index >= start_date) & (price_data.index <= end_date)]
    
    # 筛选模型仓位到相同日期范围
    for model_name in model_positions:
        model_positions[model_name] = model_positions[model_name].loc[
            (model_positions[model_name].index >= start_date) & 
            (model_positions[model_name].index <= end_date)
        ]
    
    # 创建图表
    fig, axs = plt.subplots(len(instruments), 1, figsize=(15, 6 * len(instruments)), sharex=True)
    fig.suptitle(f"Price and Position Data: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}", 
                 fontsize=16, y=0.95)
    
    # 确保axs是一个列表，即使instruments只有一个元素
    if len(instruments) == 1:
        axs = [axs]
    
    for i, instrument in enumerate(instruments):
        # 主坐标轴(价格)
        ax1 = axs[i]
        
        # 绘制价格数据
        price_series = price_data[instrument].dropna()
        if not price_series.empty:
            # 使用arange作为x轴而不是datetime
            x = np.arange(len(price_series))
            x_labels = price_series.index.strftime('%Y-%m-%d %H:%M')  # 转换为时间标签以供参考
            
            # 使用数值x值绘图
            ax1.plot(x, price_series.values, color='black', linewidth=1.5, label=f'{instrument} Price')
            ax1.set_ylabel(f'{instrument} Price', color='black', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # 创建次坐标轴(仓位数据)
            ax2 = ax1.twinx()
            
            # 为每个模型绘制仓位
            for model_name, model_pos in model_positions.items():
                # 对该期间和品种筛选仓位数据
                pos_period = model_pos.loc[
                    (model_pos.index >= start_date) & 
                    (model_pos.index <= end_date), 
                    instrument
                ].dropna()
                
                if not pos_period.empty:
                    # 需要将仓位数据与价格索引对齐以便绘图
                    aligned_pos = pos_period.reindex(price_series.index, method='ffill')
                    # 仅使用非NaN值
                    valid_indices = ~aligned_pos.isna()
                    if valid_indices.any():
                        # 获取有效仓位对应的x值
                        valid_x = x[valid_indices.values]
                        valid_pos = aligned_pos[valid_indices].values
                        
                        # 使用模型配置中的颜色
                        color = model_colors[model_name]
                        
                        if use_bar:
                            # 使用条形图表示仓位
                            ax2.bar(valid_x, valid_pos, color=color, alpha=0.6, 
                                   label=f'{model_name}', width=0.8)
                        else:
                            # 使用线图表示仓位
                            ax2.plot(valid_x, valid_pos, color=color, 
                                   linewidth=1.5, linestyle='-', 
                                   label=f'{model_name}')
            
            # 设置次坐标轴标签
            ax2.set_ylabel('Position', fontsize=12)
            ax2.set_ylim(-1.2, 1.2)  # 假设仓位值在-1和1之间
            ax2.tick_params(axis='y')
            
            # 在仓位0处添加水平虚线
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            
            # 找出每个9:30 AM(市场开盘)的垂直线
            nine_thirty_indices = [idx for idx, t in enumerate(price_series.index) if t.strftime('%H:%M') == '09:30']
            for idx in nine_thirty_indices:
                ax1.axvline(x=idx, color='gray', linestyle='--', linewidth=1, alpha=0.7)
                
            # 设置子图标题
            ax1.set_title(f"{instrument} Price and Positions", fontsize=14)
            
            # 添加图例
            # 左图的价格图例
            ax1.legend(loc='upper left')
            
            # 右图的仓位图例
            handles, labels = ax2.get_legend_handles_labels()
            ax2.legend(handles, labels, loc='upper right')
    
    # 为底部子图设置x刻度位置和标签
    # 显示10个均匀分布的刻度及对应的时间标签
    if len(x) > 0:
        tick_positions = np.linspace(0, len(x)-1, num=10, dtype=int)
        axs[-1].set_xticks(tick_positions)
        axs[-1].set_xticklabels([x_labels[i] for i in tick_positions], rotation=45)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图表
    plot_type = "bar" if use_bar else "line"
    save_path = save_dir / f"price_positions_{plot_type}_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Visualization completed and saved to {save_path}")


# %% 使用示例
if __name__ == "__main__":

    # 这些参数应根据你的实际环境设置
    price_path = Path(r"D:\mnt\data1\futuretwap") / f"t1min_fq1min_dl1min.parquet"
    
    # 新的模型映射格式，包含自定义颜色
    model_mapping = {
        '1_2_3_overnight': {
            'model_name': 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18',
            'test_name': 'trade_ver3_futtwap_sp1min_s240d_icim_v6',
            'prod_name': 'agg_1.2.0_3',
            'color': 'r'
        },
        '1_2_3_intraday': {
            'model_name': 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18',
            'test_name': 'trade_ver3_3_futtwap_sp1min_s240d_icim_v6',
            'prod_name': 'agg_1.2.0_4',
            'color': 'g'
        },
    }
    
    model_dir_mapping = {
        'merged_model': Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model'),
        'model': Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model')
    }
    save_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\test_update_summary')
    
    # 使用线图表示仓位
    plot_price_and_positions(
        price_path=price_path,
        model_mapping=model_mapping,
        model_dir_mapping=model_dir_mapping,
        save_dir=save_dir,
        start_date='2025-03-17',
        end_date='2025-04-18',
        use_bar=False  # 使用线图
    )
    
    # 使用条形图表示仓位
    plot_price_and_positions(
        price_path=price_path,
        model_mapping=model_mapping,
        model_dir_mapping=model_dir_mapping,
        save_dir=save_dir,
        start_date='2025-03-17',
        end_date='2025-04-18',
        use_bar=True  # 使用条形图
    )