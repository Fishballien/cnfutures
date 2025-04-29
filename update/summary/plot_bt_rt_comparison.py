# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:02:09 2025

@author: Xintang Zheng

预测与仓位对比工具
"""
# %%
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from datetime import datetime
import yaml
import argparse


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[2]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config


# %%
def create_comparison_plots(bt_pred: pd.DataFrame, rt_pred: pd.DataFrame, 
                           rt_name: str, model_name: str, save_dir: Path, date: str) -> None:
    """
    Create and save comparison plots for each common column between bt_pred and rt_pred.
    
    Args:
        bt_pred: Backtest predictions DataFrame
        rt_pred: Runtime predictions DataFrame
        rt_name: Runtime name
        model_name: Model name
        save_dir: Directory to save plots
        date: Date string
    """
    common_cols = bt_pred.columns.intersection(rt_pred.columns)
    
    # Create model-specific directory
    model_save_dir = save_dir / f"{rt_name}_vs_{model_name}"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    for col in common_cols:
        plt.figure(figsize=(12, 6))
        
        # Plot both predictions
        plt.plot(bt_pred.index, bt_pred[col], label=f'Backtest ({model_name})', linestyle='-', marker='.')
        plt.plot(rt_pred.index, rt_pred[col], label=f'Runtime ({rt_name})', linestyle='--', marker='.')
        
        # Add difference plot on secondary axis
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        diff = bt_pred[col] - rt_pred[col]
        ax2.plot(bt_pred.index, diff, label='Difference', color='red', alpha=0.5)
        ax2.set_ylabel('Difference', color='red')
        
        plt.title(f'Comparison of {col} ({date})\n{rt_name} vs {model_name}')
        plt.xlabel('Time')
        ax1.set_ylabel(col)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(model_save_dir / f"{date}_{rt_name}_vs_{model_name}_{col}_comparison.png", dpi=200)
        plt.close()


# %%
def create_pred_summary_plot(bt_pred: pd.DataFrame, rt_pred: pd.DataFrame, 
                           rt_name: str, model_name: str, save_dir: Path, date: str, color: str, rt_color: str) -> None:
    """
    Create and save a single summary plot for predictions comparison.
    
    Args:
        bt_pred: Backtest predictions DataFrame
        rt_pred: Runtime predictions DataFrame
        rt_name: Runtime name
        model_name: Model name
        save_dir: Directory to save plots
        date: Date string
        color: Color for the plots as specified in model_mapping
    """
    common_cols = bt_pred.columns.intersection(rt_pred.columns)
    
    # Create model-specific directory
    model_save_dir = save_dir / f"{rt_name}_vs_{model_name}"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a single figure with subplots for each column
    n_cols = len(common_cols)
    if n_cols == 0:
        print(f"No common columns found between bt_pred and rt_pred for {model_name}")
        return
    
    fig, axes = plt.subplots(n_cols, 1, figsize=(12, 5*n_cols))
    if n_cols == 1:
        axes = [axes]  # Make axes iterable when there's only one subplot
    
    for i, col in enumerate(common_cols):
        ax1 = axes[i]
        
        # Plot both predictions with specified color
        ax1.plot(bt_pred.index, bt_pred[col], label=f'Backtest ({model_name})', linestyle='-', marker='.', color=color)
        ax1.plot(rt_pred.index, rt_pred[col], label=f'Realtime ({rt_name})', linestyle='--', marker='.', color=rt_color, alpha=0.6)
        
        # Add difference plot on secondary axis
        ax2 = ax1.twinx()
        diff = bt_pred[col] - rt_pred[col]
        ax2.plot(bt_pred.index, diff, label='Difference', color='red', alpha=0.5)
        ax2.set_ylabel('Difference', color='red')
        
        ax1.set_title(f'Prediction Comparison of {col}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel(f'{col} Value')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        ax1.grid(True)
    
    plt.suptitle(f'Prediction Comparison ({date})\n{rt_name} vs {model_name}', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_dir / f"{date}_{rt_name}_vs_{model_name}_pred_summary_comparison.png", dpi=200)
    plt.close()


# %%
def create_pos_summary_plot(bt_pos: pd.DataFrame, rt_pos: pd.DataFrame, 
                              rt_name: str, model_name: str, save_dir: Path, date: str, color: str, rt_color: str) -> None:
    """
    Create and save a single summary plot for positions comparison.
    
    Args:
        bt_pos: Backtest positions DataFrame
        rt_pos: Runtime positions DataFrame
        rt_name: Runtime name
        model_name: Model name
        save_dir: Directory to save plots
        date: Date string
        color: Color for the plots as specified in model_mapping
    """
    common_cols = bt_pos.columns.intersection(rt_pos.columns)
    
    # Create a single figure with subplots for each column
    n_cols = len(common_cols)
    if n_cols == 0:
        print(f"No common columns found between bt_pos and rt_pos for {model_name}")
        return
    
    fig, axes = plt.subplots(n_cols, 1, figsize=(12, 5*n_cols))
    if n_cols == 1:
        axes = [axes]  # Make axes iterable when there's only one subplot
    
    for i, col in enumerate(common_cols):
        ax1 = axes[i]
        
        # Plot both positions with specified color
        ax1.plot(bt_pos.index, bt_pos[col], label=f'Backtest ({model_name})', linestyle='-', marker='.', color=color)
        ax1.plot(rt_pos.index, rt_pos[col], label=f'Realtime ({rt_name})', linestyle='--', marker='.', color=rt_color, alpha=0.6)
        
        # Add difference plot on secondary axis
        ax2 = ax1.twinx()
        diff = bt_pos[col] - rt_pos[col]
        ax2.plot(bt_pos.index, diff, label='Difference', color='red', alpha=0.5)
        ax2.set_ylabel('Difference', color='red')
        
        ax1.set_title(f'Position Comparison of {col}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel(f'{col} Position')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        ax1.grid(True)
    
    plt.suptitle(f'Position Comparison ({date})\n{rt_name} vs {model_name}', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_dir / f"{date}_{rt_name}_vs_{model_name}_pos_summary_comparison.png", dpi=200)
    plt.close()


# %%
def load_and_generate_comparisons(model_mapping: Dict, rt_perdist_dir: Path, model_dir: Path, 
                                date: str, save_dir: Path) -> None:
    """
    Load data and generate comparison plots for all models in model_mapping.
    
    Args:
        model_mapping: Dictionary containing model information
        rt_perdist_dir: Directory with runtime persistence data
        model_dir: Directory with model data
        date: Date string
        project_dir: Project directory
    """
    
    for tag_name, model_info in model_mapping.items():
        model_name = model_info['model_name']
        test_name = model_info['test_name']
        prod_name = model_info['prod_name']
        color = model_info['color']  # 从model_mapping中获取颜色
        rt_color = model_info['rt_color']
        
        # Load runtime predictions and positions
        date_format = datetime.strftime(datetime.strptime(date, '%Y-%m-%d'), '%Y%m%d')
        rt_pred_path = rt_perdist_dir / f'{prod_name}/records/{date_format}.h5'
        store = pd.HDFStore(rt_pred_path, 'r')
        rt_pred = store['predict']
        rt_pos = store['pos']
        store.close()
        
        # Load backtest predictions and positions
        bt_pred_path = model_dir / f'{model_name}/predict/predict_{model_name}.parquet'
        bt_pred_his = pd.read_parquet(bt_pred_path)
        bt_pred = bt_pred_his.loc[date]
        bt_pos_path = model_dir / model_name / 'test' / test_name / 'data' / f'pos_predict_{model_name}.parquet'
        bt_pos_his = pd.read_parquet(bt_pos_path)
        bt_pos = bt_pos_his.loc[date]
        
        # 调用汇总对比函数生成图表
        create_pred_summary_plot(bt_pred, rt_pred, prod_name, model_name, save_dir, date, color, rt_color)
        create_pos_summary_plot(bt_pos, rt_pos, prod_name, model_name, save_dir, date, color, rt_color)
        
        print(f"Generated comparison plots for {tag_name} ({prod_name} vs {model_name})")


# %% 主程序
if __name__ == "__main__":

    date = '2025-04-02'
    
    # 定义目录路径
    rt_perdist_dir = Path(r'D:\CNIndexFutures\timeseries\prod\merge_to_pos\persist')
    model_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model')
    save_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\test_update_summary')

    # 模型映射配置
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
    
    # 生成对比图表
    load_and_generate_comparisons(model_mapping, rt_perdist_dir, model_dir, date, save_dir)