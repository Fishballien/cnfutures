# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:02:09 2025

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
from typing import Dict, List, Tuple, Any
from datetime import datetime
import yaml
import argparse


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.logutils import FishStyleLogger
from update.database_handler import DailyUpdateSender, DailyUpdateReader, DailyUpdateMsgSender
from update.loop_check import CheckDb, ProcessUpdateCoordinator
from utils.dateutils import get_previous_n_trading_day


# %%
def load_prediction_data(date: str, rt_name: str, model_name: str, path_config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load prediction data for both runtime and backtest predictions.
    
    Args:
        date: Date string in format 'YYYYMMDD'
        rt_name: Runtime name
        model_name: Model name
        path_config: Dictionary containing path configurations
        
    Returns:
        Tuple of (backtest_predictions, runtime_predictions)
    """
    # Load runtime predictions
    rt_pred_path = Path(path_config['rt_persist']) / f'{rt_name}/records/{date}.h5'
    store = pd.HDFStore(rt_pred_path, 'r')
    rt_pred = store['predict']
    store.close()
    
    # Load backtest predictions
    bt_pred_path = Path(path_config['result']) / f'model/{model_name}/predict/predict_{model_name}.parquet'
    bt_pred_his = pd.read_parquet(bt_pred_path)
    bt_pred = bt_pred_his.loc[date]
    
    return bt_pred, rt_pred

def calculate_diff_statistics(bt_pred: pd.DataFrame, rt_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics on differences between backtest and runtime predictions.
    
    Args:
        bt_pred: Backtest predictions DataFrame
        rt_pred: Runtime predictions DataFrame
        
    Returns:
        DataFrame with statistics for each common column
    """
    common_cols = bt_pred.columns.intersection(rt_pred.columns)
    stats = []
    
    for col in common_cols:
        # Align indexes if they're not identical
        bt_series = bt_pred[col]
        rt_series = rt_pred[col]
        
        # Calculate difference
        diff = (bt_series - rt_series).abs()
        
        # Calculate statistics
        col_stats = {
            'column': col,
            'mean_diff': diff.mean(),
            'max_diff': diff.max(),
            'min_diff': diff.min(),
            'std_diff': diff.std(),
            'abs_mean_diff': diff.abs().mean(),
            'max_abs_diff': diff.abs().max(),
            'mean_relative_diff': (diff / bt_series.abs().clip(lower=1e-10)).mean(),
            'max_relative_diff': (diff / bt_series.abs().clip(lower=1e-10)).max(),
            'correlation': bt_series.corr(rt_series),
            'num_samples': len(diff)
        }
        stats.append(col_stats)
    
    return pd.DataFrame(stats)

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
        plt.savefig(model_save_dir / f"{date}_{col}_comparison.png", dpi=200)
        plt.close()

def save_diff_statistics(stats_df: pd.DataFrame, rt_name: str, model_name: str, 
                         save_dir: Path, date: str) -> Path:
    """
    Save difference statistics to a CSV file.
    
    Args:
        stats_df: DataFrame with statistics
        rt_name: Runtime name
        model_name: Model name
        save_dir: Directory to save statistics
        date: Date string
        
    Returns:
        Path to the saved statistics file
    """
    # Create model-specific directory
    model_save_dir = save_dir / f"{rt_name}_vs_{model_name}"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save statistics
    stats_file = model_save_dir / f"{date}_diff_statistics.csv"
    stats_df.to_csv(stats_file, index=False)
    
    # Also save as a formatted markdown table for easy reading
    markdown_file = model_save_dir / f"{date}_diff_statistics.md"
    with open(markdown_file, 'w') as f:
        f.write(f"# Difference Statistics: {rt_name} vs {model_name} ({date})\n\n")
        f.write(stats_df.to_markdown(index=False))
    
    return stats_file

def compare_model_pairs(date: str, model_pairs: List[Dict[str, str]], project_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple pairs of runtime and model predictions.
    
    Args:
        date: Date string in format 'YYYYMMDD'
        model_pairs: List of dictionaries with 'rt_name' and 'model_name' keys
        project_dir: Project directory path
        
    Returns:
        Dictionary with statistics for each pair
    """
    # Load path configuration
    path_config = load_path_config(project_dir)
    
    # Setup save directory
    verify_dir = Path(path_config['result']) / 'verify'
    
    # Dictionary to store results
    results = {}
    
    for pair in model_pairs:
        rt_name = pair['rt_name']
        model_name = pair['model_name']
        
        # Create directory for this runtime
        save_dir = verify_dir / rt_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load prediction data
        try:
            bt_pred, rt_pred = load_prediction_data(date, rt_name, model_name, path_config)
            
            # Calculate difference statistics
            stats_df = calculate_diff_statistics(bt_pred, rt_pred)
            
            # Create comparison plots
            create_comparison_plots(bt_pred, rt_pred, rt_name, model_name, save_dir, date)
            
            # Save statistics
            stats_file = save_diff_statistics(stats_df, rt_name, model_name, save_dir, date)
            
            # Store results
            results[f"{rt_name}_vs_{model_name}"] = {
                "bt_pred": bt_pred,
                "rt_pred": rt_pred,
                "stats_df": stats_df,
                "stats_file": stats_file
            }
            
            print(f"✓ Completed comparison for {rt_name} vs {model_name}")
            
        except Exception as e:
            print(f"✗ Error comparing {rt_name} vs {model_name}: {e}")
            results[f"{rt_name}_vs_{model_name}"] = {"error": str(e)}
    
    return results


def daily_check(verify_name, delay=1):
    # 更新至 ————
    date_today = datetime.today().strftime('%Y%m%d')
    target_date = get_previous_n_trading_day(date_today, delay)
    
    # 读取路径配置
    path_config = load_path_config(project_dir)
    param_dir = Path(path_config['workflow_param'])
    
    # 读取参数
    with open(param_dir / 'daily_verify' / f'{verify_name}.yaml', "r") as file:
        params = yaml.safe_load(file)
    
    # 数据库交互
    # Initialize logger and senders
    mysql_name = params['mysql_name']
    author = params['author']
    log = FishStyleLogger()
    daily_update_sender = DailyUpdateSender(mysql_name, author, log=log)
    daily_update_reader = DailyUpdateReader(mysql_name, log=log)
    msg_sender = DailyUpdateMsgSender(mysql_name, author, log=log)
    
    # Initialize check database and coordinator
    check_db_params = params['check_db_params']
    check_db = CheckDb(daily_update_reader, msg_sender, log, config=check_db_params)
    coordinator = ProcessUpdateCoordinator(check_db, daily_update_sender, msg_sender, log)
    coordinator.set_target_date(target_date)
    
    ## update
    
    # 更新近10-20天股票ind
    output = params['output']
    dependency = params['dependency']
    model_pairs = params['model_pairs']
    
    with coordinator(output, dependency):
        if not coordinator.skip_task:
            # Run comparisons
            results = compare_model_pairs(target_date, model_pairs, project_dir)
            
            # Print summary
            print("\nSummary of comparisons:")
            for pair_name, pair_results in results.items():
                if "error" in pair_results:
                    print(f"  - {pair_name}: Failed with error: {pair_results['error']}")
                    msg = f"{pair_name}: Failed with error: {pair_results['error']}"
                    msg_sender.insert('error', output['theme'], msg)
                    raise 
                else:
                    stats = pair_results["stats_df"]
                    print(f"  - {pair_name}: Compared {len(stats)} columns")
                    print(f"    Average absolute difference: {stats['abs_mean_diff'].mean():.6f}")
                    print(f"    Average correlation: {stats['correlation'].mean():.4f}")
                    print(f"    Statistics saved to: {pair_results['stats_file']}")
                    if (stats['max_diff'].max() > 0.1):
                        msg_sender.insert('warning', '发现异常，请检查实盘回测核对', stats)
    
    
def main():
    '''read args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-vn', '--verify_name', type=str, help='verify_name')
    parser.add_argument('-dl', '--delay', type=int, help='delay')

    args = parser.parse_args()
    verify_name, delay = args.verify_name, args.delay
    
    daily_check(verify_name, delay)
    

# %%
if __name__ == "__main__":
    main()