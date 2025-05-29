# -*- coding: utf-8 -*-
"""
Created on Wed May 28 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import sys
from pathlib import Path
from datetime import datetime, timedelta
import toml
from tqdm import tqdm
import warnings
import concurrent.futures
warnings.filterwarnings("ignore")


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from utils.timeutils import get_lb_fit_periods, get_rolling_dates
from apply_filter.apply_filters_on_merged import FilterApplier


# %% Helper function for concurrent processing
def process_period_with_filters(fa, period_info, period_idx, total_periods):
    """
    Helper function to process a single period for parallel execution with filter application.
    
    Parameters
    ----------
    fa : FilterApplier
        Instance of FilterApplier class to use for processing
    period_info : tuple
        Tuple containing the period information
    period_idx : int
        Index of the current period (1-based)
    total_periods : int
        Total number of periods to process
        
    Returns
    -------
    dict
        Dictionary containing the processing result information
    """
    print(f"Processing filter application for period {period_idx}/{total_periods}: {period_info}")
    try:
        fa.run_one_period(*period_info)
        status = "success"
        message = f"Completed filter application for period {period_idx}/{total_periods}"
    except Exception as e:
        status = "error"
        message = f"Error processing filter application for period {period_idx}/{total_periods}: {str(e)}"
    
    print(message)
    return {
        "period_idx": period_idx,
        "period_info": period_info,
        "status": status,
        "message": message
    }

                
# %%
def run_rolling_apply_filters_on_merged(apply_filters_name: str, merge_name: str, 
                                      rolling_name: str, pstart: str = '20230701', puntil: str = None, 
                                      mode: str = 'rolling', max_workers: int = None, n_workers: int = 1):
    """
    对合并后的因子执行滚动过滤器应用操作
    
    此函数通过滚动窗口方法或仅更新最近期间来执行因子过滤器应用。它从TOML文件加载配置参数
    并使用FilterApplier类来执行操作。
    
    Parameters
    ----------
    apply_filters_name : str
        要使用的过滤器应用配置名称
    merge_name : str
        要使用的合并配置名称
    rolling_name : str
        滚动配置文件名称（不含.toml扩展名）
    pstart : str, optional
        处理期间的开始日期，格式为'YYYYMMDD' (默认: '20230701')
    puntil : str, optional
        处理期间的结束日期，格式为'YYYYMMDD' (默认: 当前日期)
    mode : str, optional
        处理模式，'rolling'（处理所有期间）或'update'（仅处理最新期间）
        (默认: 'rolling')
    max_workers : int, optional
        用于因子过滤器应用操作的最大工作进程数 (默认: None)
    n_workers : int, optional
        用于并行期间处理的工作进程数 (默认: 1)
        如果设置为1，期间将按顺序处理
        
    Returns
    -------
    None
        结果保存到配置的输出位置
    
    Notes
    -----
    此函数依赖于'param/rolling_single_period'目录中的配置文件
    并使用FilterApplier类来执行实际的过滤器应用操作。
    """
    # Initialize variables
    if puntil is None:
        puntil = datetime.utcnow().date().strftime('%Y%m%d')
    
    print(f"Starting {mode} filter application with parameters:")
    print(f"- Apply filters name: {apply_filters_name}")
    print(f"- Merge name: {merge_name}")
    print(f"- Rolling config: {rolling_name}")
    print(f"- Period: {pstart} to {puntil}")
    print(f"- Parallel workers: {n_workers}")
    
    # Load path configuration
    path_config = load_path_config(project_dir)
    print(f"Loaded path configuration from {project_dir}")
    
    # Initialize directories
    param_dir = Path(path_config['param'])
    print(f"Using parameter directory: {param_dir}")
    
    # Load parameters
    params_file = param_dir / 'rolling_single_period' / f'{rolling_name}.toml'
    print(f"Loading parameters from {params_file}")
    params = toml.load(params_file)
    
    # Get rolling dates
    fstart = params['fstart']
    rolling_dates = get_rolling_dates(fstart, pstart, puntil)
    
    # Get lookback fit periods
    rolling_params = params['rolling_params']
    lb = params['lb']
    lb_fit_periods = get_lb_fit_periods(rolling_dates, rolling_params, lb)
    print(f"Created {len(lb_fit_periods)} lookback fit periods with lookback windows")
    
    # Initialize the filter applier
    fa = FilterApplier(apply_filters_name, merge_name, max_workers=max_workers)
    print(f"Initialized FilterApplier with configuration: {apply_filters_name}, "
          f"merge_name: {merge_name}")
    
    if mode == 'rolling':
        print(f"Running in ROLLING mode - processing all {len(lb_fit_periods)} periods for filter application")
        
        # Sequential processing (original behavior)
        if n_workers == 1:
            print("Using sequential processing (n_workers=1)")
            for i, fp in enumerate(tqdm(lb_fit_periods, desc='Rolling filter application progress')):
                process_period_with_filters(fa, fp, i+1, len(lb_fit_periods))
        
        # Parallel processing
        else:
            print(f"Using parallel processing with {n_workers} workers")
            n_workers = min(n_workers, len(lb_fit_periods))  # Don't use more workers than periods
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Prepare the tasks
                future_to_period = {
                    executor.submit(
                        process_period_with_filters, 
                        FilterApplier(apply_filters_name, merge_name, max_workers=max_workers),  # Create new instance for each worker
                        fp, i+1, len(lb_fit_periods)
                    ): i for i, fp in enumerate(lb_fit_periods)
                }
                
                # Process results as they complete
                results = []
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_period), 
                    total=len(future_to_period),
                    desc='Rolling filter application progress'
                ):
                    period_idx = future_to_period[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Period {period_idx+1} filter application failed: {str(e)}")
                
                # Print summary of results
                success_count = sum(1 for r in results if r['status'] == 'success')
                print(f"Completed {success_count}/{len(lb_fit_periods)} periods successfully")
    
    elif mode == 'update':
        print("Running in UPDATE mode - processing only the latest period for filter application")
        fp = lb_fit_periods[-1]
        print(f"Processing latest period: {fp}")
        fa.run_one_period(*fp)
        print("Completed filter application for latest period")
    
    print(f"Finished {mode} filter application operation for {rolling_name}")


# %%
def main():
    """
    主函数，提供使用示例
    """
    # 示例调用
    run_rolling_apply_filters_on_merged(
        apply_filters_name='momentum_filters',
        merge_name='merge_v1',
        rolling_name='monthly_rolling',
        pstart='20240101',
        puntil='20241231',
        mode='rolling',
        max_workers=4,
        n_workers=2
    )


if __name__ == "__main__":
    # 这里可以根据命令行参数调用或直接运行示例
    main()
