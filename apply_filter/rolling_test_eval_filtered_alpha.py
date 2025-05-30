# -*- coding: utf-8 -*-
"""
Created on Wed May 29 2025

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
from apply_filter.test_eval_filtered_alpha import TestEvalFilteredAlpha


# %% Helper function for concurrent processing
def process_period_with_test_eval(tefa, period_info, eval_period_info, period_idx, total_periods):
    """
    Helper function to process a single period for parallel execution with test evaluation.
    
    Parameters
    ----------
    tefa : TestEvalFilteredAlpha
        Instance of TestEvalFilteredAlpha class to use for processing
    period_info : tuple
        Tuple containing the period information (date_start, date_end)
    eval_period_info : tuple
        Tuple containing the evaluation period information (eval_date_start, eval_date_end)
    period_idx : int
        Index of the current period (1-based)
    total_periods : int
        Total number of periods to process
        
    Returns
    -------
    dict
        Dictionary containing the processing result information
    """
    print(f"Processing test evaluation for period {period_idx}/{total_periods}: {period_info} -> eval: {eval_period_info}")
    try:
        tefa.run_one_period(*period_info, *eval_period_info)
        status = "success"
        message = f"Completed test evaluation for period {period_idx}/{total_periods}"
    except Exception as e:
        status = "error"
        message = f"Error processing test evaluation for period {period_idx}/{total_periods}: {str(e)}"
    
    print(message)
    return {
        "period_idx": period_idx,
        "period_info": period_info,
        "eval_period_info": eval_period_info,
        "status": status,
        "message": message
    }

                
# %%
def run_rolling_test_eval_filtered_alpha(test_eval_filtered_alpha_name: str, merge_name: str, 
                                       rolling_name: str, eval_rolling_name: str, 
                                       pstart: str = '20230701', puntil: str = None, 
                                       mode: str = 'rolling', max_workers: int = None, 
                                       n_workers: int = 1):
    """
    对过滤后的alpha执行滚动测试评估操作
    
    此函数通过滚动窗口方法或仅更新最近期间来执行过滤后alpha的测试评估。它从TOML文件加载配置参数
    并使用TestEvalFilteredAlpha类来执行操作。
    
    Parameters
    ----------
    test_eval_filtered_alpha_name : str
        要使用的测试评估配置名称
    merge_name : str
        要使用的合并配置名称
    rolling_name : str
        滚动配置文件名称（不含.toml扩展名）
    eval_rolling_name : str
        评估滚动配置文件名称（不含.toml扩展名）
    pstart : str, optional
        处理期间的开始日期，格式为'YYYYMMDD' (默认: '20230701')
    puntil : str, optional
        处理期间的结束日期，格式为'YYYYMMDD' (默认: 当前日期)
    mode : str, optional
        处理模式，'rolling'（处理所有期间）或'update'（仅处理最新期间）
        (默认: 'rolling')
    max_workers : int, optional
        用于测试评估操作的最大工作进程数 (默认: None)
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
    并使用TestEvalFilteredAlpha类来执行实际的测试评估操作。
    """
    # Initialize variables
    if puntil is None:
        puntil = datetime.utcnow().date().strftime('%Y%m%d')
    
    print(f"Starting {mode} test evaluation for filtered alpha with parameters:")
    print(f"- Test eval config name: {test_eval_filtered_alpha_name}")
    print(f"- Merge name: {merge_name}")
    print(f"- Rolling config: {rolling_name}")
    print(f"- Eval rolling config: {eval_rolling_name}")
    print(f"- Period: {pstart} to {puntil}")
    print(f"- Parallel workers: {n_workers}")
    
    # Load path configuration
    path_config = load_path_config(project_dir)
    print(f"Loaded path configuration from {project_dir}")
    
    # Initialize directories
    param_dir = Path(path_config['param'])
    print(f"Using parameter directory: {param_dir}")
    
    # Load parameters for main rolling
    params_file = param_dir / 'rolling_single_period' / f'{rolling_name}.toml'
    print(f"Loading parameters from {params_file}")
    params = toml.load(params_file)
    
    # Load parameters for evaluation rolling
    eval_params_file = param_dir / 'rolling_single_period' / f'{eval_rolling_name}.toml'
    print(f"Loading evaluation parameters from {eval_params_file}")
    eval_params = toml.load(eval_params_file)
    
    # Get rolling dates for main periods
    fstart = params['fstart']
    rolling_dates = get_rolling_dates(fstart, pstart, puntil)
    
    # Get rolling dates for evaluation periods
    eval_fstart = eval_params['fstart']
    eval_rolling_dates = get_rolling_dates(eval_fstart, pstart, puntil)
    
    # Get lookback fit periods for main periods
    rolling_params = params['rolling_params']
    lb = params['lb']
    lb_fit_periods = get_lb_fit_periods(rolling_dates, rolling_params, lb)
    print(f"Created {len(lb_fit_periods)} main lookback fit periods with lookback windows")
    
    # Get lookback fit periods for evaluation periods
    eval_rolling_params = eval_params['rolling_params']
    eval_lb = eval_params['lb']
    eval_lb_fit_periods = get_lb_fit_periods(eval_rolling_dates, eval_rolling_params, eval_lb)
    print(f"Created {len(eval_lb_fit_periods)} evaluation lookback fit periods with lookback windows")
    
    # Check if the periods can be zipped together
    if len(lb_fit_periods) != len(eval_lb_fit_periods):
        print(f"Warning: Main periods ({len(lb_fit_periods)}) and eval periods ({len(eval_lb_fit_periods)}) have different lengths")
        # Take the minimum length to avoid index errors
        min_length = min(len(lb_fit_periods), len(eval_lb_fit_periods))
        lb_fit_periods = lb_fit_periods[:min_length]
        eval_lb_fit_periods = eval_lb_fit_periods[:min_length]
        print(f"Using {min_length} periods for processing")
    
    # Initialize the test evaluator
    tefa = TestEvalFilteredAlpha(
        test_eval_filtered_alpha_name, 
        merge_name, 
        max_workers=max_workers
    )
    print(f"Initialized TestEvalFilteredAlpha with configuration: {test_eval_filtered_alpha_name}, "
          f"merge_name: {merge_name}")
    tefa = TestEvalFilteredAlpha(
        test_eval_filtered_alpha_name, 
        merge_name, 
        max_workers=max_workers
    )
    print(f"Initialized TestEvalFilteredAlpha with configuration: {test_eval_filtered_alpha_name}, "
          f"merge_name: {merge_name}")
    
    if mode == 'rolling':
        print(f"Running in ROLLING mode - processing all {len(lb_fit_periods)} periods for test evaluation")
        
        # Sequential processing (original behavior)
        if n_workers == 1:
            print("Using sequential processing (n_workers=1)")
            for i, (fp, eval_fp) in enumerate(tqdm(zip(lb_fit_periods, eval_lb_fit_periods), 
                                                  desc='Rolling test evaluation progress')):
                print(fp, eval_fp, i+1, len(lb_fit_periods))
                process_period_with_test_eval(tefa, fp, eval_fp, i+1, len(lb_fit_periods))
        
        # Parallel processing
        else:
            print(f"Using parallel processing with {n_workers} workers")
            n_workers = min(n_workers, len(lb_fit_periods))  # Don't use more workers than periods
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Prepare the tasks
                future_to_period = {
                    executor.submit(
                        process_period_with_test_eval, 
                        TestEvalFilteredAlpha(test_eval_filtered_alpha_name, merge_name, max_workers=max_workers),  # Create new instance for each worker
                        fp, eval_fp, i+1, len(lb_fit_periods)
                    ): i for i, (fp, eval_fp) in enumerate(zip(lb_fit_periods, eval_lb_fit_periods))
                }
                
                # Process results as they complete
                results = []
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_period), 
                    total=len(future_to_period),
                    desc='Rolling test evaluation progress'
                ):
                    period_idx = future_to_period[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Period {period_idx+1} test evaluation failed: {str(e)}")
                
                # Print summary of results
                success_count = sum(1 for r in results if r['status'] == 'success')
                print(f"Completed {success_count}/{len(lb_fit_periods)} periods successfully")
    
    elif mode == 'update':
        print("Running in UPDATE mode - processing only the latest period for test evaluation")
        fp = lb_fit_periods[-1]
        eval_fp = eval_lb_fit_periods[-1]
        print(f"Processing latest period: {fp} -> eval: {eval_fp}")
        tefa.run_one_period(*fp, *eval_fp)
        print("Completed test evaluation for latest period")
    
    print(f"Finished {mode} test evaluation operation for {rolling_name}")


# %%
def main():
    """
    主函数，提供使用示例
    """
    # 示例调用
    run_rolling_test_eval_filtered_alpha(
        test_eval_filtered_alpha_name='momentum_test_eval',
        merge_name='merge_v1',
        rolling_name='monthly_rolling',
        eval_rolling_name='monthly_eval_rolling',
        pstart='20240101',
        puntil='20241231',
        mode='rolling',
        max_workers=4,
        n_workers=2
    )


if __name__ == "__main__":
    # 这里可以根据命令行参数调用或直接运行示例
    main()