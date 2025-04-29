# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 15:42:18 2025

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
import pandas as pd
from functools import partial
from tqdm import tqdm
import concurrent.futures


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[2]
sys.path.append(str(project_dir))


# %%
from utils.datautils import align_and_sort_columns
from utils.market import index_to_futures


from utils.timeutils import parse_time_string
from utils.trade_rules import *
from data_processing.ts_trans import *
from test_and_eval.factor_tester import FactorTesterByDiscrete
from test_and_eval.factor_evaluation import eval_one_factor_one_period_net_public


# %%
version_name = "v7"  # 更新版本号
model_name = 'avg_agg_250218_3_fix_tfe_by_trade_net_v4'
factor_name = f'predict_{model_name}'
direction = 1

# factor_name = 'LargeOrderAmountByValue_p1.0_v40000-wavg_imb04_dpall-mean_w30min'
# direction = -1
# factor_dir = Path(r'D:/mnt/CNIndexFutures/timeseries/factor_test/sample_data/factors/low_freq')


# %%
test_name = 'traded_futtwap_sp1min_s240d_icim_v6_noscale'
price_name = 't1min_fq1min_dl1min'

scale_method = 'minmax_scale'
scale_window = '240d'
scale_quantile = 0.02
sp = '1min'

trade_rule_name = 'trade_rule_by_reversal_v5'  # 更新为使用新的V4版本交易规则

fee = 0.00024

# v7 参数设置 - 更新为v4需要的参数
threshold_list = [0.6, 0.7, 0.8, 0.9]  # 触发阈值
observation_period_list = [15]  # 观察期长度
reversal_value_list = [0.5, 0.75, 1]  # 新增：从极值点回落/上升的数值
holding_period_list = [60, 120]  # 持仓期
close_on_opposite_threshold_list = [True]  # 在触发对侧阈值时平仓
time_gap_minutes_list = [60]  # 隔夜或午休的时间间隔阈值(分钟)
cooldown_minutes_list = [0, 5]  # 隔夜或午休后的冷却期(分钟)
lookback_periods_list = [0, 5, 30]  # 触发观察前需要检查的前n分钟

date_start = '20160101'
date_end = '20250101'

gen_workers = 20
test_workers = 50


# %%
# fut_dir = Path('/mnt/data1/future_twap')
# factor_dir = Path(rf'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model\{model_name}\predict')
# analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\lob_reversal')

fut_dir = Path('/mnt/data1/futuretwap')
factor_dir = Path(rf'/mnt/data1/xintang/CNIndexFutures/timeseries/factor_test/results/model/{model_name}/predict')
analysis_dir = Path(r'/mnt/data1/xintang/CNIndexFutures/timeseries/factor_test/results/analysis/lob_reversal')


version_dir = analysis_dir / factor_name / trade_rule_name / version_name
pos_dir = version_dir / 'pos'
pos_dir.mkdir(parents=True, exist_ok=True)
test_dir = version_dir
test_dir.mkdir(parents=True, exist_ok=True)
summary_dir = version_dir / 'summary'
summary_dir.mkdir(parents=True, exist_ok=True)


# %%
factor_data = pd.read_parquet(factor_dir / f'{factor_name}.parquet')
price_data = pd.read_parquet(fut_dir / f'{price_name}.parquet')
factor_data = factor_data.rename(columns=index_to_futures)[['IC', 'IF', 'IM']]
factor_data, price_data = align_and_sort_columns([factor_data, price_data])

price_data = price_data.loc[factor_data.index.min():factor_data.index.max()] # 按factor头尾截取
factor_data = factor_data.reindex(price_data.index) # 按twap reindex，确保等长


# %%
scale_func = globals()[scale_method]
scale_step = int(parse_time_string(scale_window) / parse_time_string(sp))
# factor_scaled = ts_quantile_scale(factor, window=scale_step, quantile=scale_quantile)
if scale_method in ['minmax_scale', 'minmax_scale_separate']:
    factor_scaled = scale_func(factor_data, window=scale_step, quantile=scale_quantile)
elif scale_method in ['minmax_scale_adj_by_his_rtn', 'zscore_adj_by_his_rtn_and_minmax']:
    factor_scaled = scale_func(factor, rtn_1p, window=scale_step, rtn_window=pp_by_sp, quantile=scale_quantile)
elif scale_method in ['rolling_percentile']:
    factor_scaled = scale_func(factor, window=scale_step)
elif scale_method in ['percentile_adj_by_his_rtn']:
    factor_scaled = scale_func(factor, rtn_1p, window=scale_step, rtn_window=pp_by_sp)

factor_scaled = (factor_scaled - 0.5) * 2 * direction


# %% 更新process_combination函数，使用v4版本参数
def process_combination(params):
    """Process a single parameter combination."""
    (threshold, observation_period, reversal_value, 
     holding_period, close_on_opposite_threshold,
     time_gap_minutes, cooldown_minutes, lookback_periods) = params
    
    # 设置交易规则参数 - 更新为v4版本所需参数
    trade_rule_param = {
        'threshold': threshold,
        'observation_period': observation_period,
        'reversal_value': reversal_value,  # 新参数：从极值点回落/上升的数值
        'holding_period': holding_period,
        'close_on_opposite_threshold': close_on_opposite_threshold,
        'time_gap_minutes': time_gap_minutes,
        'cooldown_minutes': cooldown_minutes,
        'lookback_periods': lookback_periods
    }
    
    # 创建部分函数用于应用交易规则
    trade_rule_func = partial(globals()[trade_rule_name], **trade_rule_param)
    
    # 对每个品种应用交易规则生成仓位
    actual_pos = pd.DataFrame(index=factor_scaled.index, columns=factor_scaled.columns)
    
    for col in factor_scaled.columns:
        signal = factor_scaled[col]
        actual_pos[col] = trade_rule_func(signal=signal)
    
    # 生成文件名 - 更新为v4版本参数
    file_name = f'th{threshold}_obs{observation_period}_rev{reversal_value}_hold{holding_period}_oppo{str(close_on_opposite_threshold)[0]}_gap{time_gap_minutes}_cool{cooldown_minutes}_look{lookback_periods}'
    
    # 保存仓位数据
    actual_pos.to_parquet(pos_dir / f'{file_name}.parquet')
    
    return trade_rule_param


# %%
# 创建一个结果汇总数据框
results_summary = []

# %% 更新并行处理部分，使用v4版本参数
if gen_workers == 1:
    # 使用tqdm显示进度条
    total_combinations = (
        len(threshold_list) * len(observation_period_list) * 
        len(reversal_value_list) * len(holding_period_list) * 
        len(close_on_opposite_threshold_list) * len(time_gap_minutes_list) *
        len(cooldown_minutes_list) * len(lookback_periods_list)
    )
    
    with tqdm(total=total_combinations) as pbar:
        for threshold in threshold_list:
            for observation_period in observation_period_list:
                for reversal_value in reversal_value_list:
                    for holding_period in holding_period_list:
                        for close_on_opposite_threshold in close_on_opposite_threshold_list:
                            for time_gap_minutes in time_gap_minutes_list:
                                for cooldown_minutes in cooldown_minutes_list:
                                    for lookback_periods in lookback_periods_list:
                                        # 设置交易规则参数 - 更新为v4版本所需参数
                                        trade_rule_param = {
                                            'threshold': threshold,
                                            'observation_period': observation_period,
                                            'reversal_value': reversal_value,  # 新参数
                                            'holding_period': holding_period,
                                            'close_on_opposite_threshold': close_on_opposite_threshold,
                                            'time_gap_minutes': time_gap_minutes,
                                            'cooldown_minutes': cooldown_minutes,
                                            'lookback_periods': lookback_periods
                                        }
                                                
                                        # 创建部分函数用于应用交易规则
                                        trade_rule_func = partial(globals()[trade_rule_name], **trade_rule_param)
                                        
                                        # 对每个品种应用交易规则生成仓位
                                        actual_pos = pd.DataFrame(index=factor_scaled.index, columns=factor_scaled.columns)
                                        
                                        for col in factor_scaled.columns:
                                            signal = factor_scaled[col]
                                            actual_pos[col] = trade_rule_func(signal=signal)
                                                
                                        # 生成文件名 - 更新为v4版本参数
                                        file_name = f'th{threshold}_obs{observation_period}_rev{reversal_value}_hold{holding_period}_oppo{str(close_on_opposite_threshold)[0]}_gap{time_gap_minutes}_cool{cooldown_minutes}_look{lookback_periods}'
                                                
                                        # 保存仓位数据
                                        actual_pos.to_parquet(pos_dir / f'{file_name}.parquet')
                                        
                                        pbar.update(1)
                            
else:
    # 生成所有参数组合 - 更新为v4版本参数
    param_combinations = [
        (threshold, observation_period, reversal_value, 
         holding_period, close_on_opposite_threshold,
         time_gap_minutes, cooldown_minutes, lookback_periods)
        for threshold in threshold_list
        for observation_period in observation_period_list
        for reversal_value in reversal_value_list
        for holding_period in holding_period_list
        for close_on_opposite_threshold in close_on_opposite_threshold_list
        for time_gap_minutes in time_gap_minutes_list
        for cooldown_minutes in cooldown_minutes_list
        for lookback_periods in lookback_periods_list
    ]
    
    # 计算总组合数
    total_combinations = len(param_combinations)
    
    # 使用ProcessPoolExecutor进行并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=gen_workers) as executor:
        # 提交所有任务并使用tqdm显示进度
        futures = [executor.submit(process_combination, params) for params in param_combinations]
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_combinations):
            try:
                result = future.result()
                results_summary.append(result)  # 收集结果
            except Exception as e:
                print(f"处理时出错: {e}")
                        

# %%
tester = FactorTesterByDiscrete(None, None, pos_dir, test_name=test_name, 
                                result_dir=test_dir, n_workers=test_workers)
tester.test_multi_factors(skip_exists=True)


# %%
test_data_dir = test_dir / 'test' / test_name / 'data'
# 更新评估部分，加入止损参数
res_list = []

# 遍历所有交易规则参数组合
for threshold in threshold_list:
    for observation_period in observation_period_list:
        for reversal_value in reversal_value_list:
            for holding_period in holding_period_list:
                for close_on_opposite_threshold in close_on_opposite_threshold_list:
                    for time_gap_minutes in time_gap_minutes_list:
                        for cooldown_minutes in cooldown_minutes_list:
                            for lookback_periods in lookback_periods_list:
                                # 基本结果信息 - 更新为v4版本参数
                                res_info = {
                                    'threshold': threshold,
                                    'observation_period': observation_period,
                                    'reversal_value': reversal_value,  # 新参数
                                    'holding_period': holding_period,
                                    'close_on_opposite_threshold': close_on_opposite_threshold,
                                    'time_gap_minutes': time_gap_minutes,
                                    'cooldown_minutes': cooldown_minutes,
                                    'lookback_periods': lookback_periods
                                }
                                
                                # 生成预测名称 - 更新为v4版本参数
                                pred_name = f'th{threshold}_obs{observation_period}_rev{reversal_value}_hold{holding_period}_oppo{str(close_on_opposite_threshold)[0]}_gap{time_gap_minutes}_cool{cooldown_minutes}_look{lookback_periods}'

                                # 评估结果
                                res_dict = eval_one_factor_one_period_net_public(
                                    pred_name, res_info, test_data_dir, date_start, date_end, fee)
                                
                                res_list.append(res_dict)

res_df = pd.DataFrame(res_list)
res_df.to_csv(summary_dir / f'eval_summary_{version_name}.csv', index=None)