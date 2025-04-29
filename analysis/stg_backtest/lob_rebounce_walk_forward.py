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
from tqdm import tqdm
from datetime import datetime


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[2]
sys.path.append(str(project_dir))


# %%
from utils.trade_rules import *
from data_processing.ts_trans import *
from test_and_eval.factor_tester import FactorTesterByDiscrete
from test_and_eval.factor_evaluation import eval_one_factor_one_period_net_public
from utils.timeutils import RollingPeriods, period_shortcut
from synthesis.filter_methods import *
from utils.datautils import compute_dataframe_dict_average, add_dataframe_to_dataframe_reindex


# %%
to_eval = True
to_filter = True
to_predict = True


# %%
wk_name = '07_only_top2'
version_name = "v5"
model_name = 'avg_agg_250218_3_fix_tfe_by_trade_net_v4'
factor_name = f'predict_{model_name}'
factor_dir = Path(rf'/mnt/data1/xintang/CNIndexFutures/timeseries/factor_test/results/model/{model_name}/predict')
direction = 1

# factor_name = 'LargeOrderAmountByValue_p1.0_v40000-avg_imb01_dp2-rollingAggMinuteMinMaxScale_w30d_q0_i30'
# factor_name = 'ValueTimeDecayOrderAmount_p1.0_v40000_d0.1-wavg_imb04_dp2-rollingAggMinuteMinMaxScale_w30d_q0.02_i5'
# direction = -1
# factor_dir = Path(r'/mnt/data1/xintang/index_factors/Batch10_fix_best_241218_selected_f64/v1.2_all_trans_3')


# %%
test_name = 'traded_futtwap_sp1min_s240d_icim_v6_noscale'
trade_rule_name = 'trade_rule_by_reversal_v3'

fee = 0.00024

# v5
# threshold_list = [0.7, 0.8, 0.9, 0.99]  # 触发阈值
threshold_list = [0.7]  # 触发阈值
observation_period_list = [10, 15]  # 观察期长度
min_observation_periods_list = [5]  # 最小观察期
slope_threshold_list = [0.1, 0.15]  # 斜率阈值
max_slope_periods_list = [5, 7]  # 新增：斜率计算的最大周期数
holding_period_list = [60, 120]  # 持仓期
close_on_opposite_threshold_list = [True]  # 在触发对侧阈值时平仓
time_gap_minutes_list = [240]  # 隔夜或午休的时间间隔阈值(分钟)
cooldown_minutes_list = [0, 3, 5]  # 隔夜或午休后的冷却期(分钟)
lookback_periods_list = [0, 5]  # 触发观察前需要检查的前n分钟

fstart = '20150101'
pstart = '20170101'
puntil = '20250201'
window_kwargs = {'months': 96}
rrule_kwargs = {'freq': 'M', 'interval': 1}
end_by = 'date'

filter_name = 'filter_func_v17'


# %%
# fut_dir = Path('/mnt/data1/future_twap')
# factor_dir = Path(rf'D:\mnt\CNIndexFutures\timeseries\factor_test\results\model\{model_name}\predict')
# analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\lob_reversal')

fut_dir = Path('/mnt/data1/futuretwap')
analysis_dir = Path(r'/mnt/data1/xintang/CNIndexFutures/timeseries/factor_test/results/analysis/lob_reversal')


version_dir = analysis_dir / factor_name / trade_rule_name / version_name
pos_dir = version_dir / 'pos'
pos_dir.mkdir(parents=True, exist_ok=True)
test_dir = version_dir
test_dir.mkdir(parents=True, exist_ok=True)
save_dir = version_dir / 'walk_forward' / wk_name
eval_dir = save_dir / 'eval'
filtered_dir = save_dir / 'filtered'
predict_dir = save_dir / 'predict' 
predict_test_dir = save_dir / 'test' 
eval_dir.mkdir(parents=True, exist_ok=True)
filtered_dir.mkdir(parents=True, exist_ok=True)
predict_dir.mkdir(parents=True, exist_ok=True)
predict_test_dir.mkdir(parents=True, exist_ok=True)


# %%
rolling = RollingPeriods(
    fstart=datetime.strptime(fstart, '%Y%m%d'),
    pstart=datetime.strptime(pstart, '%Y%m%d'),
    puntil=datetime.strptime(puntil, '%Y%m%d'),
    window_kwargs=window_kwargs,
    rrule_kwargs=rrule_kwargs,
    end_by=end_by,
    )


filter_params = {'min_count': 2}


# %% eval
test_data_dir = test_dir / 'test' / test_name / 'data'


def eval_one_period(date_start, date_end):
    # Update the evaluation part
    res_list = []
    
    # 遍历交易规则参数
    for threshold in threshold_list:
        for observation_period in observation_period_list:
            for min_observation_periods in min_observation_periods_list:
                for slope_threshold in slope_threshold_list:
                    for max_slope_periods in max_slope_periods_list:
                        for holding_period in holding_period_list:
                            for close_on_opposite_threshold in close_on_opposite_threshold_list:
                                for time_gap_minutes in time_gap_minutes_list:
                                    for cooldown_minutes in cooldown_minutes_list:
                                        for lookback_periods in lookback_periods_list:
                                            
                                            # 生成预测名称
                                            pred_name = f'th{threshold}_obs{observation_period}_minobs{min_observation_periods}_slope{slope_threshold}_maxsp{max_slope_periods}_hold{holding_period}_oppo{str(close_on_opposite_threshold)[0]}_gap{time_gap_minutes}_cool{cooldown_minutes}_look{lookback_periods}'
                                            
                                            # 基本结果信息
                                            res_info = {
                                                'pred_name': pred_name, 
                                                'threshold': threshold,
                                                'observation_period': observation_period,
                                                'min_observation_periods': min_observation_periods,
                                                'slope_threshold': slope_threshold,
                                                'max_slope_periods': max_slope_periods,
                                                'holding_period': holding_period,
                                                'close_on_opposite_threshold': close_on_opposite_threshold,
                                                'time_gap_minutes': time_gap_minutes,
                                                'cooldown_minutes': cooldown_minutes,
                                                'lookback_periods': lookback_periods
                                            }
                                            
                                            
                                            # 评估结果
                                            res_dict = eval_one_factor_one_period_net_public(
                                                pred_name, res_info, test_data_dir, date_start, date_end, fee)
                                            
                                            res_list.append(res_dict)
    
    res_df = pd.DataFrame(res_list)
    return res_df


# %% rolling eval
if to_eval:
    for fp in tqdm(rolling.fit_periods, 'rolling eval'):
        fit_period = period_shortcut(*fp)
        eval_res = eval_one_period(*fp)
        eval_res.to_csv(eval_dir / f'eval_summary_{version_name}_{fit_period}.csv', index=None)
    

# %% rolling filter
if to_filter:
    filter_func = globals()[filter_name]
    for fp in tqdm(rolling.fit_periods, 'rolling filter'):
        fit_period = period_shortcut(*fp)
        eval_res = pd.read_csv(eval_dir / f'eval_summary_{version_name}_{fit_period}.csv')
        filtered = eval_res[filter_func(eval_res, **filter_params)].reset_index()
        filtered.to_csv(filtered_dir / f'filtered_{version_name}_{fit_period}.csv', index=None)
    
    
# %%
pos_all = pd.DataFrame()
for fp, pp in tqdm(list(zip(rolling.fit_periods, rolling.predict_periods)), desc='rolling predict'):
    fit_period = period_shortcut(*fp)
    filtered = pd.read_csv(filtered_dir / f'filtered_{version_name}_{fit_period}.csv')
    pos_dict, weight_dict = {}, {}
    for pred_name in filtered['pred_name']:
        pos_path = pos_dir / f'{pred_name}.parquet'
        pos = pd.read_parquet(pos_path)
        pos_dict[pred_name] = pos.loc[pp[0]:pp[1]]
        weight_dict[pred_name] = 1
    pos_avg_period = compute_dataframe_dict_average(pos_dict, weight_dict)
    pos_all = add_dataframe_to_dataframe_reindex(pos_all, pos_avg_period)
    
pos_all.to_csv(predict_dir / f'pos_{version_name}.csv')
pos_all.to_parquet(predict_dir / f'pos_{version_name}.parquet')


# %%
tester = FactorTesterByDiscrete(None, None, predict_dir, test_name=test_name, 
                                result_dir=predict_test_dir)
tester.test_one_factor(f'pos_{version_name}')
