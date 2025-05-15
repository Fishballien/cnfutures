# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:15:44 2025

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
import toml
import copy
import pickle
import toml
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from test_and_eval.factor_tester import FactorTesterByDiscrete, FactorTesterByContinuous
from test_and_eval.scores import get_general_return_metrics
from test_and_eval.factor_evaluation import eval_one_factor_one_period_net_public, eval_one_factor_one_period_by_net
from utils.timeutils import RollingPeriods, period_shortcut
from utils.datautils import add_dataframe_to_dataframe_reindex


# %%
model_name = 'avg_agg_250218_3_fix_fut_fr15_by_trade_net_v18'
suffix = '1_2_0'
process_name = 'final_predict'
tag_name = 'test_trade_rules'
test_name = 'trade_ver3_3_futtwap_sp1min_s240d_icim_v6'
pred_name = f'predict_{model_name}'
multi_test_name = 'trade_ver3_3_futtwap_sp1min_s240d_icim_v6'
multi_test_func_name = 'trade_rule_by_trigger_v3_4'
final_test_list = [
    {
     'mode': 'trade',
     'test_name': 'traded_futtwap_sp1min_s240d_icim_v6_noscale',
     }
    ]
fee = 0.00024


fstart = '20150101'
pstart = '20170101'
puntil = '20250515'
window_kwargs = {'months': 96}
rrule_kwargs = {'freq': 'm', 'interval': 1, 'bymonthday': 1}
end_by = 'date'

version_name = f"v6_{suffix}"
version_params = {
    'neighbor': True,
    'net_sharpe_ratio': 1.9,
    'profit_per_trade': 1.2,
    'net_max_dd': 0.2,
    'net_burke_ratio': 5,
    }

# =============================================================================
# version_name = f"v7_{suffix}"
# version_params = {
#     # 'neighbor': True,
#     'net_sharpe_ratio': 1.9,
#     'profit_per_trade': 0,
#     'net_max_dd': 1,
#     'net_burke_ratio': 0,
#     }
# 
# 
# version_name = f"v8_{suffix}"
# version_params = {
#     'neighbor': True,
#     'net_sharpe_ratio': 1.9,
#     'profit_per_trade': 0,
#     'net_max_dd': 1,
#     'net_burke_ratio': 0,
#     }
# =============================================================================


# %%
path_config = load_path_config(project_dir)
param_dir = Path(path_config['param'])
result_dir = Path(path_config['result'])
model_dir = result_dir / 'model'
rolling_model_dir = result_dir / 'rolling_model' / version_name
pos_dir = rolling_model_dir / 'pos'
pos_dir.mkdir(parents=True, exist_ok=True)
factor_data_dir = model_dir / model_name
summary_dir = factor_data_dir / 'trade_rule_summary'
summary_dir.mkdir(parents=True, exist_ok=True)


# %%
test_param_path = param_dir / 'test' / f'{test_name}.toml'
test_param = toml.load(test_param_path)


# %%
openthres_list = list(reversed([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]))
# openthres_list = [-0.1, 0, 0.1]
closethres_list = [-0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# closethres_list = [0.3, 0.4, 0.5, 0.6]


# %% test
for openthres in tqdm(openthres_list, desc='test_open'):
    for closethres in closethres_list:
        test_pr = copy.deepcopy(test_param)
        test_pr['trade_rule_param']['threshold_combinations'] = [[openthres, closethres]]
        
        new_test_name = f'{test_name}_op{openthres}_cl{closethres}'
        
        tester = FactorTesterByDiscrete(process_name, tag_name, factor_data_dir, test_name=new_test_name, 
                                        result_dir=result_dir, params=test_pr)
        tester.test_multi_factors()


# %%
rolling = RollingPeriods(
    fstart=datetime.strptime(fstart, '%Y%m%d'),
    pstart=datetime.strptime(pstart, '%Y%m%d'),
    puntil=datetime.strptime(puntil, '%Y%m%d'),
    window_kwargs=window_kwargs,
    rrule_kwargs=rrule_kwargs,
    end_by=end_by,
    )
        
        
# %%
def eval_all_open_close_thres(date_start, date_end):
    res_list = []
    
    for openthres in openthres_list:
        for closethres in closethres_list:
            res_info = {
                'openthres': openthres, 
                'closethres': closethres, 
                }
            pred_name = f'predict_{model_name}'
            new_test_name = f'{test_name}_op{openthres}_cl{closethres}'
            test_data_dir = factor_data_dir / 'test' / new_test_name / tag_name / process_name / 'data'
            
            res_dict = eval_one_factor_one_period_net_public(
                pred_name, res_info, test_data_dir, date_start, date_end, fee)
            
            res_list.append(res_dict)
            
    res_df = pd.DataFrame(res_list)
    return res_df
        

# %%
def generate_condition(data, params):
    condition = (
        (data['net_sharpe_ratio'] > params['net_sharpe_ratio']) & 
        (data['profit_per_trade'] > params['profit_per_trade']) & 
        (data['net_max_dd'] < params['net_max_dd']) & 
        (data['net_burke_ratio'] > params['net_burke_ratio'])
    )
    return condition


def generate_condition_with_neighbors(data, params, openthres_list, closethres_list):
    """
    根据相邻参数生成条件，判断当前参数与其相邻参数是否满足基本条件。
    如果相邻参数不存在，则默认满足条件。
    """
    condition = pd.Series(True, index=data.index)  # 初始化条件为True
    
    # 获取 openthres 和 closethres 的索引位置
    openthres_idx = {v: i for i, v in enumerate(openthres_list)}
    closethres_idx = {v: i for i, v in enumerate(closethres_list)}

    for idx, row in data.iterrows():
        openthres, closethres = row.name
        
        # 获取相邻的 openthres 和 closethres 的位置，根据 openthres_idx 和 closethres_idx 来偏移
        # 直接列出五种情况
        neighbors = [
            data.loc[(openthres, closethres)],  # 当前值 (open, close)
            data.loc[(openthres_list[openthres_idx[openthres] + 1] if openthres_idx[openthres] + 1 < len(openthres_list) else openthres, closethres)],  # open + 1
            data.loc[(openthres_list[openthres_idx[openthres] - 1] if openthres_idx[openthres] - 1 >= 0 else openthres, closethres)],  # open - 1
            data.loc[(openthres, closethres_list[closethres_idx[closethres] + 1] if closethres_idx[closethres] + 1 < len(closethres_list) else closethres)],  # close + 1
            data.loc[(openthres, closethres_list[closethres_idx[closethres] - 1] if closethres_idx[closethres] - 1 >= 0 else closethres)]  # close - 1
        ]
        
        # 检查相邻参数
        for neighbor in neighbors:
            # 检查相邻参数是否满足条件，若不满足条件，则当前参数也不满足
            if not (
                neighbor['net_sharpe_ratio'] > params['net_sharpe_ratio'] and
                neighbor['profit_per_trade'] > params['profit_per_trade'] and
                neighbor['net_max_dd'] < params['net_max_dd'] and
                neighbor['net_burke_ratio'] > params['net_burke_ratio']
            ):
                condition[idx] = False
                break
        
        # 检查当前参数是否满足条件，若不满足，则该位置也为 False
        if not (
            row['net_sharpe_ratio'] > params['net_sharpe_ratio'] and
            row['profit_per_trade'] > params['profit_per_trade'] and
            row['net_max_dd'] < params['net_max_dd'] and
            row['net_burke_ratio'] > params['net_burke_ratio']
        ):
            condition[idx] = False

    return condition

# 生成条件的描述文本
def generate_conditions_text(params):
    conditions_text = "Filter conditions:\n"
    for key, value in params.items():
        if key == 'net_max_dd':
            conditions_text += f"- {key} < {value}\n"
        else:
            conditions_text += f"- {key} > {value}\n"
    return conditions_text


def filter_conditions(res_df, date_start, date_end, to_plot=[]):
    period = period_shortcut(date_start, date_end)
    # 设置 'openthres' 和 'closethres' 为索引
    heatmap_data = res_df.set_index(['openthres', 'closethres'])
    
    # 生成condition并输出条件文本
    if version_params.get('neighbor'):
        condition = generate_condition_with_neighbors(heatmap_data, version_params, list(reversed(openthres_list)), closethres_list)
    else:
        condition = generate_condition(heatmap_data, version_params)
    conditions_text = generate_conditions_text(version_params)
    
    # 提取符合条件的 openthres 和 closethres 的组合
    valid_pairs = condition[condition].index.tolist()
    print(f"Version {version_name} {period} valid pairs: {valid_pairs}")
    
    # 根据筛选条件绘制热力图并保存到 summary_dir
    mask = ~condition.unstack()  # 转换成unstack后的形状，并反转
    
    to_plot = to_plot or heatmap_data.columns
    for column in to_plot:
        fig, ax = plt.subplots(figsize=(12, 8))  # 增加宽度，预留文本区域
        
        # 使用 unstack() 将数据转换为适合绘制热力图的格式
        heatmap_matrix = heatmap_data[column].unstack()
        
        # 应用 mask
        sns.heatmap(heatmap_matrix, annot=True, cmap='coolwarm', fmt='.2f', mask=mask, ax=ax)
    
        # 调整轴标签和标题
        plt.title(f'Masked Heatmap for {column} {period} (filtered by conditions) - {version_name}')
        plt.xlabel('closethres')
        plt.ylabel('openthres')
    
        # 在图右侧添加过滤条件文本
        plt.figtext(1.05, 0.5, conditions_text, fontsize=12, verticalalignment="center", bbox=dict(facecolor='white', alpha=0.5))
    
        # 调整图像布局，避免标签遮挡
        plt.tight_layout()
    
        # 保存图像
        img_filename = summary_dir / f'{version_name}_{period}_{column}_masked_heatmap.png'
        plt.savefig(img_filename, bbox_inches='tight')  # bbox_inches='tight' 确保文本不被裁剪
        plt.show()
        plt.close()  # 关闭当前图像，防止重叠
        
    test_pr = copy.deepcopy(test_param)
    test_pr['trade_rule_name'] = multi_test_func_name
    test_pr['trade_rule_param'] = {'threshold_combinations': valid_pairs}
    
    new_test_name = f'{multi_test_name}_{version_name}_{period}'
    
    with open(param_dir / 'test' / f"{new_test_name}.toml", "w") as f:
        toml.dump(test_pr, f)
    
    tester = FactorTesterByDiscrete(None, None, factor_data_dir / 'predict', test_name=new_test_name, 
                                    result_dir=factor_data_dir, params=test_pr)
    tester.test_one_factor(pred_name)

    
# %%
for fp in tqdm(rolling.fit_periods, 'rolling filter'):
    eval_res = eval_all_open_close_thres(*fp)
    filter_conditions(eval_res, *fp, to_plot=['net_sharpe_ratio'])
    
pos_all = pd.DataFrame()
for fp, pp in tqdm(list(zip(rolling.fit_periods, rolling.predict_periods)), desc='concat prediction'):
    period = period_shortcut(*fp)
    new_test_name = f'{multi_test_name}_{version_name}_{period}'
    test_data_dir = factor_data_dir / 'test' / new_test_name / 'data'
    pos_filename = f'pos_predict_{model_name}'
    pos_path = test_data_dir / f'{pos_filename}.parquet'
    pos = pd.read_parquet(pos_path)
    pos_to_predict = pos.loc[pp[0]:pp[1]]
    
    pos_all = add_dataframe_to_dataframe_reindex(pos_all, pos_to_predict)
    
pos_all.to_csv(pos_dir / f'pos_{version_name}.csv')
pos_all.to_parquet(pos_dir / f'pos_{version_name}.parquet')


# %%
for test_info in final_test_list:
    mode = test_info['mode']
    test_name = test_info['test_name']
    date_start = test_info.get('date_start')

    if mode == 'test':
        test_class = FactorTesterByContinuous
    elif mode == 'trade':
        test_class = FactorTesterByDiscrete
    else:
        raise NotImplementedError()

    ft = test_class(None, None, pos_dir, test_name=test_name, result_dir=rolling_model_dir)
    ft.test_one_factor(f'pos_{version_name}')