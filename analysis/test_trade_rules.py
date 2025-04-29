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


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config
from test_and_eval.factor_tester import FactorTesterByDiscrete
from test_and_eval.scores import get_general_return_metrics


# %%
model_name = 'avg_agg_250218_3_fix_tfe_by_trade_net_v4'
suffix = '1_2_0'
process_name = 'final_predict'
tag_name = 'test_trade_rules'
factor_data_dir = rf'D:/mnt/CNIndexFutures/timeseries/factor_test/results/model/{model_name}'
test_name = 'trade_ver0_futtwap_sp1min_s240d_icim'
pred_name = f'predict_{model_name}'
multi_test_name = 'trade_ver3_futtwap_sp1min_s240d_icim'


# %%
path_config = load_path_config(project_dir)
param_dir = Path(path_config['param'])


# %%
result_dir = Path(factor_data_dir)
summary_dir = result_dir / 'trade_rule_summary'
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
# openthres = 0.6
# closethres = 0.0

# =============================================================================
# for openthres in tqdm(openthres_list, desc='test_open'):
#     for closethres in closethres_list:
#         test_pr = copy.deepcopy(test_param)
#         test_pr['trade_rule_param']['openthres'] = openthres
#         test_pr['trade_rule_param']['closethres'] = closethres
#         
#         new_test_name = f'{test_name}_op{openthres}_cl{closethres}'
#         
#         tester = FactorTesterByDiscrete(process_name, tag_name, factor_data_dir, test_name=new_test_name, 
#                                         result_dir=result_dir, params=test_pr)
#         tester.test_multi_factors()
# =============================================================================
        
        
# %%
fee = 0.00024

res_list = []
net_list = []
for openthres in openthres_list:
    for closethres in closethres_list:
        res_dict = {
            'openthres': openthres, 
            'closethres': closethres, 
            }
        new_test_name = f'{test_name}_op{openthres}_cl{closethres}'
        test_data_dir = result_dir / 'test' / new_test_name / tag_name / process_name / 'data'
        
        test_data = {}
        for data_type in ('gpd', 'hsr'):
            data_path = test_data_dir / f'{data_type}_{pred_name}.pkl'
            with open(data_path, 'rb') as f:
                test_data[data_type] = pickle.load(f)
                
        df_gp = test_data['gpd']['all']
        df_hsr = test_data['hsr']['all']
        
        net = (df_gp['return'] - fee * df_hsr['avg']).fillna(0)
        metrics = get_general_return_metrics(net)
        renamed_metrics = {f'net_{m}': v for m, v in metrics.items()}
        res_dict.update(renamed_metrics)
        
        profit_per_trade = df_gp["return"].sum() / df_hsr["avg"].sum()
        res_dict.update({'profit_per_trade': profit_per_trade*1000})
        
        res_list.append(res_dict)
        net_list.append(net)
        
res_df = pd.DataFrame(res_list)
        
        
# %%
import seaborn as sns
import matplotlib.pyplot as plt

# 直接将 'openthres' 和 'closethres' 设置为索引
res_df_cp = res_df.copy()
heatmap_data = res_df_cp.set_index(['openthres', 'closethres'])

# 绘制每一列的热力图，并保存到 summary_dir
for column in heatmap_data.columns:
    plt.figure(figsize=(10, 8))
    
    # 使用 unstack() 将数据转换为适合绘制热力图的格式
    sns.heatmap(heatmap_data[column].unstack(), annot=True, cmap='coolwarm', fmt='.2f')

    # 调整轴标签和标题
    plt.title(f'Heatmap for {column}', pad=15)
    plt.xlabel('closethres')  # 原来是 openthres，现在改为 closethres
    plt.ylabel('openthres')  # 原来是 closethres，现在改为 openthres
    
    # 调整图像布局，避免标签遮挡
    plt.tight_layout()

    # 保存图像
    img_filename = summary_dir / f'{column}_heatmap.png'
    plt.savefig(img_filename)
    plt.show()
    plt.close()  # 关闭当前图像，防止重叠
        
        
# %%
# =============================================================================
# # 将 net_list 转换为 DataFrame
# # 每个 net_list 元素是一个 pandas Series，所以我们将它们组合成一个 DataFrame
# net_df = pd.DataFrame({f'net_{i}': net for i, net in enumerate(net_list)})
# 
# # 计算两两之间的相关性
# corr_matrix = net_df.corr()
# 
# # 绘制相关性热力图
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', 
#             xticklabels=False, yticklabels=False)
# 
# # 添加标题
# plt.title('Correlation Heatmap for net_list')
# 
# # 调整图像布局，避免标签遮挡
# plt.tight_layout()
# 
# # 保存图像
# corr_img_filename = summary_dir / 'correlation_heatmap.png'
# plt.savefig(corr_img_filename)
# plt.show()
# plt.close()  # 关闭当前图像，防止重叠
# =============================================================================


# %%
# =============================================================================
# # 设置 'openthres' 和 'closethres' 为索引
# heatmap_data = res_df.set_index(['openthres', 'closethres'])
# 
# # 定义不同版本的参数字典
# version_dict = {
#     # "v1": {
#     #     'net_sharpe_ratio': 2.0,
#     #     'profit_per_trade': 0.5,
#     #     'net_max_dd': 0.2,
#     #     'net_burke_ratio': 10,
#     # },
#     # "v2": {
#     #     'net_sharpe_ratio': 2.0,
#     #     'profit_per_trade': 0.5,
#     #     'net_max_dd': 0.18,
#     #     'net_burke_ratio': 10,
#     # },
#     # "v3": {
#     #     'net_sharpe_ratio': 2.0,
#     #     'profit_per_trade': 0.5,
#     #     'net_max_dd': 0.15,
#     #     'net_burke_ratio': 10,
#     # },
#     # "v4": {
#     #     'net_sharpe_ratio': 2.0,
#     #     'profit_per_trade': 1.0,
#     #     'net_max_dd': 0.2,
#     #     'net_burke_ratio': 10,
#     # },
#     # "v5": {
#     #     'net_sharpe_ratio': 2.0,
#     #     'profit_per_trade': 1.5,
#     #     'net_max_dd': 0.2,
#     #     'net_burke_ratio': 10,
#     # },
#     f"v6_{suffix}": {
#         'neighbor': True,
#         'net_sharpe_ratio': 1.9,
#         'profit_per_trade': 1.2,
#         'net_max_dd': 0.2,
#         'net_burke_ratio': 5,
#     },
# }
# 
# # 自动生成筛选条件
# def generate_condition(data, params):
#     condition = (
#         (data['net_sharpe_ratio'] > params['net_sharpe_ratio']) & 
#         (data['profit_per_trade'] > params['profit_per_trade']) & 
#         (data['net_max_dd'] < params['net_max_dd']) & 
#         (data['net_burke_ratio'] > params['net_burke_ratio'])
#     )
#     return condition
# 
# # def generate_condition_with_neighbors(data, params, openthres_list, closethres_list):
# #     """
# #     根据相邻参数生成条件，判断当前参数与其相邻参数是否满足基本条件。
# #     如果相邻参数不存在，则默认满足条件。
# #     """
# #     condition = pd.Series(True, index=data.index)  # 初始化条件为True
#     
# #     # 获取 openthres 和 closethres 的索引位置
# #     openthres_idx = {v: i for i, v in enumerate(openthres_list)}
# #     closethres_idx = {v: i for i, v in enumerate(closethres_list)}
# 
# #     for idx, row in data.iterrows():
# #         openthres, closethres = row.name
#         
# #         # 获取相邻的 openthres 和 closethres 的位置（前后一个位置）
# #         neighboring_openthres = [
# #             openthres_list[openthres_idx[openthres] - 1] if openthres_idx[openthres] > 0 else None,
# #             openthres_list[openthres_idx[openthres] + 1] if openthres_idx[openthres] + 1 < len(openthres_list) else None
# #         ]
#         
# #         neighboring_closethres = [
# #             closethres_list[closethres_idx[closethres] - 1] if closethres_idx[closethres] > 0 else None,
# #             closethres_list[closethres_idx[closethres] + 1] if closethres_idx[closethres] + 1 < len(closethres_list) else None
# #         ]
#         
# #         # 获取相邻参数的值，避免索引越界
# #         neighbors = []
# #         for neighbor_op in neighboring_openthres:
# #             for neighbor_cl in neighboring_closethres:
# #                 if neighbor_op is not None and neighbor_cl is not None:
# #                     neighbors.append(data.loc[(neighbor_op, neighbor_cl)])
#         
# #         # 检查相邻参数
# #         for neighbor in neighbors:
# #             # 检查相邻参数是否满足条件，若不满足条件，则当前参数也不满足
# #             if not (
# #                 neighbor['net_sharpe_ratio'] > params['net_sharpe_ratio'] and
# #                 neighbor['profit_per_trade'] > params['profit_per_trade'] and
# #                 neighbor['net_max_dd'] < params['net_max_dd'] and
# #                 neighbor['net_burke_ratio'] > params['net_burke_ratio']
# #             ):
# #                 condition[idx] = False
# #                 break
#         
# #         # 检查当前参数是否满足条件，若不满足，则该位置也为 False
# #         if not (
# #             row['net_sharpe_ratio'] > params['net_sharpe_ratio'] and
# #             row['profit_per_trade'] > params['profit_per_trade'] and
# #             row['net_max_dd'] < params['net_max_dd'] and
# #             row['net_burke_ratio'] > params['net_burke_ratio']
# #         ):
# #             condition[idx] = False
# 
# #     return condition
# 
# def generate_condition_with_neighbors(data, params, openthres_list, closethres_list):
#     """
#     根据相邻参数生成条件，判断当前参数与其相邻参数是否满足基本条件。
#     如果相邻参数不存在，则默认满足条件。
#     """
#     condition = pd.Series(True, index=data.index)  # 初始化条件为True
#     
#     # 获取 openthres 和 closethres 的索引位置
#     openthres_idx = {v: i for i, v in enumerate(openthres_list)}
#     closethres_idx = {v: i for i, v in enumerate(closethres_list)}
# 
#     for idx, row in data.iterrows():
#         openthres, closethres = row.name
#         
#         # 获取相邻的 openthres 和 closethres 的位置，根据 openthres_idx 和 closethres_idx 来偏移
#         # 直接列出五种情况
#         neighbors = [
#             data.loc[(openthres, closethres)],  # 当前值 (open, close)
#             data.loc[(openthres_list[openthres_idx[openthres] + 1] if openthres_idx[openthres] + 1 < len(openthres_list) else openthres, closethres)],  # open + 1
#             data.loc[(openthres_list[openthres_idx[openthres] - 1] if openthres_idx[openthres] - 1 >= 0 else openthres, closethres)],  # open - 1
#             data.loc[(openthres, closethres_list[closethres_idx[closethres] + 1] if closethres_idx[closethres] + 1 < len(closethres_list) else closethres)],  # close + 1
#             data.loc[(openthres, closethres_list[closethres_idx[closethres] - 1] if closethres_idx[closethres] - 1 >= 0 else closethres)]  # close - 1
#         ]
#         
#         # 检查相邻参数
#         for neighbor in neighbors:
#             # 检查相邻参数是否满足条件，若不满足条件，则当前参数也不满足
#             if not (
#                 neighbor['net_sharpe_ratio'] > params['net_sharpe_ratio'] and
#                 neighbor['profit_per_trade'] > params['profit_per_trade'] and
#                 neighbor['net_max_dd'] < params['net_max_dd'] and
#                 neighbor['net_burke_ratio'] > params['net_burke_ratio']
#             ):
#                 condition[idx] = False
#                 break
#         
#         # 检查当前参数是否满足条件，若不满足，则该位置也为 False
#         if not (
#             row['net_sharpe_ratio'] > params['net_sharpe_ratio'] and
#             row['profit_per_trade'] > params['profit_per_trade'] and
#             row['net_max_dd'] < params['net_max_dd'] and
#             row['net_burke_ratio'] > params['net_burke_ratio']
#         ):
#             condition[idx] = False
# 
#     return condition
# 
# # 生成条件的描述文本
# def generate_conditions_text(params):
#     conditions_text = "Filter conditions:\n"
#     for key, value in params.items():
#         if key == 'net_max_dd':
#             conditions_text += f"- {key} < {value}\n"
#         else:
#             conditions_text += f"- {key} > {value}\n"
#     return conditions_text
# 
# # 遍历每个版本，生成热力图和条件文本
# for version, params in version_dict.items():
#     # 生成condition并输出条件文本
#     if params.get('neighbor'):
#         condition = generate_condition_with_neighbors(heatmap_data, params, list(reversed(openthres_list)), closethres_list)
#     else:
#         condition = generate_condition(heatmap_data, params)
#     conditions_text = generate_conditions_text(params)
# 
#     # 提取符合条件的 openthres 和 closethres 的组合
#     valid_pairs = condition[condition].index.tolist()
#     print(f"Version {version} valid pairs: {valid_pairs}")
# 
#     # 根据筛选条件绘制热力图并保存到 summary_dir
#     mask = ~condition.unstack()  # 转换成unstack后的形状，并反转
# 
#     for column in heatmap_data.columns:
#         fig, ax = plt.subplots(figsize=(12, 8))  # 增加宽度，预留文本区域
#         
#         # 使用 unstack() 将数据转换为适合绘制热力图的格式
#         heatmap_matrix = heatmap_data[column].unstack()
#         
#         # 应用 mask
#         sns.heatmap(heatmap_matrix, annot=True, cmap='coolwarm', fmt='.2f', mask=mask, ax=ax)
# 
#         # 调整轴标签和标题
#         plt.title(f'Masked Heatmap for {column} (filtered by conditions) - {version}')
#         plt.xlabel('closethres')
#         plt.ylabel('openthres')
# 
#         # 在图右侧添加过滤条件文本
#         plt.figtext(1.05, 0.5, conditions_text, fontsize=12, verticalalignment="center", bbox=dict(facecolor='white', alpha=0.5))
# 
#         # 调整图像布局，避免标签遮挡
#         plt.tight_layout()
# 
#         # 保存图像
#         img_filename = summary_dir / f'{version}_{column}_masked_heatmap.png'
#         plt.savefig(img_filename, bbox_inches='tight')  # bbox_inches='tight' 确保文本不被裁剪
#         plt.show()
#         plt.close()  # 关闭当前图像，防止重叠
#         
#     test_pr = copy.deepcopy(test_param)
#     test_pr['trade_rule_name'] = 'trade_rule_by_trigger_v3'
#     test_pr['trade_rule_param'] = {'threshold_combinations': valid_pairs}
#     
#     new_test_name = f'{multi_test_name}_{version}'
#     
#     with open(param_dir / 'test' / f"{new_test_name}.toml", "w") as f:
#         toml.dump(test_pr, f)
#     
#     tester = FactorTesterByDiscrete(process_name, tag_name, factor_data_dir, test_name=new_test_name, 
#                                     result_dir=result_dir, params=test_pr)
#     tester.test_multi_factors()
# # breakpoint()
# =============================================================================
    
# %%
# =============================================================================
# version_dir_dict = {}
# version_dir_dict[test_name] = result_dir / 'test' / test_name / 'data'
# for version in version_dict:
#     new_test_name = f'{multi_test_name}_{version}'
#     test_data_dir = result_dir / 'test' / new_test_name / tag_name / process_name / 'data'
#     version_dir_dict[new_test_name] = test_data_dir
# 
# 
# final_res_list = []
# for version, version_dir in version_dir_dict.items():
#     res_dict = {
#         'version': version, 
#         }
#     new_test_name = f'{multi_test_name}_{version}'
#     test_data_dir = version_dir
#     
#     test_data = {}
#     for data_type in ('gpd', 'hsr'):
#         data_path = test_data_dir / f'{data_type}_{pred_name}.pkl'
#         with open(data_path, 'rb') as f:
#             test_data[data_type] = pickle.load(f)
#             
#     df_gp = test_data['gpd']['all']
#     df_hsr = test_data['hsr']['all']
#     
#     net = (df_gp['return'] - fee * df_hsr['avg']).fillna(0)
#     metrics = get_general_return_metrics(net)
#     renamed_metrics = {f'net_{m}': v for m, v in metrics.items()}
#     res_dict.update(renamed_metrics)
#     
#     profit_per_trade = df_gp["return"].sum() / df_hsr["avg"].sum()
#     res_dict.update({'profit_per_trade': profit_per_trade*1000})
#     
#     final_res_list.append(res_dict)
#     
# final_res = pd.DataFrame(final_res_list)
# final_res.to_csv(summary_dir / 'final_version_comparison.csv', index=None)
# =============================================================================


# %%
from collections import defaultdict

(summary_dir / 'by_year').mkdir(parents=True, exist_ok=True)

res_dict_by_year = defaultdict(list)
net_list_by_year = defaultdict(list)

for openthres in openthres_list:
    for closethres in closethres_list:
        new_test_name = f'{test_name}_op{openthres}_cl{closethres}'
        test_data_dir = result_dir / 'test' / new_test_name / tag_name / process_name / 'data'

        test_data = {}
        for data_type in ('gpd', 'hsr'):
            data_path = test_data_dir / f'{data_type}_{pred_name}.pkl'
            with open(data_path, 'rb') as f:
                test_data[data_type] = pickle.load(f)

        df_gp = test_data['gpd']['all']
        df_hsr = test_data['hsr']['all']

        # Extract year from the date column (assuming 'date' exists in df_gp and df_hsr)
        df_gp['year'] = df_gp.index.year
        df_hsr['year'] = df_hsr.index.year

        for year in df_gp['year'].unique():
            # Filter data for the specific year
            df_gp_year = df_gp[df_gp['year'] == year]
            df_hsr_year = df_hsr[df_hsr['year'] == year]

            net = (df_gp_year['return'] - fee * df_hsr_year['avg']).fillna(0)
            metrics = get_general_return_metrics(net)
            renamed_metrics = {f'net_{m}': v for m, v in metrics.items()}

            # Create dictionary for each year
            res_dict = {
                'openthres': openthres,
                'closethres': closethres,
                'year': year,
            }
            res_dict.update(renamed_metrics)

            profit_per_trade = df_gp_year["return"].sum() / df_hsr_year["avg"].sum()
            res_dict.update({'profit_per_trade': profit_per_trade * 1000})

            res_dict_by_year[year].append(res_dict)
            net_list_by_year[year].append(net)

# Now, create a DataFrame for each year and plot the heatmaps
for year, res_list in res_dict_by_year.items():
    res_df = pd.DataFrame(res_list)

    # Prepare data for heatmap (set 'openthres' and 'closethres' as indices)
    res_df_cp = res_df.copy()
    heatmap_data = res_df_cp.set_index(['openthres', 'closethres'])

    # Generate heatmaps for each column
    for column in heatmap_data.columns:
        plt.figure(figsize=(10, 8))

        # Unstack the data to create a 2D matrix for the heatmap
        sns.heatmap(heatmap_data[column].unstack(), annot=True, cmap='coolwarm', fmt='.2f')

        # Set axis labels and title
        plt.title(f'Heatmap for {column} (Year {year})', pad=15)
        plt.xlabel('closethres')
        plt.ylabel('openthres')

        # Adjust layout
        plt.tight_layout()

        # Save the heatmap image
        img_filename = summary_dir / 'by_year' / f'{column}_heatmap_year{year}.jpg'
        plt.savefig(img_filename)
        plt.show()
        plt.close()  # Close the figure to avoid overlap with the next one

