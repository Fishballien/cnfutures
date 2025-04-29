# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:53:55 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import pandas as pd


# %%
def filter_func_v0(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 2)
                      , 1)
        
        
def filter_func_v1(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 2)
                      & (r['return_annualized'] / r['hsr'] > 0.145)
                      , 1)
        
        
def filter_func_v2(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 1.75)
                      , 1)
        
            
def filter_func_v3(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio'] > 1.75)
                      , 1)
        
        
def filter_func_v4(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio'] > 1.6)
                      , 1)
        
        
def filter_func_v5(data):
    return data.apply(lambda r:
                      (r['net_sortino_ratio'] > 1.5)
                      , 1)
        

def filter_func_v6(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio'] > 1.8)
                      , 1)
        
        
def filter_func_v7(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio'] > 1.9)
                      , 1)
        
        
def filter_func_v8(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio'] > 2.0)
                      , 1)
        
def filter_func_v9(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio'] > 2.2)
                      , 1)
    
def filter_func_v10(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio'] > 1.8)
                      & (r['net_return_annualized'] / r['hsr'] / 245 > 0.001)
                      , 1)
        
def filter_func_v11(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio'] > 1.8)
                      & (r['net_return_annualized'] / r['hsr'] / 245 > 0.0012)
                      , 1)
        
def filter_func_v12(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio'] > 1.8)
                      & (r['net_calmar_ratio'] > 2.5)
                      , 1)
        
def filter_func_v13(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio'] > 1.8)
                      & (r['net_calmar_ratio'] > 4)
                      , 1)
        
def filter_func_v14(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio'] > 1.8)
                      & (r['net_calmar_ratio'] > 2.5)
                      & (r['net_return_annualized'] / r['hsr'] / 245 > 0.0012)
                      , 1)
        
def filter_func_v15(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio'] > 1.8)
                      & (r['net_calmar_ratio'] > 4)
                      & (r['net_return_annualized'] / r['hsr'] / 245 > 0.001)
                      , 1)
        
def filter_func_v16(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio'] > 1.0)
                      & (r['direction'] > 0)
                      , 1)
        
def filter_func_v17(data, min_count=3):
    # Apply original filter
    primary_filter = data.apply(lambda r:
                       (r['net_sharpe_ratio'] > 0.9)
                       & (r['direction'] > 0)
                       , 1)
    
    filtered_data = data[primary_filter]
    
    # If filtered results are fewer than min_count, select top n by net_sharpe_ratio
    if len(filtered_data) < min_count:
        # Sort by net_sharpe_ratio in descending order and take top min_count
        top_n = data[data['direction'] > 0].sort_values('net_sharpe_ratio', ascending=False).head(min_count)
        return data.index.isin(top_n.index)
    else:
        return primary_filter
    
    
def filter_func_v18(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio'] > 1.4)
                      , 1)
    

def filter_func_dynamic(data, pred_name=None, conditions=None, min_count=3, sort_target='net_sharpe_ratio', sort_ascending=False):
    """
    Filter data based on comparing values to the row where pred_name matches.
    
    Parameters:
    data (DataFrame): Input dataframe to filter
    pred_name (str, optional): The specific pred_name to use as reference. If None, will use absolute thresholds.
    conditions (list): List of dictionaries with keys:
                     - 'target': column name to compare (e.g., 'net_sharpe_ratio')
                     - 'operator': 'greater', 'less', 'equal', 'greater_equal', 'less_equal'
                     - 'threshold': absolute value or multiplier of reference value
                     - 'is_multiplier': boolean, if True, threshold is treated as multiplier of reference value
    min_count (int): Minimum number of results to return
    sort_target (str): Column name to use for sorting when selecting top n results
    sort_ascending (bool): Sort direction (False for descending, True for ascending)
    
    Returns:
    Series: Boolean mask to filter the dataframe
    """
    # Get reference row values if pred_name is provided
    reference_row = None
    if pred_name is not None:
        reference = data[data['pred_name'] == pred_name]
        
        if len(reference) > 0:
            reference_row = reference.iloc[0]
        else:
            print(f"Warning: No data found for pred_name '{pred_name}', will use absolute thresholds")
    
    # Default condition if none provided
    if conditions is None:
        conditions = [
            {'target': 'net_sharpe_ratio', 'operator': 'greater', 'threshold': 0.9, 'is_multiplier': False},
            {'target': 'direction', 'operator': 'greater', 'threshold': 0, 'is_multiplier': False}
        ]
    
    # Build filter based on conditions
    filter_mask = pd.Series(True, index=data.index)
    
    for condition in conditions:
        target = condition['target']
        operator = condition['operator']
        threshold = condition['threshold']
        is_multiplier = condition.get('is_multiplier', False)
        
        # If this is a multiplier and we have a reference row, calculate actual threshold
        if is_multiplier and reference_row is not None:
            if target in reference_row:
                actual_threshold = reference_row[target] * threshold
            else:
                print(f"Warning: Target '{target}' not found in reference row, using absolute threshold")
                actual_threshold = threshold
        else:
            actual_threshold = threshold
        
        # Apply comparison
        if operator == 'greater':
            filter_mask &= data[target] > actual_threshold
        elif operator == 'less':
            filter_mask &= data[target] < actual_threshold
        elif operator == 'equal':
            filter_mask &= data[target] == actual_threshold
        elif operator == 'greater_equal':
            filter_mask &= data[target] >= actual_threshold
        elif operator == 'less_equal':
            filter_mask &= data[target] <= actual_threshold
        else:
            raise ValueError(f"Invalid operator: {operator}")
    
    filtered_data = data[filter_mask]
    
    # If filtered results are fewer than min_count, select top n by the specified sort_target
    if len(filtered_data) < min_count:
        # Check if sort_target is in the dataframe
        if sort_target not in data.columns:
            # Choose first numerical column as fallback
            num_cols = data.select_dtypes(include='number').columns
            if len(num_cols) > 0:
                sort_col = num_cols[0]
            else:
                raise ValueError(f"Sort target '{sort_target}' not found and no numerical column available for sorting")
        else:
            sort_col = sort_target
            
        # Sort by sort_col in the specified direction and take top min_count
        top_n = data.sort_values(sort_col, ascending=sort_ascending).head(min_count)
        return data.index.isin(top_n.index)
    else:
        return filter_mask
    
        
def filter_long_only_v0(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio_long_only'] > 1.5)
                      , 1)
        
def filter_short_only_v0(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio_short_only'] > 1.5)
                      , 1)
        
def filter_short_only_v1(data):
    return data.apply(lambda r:
                      (r['net_sharpe_ratio_short_only'] > 1.7)
                      , 1)

def filter_func_rec_v0(data):
    return data.apply(lambda r:
                      (r['sharpe_ratio'] > 0)
                      , 1)
        
        
        