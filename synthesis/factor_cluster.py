# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:09:24 2024

@author: Xintang Zheng

"""
# %% imports
import yaml
import toml
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as spc
import pickle
from functools import partial


from utils.timeutils import period_shortcut
from utils.datautils import align_to_primary, check_dataframe_consistency
from synthesis.filter_methods import *


# %%
class Cluster:
    
    def __init__(self, cluster_name, check_consistency=True, base_cluster_name=None):
        self.cluster_name = cluster_name
        self.check_consistency = check_consistency
        self.base_cluster_name = base_cluster_name

        self._load_paths()
        self._load_cluster_params()
        self._load_test_name()
        self._init_dirs()
        self._init_filter_func()
        
    def _load_paths(self):
        file_path = Path(__file__).resolve()
        file_dir = file_path.parents[1]
        path_config_path = file_dir / '.path_config.yaml'
        with path_config_path.open('r') as file:
            path_config = yaml.safe_load(file)

        self.result_dir = Path(path_config['result'])
        self.param_dir = Path(path_config['param'])
        
    def _load_cluster_params(self):
        param_path = (self.param_dir / 'cluster' / f'{self.cluster_name}.toml'
                      if self.base_cluster_name is None
                      else self.param_dir / 'cluster' / self.base_cluster_name / f'{self.cluster_name}.toml')
        self.params = toml.load(param_path)
        
    def _load_test_name(self):
        feval_name = self.params.get('feval_name')
        pool_name = self.params.get('pool_name')
        
        if feval_name is not None and pool_name is None:
            feval_params = toml.load(self.param_dir / 'feval' / f'{feval_name}.toml')
            self.test_name = feval_params['test_name']
        elif pool_name is not None and feval_name is None:
            pool_param_dir = self.param_dir / 'features_of_factors' / 'generate'
            pool_params = toml.load(pool_param_dir / f'{pool_name}.toml')
            self.test_name = pool_params['test_data']['test_name']
        
    def _init_dirs(self):
        feval_name = self.params.get('feval_name')
        
        self.feval_dir = (self.result_dir / 'factor_evaluation' / feval_name 
                          if feval_name is not None else None)
        self.cluster_dir = (self.result_dir / 'cluster' / f'{self.cluster_name}'
                            if self.base_cluster_name is None
                            else self.result_dir / 'cluster' / self.base_cluster_name / f'{self.cluster_name}')
        self.cluster_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir = self.result_dir / 'cluster' / 'debug' / f'{self.cluster_name}'
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir = self.result_dir / 'test'
        
    def _init_filter_func(self):
        filter_name = self.params['filter_func']
        filter_func_param = self.params.get('filter_func_param')
        rec_filter_name = self.params.get('rec_filter_func', None)
        super_rec_filter_name = self.params.get('super_rec_filter_func', None)
        filter_func = globals()[filter_name]
        self.filter_func = partial(filter_func, **filter_func_param) if filter_func_param is not None else filter_func
        self.rec_filter_func = globals()[rec_filter_name] if rec_filter_name is not None else None
        self.super_rec_filter_func = globals()[super_rec_filter_name] if super_rec_filter_name is not None else None

    def cluster_one_period(self, date_start, date_end, rec_date_start, rec_date_end,
                           super_rec_date_start, super_rec_date_end):
        filter_func = self.filter_func
        rec_filter_func = self.rec_filter_func
        super_rec_filter_func = self.super_rec_filter_func
        feval_dir = self.feval_dir
        cluster_dir = self.cluster_dir
        params = self.params
        feval_name = self.params.get('feval_name')
        corr_target = params.get('corr_target')
        pool_name = params.get('pool_name')
        cluster_params = params.get('cluster_params')
        linkage_method = params.get('linkage_method', 'average')
        
        period_name = period_shortcut(date_start, date_end)
        
        if feval_name is not None and pool_name is None:
            factor_eval = pd.read_csv(feval_dir / f'factor_eval_{period_name}.csv')
        elif pool_name is not None and feval_name is None:
            print(date_end, self.predict[self.predict.index <= date_end])
            latest_direction = self.directions[self.directions.index <= date_end].iloc[-1]
            latest_pred = self.predict[self.predict.index <= date_end].iloc[-1]
            factor_eval = self.factor_mapping.copy()
            factor_eval['direction'] = latest_direction
            factor_eval['predict'] = latest_pred
        else:
            raise NotImplementedError()
        
        selected_idx = filter_func(factor_eval)
        if rec_filter_func is not None:
            rec_period_name = period_shortcut(rec_date_start, rec_date_end)
            rec_factor_eval = pd.read_csv(feval_dir / f'factor_eval_{rec_period_name}.csv')
            rec_factor_eval = align_to_primary(factor_eval, rec_factor_eval, 'process_name', 'factor')
            selected_idx_rec = rec_filter_func(rec_factor_eval)
            selected_idx = selected_idx & selected_idx_rec
        if super_rec_filter_func is not None:
            super_rec_period_name = period_shortcut(super_rec_date_start, super_rec_date_end)
            super_rec_factor_eval = pd.read_csv(feval_dir / f'factor_eval_{super_rec_period_name}.csv')
            super_rec_factor_eval = align_to_primary(factor_eval, super_rec_factor_eval, 'process_name', 'factor')
            selected_idx_super_rec = super_rec_filter_func(super_rec_factor_eval)
            selected_idx = selected_idx & selected_idx_super_rec
        info_list = ['root_dir',  'test_name', 'tag_name', 'process_name', 'factor', 'direction']
        selected_factor_info = factor_eval[selected_idx][info_list].reset_index(drop=True)
        
        if corr_target is not None:
            if corr_target == 'factor':
                distance_matrix = self._calc_corr_by_factor_value(selected_factor_info, period_name, date_start, date_end)
            elif corr_target == 'gpd':
                distance_matrix = self._calc_corr_by_gp(selected_factor_info, period_name, date_start, date_end)
                    
            try:
                condensed_distance_matrix = squareform(distance_matrix)
            except:
                breakpoint()
            try:
                linkage = spc.linkage(condensed_distance_matrix, method=linkage_method) # complete # average
            except:
                breakpoint()
            idx = spc.fcluster(linkage, **cluster_params)
        else:
            idx = list(range(len(selected_factor_info)))
        
        selected_factor_info['group'] = idx
        selected_factor_info.to_csv(cluster_dir / f'cluster_info_{period_name}.csv', index=None)
        
    def save_cluster_info(self, selected_factor_info, period_name):
        """
        保存聚类因子信息到CSV文件，如果debug_dir存在且已有相同文件，则先检查一致性
        
        参数:
        selected_factor_info (pd.DataFrame): 选定的因子信息DataFrame
        period_name (str): 时间段名称
        """
        # 定义保存路径
        cluster_dir = self.save_dir / 'clusters'  # 假设cluster_dir是在self.save_dir下的子目录
        save_path = cluster_dir / f'cluster_info_{period_name}.csv'
        
        # 检查是否需要进行一致性检查
        if self.check_consistency and hasattr(self, 'debug_dir') and save_path.exists():
            try:
                # 读取已存在的数据
                existing_data = pd.read_csv(save_path)
                
                # 检查一致性
                status, info = check_dataframe_consistency(existing_data, selected_factor_info)
                
                if status == "INCONSISTENT":
                    # 保存不一致的数据到debug目录
                    debug_path = self.debug_dir / f'cluster_info_{period_name}_inconsistent.csv'
                    selected_factor_info.to_csv(debug_path, index=None)
                    
                    # 构造错误信息
                    error_msg = f"DataFrame一致性检查失败! 索引: {info['index']}, 列: {info['column']}, "
                    error_msg += f"原始值: {info['original_value']}, 新值: {info['new_value']}, "
                    error_msg += f"不一致计数: {info['inconsistent_count']}。已保存到 {debug_path}"
                    
                    raise ValueError(error_msg)
            except Exception as e:
                if not isinstance(e, ValueError):  # 如果不是我们自己抛出的ValueError，则记录异常但继续执行
                    print(f"一致性检查过程中发生异常: {str(e)}")
        
        # 确保目录存在
        cluster_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存新数据
        selected_factor_info.to_csv(save_path, index=None)
        
    def _calc_corr_by_gp(self, selected_factor_info, period_name, date_start, date_end):
        params = self.params
        data_to_use = params.get('data_to_use', 'gpd')
        col = params['col']
        test_dir = self.test_dir
        
        date_range = pd.date_range(start=date_start, end=date_end, freq='D')

        gp_list = []
        for n_fct in selected_factor_info.index:
            tag_name, process_name, factor_name = selected_factor_info.loc[n_fct, ['tag_name', 'process_name', 'factor']]
            process_dir = (test_dir / self.test_name / tag_name if isinstance(tag_name, str)
                          else test_dir / self.test_name)
            data_dir = process_dir / process_name / 'data' 
            try:
                data_path = data_dir / f'{data_to_use}_{factor_name}.pkl'
                with open(data_path, 'rb') as f:
                    data_dict = pickle.load(f)
                data = data_dict['all']
                gp_series = data[(data.index >= date_start) & (data.index <= date_end)
                                 ].reindex(date_range)[col].fillna(0)
            except:
                print('missing', f'{data_to_use}_{factor_name}.parquet')
                gp_series = pd.Series(np.zeros(len(date_range)), index=date_range)
            gp_list.append(gp_series)
            
        gp_matrix = np.array(gp_list)
        corr_matrix = np.corrcoef(gp_matrix)
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # 处理数据精度问题
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # 强制对称性
        for i in range(distance_matrix.shape[0]):
            for j in range(i + 1, distance_matrix.shape[1]):
                avg_value = (distance_matrix[i, j] + distance_matrix[j, i]) / 2
                distance_matrix[i, j] = avg_value
                distance_matrix[j, i] = avg_value
        
        # 确保对角线为零
        np.fill_diagonal(distance_matrix, 0)

        return distance_matrix
    
    
# %%
def cluster_by_correlation(data_array, cluster_params=None, linkage_method='average'):
    """
    Calculate correlation distances for an array and output clustering groups.
    
    Parameters:
    -----------
    data_array : numpy.ndarray
        Input array with shape (n_samples, n_features) where each row is a data point to cluster.
    cluster_params : dict, optional
        Parameters to pass to scipy.cluster.hierarchy.fcluster.
        Example: {'t': 0.5, 'criterion': 'distance'}
    linkage_method : str, optional
        Method for calculating linkage. Options include 'single', 'complete', 'average', 'weighted', etc.
        Default is 'average'.
        
    Returns:
    --------
    tuple
        (cluster_indices, distance_matrix, linkage)
        - cluster_indices: array of cluster assignments for each data point
        - distance_matrix: distance matrix based on correlation
        - linkage: hierarchical clustering linkage matrix
    """
    if cluster_params is None:
        cluster_params = {'t': 0.5, 'criterion': 'distance'}
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(data_array)
    
    # Convert to distance matrix (1 - |correlation|)
    distance_matrix = 1 - np.abs(corr_matrix)
    
    # Handle numerical precision issues and ensure symmetry
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # Ensure diagonal elements are zero
    np.fill_diagonal(distance_matrix, 0)
    
    # Convert to condensed distance matrix for linkage calculation
    try:
        condensed_distance_matrix = squareform(distance_matrix)
    except ValueError as e:
        # Handle potential errors in distance matrix
        print(f"Error in distance matrix conversion: {e}")
        # Try to fix common issues with distance matrices
        for i in range(distance_matrix.shape[0]):
            for j in range(i+1, distance_matrix.shape[1]):
                avg_value = (distance_matrix[i, j] + distance_matrix[j, i]) / 2
                distance_matrix[i, j] = avg_value
                distance_matrix[j, i] = avg_value
        np.fill_diagonal(distance_matrix, 0)
        condensed_distance_matrix = squareform(distance_matrix)
    
    # Calculate linkage
    try:
        linkage = spc.linkage(condensed_distance_matrix, method=linkage_method)
    except Exception as e:
        print(f"Error in linkage calculation: {e}")
        raise
    
    # Form flat clusters
    cluster_indices = spc.fcluster(linkage, **cluster_params)
    
    return cluster_indices, distance_matrix, linkage

# Example usage:
# data = np.random.rand(10, 100)  # 10 samples, 100 features each
# cluster_params = {'t': 0.5, 'criterion': 'distance'}
# clusters, distances, linkage = cluster_by_correlation(data, cluster_params)


# %%
def extract_gp_matrix(selected_factor_info, date_start, date_end, test_dir, data_to_use='gpd', use_direction='all',
                      col='return'):
    """
    从文件中提取指定因子和时间段的GP矩阵。
    
    参数:
    -----------
    selected_factor_info : pandas.DataFrame
        包含所选因子信息的DataFrame，必须包含以下列:
        ['tag_name', 'process_name', 'factor', 'test_name', 'direction']
        
    date_start : str
        起始日期，格式为'YYYY-MM-DD'
        
    date_end : str
        结束日期，格式为'YYYY-MM-DD'
        
    test_dir : Path or str
        包含测试数据的目录路径
        
    data_to_use : str, optional
        要使用的数据类型，默认为'gpd'
        
    use_direction : str, optional
        使用的方向类型，可选值为:
        - 'all': 使用所有数据
        - 'long_only': 仅使用多头数据（根据direction值确定是pos还是neg）
        - 'short_only': 仅使用空头数据（根据direction值确定是neg还是pos）
        默认为'all'
        
    col : str, optional
        从GP数据中提取的列名，默认为'avg'
    
    返回值:
    --------
    numpy.ndarray
        GP矩阵，形状为(n_factors, n_days)，其中每一行对应一个因子的时间序列数据
    """
    test_dir = Path(test_dir) if not isinstance(test_dir, Path) else test_dir
    
    # Create date range for the specified period
    date_range = pd.date_range(start=date_start, end=date_end, freq='D')
    
    gp_list = []
    missing_files = []
    
    # Process each factor
    for n_fct in selected_factor_info.index:
        tag_name = selected_factor_info.loc[n_fct, 'tag_name']
        process_name = selected_factor_info.loc[n_fct, 'process_name']
        factor_name = selected_factor_info.loc[n_fct, 'factor']
        test_name = selected_factor_info.loc[n_fct, 'test_name']
        direction = selected_factor_info.loc[n_fct, 'direction']
        
        # Determine process directory based on tag_name type
        process_dir = (test_dir / test_name / tag_name if isinstance(tag_name, str)
                      else test_dir / test_name)
        data_dir = process_dir / process_name / 'data'
        
        try:
            # Attempt to read data file
            data_path = data_dir / f'{data_to_use}_{factor_name}.pkl'
            
            with open(data_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            # Extract and filter data for the specified period
            direction_dict = {
                'all': 'all',
                'long_only': 'pos' if direction == 1 else 'neg',
                'short_only': 'neg' if direction == 1 else 'pos',
                }
            data = data_dict[direction_dict[use_direction]]
            gp_series = data[(data.index >= date_start) & (data.index <= date_end)
                             ].reindex(date_range)[col].fillna(0)
                             
        except Exception as e:
            # Handle missing or corrupt files
            print(f'Missing or error in file: {data_to_use}_{factor_name}.pkl - {str(e)}')
            missing_files.append(f'{data_to_use}_{factor_name}.pkl')
            gp_series = pd.Series(np.zeros(len(date_range)), index=date_range)
            
        gp_list.append(gp_series)
    
    # Convert list of series to matrix
    gp_matrix = np.array(gp_list)
    
    # Report any missing files
    if missing_files:
        print(f"Warning: {len(missing_files)} files were missing or could not be read.")
    
    return gp_matrix


def cluster_factors(selected_factor_info, date_start, date_end, test_dir, 
                    cluster_params=None, linkage_method='average',
                    data_to_use='gpd', use_direction='all', col='return'):
    """
    对所选因子进行聚类分析，返回聚类分组结果。
    
    参数:
    -----------
    selected_factor_info : pandas.DataFrame
        包含所选因子信息的DataFrame，必须包含以下列:
        ['tag_name', 'process_name', 'factor', 'test_name', 'direction']
        
    date_start : str
        起始日期，格式为'YYYY-MM-DD'
        
    date_end : str
        结束日期，格式为'YYYY-MM-DD'
        
    test_dir : Path or str
        包含测试数据的目录路径
        
    cluster_params : dict, optional
        传递给scipy.cluster.hierarchy.fcluster的参数。
        例如: {'t': 0.5, 'criterion': 'distance'}
        默认为None时会使用{'t': 0.5, 'criterion': 'distance'}
        
    linkage_method : str, optional
        计算连接时使用的方法。选项包括'single', 'complete', 'average', 'weighted'等。
        默认为'average'
        
    data_to_use : str, optional
        要使用的数据类型，默认为'gpd'
        
    use_direction : str, optional
        使用的方向类型，可选值为:
        - 'all': 使用所有数据
        - 'long_only': 仅使用多头数据
        - 'short_only': 仅使用空头数据
        默认为'all'
        
    col : str, optional
        从GP数据中提取的列名，默认为'avg'
    
    返回值:
    --------
    pandas.DataFrame
        添加了聚类分组结果的因子信息DataFrame，增加了'cluster'列表示聚类分组
    """
    # 提取GP矩阵
    gp_matrix = extract_gp_matrix(
        selected_factor_info, 
        date_start, 
        date_end, 
        test_dir, 
        data_to_use, 
        use_direction,
        col
    )
    
    # 执行聚类
    cluster_indices, distance_matrix, linkage = cluster_by_correlation(
        gp_matrix,
        cluster_params,
        linkage_method
    )
    
    return cluster_indices


# %%
def select_best_test_name(df, metric='net_sharpe_ratio_long_only', ascending=False):
    """
    对每个相同的process_name和factor组合，只保留指定指标下test_name最佳的那一个
    
    Parameters:
    -----------
    df : pandas.DataFrame
        原始因子评估数据框
    metric : str
        用于筛选的指标列名
    ascending : bool
        如果为True，则保留最小值；如果为False，则保留最大值
        
    Returns:
    --------
    pandas.DataFrame
        筛选后的数据框
    """
    # 确保指定的metric存在于数据框中
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame columns")
    
    # 对每个process_name和factor组合，根据metric选择最佳的test_name
    result = df.sort_values(by=metric, ascending=ascending).groupby(['process_name', 'factor']).first().reset_index()
    
    return result


def select_topn_per_group(df, metric='net_sharpe_ratio_long_only', n=1, ascending=False):
    """
    Select top N rows from each group based on a specified metric.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with a 'group' column
    metric : str
        Column name to sort by
    n : int
        Number of top entries to select from each group
    ascending : bool
        If True, smallest values are at top (False by default, so largest values at top)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing only the top N rows from each group
    """
    # Ensure the metric exists in the dataframe
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame columns")
    
    # Group by 'group', sort by the metric, and take the top N
    result = df.groupby('group').apply(
        lambda x: x.sort_values(by=metric, ascending=ascending).head(n)
    ).reset_index(drop=True)
    
    return result

# =============================================================================
# # Get top 1 factor per group based on net_sharpe_ratio_long_only
# top_factors = select_topn_per_group(
#     first_selected, 
#     metric='net_sharpe_ratio_long_only',  # Change to your desired metric
#     n=1,                                 # Change to your desired N
#     ascending=False                      # True if lower values are better
# )
# =============================================================================


def remove_ma_suffix_from_factors(top_factors_df):
    # 创建结果的副本，避免修改原始数据
    result_df = top_factors_df.copy()
    
    for idx, row in result_df.iterrows():
        process_name = row['process_name']
        factor = row['factor']
        
        # 找到最后一个 TS 的位置
        last_ts_pos = process_name.rfind('TS')
        
        # 如果存在 TS 并且最后一个 TS 后面的字符串包含 'ma'
        if last_ts_pos != -1 and 'ma' in process_name[last_ts_pos:]:
            # 找到最后一个 TS 前的下划线位置
            underscore_pos = process_name.rfind('_', 0, last_ts_pos)
            
            if underscore_pos != -1:
                # 截断 process_name 到最后一个 TS 前的下划线位置
                result_df.at[idx, 'process_name'] = process_name[:underscore_pos]
            else:
                # 如果没有下划线，直接截断到 TS 前
                result_df.at[idx, 'process_name'] = process_name[:last_ts_pos]
            
            # 处理对应的 factor，去掉最后一个 '-' 后面的部分
            last_dash_pos = factor.rfind('-')
            if last_dash_pos != -1:
                result_df.at[idx, 'factor'] = factor[:last_dash_pos]
    
    return result_df