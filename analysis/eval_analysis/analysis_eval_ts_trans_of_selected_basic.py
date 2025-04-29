# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 13:27:45 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pathlib import Path


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[2]
sys.path.append(str(project_dir))


# %%
from utils.dirutils import load_path_config


# %%
simple_eval_name = 'basis_oad_z1_s4'
eval_name = 'basis_pct_250416_org_TS_oad_0931_batch_250419_batch_test_v1_s4_ts_v1_batch_test_v1_overnight_default_v0'
period = '160101_250101'


# %%
path_config = load_path_config(project_dir)
result_dir = Path(path_config['result'])
eval_dir = result_dir / 'factor_evaluation'
analysis_dir = result_dir / 'analysis/eval_analysis'
summary_dir = analysis_dir / f'{simple_eval_name}/factor_eval_{period}'
summary_dir.mkdir(parents=True, exist_ok=True)


# %%
path = eval_dir / eval_name / f'factor_eval_{period}.csv'
eval_res = pd.read_csv(path)
eval_res['basic_fac'] = eval_res['process_name'].apply(lambda x: x.split('/')[-2])
eval_res['ts_trans_fac'] = eval_res['factor'].apply(lambda x: (x.split('-')[-1]).split('_')[0])
eval_res = eval_res[eval_res['ts_trans_fac'] != 'termtrend']
eval_res = eval_res.dropna(how='any', axis=0)


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.gridspec as GridSpec

def create_pairwise_heatmaps(df, metric='net_sharpe_ratio', col1='basic_fac', col2='ts_trans_fac', 
                            figsize=(42, 25), mask_negative=True, save_path=None, 
                            rotation_col=45, font_size=20):
    """
    Create heatmaps for a pair of grouping variables.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The factor evaluation results dataframe
    metric : str
        The metric to analyze (column name in df)
    col1, col2 : str
        The columns to group by
    figsize : tuple
        Figure size (width, height)
    mask_negative : bool
        Whether to mask negative values in the heatmap
    save_path : str or Path, optional
        If provided, saves the figure to this path
    rotation_col : int
        Rotation angle for column labels
    font_size : int
        Font size for annotations and labels
        
    Returns:
    --------
    fig : matplotlib figure
        The figure with mean and 90th percentile heatmaps
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Calculate mean and 90th percentile for the groups
    mean_pivot = df.pivot_table(index=col1, columns=col2, values=metric, aggfunc='mean')
    p90_pivot = df.pivot_table(index=col1, columns=col2, values=metric, aggfunc=lambda x: np.percentile(x, 90))
    
    # Create masks for negative values if requested
    if mask_negative:
        mean_mask = mean_pivot <= 0
        p90_mask = p90_pivot <= 0
    else:
        mean_mask = np.zeros_like(mean_pivot, dtype=bool)
        p90_mask = np.zeros_like(p90_pivot, dtype=bool)
    
    # Determine whether to use annotations based on size
    use_annot = (mean_pivot.shape[0] <= 100 and mean_pivot.shape[1] <= 100)
    
    # Create the figure with two subplots with increased size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Set font sizes for all elements
    plt.rcParams.update({'font.size': font_size})
    
    # Create the heatmaps with appropriate spacing
    sns.heatmap(mean_pivot, annot=use_annot, fmt='.1f', cmap='YlGnBu', mask=mean_mask, 
                cbar_kws={'label': f'Mean {metric}'}, ax=ax1, annot_kws={"size": font_size})
    ax1.set_title(f'Mean {metric} by {col1} and {col2}', fontsize=font_size+2)
    ax1.set_xlabel(col2, fontsize=font_size+1)
    ax1.set_ylabel(col1, fontsize=font_size+1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=rotation_col, ha='right', fontsize=font_size)
    
    sns.heatmap(p90_pivot, annot=use_annot, fmt='.1f', cmap='YlOrRd', mask=p90_mask, 
                cbar_kws={'label': f'90th Percentile {metric}'}, ax=ax2, annot_kws={"size": font_size})
    ax2.set_title(f'90th Percentile {metric} by {col1} and {col2}', fontsize=font_size+2)
    ax2.set_xlabel(col2, fontsize=font_size+1)
    ax2.set_ylabel(col1, fontsize=font_size+1)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=rotation_col, ha='right', fontsize=font_size)
    
    # Adjust layout to ensure all labels are visible
    plt.tight_layout(pad=3.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def create_category_distributions(df, metric='net_sharpe_ratio', category='basic_fac', 
                              figsize=(16, 16), save_path=None):
    """
    Create a single plot showing distribution curves for each value in a category.
    Sort categories by mean value and place legend outside the plot on the right.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The factor evaluation results dataframe
    metric : str
        The metric to analyze (column name in df)
    category : str
        The category column to group by
    figsize : tuple
        Figure size (width, height)
    save_path : str or Path, optional
        If provided, saves the figure to this path
        
    Returns:
    --------
    fig : matplotlib figure
        The figure with distribution curves
    """
    from scipy import stats
    
    # Calculate mean value for each category to sort by
    category_means = {}
    for cat_value in df[category].unique():
        cat_data = df[df[category] == cat_value][metric].dropna()
        if len(cat_data) > 1:  # Need at least 2 points for KDE
            category_means[cat_value] = cat_data.mean()
    
    # Sort categories by mean value (high to low)
    sorted_categories = sorted(category_means.keys(), key=lambda x: category_means[x], reverse=True)
    n_categories = len(sorted_categories)
    
    # Create figure with adjusted size to accommodate legend
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use different line styles for better distinction
    line_styles = ['-', '--', ':', '-.'] * 10  # Repeat styles if needed
    colors = plt.cm.tab20(np.linspace(0, 1, n_categories))
    
    # Store lines and labels for legend
    legend_lines = []
    legend_labels = []
    
    # Plot distribution curves for each category value
    for i, cat_value in enumerate(sorted_categories):
        cat_data = df[df[category] == cat_value][metric].dropna()
        if len(cat_data) > 1:  # Need at least 2 points for KDE
            # Use gaussian KDE for smooth curves
            x = np.linspace(min(cat_data), max(cat_data), 100)
            kde = stats.gaussian_kde(cat_data)
            y = kde(x)
            line = ax.plot(x, y, color=colors[i], 
                         linewidth=2, linestyle=line_styles[i % len(line_styles)])
            
            # Add vertical line for the mean
            mean_val = cat_data.mean()
            ax.axvline(mean_val, color=colors[i], linestyle='--', alpha=0.5)
            
            # Prepare legend items with mean value
            legend_lines.append(line[0])
            legend_labels.append(f'{cat_value} ({mean_val:.2f})')
    
    # Add labels and title
    ax.set_xlabel(metric)
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of {metric} by {category} (Sorted by Mean Value)')
    
    # Place legend outside the plot on the right
    ax.legend(legend_lines, legend_labels, 
             loc='center left', bbox_to_anchor=(1.02, 0.5), 
             title=f"{category} (Mean {metric})")
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Adjust to leave space for legend
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def analyze_factor_performance(df, metrics, base_save_path=None):
    """
    Create comprehensive factor analysis visualizations.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The factor evaluation results dataframe
    metrics : list of str
        List of metrics to analyze
    base_save_path : str or Path, optional
        Base path for saving figures. If provided, will save figures there.
    """
    # Define pairs for heatmaps
    pairs = [
        ('basic_fac', 'ts_trans_fac'),
        ('basic_fac', 'test_name'),
        ('ts_trans_fac', 'test_name')
    ]
    
    # Define categories for distribution plots
    categories = ['basic_fac', 'ts_trans_fac', 'test_name']
    
    for metric in metrics:
        print(f"Creating visualizations for {metric}...")
        
        # Create pairwise heatmaps
        for col1, col2 in pairs:
            pair_name = f"{col1}_by_{col2}"
            print(f"  Creating heatmap for {pair_name}...")
            
            fig = create_pairwise_heatmaps(df, metric=metric, col1=col1, col2=col2)
            
            if base_save_path:
                save_path = f"{base_save_path}/{metric}_{pair_name}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"    Saved to {save_path}")
            
            plt.show()
            plt.close()
        
        # Create category distribution plots
        for category in categories:
            print(f"  Creating distribution plot for {category}...")
            
            fig = create_category_distributions(df, metric=metric, category=category)
            
            if base_save_path:
                save_path = f"{base_save_path}/{metric}_{category}_dist.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"    Saved to {save_path}")
            
            plt.show()
            plt.close()

# Example usage:
# metrics_to_analyze = ['net_sharpe_ratio', 'net_calmar_ratio', 'net_sortino_ratio']
# analyze_factor_performance(eval_res, metrics_to_analyze, base_save_path=str(summary_dir / 'factor_analysis'))

# Example usage:
metrics_to_analyze = ['net_sharpe_ratio_long_only']
analyze_factor_performance(eval_res, metrics_to_analyze, base_save_path=str(summary_dir))
