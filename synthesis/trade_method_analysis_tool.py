# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 2025

@author: Claude

交易方法测试结果分析工具
整合compare_factors和update.plot_pos的分析功能，对select_trade_method的测试结果进行综合分析

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from pathlib import Path
import seaborn as sns
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Add project directory to path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))

from utils.dirutils import load_path_config
from test_and_eval.scores import get_general_return_metrics


class TradeMethodAnalyzer:
    """
    交易方法测试结果分析器
    
    对select_trade_method的最终测试结果进行综合分析，包括：
    1. 月度收益热力图
    2. 价格与仓位时序分析
    3. 滚动净值夏普比分析
    """
    
    def __init__(self, select_folder_name, test_name, config_name=None):
        """
        初始化分析器
        
        Parameters:
        -----------
        select_folder_name : str
            select_trade_method的文件夹名
        test_name : str
            对应的测试名称
        config_name : str, optional
            配置文件名（不含.yaml后缀），默认为'default'
        """
        self.select_folder_name = select_folder_name
        self.test_name = test_name
        self.config_name = config_name or 'default'
        
        # Load paths from config
        self.path_config = load_path_config(project_dir)
        self.result_dir = Path(self.path_config['result'])
        self.param_dir = Path(self.path_config['param'])
        
        # Set up directories
        self.select_result_dir = self.result_dir / 'select_trade_method' / select_folder_name
        self.test_result_dir = self.select_result_dir / 'test' / test_name
        self.data_dir = self.test_result_dir / 'data'
        
        # Create analysis directory
        self.analysis_dir = self.test_result_dir / 'analysis'
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Set fee from config
        self.fee = self.config.get('fee', 2.4e-4)
        
        print(f"Initialized TradeMethodAnalyzer for {select_folder_name}/{test_name}")
        print(f"Data directory: {self.data_dir}")
        print(f"Analysis directory: {self.analysis_dir}")
    
    def _load_config(self):
        """加载配置文件"""
        config_path = self.param_dir / 'analysis_trade' / f'{self.config_name}.yaml'
        
        # Create default config if not exists
        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            default_config = {
                'fee': 2.4e-4,
                'monthly_heatmap': {
                    'enabled': True
                },
                'weekly_position_analysis': {
                    'enabled': True,
                    'date_start': '2024-01-01',
                    'date_end': '2024-12-31',
                    'instruments': ['IC', 'IF', 'IM']
                },
                'rolling_sharpe_analysis': {
                    'enabled': True,
                    'rolling_window': '60d',
                    'date_start': '2024-01-01',
                    'date_end': '2024-12-31'
                },
                'price_data': {
                    'price_name': 't1min_fq1min_dl1min',
                    'price_dir': None  # Will use default from path_config
                }
            }
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
            print(f"Created default config at: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _load_test_data(self):
        """加载测试数据 (gp, hsr, pos)"""
        factor_name = f'pos_{self.select_folder_name}'
        
        # Load gp data
        gp_path = self.data_dir / f'gpd_{factor_name}.pkl'
        hsr_path = self.data_dir / f'hsr_{factor_name}.pkl'
        pos_path = self.data_dir / f'pos_{factor_name}.parquet'
        
        if not all([gp_path.exists(), hsr_path.exists(), pos_path.exists()]):
            missing_files = [str(p) for p in [gp_path, hsr_path, pos_path] if not p.exists()]
            raise FileNotFoundError(f"Missing required data files: {missing_files}")
        
        # Load data
        with open(gp_path, 'rb') as f:
            gp_dict = pickle.load(f)
        
        with open(hsr_path, 'rb') as f:
            hsr_dict = pickle.load(f)
        
        pos_data = pd.read_parquet(pos_path)
        
        return gp_dict, hsr_dict, pos_data
    
    def _load_price_data(self):
        """加载价格数据"""
        price_config = self.config['price_data']
        price_name = price_config['price_name']
        price_dir = price_config.get('price_dir')
        
        if price_dir is None:
            price_dir = Path(self.path_config['future_twap'])
        else:
            price_dir = Path(price_dir)
        
        price_path = price_dir / f'{price_name}.parquet'
        if not price_path.exists():
            raise FileNotFoundError(f"Price data not found: {price_path}")
        
        price_data = pd.read_parquet(price_path)
        return price_data
    
    def _calculate_net_returns(self, gp_dict, hsr_dict, direction='all'):
        """计算净收益"""
        if direction not in gp_dict or direction not in hsr_dict:
            raise ValueError(f"Direction '{direction}' not found in data")
        
        df_gp = gp_dict[direction]
        df_hsr = hsr_dict[direction]
        
        # Calculate net return
        net = (df_gp['return'] - self.fee * df_hsr['avg']).fillna(0)
        
        return net
    
    def analyze_monthly_heatmap(self, direction='all'):
        """
        1. 月度收益热力图分析
        """
        if not self.config['monthly_heatmap']['enabled']:
            print("Monthly heatmap analysis disabled in config")
            return
        
        print("Generating monthly return heatmap...")
        
        # Load data
        gp_dict, hsr_dict, _ = self._load_test_data()
        
        # Calculate net returns
        net = self._calculate_net_returns(gp_dict, hsr_dict, direction)
        
        # Resample to monthly returns
        monthly_returns = net.resample('M').sum()
        
        # Create DataFrame with Year and Month for heatmap
        monthly_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        # Convert to pivot table
        pivot_table = monthly_df.pivot_table(index='Year', columns='Month', values='Return')
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot heatmap
        sns.heatmap(pivot_table, annot=True, fmt='.1%', cmap='RdYlGn', 
                   center=0, cbar_kws={'label': 'Monthly Return'})
        
        # Set title and labels
        plt.title(f"Monthly Returns: {self.select_folder_name} - {self.test_name}", fontsize=16)
        plt.xlabel("Month", fontsize=12)
        plt.ylabel("Year", fontsize=12)
        
        # Set month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.gca().set_xticklabels(month_names, rotation=0)
        
        # Calculate and add metrics
        metrics = get_general_return_metrics(net)
        metrics_text = (
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_dd']*100:.2f}%\n"
            f"Annualized Return: {metrics['return_annualized']*100:.2f}%"
        )
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.analysis_dir / f"monthly_heatmap_{direction}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"Monthly heatmap saved to: {save_path}")
    
    def analyze_weekly_positions(self):
        """
        2. 按周分析价格与仓位
        """
        config = self.config['weekly_position_analysis']
        if not config['enabled']:
            print("Weekly position analysis disabled in config")
            return
        
        print("Generating weekly position analysis...")
        
        # Load data
        _, _, pos_data = self._load_test_data()
        price_data = self._load_price_data()
        
        # Get parameters from config
        date_start = pd.Timestamp(config['date_start'])
        date_end = pd.Timestamp(config['date_end'])
        instruments = config['instruments']
        
        # Filter data
        price_data = price_data[instruments]
        price_data = price_data.loc[date_start:date_end]
        pos_data = pos_data[instruments]
        pos_data = pos_data.loc[date_start:date_end]
        
        # Create weekly analysis directory
        weekly_dir = self.analysis_dir / 'weekly_positions'
        weekly_dir.mkdir(parents=True, exist_ok=True)
        
        # Group by weeks
        for week_start, week_price in price_data.groupby(pd.Grouper(freq='W-MON', label='left', closed='left')):
            if week_start < date_start or week_start > date_end:
                continue
                
            # Get corresponding week data
            week_end = week_start + pd.Timedelta(weeks=1)
            week_pos = pos_data.loc[week_start:week_end]
            
            if week_price.empty or week_pos.empty:
                continue
            
            # Align position data with price data
            week_pos_aligned = week_pos.reindex(week_price.index, method='ffill')
            
            # Create plot
            self._plot_weekly_price_position(week_price, week_pos_aligned, week_start, weekly_dir)
        
        print(f"Weekly position analysis saved to: {weekly_dir}")
    
    def _plot_weekly_price_position(self, price_data, pos_data, week_start, save_dir):
        """绘制单周的价格与仓位图"""
        instruments = price_data.columns
        
        # Generate sequential x-axis
        x = np.arange(len(price_data))
        x_labels = price_data.index.strftime('%Y-%m-%d %H:%M')
        
        # Find 9:30 positions for vertical lines
        nine_thirty_indices = [i for i, t in enumerate(price_data.index) if t.strftime('%H:%M') == '09:30']
        
        # Create figure
        fig, axs = plt.subplots(len(instruments), 1, figsize=(15, 4 * len(instruments)), sharex=True)
        if len(instruments) == 1:
            axs = [axs]
        
        fig.suptitle(f"Price and Position for Week Starting {week_start.strftime('%Y-%m-%d')}", fontsize=16)
        
        for i, instrument in enumerate(instruments):
            ax1 = axs[i]
            
            # Plot price
            price_series = price_data[instrument] #.dropna()
            ax1.plot(x, price_series, color='black', linewidth=1.5, label=f'{instrument} Price')
            ax1.set_ylabel(f'{instrument} Price', color='black', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Add vertical lines for market open
            for idx in nine_thirty_indices:
                ax1.axvline(x=idx, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            
            # Create secondary axis for positions
            ax2 = ax1.twinx()
            
            # Plot positions
            pos_series = pos_data[instrument].dropna()
            if not pos_series.empty:
                # Align with price index
                pos_aligned = pos_series.reindex(price_series.index, method='ffill')
                valid_mask = ~pos_aligned.isna()
                
                if valid_mask.any():
                    valid_x = x[valid_mask.values]
                    valid_pos = pos_aligned[valid_mask].values
                    
                    ax2.plot(valid_x, valid_pos, color='red', linewidth=2, 
                           label=f'{instrument} Position')
            
            # Format secondary axis
            ax2.set_ylabel('Position', fontsize=12)
            ax2.set_ylim(-1.2, 1.2)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add legends
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            ax1.set_title(f"{instrument}", fontsize=14)
        
        # Set x-axis ticks
        if len(x) > 0:
            tick_positions = np.linspace(0, len(x)-1, num=10, dtype=int)
            axs[-1].set_xticks(tick_positions)
            axs[-1].set_xticklabels([x_labels[i] for i in tick_positions], rotation=45)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save plot
        filename = f"week_{week_start.strftime('%Y-%m-%d')}.png"
        plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_rolling_sharpe(self, direction='all'):
        """
        3. 滚动夏普比分析
        """
        config = self.config['rolling_sharpe_analysis']
        if not config['enabled']:
            print("Rolling Sharpe analysis disabled in config")
            return
        
        print("Generating rolling Sharpe ratio analysis...")
        
        # Load data
        gp_dict, hsr_dict, _ = self._load_test_data()
        
        # Calculate net returns
        net = self._calculate_net_returns(gp_dict, hsr_dict, direction)
        
        # Get parameters from config
        rolling_window = config['rolling_window']
        date_start = pd.Timestamp(config['date_start'])
        date_end = pd.Timestamp(config['date_end'])
        
        # Filter data
        net = net.loc[date_start:date_end]
        
        # Calculate rolling Sharpe ratio
        rolling_window_days = pd.Timedelta(rolling_window).days
        rolling_sharpe = net.rolling(window=f'{rolling_window_days}d').apply(
            lambda x: get_general_return_metrics(x)['sharpe_ratio'] if len(x.dropna()) > 10 else np.nan
        )
        
        # Create plot
        plt.figure(figsize=(15, 8))
        
        # Plot rolling Sharpe
        plt.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='blue')
        plt.title(f"Rolling Sharpe Ratio ({rolling_window}) - {self.select_folder_name}/{self.test_name}", 
                 fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Sharpe Ratio", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add horizontal line at 0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add statistics
        stats_text = (
            f"Mean Sharpe: {rolling_sharpe.mean():.2f}\n"
            f"Std Sharpe: {rolling_sharpe.std():.2f}\n"
            f"Min Sharpe: {rolling_sharpe.min():.2f}\n"
            f"Max Sharpe: {rolling_sharpe.max():.2f}"
        )
        plt.figtext(0.02, 0.7, stats_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.analysis_dir / f"rolling_sharpe_{rolling_window}_{direction}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"Rolling Sharpe analysis saved to: {save_path}")
    
    def run_full_analysis(self, direction='all'):
        """
        运行完整分析流程
        
        Parameters:
        -----------
        direction : str
            分析方向，可选 'all', 'pos', 'neg'
        """
        print(f"Starting full analysis for {self.select_folder_name}/{self.test_name}")
        print(f"Direction: {direction}")
        print("="*60)
        
        try:
            # 1. Monthly heatmap analysis
            if self.config['monthly_heatmap']['enabled']:
                self.analyze_monthly_heatmap(direction)
                print()
            
            # 2. Weekly position analysis
            if self.config['weekly_position_analysis']['enabled']:
                self.analyze_weekly_positions()
                print()
            
            # 3. Rolling Sharpe analysis
            if self.config['rolling_sharpe_analysis']['enabled']:
                self.analyze_rolling_sharpe(direction)
                print()
            
            print("="*60)
            print(f"✅ Full analysis completed successfully!")
            print(f"📁 Results saved to: {self.analysis_dir}")
            
        except Exception as e:
            print(f"❌ Analysis failed with error: {str(e)}")
            raise


# 使用示例
if __name__ == "__main__":
    # 示例参数 - 根据实际情况修改
    select_folder_name = "merged_factors_select_trade_v1_rolling_select_v1"
    test_name = "trade_ver3_futtwap_sp1min_s240d_icim_v6"
    
    # 创建分析器
    analyzer = TradeMethodAnalyzer(
        select_folder_name=select_folder_name,
        test_name=test_name,
        config_name='default'  # 可选，使用默认配置
    )
    
    # 运行完整分析
    analyzer.run_full_analysis(direction='all')