# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 12:31:21 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %%
from pathlib import Path
import pandas as pd


# %%
fut='IC'
vol_threshold = 0.012
slope_threshold = 2e-5
pen = 30000


price_name = 't1min_fq1min_dl1min'


# %%
label_name = f'rv{vol_threshold}_slp{slope_threshold}_pen{pen}'
label_dir = Path('/mnt/data1/labels')


# %%
labels_df = pd.read_parquet(label_dir / f'{label_name}.parquet')


# %%
fut_dir = Path('/mnt/data1/future_twap')
analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\label_plot')
summary_dir = analysis_dir / label_name
summary_dir.mkdir(parents=True, exist_ok=True)


# %%
# =============================================================================
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from pathlib import Path
# import os
# 
# label_name = f'rv{vol_threshold}_slp{slope_threshold}_pen{pen}'
# 
# # Directories
# label_dir = Path('/mnt/data1/labels')
# fut_dir = Path('/mnt/data1/future_twap')
# analysis_dir = Path(r'D:\mnt\CNIndexFutures\timeseries\factor_test\results\analysis\label_plot')
# summary_dir = analysis_dir / label_name
# summary_dir.mkdir(parents=True, exist_ok=True)
# 
# # Load data
# print(f"Loading labels from {label_dir / f'{label_name}.parquet'}")
# labels_df = pd.read_parquet(label_dir / f'{label_name}.parquet')
# print(f"Loading prices from {fut_dir / f'{price_name}.parquet'}")
# twap_price = pd.read_parquet(fut_dir / f'{price_name}.parquet')
# 
# # Ensure the index is datetime
# if not isinstance(labels_df.index, pd.DatetimeIndex):
#     labels_df.index = pd.to_datetime(labels_df.index)
# if not isinstance(twap_price.index, pd.DatetimeIndex):
#     twap_price.index = pd.to_datetime(twap_price.index)
# 
# # Check if the specified future column exists in both dataframes
# if fut not in labels_df.columns:
#     raise KeyError(f"Future column '{fut}' not found in labels dataframe. Available columns: {labels_df.columns.tolist()}")
# if fut not in twap_price.columns:
#     raise KeyError(f"Future column '{fut}' not found in price dataframe. Available columns: {twap_price.columns.tolist()}")
# 
# print(f"Using future column: {fut}")
# 
# # Make sure the indices match
# common_index = labels_df.index.intersection(twap_price.index)
# labels_df = labels_df.loc[common_index]
# twap_price = twap_price.loc[common_index]
# 
# # Get quarter information for each date
# twap_price['year_quarter'] = twap_price.index.to_period('Q')
# 
# # Set colors for different classification labels - using a more aesthetically pleasing palette
# colors = {
#     "Low_Vol_Range": "#4287f5",    # Soft blue
#     "Low_Vol_Trend": "#42c2a8",    # Teal
#     "High_Vol_Range": "#f56042",   # Soft red/coral
#     "High_Vol_Trend": "#f5a742",   # Amber
#     "Unknown": "#a8a8a8"           # Light gray
# }
# 
# # Get unique quarters
# quarters = twap_price['year_quarter'].unique()
# 
# # Process each quarter
# for quarter in sorted(quarters):
#     print(f"Processing quarter: {quarter}")
#     
#     # Filter data for this quarter
#     quarter_mask = twap_price['year_quarter'] == quarter
#     
#     # Get price and label data for this quarter
#     quarter_price = twap_price.loc[quarter_mask][fut].copy()  # Use .copy() to avoid view vs copy issues
#     
#     # Check if the corresponding label data exists for every price point
#     common_indices = quarter_price.index.intersection(labels_df.index)
#     
#     # Use only the common indices to ensure we have matching data
#     quarter_price = quarter_price.loc[common_indices]
#     quarter_labels = labels_df.loc[common_indices][fut].copy()  # Use the specified future column
#     
#     # Skip this quarter if no data or very little data
#     if len(quarter_price) < 10:
#         print(f"Insufficient data for {fut} in quarter {quarter}, skipping...")
#         plt.close()
#         continue
#         
#     # Print data alignment information
#     print(f"Quarter {quarter}: {len(quarter_price)} price points, {len(quarter_labels)} label points")
#     
#     if len(quarter_price) == 0:
#         print(f"No data for quarter {quarter}, skipping...")
#         continue
#     
#     # Create figure with enhanced aesthetics
#     plt.figure(figsize=(15, 8), facecolor='#f9f9f9')
#     plt.rcParams.update({
#         'font.family': 'Arial',
#         'font.size': 10,
#         'axes.titlesize': 14,
#         'axes.labelsize': 12
#     })
#     
#     # Skip this quarter if no data
#     if len(quarter_price) == 0:
#         print(f"No data for {fut} in quarter {quarter}, skipping...")
#         plt.close()
#         continue
#     ax = plt.gca()
#     
#     # Set a nice style for the plot
#     plt.style.use('seaborn-v0_8-whitegrid')
# 
#     # Create arrays for plotting - using continuous index
#     index_array = np.arange(len(quarter_price))
#     
#     # Get the actual dates for labeling
#     date_indices = quarter_price.index
#     
#     # Plot price with continuous index for better visualization
#     plt.plot(index_array, quarter_price.values, label='Price', color='#2e2e2e', linewidth=1.5)
#     
#     # Make sure we're using the same length of data for labels and prices
#     # This ensures we don't miss any classification at the end
#     if len(quarter_labels) != len(quarter_price):
#         print(f"Warning: Length mismatch between prices ({len(quarter_price)}) and labels ({len(quarter_labels)})")
#         # Use the minimum length to avoid index errors
#         min_len = min(len(quarter_price), len(quarter_labels))
#         quarter_labels = quarter_labels.iloc[:min_len]
#         quarter_price = quarter_price.iloc[:min_len]
#         index_array = index_array[:min_len]
#     
#     # Create label changes based on continuous index
#     if len(quarter_labels) > 1:  # Make sure we have at least 2 points to find changes
#         label_changes = np.where(quarter_labels.values[1:] != quarter_labels.values[:-1])[0] + 1
#         segment_points = np.concatenate(([0], label_changes, [len(quarter_labels)]))
#     else:
#         # If we have only one point (or none), create a single segment
#         segment_points = np.array([0, len(quarter_labels)])
#     
#     # Fill each segment with appropriate color using continuous index
#     for i in range(len(segment_points)-1):
#         start_idx = segment_points[i]
#         end_idx = segment_points[i+1]
#         
#         if start_idx >= len(quarter_labels) or start_idx == end_idx:
#             continue
#             
#         # Get the label for this segment
#         if start_idx < len(quarter_labels):
#             label = quarter_labels.values[start_idx]
#             # Check if label contains valid classification value
#             if pd.isna(label) or label == '' or label not in colors:
#                 print(f"Warning: Invalid label '{label}' at index {start_idx}, using gray")
#                 color = colors.get("Unknown", "gray")
#             else:
#                 color = colors.get(label, "gray")
#         else:
#             print(f"Warning: Start index {start_idx} out of bounds for labels array (length {len(quarter_labels)})")
#             color = colors.get("Unknown", "gray")
#         
#         # Add colored background using index positions - ensure we cover the entire range
#         ax.axvspan(start_idx, min(end_idx, len(index_array)-1), alpha=0.25, color=color, edgecolor=color, linewidth=0.5)
#     
#     # Add a legend for segment types with improved aesthetics
#     legend_elements = [
#         patches.Patch(color=colors["Low_Vol_Range"], alpha=0.3, label="Low Volatility Range"),
#         patches.Patch(color=colors["Low_Vol_Trend"], alpha=0.3, label="Low Volatility Trend"),
#         patches.Patch(color=colors["High_Vol_Range"], alpha=0.3, label="High Volatility Range"),
#         patches.Patch(color=colors["High_Vol_Trend"], alpha=0.3, label="High Volatility Trend")
#     ]
#     
#     # Enhanced title with adjusted padding to move it up
#     plt.title(f"{fut.upper()} Price Series Classification - {quarter}", fontweight='bold', pad=20)
#     plt.xlabel("Time", labelpad=10)
#     plt.ylabel(f"{fut.upper()} Price", labelpad=10)
#     
#     # Improved legend
#     legend = plt.legend(handles=legend_elements, loc='upper left', framealpha=0.9, 
#                         facecolor='white', edgecolor='#dddddd')
#     legend.get_frame().set_linewidth(0.5)
#     
#     # Enhance grid 
#     plt.grid(True, linestyle='--', alpha=0.7)
#     
#     # Format axes
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_linewidth(0.5)
#     ax.spines['bottom'].set_linewidth(0.5)
#     
#     # Add a subtitle with trading days information - moved down to avoid overlap
#     unique_days = pd.Series(date_indices.date).nunique()
#     plt.figtext(0.5, 0.89, f"Trading Days: {unique_days} | Total Minutes: {len(quarter_price)}", 
#                 ha="center", fontsize=10, style='italic')
#     
#     # Format the plot for better readability with trading hour gaps
#     # Increased top margin to give more space for title and subtitle
#     plt.tight_layout(rect=[0, 0.03, 1, 0.87])  # Adjust layout to make room for the annotations
#     
#     # Format x-axis with actual dates
#     # Select evenly spaced dates for x-ticks to avoid overcrowding
#     n_ticks = min(10, len(date_indices))  # Maximum of 10 ticks
#     tick_positions = np.linspace(0, len(index_array)-1, n_ticks, dtype=int)
#     
#     plt.xticks(tick_positions, [date_indices[i].strftime('%Y-%m-%d') for i in tick_positions], rotation=45)
#     
#     # Add more specific time labels for the first and last days
#     if len(date_indices) > 0:
#         first_day = date_indices[0].strftime('%Y-%m-%d %H:%M')
#         last_day = date_indices[-1].strftime('%Y-%m-%d %H:%M')
#         plt.figtext(0.1, 0.01, f"Start: {first_day}", ha="left", fontsize=9)
#         plt.figtext(0.9, 0.01, f"End: {last_day}", ha="right", fontsize=9)
#     
#     # Save the figure with high DPI and transparent background for better quality
#     output_file = summary_dir / f"{fut}_price_classification_{quarter}.png"
#     plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
#     print(f"Saved figure to {output_file}")
#     
#     # Add statistics about segments to the plot
#     segment_counts = {}
#     valid_labels = [l for l in quarter_labels if not pd.isna(l) and l != '' and l in colors]
#     
#     for label in colors.keys():
#         count = sum(1 for l in valid_labels if l == label)
#         if count > 0:  # Only include labels that appear in this quarter
#             segment_counts[label] = count
#     
#     # Add text with segment statistics
#     stats_text = "Segment Statistics:\n"
#     for label, count in segment_counts.items():
#         percentage = count/len(valid_labels)*100 if valid_labels else 0
#         stats_text += f"{label}: {count} mins ({percentage:.1f}%)\n"
#     
#     # Add text box with statistics to the plot
#     props = dict(boxstyle='round', facecolor='white', alpha=0.7)
#     plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
#                 bbox=props, verticalalignment='bottom')
# 
# print("All quarters processed successfully.")


# %%

# =============================================================================
