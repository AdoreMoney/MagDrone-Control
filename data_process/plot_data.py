import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ================= User Configurable Parameters =================
# Specify the raw reward CSV file paths for PPO/SAC/TD3/PPO_only algorithms respectively
ppo_csv_path = r'C:\Users\a1fla\Desktop\MagDrone\logs\ppo\MagDrone_ppo_tensorboard\ppo_training_logs\ppo_reward_data_original.csv'
sac_csv_path = r'C:\Users\a1fla\Desktop\MagDrone\logs\sac\MagDrone_sac_tensorboard\sac_training_logs\sac_reward_data_original.csv'
td3_csv_path = r'C:\Users\a1fla\Desktop\MagDrone\logs\td3\MagDrone_td3_tensorboard\td3_training_logs\td3_reward_data_original.csv'
ppo_only_csv_path = r'C:\Users\a1fla\Desktop\MagDrone\logs\ppo\MagDrone_ppo_tensorboard\ppo_only_training_logs\ppo_reward_data_original.csv'

# Save path for the multi-algorithm reward comparison plot
compare_save_path = r'C:\Users\a1fla\Desktop\MagDrone\logs\reward_compare.png'
encoding = 'utf-8-sig'  # CSV encoding (consistent with the format used in previous data saving)
window_size = 20  # Rolling window size (used for calculating confidence intervals, adjustable based on data volume)
# =================================================

def read_reward_csv(file_path, algo_name, encoding):
    """Read and preprocess the reward CSV file for a single reinforcement learning algorithm, returning a cleaned DataFrame.
    
    This function handles file existence checks, data loading, column standardization, type conversion,
    deduplication, and calculation of rolling statistics for confidence interval visualization.
    
    Args:
        file_path (str): Full path to the algorithm's reward CSV file
        algo_name (str): Name of the reinforcement learning algorithm (for logging and error reporting)
        encoding (str): Character encoding used to read the CSV file
        
    Returns:
        pd.DataFrame: Cleaned and processed DataFrame containing episode, reward, and rolling statistics
        
    Raises:
        FileNotFoundError: If the specified file path does not exist
        ValueError: If required columns are missing from the CSV file
        RuntimeError: If any error occurs during file reading or data processing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{algo_name} reward data file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        # Standardize column names: remove whitespace and convert to lowercase to avoid case/space mismatches
        df.columns = df.columns.str.strip().str.lower()
        # Validate presence of required columns for reward analysis
        required_columns = {'episode', 'original_eval_reward'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"{algo_name} file is missing required columns: {missing_columns}. Required columns are 'Episode' and 'Original_Eval_Reward'.")
        
        # Data type conversion and deduplication to ensure data integrity
        df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
        df['original_eval_reward'] = pd.to_numeric(df['original_eval_reward'], errors='coerce')
        df = df.dropna(subset=['episode', 'original_eval_reward'])
        df = df.sort_values('episode').drop_duplicates(subset='episode', keep='last')
        
        # Calculate rolling mean and rolling standard deviation for confidence interval visualization
        # Rolling window is centered with minimum 1 observation to handle edge cases
        df['rolling_mean'] = df['original_eval_reward'].rolling(window=window_size, center=True, min_periods=1).mean()
        df['rolling_std'] = df['original_eval_reward'].rolling(window=window_size, center=True, min_periods=1).std()
        # Confidence interval (mean ± 1 standard deviation) - commonly used in academic publications
        df['upper'] = df['rolling_mean'] + df['rolling_std']
        df['lower'] = df['rolling_mean'] - df['rolling_std']
        
        reward_min = df['original_eval_reward'].min()
        reward_max = df['original_eval_reward'].max()
        print(f"Successfully loaded and processed {algo_name} data: {len(df)} episodes total, reward range [{reward_min:.2f}, {reward_max:.2f}]")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to process {algo_name} reward data: {str(e)}")

# 1) Load raw reward data for all four algorithms
ppo_df = read_reward_csv(ppo_csv_path, "PPO", encoding)
sac_df = read_reward_csv(sac_csv_path, "SAC", encoding)
td3_df = read_reward_csv(td3_csv_path, "TD3", encoding)
ppo_only_df = read_reward_csv(ppo_only_csv_path, "PPO_only", encoding)  # Load PPO_only data

# 2) Determine maximum episode count across all algorithms for consistent x-axis range
# No truncation of PPO/SAC data - full training progress is preserved
max_episode_ppo = ppo_df['episode'].max()
max_episode_sac = sac_df['episode'].max()
max_episode_td3 = td3_df['episode'].max()
max_episode_ppo_only = ppo_only_df['episode'].max()  # Get max episode for PPO_only

# Set x-axis maximum to the largest episode count across all algorithms for complete visualization
x_max = max(max_episode_ppo, max_episode_sac, max_episode_td3, max_episode_ppo_only)

# 3) Generate multi-algorithm reward comparison plot (academic publication style)
# Configure matplotlib settings for academic aesthetics
plt.rcParams['font.family'] = 'Arial'  # Sans-serif font commonly used in academic papers
plt.rcParams['font.size'] = 14  # Compact font size for academic figures
plt.rcParams['axes.linewidth'] = 0.8  # Thinner axis lines consistent with academic style
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.figure(figsize=(10, 6))  # Standard figure size for academic publications (10x6 inches)

# Define color palette with distinct primary colors and light fill shades (academic aesthetic)
color_config = {
    'PPO': {'line': '#d62728', 'fill': '#f8bbd0'},    # Red palette (dark line, light fill)
    'SAC': {'line': '#2ca02c', 'fill': '#c8e6c9'},     # Green palette
    'TD3': {'line': '#1f77b4', 'fill': '#bbdefb'},     # Blue palette
    'PPO_only': {'line': '#ff7f0e', 'fill': '#ffe0b2'} # Orange palette
}

# Plot confidence interval shading first (behind lines to avoid occlusion)
# PPO (PPO+PID)
plt.fill_between(ppo_df['episode'], ppo_df['lower'], ppo_df['upper'], 
                 color=color_config['PPO']['fill'], alpha=0.5, label='_nolegend_')  # Shading excluded from legend
plt.plot(ppo_df['episode'], ppo_df['rolling_mean'], 
         color=color_config['PPO']['line'], alpha=1.0, linewidth=1.5, label='PPO+PID')

# SAC (SAC+PID)
plt.fill_between(sac_df['episode'], sac_df['lower'], sac_df['upper'], 
                 color=color_config['SAC']['fill'], alpha=0.5, label='_nolegend_')
plt.plot(sac_df['episode'], sac_df['rolling_mean'], 
         color=color_config['SAC']['line'], alpha=1.0, linewidth=1.5, label='SAC+PID')

# TD3 (TD3+PID)
plt.fill_between(td3_df['episode'], td3_df['lower'], td3_df['upper'], 
                 color=color_config['TD3']['fill'], alpha=0.5, label='_nolegend_')
plt.plot(td3_df['episode'], td3_df['rolling_mean'], 
         color=color_config['TD3']['line'], alpha=1.0, linewidth=1.5, label='TD3+PID')

# PPO_only
plt.fill_between(ppo_only_df['episode'], ppo_only_df['lower'], ppo_only_df['upper'], 
                 color=color_config['PPO_only']['fill'], alpha=0.5, label='_nolegend_')
plt.plot(ppo_only_df['episode'], ppo_only_df['rolling_mean'], 
         color=color_config['PPO_only']['line'], alpha=1.0, linewidth=1.5, label='PPO')

# Optimize plot style to meet academic publication standards
plt.xlabel('Episode', fontsize=14, fontweight='normal')  # Normal font weight (bold not recommended for academic figures)
plt.ylabel('Reward', fontsize=14, fontweight='normal')
plt.xlim(0, x_max)
plt.ylim()  # Auto-adaptive y-axis range (can be manually set with plt.ylim(min_reward, max_reward))
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)  # Thin grid lines with low transparency for readability
# Optimize legend style - clean, semi-transparent with white border
plt.legend(loc='lower right', fontsize=13, frameon=True, framealpha=0.8, edgecolor='white')
plt.tight_layout(pad=0.5)  # Compact layout to minimize white space and prevent label cutoff

# Adjust axis spines for classic academic figure style (remove top/right spines)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# 4) Save the plot in formats suitable for academic use
# High-DPI PNG for digital display and most journals; EPS for LaTeX publications (vector graphics)
plt.savefig(compare_save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
# Optional: Save as EPS vector format (recommended for LaTeX academic papers)
# plt.savefig(compare_save_path.replace('.png', '.eps'), format='eps', bbox_inches='tight', facecolor='white')
print(f"\nMulti-algorithm reward comparison plot (with confidence intervals) saved to: {compare_save_path}")
plt.show()

# 5) Calculate and display key reward metrics for quantitative analysis
def get_key_metrics(df, algo_name):
    """Calculate key quantitative metrics for reinforcement learning algorithm performance evaluation.
    
    Computes critical metrics including maximum reward, final reward, average reward, and reward variability
    to enable objective comparison between different algorithms.
    
    Args:
        df (pd.DataFrame): Processed DataFrame containing episode and reward data
        algo_name (str): Name of the algorithm for metric labeling
        
    Returns:
        dict: Dictionary containing key performance metrics for the algorithm
    """
    max_reward = df['original_eval_reward'].max()
    # Get the first episode where maximum reward is achieved (handle potential ties)
    max_episode = df[df['original_eval_reward'] == max_reward]['episode'].values[0]
    final_reward = df['original_eval_reward'].iloc[-1]
    mean_reward = df['original_eval_reward'].mean()
    std_reward = df['original_eval_reward'].std()
    max_episode_count = int(df['episode'].max())
    
    return {
        'Algorithm': algo_name,
        'Maximum Reward': round(max_reward, 2),
        'Episode at Max Reward': int(max_episode),
        'Final Reward': round(final_reward, 2),
        'Average Reward': round(mean_reward, 2),
        'Reward Standard Deviation': round(std_reward, 2),
        'Data Length (Episodes)': max_episode_count
    }

# Compile metrics for all four algorithms and display as formatted table
metrics = pd.DataFrame([
    get_key_metrics(ppo_df, 'PPO+PID'),
    get_key_metrics(sac_df, 'SAC+PID'),
    get_key_metrics(td3_df, 'TD3+PID'),
    get_key_metrics(ppo_only_df, 'PPO')
])
print("\n========== Key Reward Metrics for All Algorithms ==========")
print(metrics.to_string(index=False))