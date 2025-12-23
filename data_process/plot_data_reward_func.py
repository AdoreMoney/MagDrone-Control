import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ================= User Configurable Parameters =================
# Specify the raw data CSV file paths for 7 PID structures respectively
p_csv_path = r'C:\Users\a1fla\Desktop\MagDrone\logs\ppo\Kp\MagDrone_ppo_tensorboard\ppo_reward_data_original.csv'
i_csv_path = r'C:\Users\a1fla\Desktop\MagDrone\logs\ppo\Ki\MagDrone_ppo_tensorboard\ppo_reward_data_original.csv'
d_csv_path = r'C:\Users\a1fla\Desktop\MagDrone\logs\ppo\Kd\MagDrone_ppo_tensorboard\ppo_reward_data_original.csv'
pi_csv_path = r'C:\Users\a1fla\Desktop\MagDrone\logs\ppo\KpKi\MagDrone_ppo_tensorboard\ppo_reward_data_original.csv'
pd_csv_path = r'C:\Users\a1fla\Desktop\MagDrone\logs\ppo\KpKd\MagDrone_ppo_tensorboard\ppo_reward_data_original.csv'
id_csv_path = r'C:\Users\a1fla\Desktop\MagDrone\logs\ppo\KiKd\MagDrone_ppo_tensorboard\ppo_reward_data_original.csv'
pid_csv_path = r'C:\Users\a1fla\Desktop\MagDrone\logs\ppo\MagDrone_ppo_tensorboard\ppo_training_logs\ppo_reward_data_original.csv'

# Save path for the comparison plot
compare_save_path = r'C:\Users\a1fla\Desktop\MagDrone\logs\pid_structures_reward_compare.png'
encoding = 'utf-8-sig'  # CSV encoding (consistent with the previously saved format)
window_size = 20  # Rolling window size (used for calculating confidence intervals, adjustable based on data volume)
# =================================================

def read_reward_csv(file_path, algo_name, encoding):
    """Read the reward CSV file for a single PID structure and return a cleaned DataFrame"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{algo_name} data file does not exist: {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        # Standardize column names (prevent issues with case sensitivity/whitespace)
        df.columns = df.columns.str.strip().str.lower()
        # Check for required columns
        if 'episode' not in df.columns or 'original_eval_reward' not in df.columns:
            raise ValueError(f"{algo_name} file is missing required columns; it must contain 'Episode' and 'Original_Eval_Reward'")
        
        # Type conversion and deduplication
        df['episode'] = pd.to_numeric(df['episode'], errors='coerce')
        df['original_eval_reward'] = pd.to_numeric(df['original_eval_reward'], errors='coerce')
        df = df.dropna(subset=['episode', 'original_eval_reward'])
        df = df.sort_values('episode').drop_duplicates(subset='episode', keep='last')
        
        # Calculate rolling mean and rolling standard deviation (for confidence intervals)
        df['rolling_mean'] = df['original_eval_reward'].rolling(window=window_size, center=True, min_periods=1).mean()
        df['rolling_std'] = df['original_eval_reward'].rolling(window=window_size, center=True, min_periods=1).std()
        # Confidence interval (mean ± 1 standard deviation, commonly used in academia)
        df['upper'] = df['rolling_mean'] + df['rolling_std']
        df['lower'] = df['rolling_mean'] - df['rolling_std']
        
        print(f"Successfully read {algo_name} data: {len(df)} episodes in total, reward range [{df['original_eval_reward'].min():.2f}, {df['original_eval_reward'].max():.2f}]")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read {algo_name} data: {e}")

# 1) Read raw data for 7 PID structures
p_df = read_reward_csv(p_csv_path, "P_only", encoding)
i_df = read_reward_csv(i_csv_path, "I_only", encoding)
d_df = read_reward_csv(d_csv_path, "D_only", encoding)
pi_df = read_reward_csv(pi_csv_path, "PI", encoding)
pd_df = read_reward_csv(pd_csv_path, "PD", encoding)
id_df = read_reward_csv(id_csv_path, "ID", encoding)
pid_df = read_reward_csv(pid_csv_path, "PID", encoding)

# 2) Get the maximum episode for each structure (used to set the x-axis range)
max_episode_p = p_df['episode'].max()
max_episode_i = i_df['episode'].max()
max_episode_d = d_df['episode'].max()
max_episode_pi = pi_df['episode'].max()
max_episode_pd = pd_df['episode'].max()
max_episode_id = id_df['episode'].max()
max_episode_pid = pid_df['episode'].max()

# Set x-axis range to the maximum episode across all structures (to ensure complete display)
x_max = max(max_episode_p, max_episode_i, max_episode_d, 
            max_episode_pi, max_episode_pd, max_episode_id, max_episode_pid)

# 3) Plot the multi-structure comparison graph (academic style)
plt.rcParams['font.family'] = 'Arial'  # Common sans-serif font for academic papers
plt.rcParams['font.size'] = 14  # Slightly reduce font size due to the large number of structures
plt.rcParams['axes.linewidth'] = 0.8  # Thinner axis lines, consistent with academic style
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.figure(figsize=(10, 6))  # Widen the figure to accommodate 7 curves

# Define color configuration (highly distinguishable color palette, consistent with academic aesthetics)
color_config = {
    'P_only':    {'line': '#e74c3c', 'fill': '#fadbd8'},  # Pure red
    'I_only':    {'line': '#8e44ad', 'fill': '#ebdaf8'},  # Purple
    'D_only':    {'line': '#27ae60', 'fill': '#d5e8d4'},  # Pure green
    'PI':        {'line': '#f39c12', 'fill': '#fef5e7'},  # Orange
    'PD':        {'line': '#3498db', 'fill': '#d6eaf8'},  # Pure blue
    'ID':        {'line': '#ff69b4', 'fill': '#ffc0cb'},  # Pink
    'PID':       {'line': '#000000', 'fill': '#f2f2f2'}   # Black (to highlight the complete PID structure)
}

# Plot curves + confidence interval shading (plot shading first, then curves to avoid occlusion)
# P_only
plt.fill_between(p_df['episode'], p_df['lower'], p_df['upper'], 
                 color=color_config['P_only']['fill'], alpha=0.5, label='_nolegend_')
plt.plot(p_df['episode'], p_df['rolling_mean'], 
         color=color_config['P_only']['line'], alpha=1.0, linewidth=1.2, label='P')

# I_only
plt.fill_between(i_df['episode'], i_df['lower'], i_df['upper'], 
                 color=color_config['I_only']['fill'], alpha=0.5, label='_nolegend_')
plt.plot(i_df['episode'], i_df['rolling_mean'], 
         color=color_config['I_only']['line'], alpha=1.0, linewidth=1.2, label='I')

# D_only
plt.fill_between(d_df['episode'], d_df['lower'], d_df['upper'], 
                 color=color_config['D_only']['fill'], alpha=0.5, label='_nolegend_')
plt.plot(d_df['episode'], d_df['rolling_mean'], 
         color=color_config['D_only']['line'], alpha=1.0, linewidth=1.2, label='D')

# PI
plt.fill_between(pi_df['episode'], pi_df['lower'], pi_df['upper'], 
                 color=color_config['PI']['fill'], alpha=0.5, label='_nolegend_')
plt.plot(pi_df['episode'], pi_df['rolling_mean'], 
         color=color_config['PI']['line'], alpha=1.0, linewidth=1.2, label='PI')

# PD
plt.fill_between(pd_df['episode'], pd_df['lower'], pd_df['upper'], 
                 color=color_config['PD']['fill'], alpha=0.5, label='_nolegend_')
plt.plot(pd_df['episode'], pd_df['rolling_mean'], 
         color=color_config['PD']['line'], alpha=1.0, linewidth=1.2, label='PD')

# ID
plt.fill_between(id_df['episode'], id_df['lower'], id_df['upper'], 
                 color=color_config['ID']['fill'], alpha=0.5, label='_nolegend_')
plt.plot(id_df['episode'], id_df['rolling_mean'], 
         color=color_config['ID']['line'], alpha=1.0, linewidth=1.2, label='ID')

# PID
plt.fill_between(pid_df['episode'], pid_df['lower'], pid_df['upper'], 
                 color=color_config['PID']['fill'], alpha=0.5, label='_nolegend_')
plt.plot(pid_df['episode'], pid_df['rolling_mean'], 
         color=color_config['PID']['line'], alpha=1.0, linewidth=1.5, label='PID')  # Bold to highlight PID

# Figure style optimization (academic paper standard)
plt.xlabel('Episode', fontsize=14, fontweight='normal')
plt.ylabel('Reward', fontsize=14, fontweight='normal')
plt.xlim(0, x_max)
plt.ylim()  # Auto-adapt y-axis range; can also be set manually: plt.ylim(min_reward, max_reward)
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)  # Thin grid lines with low transparency
# Legend optimization (display in 2 columns to avoid overlap)
plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.1), fontsize=14, frameon=True, framealpha=0.8, 
           edgecolor='white', ncol=2, columnspacing=1.0)
plt.tight_layout(pad=0.5)  # Compact layout to reduce white space

# Adjust axis style (academic style)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# 4) Save (common formats for academic papers: PNG with high DPI, or EPS vector graphics)
plt.savefig(compare_save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

# plt.savefig(compare_save_path.replace('.png', '.eps'), format='eps', bbox_inches='tight', facecolor='white')
print(f"\nPID structure reward comparison plot (with confidence intervals) saved: {compare_save_path}")
plt.show()

# 5) Output key metric comparisons (for easy analysis)
def get_key_metrics(df, algo_name):
    """Calculate key reward metrics"""
    max_reward = df['original_eval_reward'].max()
    max_episode = df[df['original_eval_reward'] == max_reward]['episode'].values[0]
    final_reward = df['original_eval_reward'].iloc[-1]
    mean_reward = df['original_eval_reward'].mean()
    std_reward = df['original_eval_reward'].std()
    return {
        'PID Structure': algo_name,
        'Maximum Reward': round(max_reward, 2),
        'Episode at Max Reward': int(max_episode),
        'Final Reward': round(final_reward, 2),
        'Average Reward': round(mean_reward, 2),
        'Reward Standard Deviation': round(std_reward, 2),
        'Data Length (Episodes)': int(df['episode'].max())
    }

# Summarize and print key metrics
metrics = pd.DataFrame([
    get_key_metrics(p_df, 'P_only'),
    get_key_metrics(i_df, 'I_only'),
    get_key_metrics(d_df, 'D_only'),
    get_key_metrics(pi_df, 'PI'),
    get_key_metrics(pd_df, 'PD'),
    get_key_metrics(id_df, 'ID'),
    get_key_metrics(pid_df, 'PID')
])
print("\n========== Key Reward Metrics for Each PID Structure ==========")
print(metrics.to_string(index=False))