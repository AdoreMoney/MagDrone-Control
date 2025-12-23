import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ================= User Configurable Parameters =================
#directory = r'C:\Users\a1fla\Desktop\MagDrone\logs\ppo\MagDrone_ppo_tensorboard\ppo_training_logs'  # Data directory
directory = r'C:\Users\a1fla\Desktop\MagDrone\logs\ppo\KiKd\MagDrone_ppo_tensorboard'
file_pattern = os.path.join(directory, 'ppo_*.csv')                                        # File matching pattern
step_per_episode = 2000                                                               # Step to episode conversion factor
save_path = os.path.join(directory, 'ppo_reward_curve.png')                 # Image save path
data_save_path = os.path.join(directory, 'ppo_reward_data_original.csv')    # Raw data save path (CSV)
encoding = None                                            # CSV encoding: None=auto, 'utf-8-sig'/'gbk' as alternatives
# =================================================

def read_file(f, encoding):
    """Read file by extension, return DataFrame or None"""
    try:
        if f.lower().endswith('.csv'):
            return pd.read_csv(f, engine='python', encoding=encoding)
        elif f.lower().endswith(('.xls', '.xlsx')):
            return pd.read_excel(f)
        else:
            print(f"Skipping unsupported format: {f}")
            return None
    except Exception as e:
        print(f"Failed to read: {f}, Error: {e}")
        return None

def resample_to_episode(df, step_col='step', episode_col='episode', reward_col='eval/mean_reward'):
    """Resample step-based data into an evenly spaced episode-based sequence (keep the last point of each episode)"""
    df = df.sort_values(step_col).reset_index(drop=True)
    df[episode_col] = (df[step_col] // step_per_episode).astype(int)
    # Keep the last step's evaluation value for each episode
    grouped = df.groupby(episode_col, as_index=False).last()[[episode_col, reward_col]]
    return grouped

# 1) Find files
files = glob.glob(file_pattern)
if not files:
    raise FileNotFoundError(f"No matching files found in directory: {file_pattern}")

merged = pd.DataFrame()

# 2) Read and merge files
for f in files:
    df = read_file(f, encoding)
    if df is None:
        continue

    # Standardize column names: remove BOM, trim leading/trailing spaces
    df.columns = (df.columns.astype(str)
                  .str.replace('\ufeff', '', regex=False)
                  .str.strip())

    # Check for required columns
    needed = {'step', 'eval/mean_reward'}
    miss = needed - set(df.columns)
    if miss:
        print(f"Skipping file {os.path.basename(f)} with missing columns: {miss}")
        continue

    # Type conversion and data cleaning
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df['eval/mean_reward'] = pd.to_numeric(df['eval/mean_reward'], errors='coerce')
    df = df.dropna(subset=['step', 'eval/mean_reward'])

    # Deduplicate by step: keep the last entry
    df = df.sort_values('step').drop_duplicates(subset='step', keep='last')

    merged = pd.concat([merged, df[['step', 'eval/mean_reward']]], ignore_index=True)

# 3) Check merged result
if merged.empty:
    raise ValueError("No valid data after merging, please check file contents and column names.")

# 4) Convert to episodes and deduplicate (keep the last evaluation point per episode)
merged = resample_to_episode(merged)

# 5) Sort (keep only raw data)
merged = merged.sort_values('episode').reset_index(drop=True)

# ========== Save raw data (only episode and original reward) ==========
# Rename columns (more intuitive)
output_df = merged.rename(columns={
    'episode': 'Episode',
    'eval/mean_reward': 'Original_Eval_Reward'
})

# Save as CSV file (universal format, readable by all tools)
output_df.to_csv(data_save_path, index=False, encoding='utf-8-sig')
print(f"PPO raw data CSV file saved: {data_save_path}")

# ========== Plotting (only raw curve) ==========
plt.figure(figsize=(10, 6))
plt.plot(merged['episode'], merged['eval/mean_reward'],
         color='tab:red', alpha=1.0, linewidth=1.5, label='Original Eval Reward (PPO)')

plt.xlabel('episode', fontsize=14)
plt.ylabel('eval/mean_reward', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()

# Save and display the image
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"PPO reward curve image saved: {save_path}")
plt.show()

# ========== Print raw data preview ==========
print("\nPPO raw data file - first 10 rows preview:")
print(output_df.head(10))
print(f"\nTotal rows of PPO raw data: {len(output_df)}")
print(f"PPO raw reward range: [{output_df['Original_Eval_Reward'].min():.2f}, {output_df['Original_Eval_Reward'].max():.2f}]")