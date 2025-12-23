import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ================= User Configurable Parameters =================
directory = r'C:\Users\a1fla\Desktop\MagDrone\logs\td3\MagDrone_td3_tensorboard\td3_training_logs'  # Directory storing TD3 training log files
file_pattern = os.path.join(directory, 'td3_*.csv')                                        # File pattern to match TD3 log CSV files
step_per_episode = 2000                                                               # Conversion factor for mapping training steps to episodes
save_path = os.path.join(directory, 'td3_reward_curve.png')                 # Output path for saving the TD3 reward curve visualization
data_save_path = os.path.join(directory, 'td3_reward_data_original.csv')    # Output path for saving processed raw TD3 reward data in CSV format
encoding = None                                            # CSV file encoding: None for auto-detection, alternatives include 'utf-8-sig' and 'gbk'
# =================================================

def read_file(f, encoding):
    """Read input file by its extension and return a pandas DataFrame, or None if reading fails or format is unsupported.
    
    Args:
        f (str): Full path to the input file
        encoding (str, optional): Character encoding used for reading text-based files (e.g., CSV)
        
    Returns:
        pd.DataFrame or None: Parsed data as DataFrame if reading succeeds, None otherwise
    """
    try:
        if f.lower().endswith('.csv'):
            return pd.read_csv(f, engine='python', encoding=encoding)
        elif f.lower().endswith(('.xls', '.xlsx')):
            return pd.read_excel(f)
        else:
            print(f"Skipping unsupported file format: {f}")
            return None
    except Exception as e:
        print(f"File reading failed for: {f}, Error details: {e}")
        return None

def resample_to_episode(df, step_col='step', episode_col='episode', reward_col='eval/mean_reward'):
    """Resample step-based reinforcement learning training data into an episode-based sequence by retaining the last data point per episode.
    
    This function converts raw step-wise training logs into a structured episode-wise format, which is more intuitive
    for analyzing the training progress and performance trends of TD3 (Twin Delayed Deep Deterministic Policy Gradient).
    
    Args:
        df (pd.DataFrame): Input DataFrame containing step-indexed training data
        step_col (str): Column name corresponding to training step indices
        episode_col (str): Column name for the generated episode indices (will be added to the DataFrame)
        reward_col (str): Column name corresponding to the evaluation mean reward values
        
    Returns:
        pd.DataFrame: Episode-based DataFrame with aggregated reward values (one row per episode)
    """
    df = df.sort_values(step_col).reset_index(drop=True)
    df[episode_col] = (df[step_col] // step_per_episode).astype(int)
    # Retain only the final evaluation reward value for each training episode to ensure consistency
    grouped = df.groupby(episode_col, as_index=False).last()[[episode_col, reward_col]]
    return grouped

# 1) Locate all TD3 log files matching the specified pattern
files = glob.glob(file_pattern)
if not files:
    raise FileNotFoundError(f"No files matching the pattern were found in the directory: {file_pattern}")

merged = pd.DataFrame()

# 2) Load, validate, and merge data from all matched files
for f in files:
    df = read_file(f, encoding)
    if df is None:
        continue

    # Standardize column names: remove BOM markers and trim leading/trailing whitespace to avoid mismatches
    df.columns = (df.columns.astype(str)
                  .str.replace('\ufeff', '', regex=False)
                  .str.strip())

    # Validate presence of required columns for TD3 reward analysis
    required_columns = {'step', 'eval/mean_reward'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Skipping file {os.path.basename(f)} due to missing required columns: {missing_columns}")
        continue

    # Data type conversion and cleaning: remove invalid or missing values
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df['eval/mean_reward'] = pd.to_numeric(df['eval/mean_reward'], errors='coerce')
    df = df.dropna(subset=['step', 'eval/mean_reward'])

    # Deduplicate records by training step, keeping only the most recent entry (last recorded value for each step)
    df = df.sort_values('step').drop_duplicates(subset='step', keep='last')

    # Merge current file's valid data into the consolidated DataFrame
    merged = pd.concat([merged, df[['step', 'eval/mean_reward']]], ignore_index=True)

# 3) Validate the integrity of the merged dataset
if merged.empty:
    raise ValueError("No valid data remains after merging all files. Please verify file contents and column names are correct.")

# 4) Convert step-based data to episode-based format and remove any residual duplicates
merged = resample_to_episode(merged)

# 5) Sort the final dataset by episode for consistent visualization and analysis
merged = merged.sort_values('episode').reset_index(drop=True)

# ========== Save Processed Raw Data (Episode and Original Evaluation Reward Only) ==========
# Rename columns to more intuitive, human-readable names for better usability
output_df = merged.rename(columns={
    'episode': 'Episode',
    'eval/mean_reward': 'Original_Eval_Reward'
})

# Save data in CSV format (universally compatible with data analysis tools like Pandas, Excel, R)
output_df.to_csv(data_save_path, index=False, encoding='utf-8-sig')
print(f"Processed raw TD3 reward data saved to CSV: {data_save_path}")

# ========== Generate and Save Reward Curve Visualization (Raw Data Only) ==========
plt.figure(figsize=(10, 6))
plt.plot(merged['episode'], merged['eval/mean_reward'],
         color='tab:blue', alpha=1.0, linewidth=1.5, label='Original Eval Reward (TD3)')

plt.xlabel('Episode', fontsize=14)
plt.ylabel('Evaluation Mean Reward', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()  # Optimize layout to prevent label cutoff

# Save visualization with high DPI (300) for publication-quality or high-resolution display
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"TD3 reward curve visualization saved to: {save_path}")
plt.show()

# ========== Print Preview of the Processed Raw Data ==========
print("\nPreview of the first 10 rows of processed TD3 raw reward data:")
print(output_df.head(10))
print(f"\nTotal number of episodes in processed TD3 data: {len(output_df)}")
print(f"TD3 evaluation reward range: [{output_df['Original_Eval_Reward'].min():.2f}, {output_df['Original_Eval_Reward'].max():.2f}]")