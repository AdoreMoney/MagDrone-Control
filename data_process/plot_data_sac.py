import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ================= User Configurable Parameters =================
directory = r'C:\Users\a1fla\Desktop\MagDrone\logs\sac\MagDrone_sac_tensorboard\sac_training_logs'  # Directory for storing input data files
file_pattern = os.path.join(directory, 'sac_*.csv')                                        # File matching pattern to locate SAC log files
step_per_episode = 2000                # Conversion factor for mapping training steps to episodes
save_path = os.path.join(directory, 'sac_reward_curve.png')                 # Output path for saving the reward curve visualization
data_save_path = os.path.join(directory, 'sac_reward_data_original.csv')    # Output path for saving the processed raw reward data in CSV format
encoding = None                                            # Encoding for CSV files: None for auto-detection, alternatives include 'utf-8-sig' and 'gbk'
# =================================================

def read_file(f, encoding):
    """Read data file by its extension and return a pandas DataFrame. Return None if the file format is unsupported or reading fails.
    
    Args:
        f (str): Path to the input file
        encoding (str, optional): Encoding used for reading text files
        
    Returns:
        pd.DataFrame or None: Loaded and parsed data as DataFrame, or None if reading fails
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
        print(f"File reading failed for: {f}, Error message: {e}")
        return None

def resample_to_episode(df, step_col='step', episode_col='episode', reward_col='eval/mean_reward'):
    """Resample step-based training data into episode-based sequence by retaining the last data point of each episode.
    
    This function converts raw step-wise training logs into a structured episode-wise format, which is more intuitive
    for analyzing reinforcement learning training progress.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing step-based training data
        step_col (str): Column name for training step indices
        episode_col (str): Column name for the generated episode indices
        reward_col (str): Column name for the evaluation reward values
        
    Returns:
        pd.DataFrame: Resampled episode-based DataFrame with aggregated reward values
    """
    df = df.sort_values(step_col).reset_index(drop=True)
    df[episode_col] = (df[step_col] // step_per_episode).astype(int)
    # Retain only the final evaluation reward value for each training episode
    grouped = df.groupby(episode_col, as_index=False).last()[[episode_col, reward_col]]
    return grouped

# 1) Locate all matching log files
files = glob.glob(file_pattern)
if not files:
    raise FileNotFoundError(f"No matching files found in directory with pattern: {file_pattern}")

merged = pd.DataFrame()

# 2) Load and merge data from all found files
for f in files:
    df = read_file(f, encoding)
    if df is None:
        continue

    # Standardize column names: remove BOM characters and leading/trailing whitespace
    df.columns = (df.columns.astype(str)
                  .str.replace('\ufeff', '', regex=False)
                  .str.strip())

    # Validate required columns for further processing
    needed_columns = {'step', 'eval/mean_reward'}
    missing_columns = needed_columns - set(df.columns)
    if missing_columns:
        print(f"Skipping file {os.path.basename(f)} due to missing required columns: {missing_columns}")
        continue

    # Data type conversion and cleaning
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df['eval/mean_reward'] = pd.to_numeric(df['eval/mean_reward'], errors='coerce')
    df = df.dropna(subset=['step', 'eval/mean_reward'])

    # Deduplicate records by step, keeping only the most recent entry
    df = df.sort_values('step').drop_duplicates(subset='step', keep='last')

    merged = pd.concat([merged, df[['step', 'eval/mean_reward']]], ignore_index=True)

# 3) Validate merged dataset integrity
if merged.empty:
    raise ValueError("No valid data remaining after merging. Please verify file contents and column names.")

# 4) Convert step-based data to episode-based format and remove duplicates
merged = resample_to_episode(merged)

# 5) Sort data by episode for consistent visualization
merged = merged.sort_values('episode').reset_index(drop=True)

# ========== Save processed raw data (episode and original reward only) ==========
# Rename columns for better readability and usability
output_df = merged.rename(columns={
    'episode': 'Episode',
    'eval/mean_reward': 'Original_Eval_Reward'
})

# Save data in CSV format (universally compatible with most data analysis tools)
output_df.to_csv(data_save_path, index=False, encoding='utf-8-sig')
print(f"SAC raw reward data saved to CSV file: {data_save_path}")

# ========== Generate and save reward curve visualization (raw data only) ==========
plt.figure(figsize=(10, 6))
plt.plot(merged['episode'], merged['eval/mean_reward'],
         color='tab:green', alpha=1.0, linewidth=1.5, label='Original Eval Reward (SAC)')

plt.xlabel('Episode', fontsize=14)
plt.ylabel('Evaluation Mean Reward', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=14)
plt.tight_layout()

# Save visualization with high DPI for publication-quality results
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"SAC reward curve visualization saved to: {save_path}")
plt.show()

# ========== Print preview of the processed raw data ==========
print("\nPreview of the first 10 rows of SAC raw reward data:")
print(output_df.head(10))
print(f"\nTotal number of rows in SAC raw data: {len(output_df)}")
print(f"SAC raw reward value range: [{output_df['Original_Eval_Reward'].min():.2f}, {output_df['Original_Eval_Reward'].max():.2f}]")