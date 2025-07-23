import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
import scienceplots

# Data preparation
data = {
    'Case': ['1', '1', '1', '2', '2', '2', '3', '3', '3', '4', '4', '4'],
    'Type': ['PID', 'PPO+PID', 'Improvement', 'PID', 'PPO+PID', 'Improvement', 
             'PID', 'PPO+PID', 'Improvement', 'PID', 'PPO+PID', 'Improvement'],
    'χ_error': [8.23, 3.46, 57.9, 6.9, 3.46, 49.9, 11.55, 1.89, 83.6, 3.83, 2.22, 41.9],
    'σ(δ_diff)': [7.826, 2.763, 64.7, 8.204, 5.138, 37.21, 7.428, 3.264, 56.1, 6.97, 2.36, 66.14],
    'σ(ϕ)': [1.855, 1.739, 6.25, 2.129, 2.254, -1.58, 1.212, 1.294, -6.77, 1.109, 1.012, 8.75],
    'σ(ψ)': [0.275, 0.224, 18.55, 0.22, 0.256, -16.4, 0.184, 0.104, 43.48, 0.113, 0.138, -22.12]
}
df = pd.DataFrame(data)

# Set Science plotting style
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({
    'font.size': 16,  # Reduce font size to fit horizontal layout
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.dpi': 600,
    'figure.figsize': (12, 4),  # Adjust canvas size to wide format
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Create 1x4 subplots
fig, axs = plt.subplots(1, 4, constrained_layout=True)
metrics = ['χ_error', 'σ(δ_diff)', 'σ(ϕ)', 'σ(ψ)']
titles = [r'$RMSE(\bar{\chi}_{error})$', 
          r'$\sigma(\delta_{e,diff})$', 
          r'$\sigma(\phi)$', 
          r'$\sigma(\psi)$']

# Plot comparison for each metric
for i, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axs[i]
    
    # Extract PID and PPO+PID data
    pid_data = df[df['Type']=='PID'][metric].values
    ppo_data = df[df['Type']=='PPO+PID'][metric].values
    
    # Bar chart parameters
    x = np.arange(4)
    width = 0.35
    
    # Plot bar chart
    bars1 = ax.bar(x - width/2, pid_data, width, label='PID', color='#1f77b4', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, ppo_data, width, label='PPO+PID', color='#ff7f0e', edgecolor='black', linewidth=0.5)
    
    # Add improvement percentage annotations
    for j in range(4):
        imp = df[df['Type']=='Improvement'][metric].iloc[j]
        color = 'green' if imp > 0 else 'red'
        ax.text(x[j], max(pid_data[j], ppo_data[j]) + 0.1, 
                f'{imp:.1f}%', ha='center', color=color, fontsize=14)
    
    # Set chart properties
    ax.set_xticks(x)
    ax.set_xticklabels(['1', '2', '3', '4'], fontsize=16)
    ax.set_xlabel('Case', fontsize=16)
    ax.set_ylabel(title, fontsize=16)
    
    # Only show legend on the last subplot (i=3)
    if i == 3:
        ax.legend(loc='upper right', frameon=True, fancybox=False, fontsize=16)
        
    # Set Y-axis limits
    if metric == 'χ_error':
        ax.set_ylim(0, 13)
    elif metric == 'σ(δ_diff)':
        ax.set_ylim(0, 9)
    else:
        ax.set_ylim(0, 2.5)

# Save the image
plt.savefig('23_Performance comparison between PPO-PID and PID control.png', bbox_inches='tight', dpi=600)