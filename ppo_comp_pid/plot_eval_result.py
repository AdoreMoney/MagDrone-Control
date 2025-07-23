import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 1. Set global styles
plt.style.use(['seaborn-v0_8-whitegrid', 'grayscale']) 
mpl.rcParams.update({
    'font.size': 16,
    'font.family': 'Times New Roman',
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 12,
    'figure.facecolor': 'white',
    'lines.markersize': 6,  
    
    'grid.color': '#dddddd',    
    'grid.alpha': 0.9,          
    'grid.linewidth': 0.8,      
})

# 2. Define color scheme
colors = {
    'pid': '#377eb8',  # Blue
    'ppo': '#e41a1c',  # Red
    'target': '#1b7c3d',  # Green
}

# Load data while ensuring correct handling of array dimensions
ppo_data = np.load('ppo_data_04.npz', allow_pickle=True)
pid_data = np.load('pid_data_04.npz', allow_pickle=True)

# Extract time steps (assuming the same sampling rate)
time_steps = np.arange(len(ppo_data['heading_deg']))

# Create 4 subplots (one for each variable)
fig, axs = plt.subplots(4, 1, figsize=(6, 8), dpi=600, sharex=True, gridspec_kw={'hspace': 0.15})
marker_interval = 200

# 1. Heading angle comparison
axs[0].plot(time_steps, ppo_data['heading_deg'], color=colors['ppo'], label='PPO+PID', linewidth=1.5,  marker='^', markevery=marker_interval)
axs[0].plot(time_steps, pid_data['heading_deg'], color=colors['pid'], linestyle='--', label='PID', linewidth=1.5, marker='o', markevery=marker_interval)
axs[0].plot(time_steps, ppo_data['target_heading_deg'], color=colors['target'], linestyle='--', label='Target', linewidth=1.5)
#axs[0].set_ylabel('Heading [deg]')
axs[0].set_title('Heading (deg)', loc='center', pad=0.2, y=1.02, fontweight='bold')
axs[0].legend(frameon=True, loc='lower right')
# Calculate and print average error
#pid_error = np.abs(pid_data['heading_deg'] - pid_data['target_heading_deg'])
#ppo_error = np.abs(ppo_data['heading_deg'] - ppo_data['target_heading_deg'])
# Calculate and print RMSE error
pid_error = np.sqrt(np.mean((pid_data['heading_deg'] - pid_data['target_heading_deg'])**2))
ppo_error = np.sqrt(np.mean((ppo_data['heading_deg'] - ppo_data['target_heading_deg'])**2))

print('\n' + '='*50)
print(f"PID average heading RMSE: {np.mean(pid_error):.2f}° ± {np.std(pid_error):.2f}°")
print(f"PPO average heading RMSE: {np.mean(ppo_error):.2f}° ± {np.std(ppo_error):.2f}°")
print(f"Error reduction ratio: {(1 - np.mean(ppo_error)/np.mean(pid_error))*100:.1f}%")
print('='*50 + '\n')

# 2. Roll angle comparison
axs[1].plot(time_steps, ppo_data['phi_deg'], color=colors['ppo'], label='PPO+PID', linewidth=1.5, marker='^', markevery=marker_interval)
axs[1].plot(time_steps, pid_data['phi_deg'], color=colors['pid'], linestyle='--', label='PID', linewidth=1.5, marker='o', markevery=marker_interval)
#axs[1].set_ylabel('Roll [deg]')
axs[1].set_title('Roll (deg)', loc='center', pad=0.2, y=1.02, fontweight='bold')
axs[1].legend(frameon=True, loc='lower right')

# 3. Yaw angle comparison
axs[2].plot(time_steps, ppo_data['psi_deg'], color=colors['ppo'], label='PPO+PID', linewidth=1.5, marker='^', markevery=marker_interval)
axs[2].plot(time_steps, pid_data['psi_deg'], color=colors['pid'], linestyle='--', label='PID', linewidth=1.5, marker='o', markevery=marker_interval)
#axs[2].set_ylabel('Yaw [deg]')
axs[2].set_title('Yaw (deg)', loc='center', pad=0.2, y=1.02, fontweight='bold')
axs[2].legend(frameon=True, loc='lower right')

# 4. Control surface deflection comparison
axs[3].plot(time_steps, ppo_data['control_deg'], color=colors['ppo'], label='PPO+PID', linewidth=1.5, marker='^', markevery=marker_interval)
axs[3].plot(time_steps, pid_data['control_deg'], color=colors['pid'], linestyle='--', label='PID', linewidth=1.5, marker='o', markevery=marker_interval)
#axs[3].set_ylabel('Elev_diff [deg]')
axs[3].set_title('Elev diff (deg)', loc='center', pad=0.2, y=1.02, fontweight='bold')
axs[3].set_xlabel('Timestep')
axs[3].legend(frameon=True, loc='lower right')

# Global adjustments
plt.tight_layout()
#plt.savefig('24_control_comparison_01.png', bbox_inches='tight', transparent=True)  # Transparent background for paper insertion
#plt.savefig('25_control_comparison_02.png', bbox_inches='tight', transparent=True)
#plt.savefig('26_control_comparison_03.png', bbox_inches='tight', transparent=True)
plt.savefig('27_control_comparison_04.png', bbox_inches='tight', transparent=True)
plt.show()