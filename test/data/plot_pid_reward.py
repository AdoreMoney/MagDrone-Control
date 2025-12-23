import numpy as np
import matplotlib.pyplot as plt
import os

# ===================== Global Configuration Switches (Core: One-Click Show/Hide Control) =====================
SHOW_LEGEND = True               # Legend switch: True=Show (single plot), False=Hide (multiple plots)
SHOW_X_AXIS_LABEL = True        # X-axis label switch: True=Show, False=Hide
SHOW_Y_AXIS_LABEL = True        # Y-axis label switch: True=Show, False=Hide
# Figure size (cm): Smaller for multiple plots, larger for single plot
FIG_WIDTH_CM = 6   
FIG_HEIGHT_CM = 4  
seed = 40          # Random seed (corresponding to NPZ file naming)
wind_type = "turbulence"        # Wind field type

# ===================== Basic Configuration (Paper-Grade Format) =====================
plt.rcParams.update({
    'font.family': 'Arial',          # Common English font for academic papers
    'font.size': 10,                 # Standard font size for journals
    'axes.linewidth': 0.8,           # Fine axis line width
    'axes.labelsize': 9,             # Axis label font size
    'xtick.labelsize': 9,            # X-axis tick label font size
    'ytick.labelsize': 9,            # Y-axis tick label font size
    'legend.fontsize': 6,            # Legend font size
    'legend.framealpha': 0.9,        # Legend background transparency
    'legend.edgecolor': 'none',      # No legend border (clean style)
    'grid.alpha': 0.3,               # Grid transparency
    'grid.linestyle': '--',          # Grid line style
    'grid.linewidth': 0.5,           # Grid line width
    'figure.dpi': 600,               # High resolution (≥300dpi)
    'savefig.dpi': 600,              # Saved figure resolution
    'savefig.facecolor': 'white'     # White background (avoid printing anomalies)
})

# ===================== File Path Configuration =====================
# NPZ file paths for each PID sub-structure (heading data)
file_paths = {
    "PID": f"./npz/15deg/{wind_type}_{seed}seed/PID_data_{wind_type}.npz",
    "P": f"./npz/15deg/{wind_type}_{seed}seed/PPO_Kp_data_{wind_type}.npz",
    "I": f"./npz/15deg/{wind_type}_{seed}seed/PPO_Ki_data_{wind_type}.npz",
    "D": f"./npz/15deg/{wind_type}_{seed}seed/PPO_Kd_data_{wind_type}.npz",
    "PI": f"./npz/15deg/{wind_type}_{seed}seed/PPO_KpKi_data_{wind_type}.npz",
    "PD": f"./npz/15deg/{wind_type}_{seed}seed/PPO_KpKd_data_{wind_type}.npz",
    "ID": f"./npz/15deg/{wind_type}_{seed}seed/PPO_KiKd_data_{wind_type}.npz"
}

# ===================== Color and Style Configuration =====================
style_config = {
    "PID":  {'linestyle': '-', 'color': '#000000', 'label': 'PID*'},          # Black to highlight full PID
    "P":    {'linestyle': '-', 'color': '#e74c3c', 'label': 'P*'},             # Red
    "I":    {'linestyle': '-', 'color': '#8e44ad', 'label': 'I*'},             # Purple
    "D":    {'linestyle': '-', 'color': '#27ae60', 'label': 'D*'},             # Green
    "PI":   {'linestyle': '-', 'color': '#f39c12', 'label': 'PI*'},            # Orange
    "PD":   {'linestyle': '-', 'color': '#3498db', 'label': 'PD*'},            # Blue
    "ID":   {'linestyle': '-', 'color': '#ff69b4', 'label': 'ID*'}            # Hot pink
}

# Check file existence and filter invalid files
valid_algos = []
heading_data_dict = {}  # Store heading data
target_heading_dict = {}  # Store target heading data

for algo, path in file_paths.items():
    abs_path = os.path.abspath(path)  # Convert to absolute path for easy debugging
    if os.path.exists(abs_path):
        valid_algos.append(algo)
        data = np.load(abs_path)
        
        # Check for heading data fields
        if 'heading_deg' not in data.files:
            print(f"⚠️ Warning: No heading_deg data in {algo} file")
            continue
        if 'target_heading_deg' not in data.files:
            print(f"⚠️ Warning: No target_heading_deg data in {algo} file")
            continue
        
        heading_data_dict[algo] = data['heading_deg']
        target_heading_dict[algo] = data['target_heading_deg']
        print(f"✅ Successfully loaded {algo} heading data: {abs_path}, data length: {len(heading_data_dict[algo])}")
    else:
        print(f"❌ Warning: {algo} file does not exist: {abs_path}, skipping this algorithm")

if not valid_algos:
    raise ValueError("No valid heading data files found, please check the paths!")

# ===================== Data Preprocessing =====================
dt = 0.01  # Simulation time step (s)
# Use the shortest data length to avoid dimension mismatch
min_length = min([len(heading_data_dict[algo]) for algo in valid_algos])
time_series = np.arange(min_length) * dt  # Time series (s)

# Extract and truncate heading/target heading data
extracted_data = {}
for algo in valid_algos:
    extracted_data[algo] = {
        'heading': heading_data_dict[algo][:min_length],
        'target_heading': target_heading_dict[algo][:min_length]
    }

# ===================== Plot Heading Comparison Graph =====================
# Convert cm to inches
fig_width_inch = FIG_WIDTH_CM * 0.3937
fig_height_inch = FIG_HEIGHT_CM * 0.3937
fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch))

# Store plotted line objects (for legend construction)
plot_handles = []
plot_labels = []
# Manually specify legend order (ensure 4 on top, 3 on bottom layout)
legend_order = ["PID", "P", "I", "D", "PI", "PD", "ID"]

# Plot heading curves for each structure (in specified order)
for algo in legend_order:
    if algo in valid_algos:
        cfg = style_config[algo]
        line = ax.plot(time_series, extracted_data[algo]['heading'],
                color=cfg['color'], linestyle=cfg['linestyle'],
                label=cfg['label'], linewidth=1.0, alpha=0.8)[0]  # Get line object
        plot_handles.append(line)
        plot_labels.append(cfg['label'])

# Axis label control
if SHOW_Y_AXIS_LABEL:
    ax.set_ylabel('Heading(°)', fontweight='normal')
else:
    ax.set_ylabel('')

if SHOW_X_AXIS_LABEL:
    ax.set_xlabel('Time (s)', fontweight='normal')
else:
    ax.set_xlabel('')

# Hide top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend control: Inside axes, two rows above x-axis (compatible with older matplotlib versions)
if SHOW_LEGEND:
    # Option 1: Use ncol=4, adjust bbox_to_anchor for line spacing (compatible with all versions)
    legend = ax.legend(
        plot_handles, plot_labels,
        loc='upper center',          # Base position at top center of axes
        bbox_to_anchor=(0.5, 0.28),  # Adjust y-value to fit two rows
        ncol=4,                      # 4 items per row, auto-split into two rows (4+3)
        frameon=False,               # Remove legend background box to integrate with axes
        fontsize=6,                  # Adapt font size
        columnspacing=1.2,           # Column spacing
        handlelength=1.2,            # Legend line length
        handletextpad=0.4,           # Spacing between line and text
        borderaxespad=0.0            # Zero spacing between legend and axes
    )
    
    # Optional: For more precise line spacing control (compatible with all versions)
    # Manually adjust vertical spacing of legend text
    plt.setp(legend.get_texts(), linespacing=1.2)  # Adjust text line spacing

# Grid control (Y-axis only)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Tight layout
plt.tight_layout(pad=0.3)

# ===================== Save and Display =====================
save_path = f'./tiff/{wind_type}_{seed}seed_pid_structures_heading.png'
plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0.05)
print(f"✅ Heading comparison plot saved to: {save_path}")

# Display plot
plt.show()

# ===================== Heading Performance Metric Calculation =====================
print("\n===== Heading Tracking Performance Metric Comparison =====")
print(f"{'Structure':<8} {'Average Heading Error (deg)':<18} {'Max Heading Error (deg)':<18} {'Error Std Dev (deg)':<18}")
print("-" * 65)
for algo in valid_algos:
    heading = extracted_data[algo]['heading']
    target_heading = extracted_data[algo]['target_heading']
    # Calculate heading error (absolute value)
    heading_error = np.abs(heading - target_heading)
    avg_error = np.mean(heading_error)  
    max_error = np.max(heading_error)   
    std_error = np.std(heading_error)
    print(f"{algo:<8} {avg_error:<18.4f} {max_error:<18.4f} {std_error:<18.4f}")

# Additional calculation of steady-state error (average error of last 10% of data)
print("\n===== Heading Steady-State Performance Metrics =====")
print(f"{'Structure':<8} {'Steady-State Error (deg)':<18} {'Error Convergence Rate (%)':<18}")
print("-" * 45)
steady_ratio = 0.1  # Use last 10% of data as steady state
for algo in valid_algos:
    heading = extracted_data[algo]['heading']
    target_heading = extracted_data[algo]['target_heading']
    heading_error = np.abs(heading - target_heading)
    
    # Calculate steady-state error
    steady_start_idx = int(len(heading_error) * (1 - steady_ratio))
    steady_error = np.mean(heading_error[steady_start_idx:])
    
    # Calculate error convergence rate ((initial error - steady error)/initial error * 100%)
    initial_error = np.mean(heading_error[:int(len(heading_error)*0.1)])
    if initial_error > 0:
        converge_rate = (initial_error - steady_error) / initial_error * 100
    else:
        converge_rate = 0.0
    
    print(f"{algo:<8} {steady_error:<18.4f} {converge_rate:<18.2f}")