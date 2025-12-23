import numpy as np
import matplotlib.pyplot as plt
import os

# ===================== Global Configuration Switches (Core: New Heading RMSE Plot Switch) =====================
SHOW_LEGEND = False              # Legend switch: True=Show (single plot), False=Hide (multiple plots)
SHOW_X_AXIS_LABEL = False         # X-axis label switch: True=Show, False=Hide
SHOW_Y_AXIS_LABEL = False         # Y-axis label switch: True=Show, False=Hide
ONLY_PLOT_WIND = False            # Wind speed plot only switch (True=Plot wind speed only)
PLOT_CONTROL_DEG = True          # Control input (control_deg) plot switch (True=Plot control input)
PLOT_PHI_DEG = False               # Roll angle (phi_deg) plot switch (True=Plot roll angle)
PLOT_HEADING_RMSE = False         # New: Heading RMSE comparison plot switch (True=Plot RMSE, higher priority than other plots)
# Figure size (cm): Smaller for multiple plots, larger for single plot
FIG_WIDTH_CM = 6   
FIG_HEIGHT_CM = 4  
wind = 40

# ===================== Basic Configuration (Paper-Grade Format + Full Border Preset) =====================
plt.rcParams.update({
    'font.family': 'Arial',          # Common English font for papers
    'font.size': 10,                 # Common font size for journals
    'axes.linewidth': 0.8,           # Axis line width (fine, uniform for full borders)
    'axes.labelsize': 10,            # Axis label font size
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
    'savefig.facecolor': 'white'     # White background (avoid printing issues)
})

# ===================== File Path Configuration =====================
file_paths = {
    "PID": f"./npz/sin/turbulence_{wind}seed/PID_data_turbulence.npz",
    "PPO": f"./npz/sin/turbulence_{wind}seed/PPO_data_turbulence.npz",
    "SAC": f"./npz/sin/turbulence_{wind}seed/SAC_data_turbulence.npz",
    "ADRC": f"./npz/sin/turbulence_{wind}seed/ADRC_data_turbulence.npz",
    "TD3": f"./npz/sin/turbulence_{wind}seed/TD3_data_turbulence.npz"
}

# Check file existence and filter invalid files
valid_algos = []
data_dict = {}
wind_data_dict = {}  # Store wind speed data
control_data_dict = {}  # Store control input (control_deg) data
phi_data_dict = {}     # Store roll angle (phi_deg) data
for algo, path in file_paths.items():
    abs_path = os.path.abspath(path)  # Convert to absolute path for easy debugging
    if os.path.exists(abs_path):
        valid_algos.append(algo)
        data = np.load(abs_path)
        data_dict[algo] = data
        
        # Extract wind speed data
        if 'wind_v_mps' in data.files:
            wind_data_dict[algo] = data['wind_v_mps']
        else:
            print(f"⚠️ Warning: No wind_v_mps data in {algo} file")
        
        # Extract control input data
        if 'control_deg' in data.files:
            control_data_dict[algo] = data['control_deg']
        else:
            print(f"⚠️ Warning: No control_deg data in {algo} file")
        
        # Extract roll angle (phi_deg) data
        if 'phi_deg' in data.files:
            phi_data_dict[algo] = data['phi_deg']
        else:
            print(f"⚠️ Warning: No phi_deg data in {algo} file")
        
        print(f"✅ Successfully loaded {algo} data: {abs_path}")
    else:
        print(f"❌ Warning: {algo} file does not exist: {abs_path}, skipping this algorithm")

if not valid_algos:
    raise ValueError("No valid data files found, please check the paths!")

# ===================== Data Preprocessing =====================
dt = 0.01  # Simulation time step
# Use the shortest data length to avoid dimension mismatch
min_length = min([len(data_dict[algo]['heading_deg']) for algo in valid_algos])
time_turbulences = np.arange(min_length) * dt  # Time series (s)
# Calculate maximum X-axis tick value (for setting tick intervals)
x_max = np.max(time_turbulences)

# Extract and truncate data
extracted_data = {}
extracted_wind_data = {}
extracted_control_data = {}
extracted_phi_data = {}
extracted_heading_rmse_data = {}  # New: Store heading RMSE related data
for algo in valid_algos:
    data = data_dict[algo]
    # Heading data
    heading = data['heading_deg'][:min_length]
    target_heading = data['target_heading_deg'][:min_length]
    extracted_data[algo] = {
        'heading': heading,          
        'target_heading': target_heading  
    }
    # New: Calculate heading error (for subsequent RMSE calculation, supports sliding window or global RMSE; first calculate point-wise squared error)
    heading_error_sq = (heading - target_heading) ** 2
    extracted_heading_rmse_data[algo] = {
        'heading_error_sq': heading_error_sq,
        'heading_error': np.abs(heading - target_heading)
    }
    # Wind speed data
    if algo in wind_data_dict:
        extracted_wind_data[algo] = wind_data_dict[algo][:min_length]
    # Control input data
    if algo in control_data_dict:
        extracted_control_data[algo] = control_data_dict[algo][:min_length]
    # Roll angle data
    if algo in phi_data_dict:
        extracted_phi_data[algo] = phi_data_dict[algo][:min_length]

# New: Preprocess heading RMSE data (supports sliding window RMSE, window size can be customized)
WINDOW_SIZE = 100  # Sliding window size, adjustable as needed
for algo in valid_algos:
    error_sq = extracted_heading_rmse_data[algo]['heading_error_sq']
    # Calculate sliding window RMSE: first compute window-wise mean squared error, then take square root
    sliding_mse = np.convolve(error_sq, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='same')
    sliding_rmse = np.sqrt(sliding_mse)
    extracted_heading_rmse_data[algo]['sliding_rmse'] = sliding_rmse

# Validate data existence
if ONLY_PLOT_WIND and not extracted_wind_data:
    raise ValueError("No valid wind_v_mps data found, cannot plot wind speed!")
if PLOT_CONTROL_DEG and not extracted_control_data:
    raise ValueError("No valid control_deg data found, cannot plot control input!")
if PLOT_PHI_DEG and not extracted_phi_data:
    raise ValueError("No valid phi_deg data found, cannot plot roll angle!")
if PLOT_HEADING_RMSE and not extracted_heading_rmse_data:
    raise ValueError("No valid heading data found, cannot plot heading RMSE!")

# ===================== Color and Style Configuration =====================
style_config = {
    "PID":  {'linestyle': '-', 'color': '#1f77b4', 'label': 'PID'},          
    "ADRC": {'linestyle': '--', 'color': '#ff7f0e', 'label': 'ADRC'},         
    "PPO":  {'linestyle': '-', 'color': '#d62728', 'label': 'PPO+PID'},     
    "SAC":  {'linestyle': '-.', 'color': '#2ca02c', 'label': 'SAC+PID'},     
    "TD3":  {'linestyle': ':', 'color': '#9467bd', 'label': 'TD3+PID'}        
}
# Wind speed plot style
wind_style = {'linestyle': '-', 'color': '#262626', 'label': 'Wind Speed (m/s)'}
# Control input plot style
control_style = style_config
# Roll angle plot style
phi_style = style_config
# New: Heading RMSE plot style (reuse existing color scheme for visual consistency)
heading_rmse_style = style_config
# New: Target heading style configuration
target_heading_style = {
    'linestyle': '-', 
    'color': '#eae936',  # Yellow: highlights target while keeping actual heading visible
    'label': 'Target Heading',
    'linewidth': 1.0,    # Consistent with actual heading line width for visual harmony
    'alpha': 0.9         # Slightly higher transparency for clear visibility
}

# ===================== Plotting (Priority: RMSE > Roll Angle > Control Input > Wind Speed > Heading) =====================
# Convert cm to inches
fig_width_inch = FIG_WIDTH_CM * 0.3937
fig_height_inch = FIG_HEIGHT_CM * 0.3937
fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch))

if PLOT_HEADING_RMSE:
    # ========== Plot Heading RMSE Comparison (Sliding Window) ==========
    for algo in valid_algos:
        if algo in extracted_heading_rmse_data:
            cfg = heading_rmse_style[algo]
            rmse_data = extracted_heading_rmse_data[algo]['sliding_rmse']
            ax.plot(time_turbulences, rmse_data,
                    color=cfg['color'], linestyle=cfg['linestyle'],
                    label=cfg['label'], linewidth=1.0, alpha=0.8)
    
    # RMSE plot Y-axis label
    if SHOW_Y_AXIS_LABEL:
        ax.set_ylabel('Heading RMSE (deg)', fontweight='normal')
    else:
        ax.set_ylabel('')

elif PLOT_PHI_DEG:
    # ========== Plot Roll Angle (phi_deg) Comparison ==========
    for algo in valid_algos:
        if algo in extracted_phi_data:
            cfg = phi_style[algo]
            ax.plot(time_turbulences, extracted_phi_data[algo],
                    color=cfg['color'], linestyle=cfg['linestyle'],
                    label=cfg['label'], linewidth=1.0, alpha=0.8)
    
    # Roll angle plot Y-axis label
    if SHOW_Y_AXIS_LABEL:
        ax.set_ylabel('Phi (deg)', fontweight='normal')
    else:
        ax.set_ylabel('')

elif PLOT_CONTROL_DEG:
    # ========== Plot Control Input (control_deg) Comparison ==========
    for algo in valid_algos:
        if algo in extracted_control_data:
            cfg = control_style[algo]
            ax.plot(time_turbulences, extracted_control_data[algo],
                    color=cfg['color'], linestyle=cfg['linestyle'],
                    label=cfg['label'], linewidth=1.0, alpha=0.8)
    
    # Control input plot Y-axis label
    if SHOW_Y_AXIS_LABEL:
        ax.set_ylabel('Control (deg)', fontweight='normal')
    else:
        ax.set_ylabel('')

elif ONLY_PLOT_WIND:
    # ========== Plot Wind Speed ==========
    plot_algo = 'PID' if 'PID' in extracted_wind_data else list(extracted_wind_data.keys())[0]
    ax.plot(time_turbulences, extracted_wind_data[plot_algo],
            color=wind_style['color'], linestyle=wind_style['linestyle'],
            label=wind_style['label'], linewidth=1.0, alpha=0.9)
    
    # Wind speed plot title
    ax.set_title(f'turbulence wind', fontsize=10, fontweight='normal', pad=8)
    
    # Wind speed plot Y-axis label
    if SHOW_Y_AXIS_LABEL:
        ax.set_ylabel('Wind Speed (m/s)', fontweight='normal')
    else:
        ax.set_ylabel('')

else:
    # ========== Plot Heading Tracking (New: Show both actual and target heading) ==========
    # Step 1: First plot target heading (underlying layer to avoid being covered by actual heading)
    # Target heading is consistent across all algorithms, use the first valid algorithm's target heading
    first_algo = valid_algos[0]
    target_heading_data = extracted_data[first_algo]['target_heading']
    ax.plot(time_turbulences, target_heading_data,
            color=target_heading_style['color'],
            linestyle=target_heading_style['linestyle'],
            label=target_heading_style['label'],
            linewidth=target_heading_style['linewidth'],
            alpha=target_heading_style['alpha'])
    
    # Step 2: Plot actual heading for each algorithm (upper layer)
    for algo in valid_algos:
        cfg = style_config[algo]
        ax.plot(time_turbulences, extracted_data[algo]['heading'],
                color=cfg['color'], linestyle=cfg['linestyle'],
                label=cfg['label'], linewidth=1.0, alpha=0.8)
    
    # Heading plot Y-axis label
    if SHOW_Y_AXIS_LABEL:
        ax.set_ylabel('Heading (deg)', fontweight='normal')
    else:
        ax.set_ylabel('')

# ===================== General Configuration (Core: Full Borders + Custom X-axis Ticks) =====================
# X-axis label control
if SHOW_X_AXIS_LABEL:
    ax.set_xlabel('Time (s)', fontweight='normal')
else:
    ax.set_xlabel('')

# Full border settings: Show all borders (unhide top and right), uniform style
ax.spines['top'].set_visible(True)    # Show top border
ax.spines['right'].set_visible(True)  # Show right border
# Uniform width and color for all borders (ensure consistent full border style)
for spine in ax.spines.values():
    spine.set_linewidth(0.8)          # Same width as left/bottom borders
    spine.set_color('black')          # Black full borders, color can be modified as needed

# Core modification: Custom X-axis ticks with 30-second intervals (0, 30, 60, ...)
# Generate tick list from 0 to x_max with step 30
xticks = np.arange(0, x_max + 1, 30)
ax.set_xticks(xticks)
# Optional: Force display of tick labels (avoid automatic omission)
ax.set_xticklabels(xticks.astype(int))

# Legend control
if SHOW_LEGEND:
    ax.legend(loc='upper right', frameon=True)
else:
    # Remove legend (avoid residual empty legend)
    if ax.get_legend():
        ax.get_legend().remove()

# Grid control (Y-axis only for cleanliness)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Tight layout
plt.tight_layout(pad=0.3)

# ===================== Save and Display =====================
# File name distinguishes plot types
if PLOT_HEADING_RMSE:
    save_path = f'./tiff/turbulence_{wind}ms_heading_rmse.tiff'
elif PLOT_PHI_DEG:
    save_path = f'./tiff/turbulence_{wind}ms_phi_deg.tiff'
elif PLOT_CONTROL_DEG:
    save_path = f'./tiff/turbulence_{wind}ms_control_deg.tiff'
elif ONLY_PLOT_WIND:
    save_path = f'./tiff/turbulence_{wind}ms_wind_only.tiff'
else:
    save_path = f'./tiff/turbulence_{wind}ms.tiff'

# Ensure save directory exists
os.makedirs('./tiff', exist_ok=True)
plt.savefig(save_path, format='tiff', bbox_inches='tight', pad_inches=0.05)
print(f"✅ Image saved to: {save_path}")

# Display image
plt.show()

# ===================== Save Standalone Legend (Core Modification: Fixed Inclusion of Target Heading) =====================
def save_standalone_legend(style_cfg, valid_algorithms, save_directory, wind_speed, fig_size=(3, 2)):
    """
    Save standalone legend image with fixed inclusion of target heading, consistent with original plot style
    :param style_cfg: Algorithm style configuration dictionary
    :param valid_algorithms: List of valid algorithms
    :param save_directory: Save directory
    :param wind_speed: Wind speed parameter (for file name)
    :param fig_size: Legend image size (inches)
    """
    # Create independent canvas for legend only
    legend_fig, legend_ax = plt.subplots(figsize=fig_size)
    legend_ax.axis('off')  # Hide axes, keep only legend
    
    # Generate legend handles matching original plot
    legend_handles = []
    
    # Fixed: Add target heading legend handle (no conditional check, force inclusion)
    target_handle = plt.Line2D(
        [], [],
        color=target_heading_style['color'],
        linestyle=target_heading_style['linestyle'],
        label=target_heading_style['label'],
        linewidth=target_heading_style['linewidth']
    )
    legend_handles.append(target_handle)
    
    # Add algorithm legend handles
    for algo_name in valid_algorithms:
        style = style_cfg[algo_name]
        # Create virtual line (for legend generation only, no actual plotting)
        handle = plt.Line2D(
            [], [],
            color=style['color'],
            linestyle=style['linestyle'],
            label=style['label'],
            linewidth=1.0  # Consistent with original plot line width
        )
        legend_handles.append(handle)
    
    # Add legend with style consistent with original configuration
    legend_fig.legend(
        handles=legend_handles,
        loc='center',  # Center legend
        frameon=False,  # No border, matching original configuration
        fontsize=6  # Consistent with original legend font size
    )
    
    # Tight layout to avoid legend cropping
    plt.tight_layout()
    
    # Construct legend save path (explicitly includes target heading identifier)
    legend_save_path = os.path.join(save_directory, f'turbulence_{wind_speed}ms_standalone_legend_with_target.tiff')
    
    # Save legend image with high resolution and consistent style
    plt.savefig(
        legend_save_path,
        format='tiff',
        dpi=600,
        bbox_inches='tight',
        pad_inches=0.05,
        facecolor='white'
    )
    print(f"✅ Standalone legend (with target heading) saved to: {legend_save_path}")
    plt.close(legend_fig)  # Close legend canvas to free memory

# Call function to save standalone legend (no conditional check, fixed inclusion of target heading)
save_standalone_legend(style_config, valid_algos, './tiff', wind)

# ===================== Performance Metric Calculation (Including RMSE) =====================
print("\n===== Heading Tracking Performance Metric Comparison (Including RMSE) =====")
# Adjust header to add RMSE column
print(f"{'Algorithm':<8} {'Average Heading Error (deg)':<18} {'Max Heading Error (deg)':<18} {'Heading Error RMSE (deg)':<18}")
print("-" * 70)
for algo in valid_algos:
    heading = extracted_data[algo]['heading']
    target_heading = extracted_data[algo]['target_heading']
    heading_error = np.abs(heading - target_heading)
    avg_error = np.mean(heading_error)  
    max_error = np.max(heading_error)   
    # Calculate RMSE (Root Mean Square Error): sqrt(mean((actual - target)^2))
    rmse_error = np.sqrt(np.mean((heading - target_heading) ** 2))
    print(f"{algo:<8} {avg_error:<18.4f} {max_error:<18.4f} {rmse_error:<18.4f}")

# Roll angle data statistics (display only when plotting roll angle)
if PLOT_PHI_DEG and extracted_phi_data:
    print("\n===== Roll Angle (phi_deg) Statistics (Including RMS) =====")
    # 1. Modify header: Replace "Max Roll Angle" with "Roll Angle RMS"
    print(f"{'Algorithm':<8} {'Average Roll Angle (deg)':<15} {'Roll Angle RMS (deg)':<15} {'Roll Angle Std Dev':<15}")
    print("-" * 55)
    for algo in valid_algos:
        if algo in extracted_phi_data:
            phi_data = extracted_phi_data[algo]
            avg_phi = np.mean(phi_data)
            # 2. New: Calculate Roll Angle RMS (Root Mean Square): sqrt(mean(phi_data^2))
            phi_rms = np.sqrt(np.mean(phi_data ** 2))
            std_phi = np.std(phi_data)
            # 3. Modify print content: Replace max roll angle with RMS
            print(f"{algo:<8} {avg_phi:<15.4f} {phi_rms:<15.4f} {std_phi:<15.4f}")

# Control input data statistics (display only when plotting control input)
elif PLOT_CONTROL_DEG and extracted_control_data:
    print("\n===== Control Input (control_deg) Statistics =====")
    print(f"{'Algorithm':<8} {'Average Control Input (deg)':<15} {'Max Control Input (deg)':<15} {'Control Input Std Dev':<15}")
    print("-" * 55)
    for algo in valid_algos:
        if algo in extracted_control_data:
            control_data = extracted_control_data[algo]
            avg_control = np.mean(control_data)
            max_control = np.max(np.abs(control_data))
            std_control = np.std(control_data)
            print(f"{algo:<8} {avg_control:<15.4f} {max_control:<15.4f} {std_control:<15.4f}")

# Wind speed data statistics (display only when plotting wind speed)
elif ONLY_PLOT_WIND:
    print("\n===== Wind Speed Data Statistics =====")
    plot_algo = 'PID' if 'PID' in extracted_wind_data else list(extracted_wind_data.keys())[0]
    wind_data = extracted_wind_data[plot_algo]
    avg_wind = np.mean(wind_data)
    max_wind = np.max(wind_data)
    min_wind = np.min(wind_data)
    std_wind = np.std(wind_data)
    print(f"Average Wind Speed: {avg_wind:.4f} m/s")
    print(f"Max Wind Speed: {max_wind:.4f} m/s")
    print(f"Min Wind Speed: {min_wind:.4f} m/s")
    print(f"Wind Speed Std Dev: {std_wind:.4f} m/s")

# New: Heading RMSE data statistics (display only when plotting RMSE)
if PLOT_HEADING_RMSE and extracted_heading_rmse_data:
    print("\n===== Heading RMSE (Sliding Window) Statistics =====")
    print(f"{'Algorithm':<8} {'Average RMSE (deg)':<15} {'Max RMSE (deg)':<15} {'RMSE Std Dev':<15}")
    print("-" * 55)
    for algo in valid_algos:
        if algo in extracted_heading_rmse_data:
            rmse_data = extracted_heading_rmse_data[algo]['sliding_rmse']
            avg_rmse = np.mean(rmse_data)
            max_rmse = np.max(rmse_data)
            std_rmse = np.std(rmse_data)
            print(f"{algo:<8} {avg_rmse:<15.4f} {max_rmse:<15.4f} {std_rmse:<15.4f}")