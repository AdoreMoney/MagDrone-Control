import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'mathtext.fontset': 'stix',  # Use Times New Roman style for mathematical formulas
    'axes.unicode_minus': False  # Properly display negative signs
})

# Model parameters (strictly matching requirements)
z0 = 0.001       # Surface roughness (m)
h_ref = 4        # Reference height (m)
W_ref = 7.5      # Wind speed at reference height (m/s)
heights = np.linspace(1, 300, 1000)  # Height range

# Logarithmic wind profile formula
def wind_speed(h, h_ref, W_ref, z0):
    numerator = np.log(h / z0)
    denominator = np.log(h_ref / z0)
    return W_ref * (numerator / denominator)

# Calculate wind speeds at various heights
wind_speeds = wind_speed(heights, h_ref, W_ref, z0)

# Create a professionally styled chart
fig, ax = plt.subplots(figsize=(6, 4))  # Adjusted to a more suitable ratio for papers

# Set high-quality output
plt.rcParams['figure.dpi'] = 600

# Plot the curve (using more professional color scheme)
line, = ax.plot(heights, wind_speeds, 
                color='#1f77b4',  # Professional blue
                linewidth=2.0,
                linestyle='-',
                label=f'Wind Speed Profile\n(z₀={z0}, U₄={W_ref}m/s)')

# Highlight the 90-110m interval
ax.axvspan(90, 110, color='#e6f2ff', alpha=0.5, zorder=0)  # Light blue shadow
ax.text(95, max(wind_speeds)*0.25, '90-110m', ha='center', fontsize=14, 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))

# Add vertical dashed lines and annotations for 90m and 110m heights
h_90 = 90
w_90 = wind_speed(h_90, h_ref, W_ref, z0)
h_110 = 110
w_110 = wind_speed(h_110, h_ref, W_ref, z0)

# Dashed line and annotation for 90m height
ax.axvline(x=h_90, color='#d62728', linestyle='--', linewidth=1.0, alpha=0.7)
ax.annotate(f'{w_90:.2f} m/s',
            xy=(h_90, w_90),
            xytext=(h_90, w_90 + 1.2),
            fontsize=14,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.8))

# Dashed line and annotation for 110m height
ax.axvline(x=h_110, color='#ff7f0e', linestyle='--', linewidth=1.0, alpha=0.7)
ax.annotate(f'{w_110:.2f} m/s',
            xy=(h_110, w_110),
            xytext=(h_110, w_110 - 1.2),
            fontsize=14,
            ha='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ff7f0e", alpha=0.8))

# Add markers for reference height and 100m height (using hollow points for enhanced professionalism)
ref_point = ax.scatter(h_ref, W_ref, 
                       color='none', 
                       edgecolor='#d62728',  # Professional red
                       s=60, 
                       linewidth=1.5,
                       zorder=5)

# Add more professional annotations (using horizontal alignment and LaTeX format)
ax.annotate(f'4m: $U$ = {W_ref} m/s',
            xy=(h_ref, W_ref),
            xytext=(h_ref + 25, W_ref),
            textcoords='data',
            fontsize=14,
            arrowprops=dict(arrowstyle='->', color='#333333', linewidth=1),
            horizontalalignment='left',
            verticalalignment='center')

h_100 = 100
w_100 = wind_speed(h_100, h_ref, W_ref, z0)

h100_point = ax.scatter(h_100, w_100, 
                        color='none', 
                        edgecolor='#ff7f0e',  # Professional orange
                        s=60, 
                        linewidth=1.5,
                        zorder=5)

ax.annotate(f'100m: $U$ = {w_100:.2f} m/s',
            xy=(h_100, w_100),
            xytext=(h_100 + 25, w_100),
            textcoords='data',
            fontsize=14,
            arrowprops=dict(arrowstyle='->', color='#333333', linewidth=1),
            horizontalalignment='left',
            verticalalignment='center')

# Set axis labels
ax.set_xlabel('Height (m)', fontsize=16, fontweight='bold')
ax.set_ylabel('Wind Speed (m/s)', fontsize=16, fontweight='bold')

# Set tick marks and grid
ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Integer tick marks
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))      # Control number of tick marks
ax.tick_params(axis='both', which='major', labelsize=16)
ax.grid(True, linestyle='--', alpha=0.5, color='#cccccc')  # Lighter grid lines

# Set axis ranges and spines
ax.set_xlim(0, 300)
ax.set_ylim(0, max(wind_speeds) * 1.1)
ax.spines['top'].set_visible(False)    # Hide top spine
ax.spines['right'].set_visible(False)  # Hide right spine
ax.spines['bottom'].set_linewidth(0.8) # Adjust spine thickness
ax.spines['left'].set_linewidth(0.8)

# Add legend (using smaller font and concise format)
ax.legend(loc='lower right', fontsize=14, frameon=False)  # Legend without frame

# Ensure compact layout
plt.tight_layout()

# Save high-quality image
plt.savefig('21_Wind shear model above the sea surface.png', dpi=600, bbox_inches='tight')

# Display the chart
plt.show()

# Output key data
print(f"90m Wind Speed: {w_90:.2f} m/s")
print(f"100m Wind Speed: {w_110:.2f} m/s")
print(f"90-110m wind changing rate: {(w_110 - w_90)/10:.4f} m/s per meter")