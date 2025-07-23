import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

def calculate_overshoot(data, target_value):

    max_value = max(data)
    if target_value == 0:
        # Handle division by zero case
        return ((max_value - target_value) / (max_value + 1e-6)) * 100
    else:
        return ((max_value - target_value) / target_value) * 100

def calculate_rise_time(data, target_value, dt, threshold=0.9):
    """
    Calculate the rise time (time from 10% to 90% of the target value).
    
    Parameters:
    data (list): System response data (e.g., pitch angle history).
    target_value (float): Target value (e.g., desired pitch angle).
    dt (float): Time step (seconds).
    threshold (float): Threshold percentage for rise time, default is 0.9 (i.e., 90%).
    
    Returns:
    float: Rise time (seconds).
    """
    # Calculate 10% and 90% thresholds of the target value
    lower_threshold = 0.1 * target_value
    upper_threshold = threshold * target_value
    
    # Find the first time points where the response crosses 10% and 90% thresholds
    reached_lower = False
    start_time = None
    end_time = None
    
    for i, value in enumerate(data):
        if not reached_lower and value >= lower_threshold:
            start_time = i * dt
            reached_lower = True
        if reached_lower and value >= upper_threshold:
            end_time = i * dt
            break
    
    if start_time is not None and end_time is not None:
        #return end_time - start_time
        return end_time
    else:
        # Return None if thresholds were not reached
        return None

def calculate_settling_time(errors, threshold=0.05, consecutive_steps=200):
    normalized_errors = [abs(e) for e in errors]
    consecutive_count = 0
    
    for i in range(len(normalized_errors)):
        if normalized_errors[i] <= threshold:
            consecutive_count += 1
            if consecutive_count >= consecutive_steps:
                return i - consecutive_steps + 1  # Return the time step when settling starts
        else:
            consecutive_count = 0
    
    return None  # Return None if no settling time is found

def calculate_steady_state_error(response, target_value, settling_time_step=None):
    if settling_time_step is None:
        errors = [(repo - target_value) for repo in response]
        settling_time_step = calculate_settling_time(errors)
    
    if settling_time_step is None:
        return None  # System is not stable, cannot calculate steady-state error
    
    # Calculate average error in the steady-state phase
    steady_state_response = response[settling_time_step:]
    steady_state_error = np.abs(target_value - np.mean(steady_state_response)) / np.abs(target_value)
    
    return steady_state_error

def plot_pid_controlled_bode(trim_calc, controller, target_phi_deg=1.0, dt=0.01):

    # Linearized model
    A_lat, B_lat = trim_calc.linearize_trim()
    
    # Define output matrix C and direct transmission term D
    C_lat = np.array([
        [0, 1, 0, 0, 0],  # First output is p
        [0, 0, 1, 0, 0],  # Second output is r
        [0, 0, 0, 1, 0],  # Third output is phi
        [0, 0, 0, 0, 1]   # Fourth output is psi
    ])
    D_lat = np.zeros((C_lat.shape[0], B_lat.shape[1]))  # Create all-zero matrix
    
    # Create state-space model
    sys_state_space = ctrl.ss(A_lat, B_lat, C_lat, D_lat)
    
    # -----------------------phi--------------------------------
    # Specify indices for input (elevator) and output (pitch angle)
    input_idx = 0  # delta_e_diff input
    output_idx = 2  # phi output
    
    # Extract A, B, C, D of the subsystem
    A_sub = sys_state_space.A  # State matrix remains unchanged
    B_sub = sys_state_space.B[:, input_idx].reshape(-1, 1)
    C_sub = sys_state_space.C[output_idx, :].reshape(1, -1)  # Only retain the row for phi
    D_sub = sys_state_space.D[output_idx, input_idx]  # Direct transmission term (usually 0)
    # Create subsystem (SISO)
    sys_sub = ctrl.ss(A_sub, B_sub, C_sub, D_sub)
    # Extract transfer function (ctrl.ss2tf)
    sys_tf = ctrl.ss2tf(sys_sub)
    
    num = [20.0]
    den = [1, 20.0]
    actuator_tf = ctrl.TransferFunction(num, den)
    
    # Construct transfer function for phiPID controller
    kp_phi = controller.roll_pid['kp']
    ki_phi = controller.roll_pid['ki']
    kd_phi = controller.roll_pid['kd']
    
    # Create transfer function for phiPID
    pid_phi_num = [kd_phi, kp_phi, ki_phi]
    pid_phi_den = [1, 0]  # Denominator is s
    pid_phi_tf = ctrl.tf(pid_phi_num, pid_phi_den)
    
    open_loop_tf = pid_phi_tf * 1 * sys_tf * actuator_tf
    closed_loop_tf = ctrl.feedback(open_loop_tf, 1) 
    
    # Calculate gain margin and phase margin
    gm, pm, wg, wp = ctrl.margin(open_loop_tf)
    gain_margin_db = 20 * np.log10(gm)

    # ------------------------psi----------------------
    # Specify indices for input (elevator) and psi
    input_idx = 0  # delta_e_diff input
    output_idx = 3  # psi output
    
    # Extract A, B, C, D for psi
    A_sub = sys_state_space.A  # State matrix remains unchanged
    B_sub = sys_state_space.B[:, input_idx].reshape(-1, 1)
    C_sub = sys_state_space.C[output_idx, :].reshape(1, -1)  # Only retain the row for psi
    D_sub = sys_state_space.D[output_idx, input_idx]  # Direct transmission term (usually 0)
    # Create subsystem (SISO)
    sys_sub = ctrl.ss(A_sub, B_sub, C_sub, D_sub)
    
    # Extract transfer function (ctrl.ss2tf)
    sys_tf = ctrl.ss2tf(sys_sub)
    
    kp_psi = 2.8
    ki_psi = 0.01
    kd_psi = 2.2
    
    # Create transfer function for psiPID
    pid_psi_num = [kd_psi, kp_psi, ki_psi]
    pid_psi_den = [1, 0]  # Denominator is s
    pid_psi_tf = ctrl.tf(pid_psi_num, pid_psi_den)
    
    open_loop_tf = pid_phi_tf * pid_psi_tf * 1 * sys_tf

    # Calculate gain margin and phase margin
    gm, pm, wg, wp = ctrl.margin(open_loop_tf)
    gain_margin_db = 20 * np.log10(gm)

    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',       # Use serif font
        'font.serif': 'Times New Roman',
        'font.size': 15,              # Base font size
        'axes.grid': True,            # Show grid
        'grid.alpha': 0.3,            # Grid transparency
        'axes.linewidth': 0.8,         # Axis line width
        # Additional font size settings for ticks and labels
        'axes.labelsize': 16,          # Axis label font size
        'xtick.labelsize': 15,         # X-axis tick font size
        'ytick.labelsize': 15,         # Y-axis tick font size
        'legend.fontsize': 15          # Legend font size
        })

    plt.figure(figsize=(8, 6), dpi=600)
    
    mag, phase, omega = ctrl.bode(open_loop_tf, dB=True, Hz=False, deg=True, plot=True)

    plt.subplot(2, 1, 1)
    plt.ylabel('Magnitude (dB)')
    plt.xlim([1e-1, 1e3]) 
    plt.ylim([-150, 50])
    plt.legend()
    plt.axhline(0, color='r', linestyle='--', alpha=0.5)  # 0dB line
    plt.axhline(-gain_margin_db, color='r', linestyle='--')  # gain dB line
    plt.axvline(wp, color='r', linestyle='--', alpha=0.5)  # wp
    plt.axvline(wg, color='r', linestyle='--',
              label=f'GM={gain_margin_db:.1f} dB')  # wp
    
    plt.annotate(f'GM={gain_margin_db:.1f} dB @ {wg:.1f} rad/s',
             xy=(wg, 0.53),                  # Arrow target position
             xycoords=('data', 'axes fraction'),  # X: data coordinates, Y: axis fraction
             xytext=(10, 0),                # Text offset (pixels)
             textcoords='offset points',
             ha='left', va='center',        # Text alignment
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
             arrowprops=dict(arrowstyle='->'))

    plt.subplot(2, 1, 2)
    plt.ylabel('Phase (deg)')
    plt.xlabel('Frequency (rad/s)')
    plt.xlim([1e-1, 1e3]) 
    plt.ylim([-360, -90])
    plt.legend()
    plt.axhline(-180 + pm, color='r', linestyle='--', alpha=0.5)  # -180° line
    plt.axhline(-180, color='r', linestyle='--')  # -180° line
    plt.axvline(wp, color='r', linestyle='--', alpha=0.5)  # wp
    plt.axvline(wg, color='r', linestyle='--')  # wg

    plt.annotate(f'PM={pm:.1f}°\n@ {wp:.3f} rad/s',
             xy=(wp, 0.75),                  # Arrow target position
             xycoords=('data', 'axes fraction'),  # X: data coordinates, Y: axis fraction
             xytext=(-10, 0),                # Text offset (pixels)
             textcoords='offset points',
             ha='right', va='center',        # Text alignment
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
             arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    #plt.savefig('heading-roll/18_Open-loop Bode plot of roll angle control.png', 
            #bbox_inches='tight', 
            #pad_inches=0.05,
            #transparent=True)
    plt.savefig('heading-roll/20_Open-loop Bode plot of heading angle control.png', 
            bbox_inches='tight', 
            pad_inches=0.05,
            transparent=True)
    #plt.show()
    plt.close()
    
    # Return system object for further analysis
    return open_loop_tf