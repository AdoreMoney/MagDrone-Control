import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from trim.trim_hrz import FixedWingDynamics, TrimCalculator, WindField

def calculate_overshoot(data, target_value):
    """
    Calculate the overshoot percentage.
    
    Parameters:
    data (list): System response data (e.g., pitch angle history).
    target_value (float): Target value (e.g., target pitch angle).
    
    Returns:
    float: Overshoot percentage.
    """
    max_value = max(data)
    if target_value == 0:
        # Avoid division by zero. Adjustments may be needed based on specific scenarios if the target value is 0
        return ((max_value - target_value) / (max_value + 1e-6)) * 100
    else:
        return ((max_value - target_value) / target_value) * 100

def calculate_rise_time(data, target_value, dt, threshold=0.9):
    """
    Calculate the rise time (time from 10% to 90% of the target value).
    
    Parameters:
    data (list): System response data (e.g., pitch angle history).
    target_value (float): Target value (e.g., target pitch angle).
    dt (float): Time step (seconds).
    threshold (float): Threshold ratio for rise time, default is 0.9 (i.e., 90%).
    
    Returns:
    float: Rise time (seconds).
    """
    # Calculate 10% and 90% of the target value
    lower_threshold = 0.1 * target_value
    upper_threshold = threshold * target_value
    
    # Find the first time points when 10% and 90% of the target value are reached
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
        return end_time - start_time
        #return end_time
    else:
        # Return None if the thresholds are not reached
        return None

def calculate_settling_time(errors, threshold=0.05, target=1.0, consecutive_steps=500):
    """
    Calculate the settling time of the system response (the number of consecutive steps where the error 
    enters and remains within the range of ±threshold of the target value)
    
    Parameters:
        errors: Error array
        target: Target value
        threshold: Stability threshold (default 5%)
        consecutive_steps: Number of consecutive steps required to meet the condition (default 500 steps)
    
    Returns:
        settling_time: Settling time (in steps)
    """
    normalized_errors = [abs(e) for e in errors]
    consecutive_count = 0
    
    for i in range(len(normalized_errors)):
        if normalized_errors[i] <= np.deg2rad(0.5):
            consecutive_count += 1
            if consecutive_count >= consecutive_steps:
                return i - consecutive_steps + 1  # Return the time step when stability starts
        else:
            consecutive_count = 0
    
    mean_error = np.mean(np.abs(errors))
    if mean_error <= threshold * target:
        return i - consecutive_steps + 1
    else:
        pass
    
    return None  # Return None if no settling time is found

def calculate_steady_state_error(response, target_value, settling_time_step=None):
    """
    Calculate the steady-state error of the system response
    
    Parameters:
        response (list): System response data (e.g., heading angle history)
        target_value (float): Target value
        settling_time_step (int): Time step corresponding to settling time, automatically calculated if not provided
    
    Returns:
        float: Steady-state error (returns None if the system is not stable)
    """
    if settling_time_step is None:
        errors = [(repo - target_value) for repo in response]
        settling_time_step = calculate_settling_time(errors)
    
    if settling_time_step is None:
        return None  # Cannot calculate steady-state error if the system is unstable
    
    # Calculate the average error in the steady-state phase
    steady_state_response = response[settling_time_step:]
    steady_state_error = np.abs(target_value - np.mean(steady_state_response)) / np.abs(target_value)
    
    return steady_state_error

def sliding_window_variance(data, window_size):
    """
    Calculate the average of the sliding window variances of the data.

    Parameters:
        data (numpy.ndarray): Input data array.
        window_size (int): Size of the sliding window.

    Returns:
        float: Average of the sliding window variances.
    """
    if len(data) < window_size:
        raise ValueError("The length of the data must be greater than or equal to the window size.")
    
    variances = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        variances.append(np.var(window))
    
    return np.mean(variances)


def plot_pid_controlled_bode(trim_calc, controller, target_phi_deg=1.0, dt=0.01):
    """
    Plot the Bode diagram of the open-loop transfer function after PID control
    
    Parameters:
    trim_calc: TrimCalculator instance
    controller: CoordinatedTurnController instance
    target_phi_deg: Target roll angle (degrees)
    dt: Time step (seconds)
    
    Returns:
    open_loop_tf: Open-loop transfer function
    """
    # Linearize the model
    A_long, B_long = trim_calc.linearize_trim()
    #print('A_long:', A_long)
    #print('B_long:', B_long)
    
    # Define output matrix C and direct transmission term D
    C_long = np.array([
        [0, 1, 0, 0, 0],  # 1st output is p
        [0, 0, 1, 0, 0],  # 2nd output is r
        [0, 0, 0, 1, 0],  # 3rd output is phi
        [0, 0, 0, 0, 1]   # 4th output is psi
    ])
    D_long = np.zeros((C_long.shape[0], B_long.shape[1]))  # Create an all-zero matrix
    
    # Create state space model
    sys_state_space = ctrl.ss(A_long, B_long, C_long, D_long)
    
    # -----------------------phi--------------------------------
    # Specify the indices of the input (elevator differential) and output (roll angle)
    input_idx = 0  # delta_e_diff input
    output_idx = 2  # phi output
    
    # Extract A, B, C, D of the subsystem
    A_sub = sys_state_space.A  # State matrix remains unchanged
    B_sub = sys_state_space.B[:, input_idx].reshape(-1, 1)
    C_sub = sys_state_space.C[output_idx, :].reshape(1, -1)  # Only keep the row for phi
    D_sub = sys_state_space.D[output_idx, input_idx]  # Direct transmission term (usually 0)
    # Create subsystem (SISO)
    sys_sub = ctrl.ss(A_sub, B_sub, C_sub, D_sub)
    # Extract transfer function (ctrl.ss2tf)
    sys_tf = ctrl.ss2tf(sys_sub)
    print("Original UAV model transfer function:")
    print(sys_tf)
    
    num = [20.0]
    den = [1, 20.0]
    actuator_tf = ctrl.TransferFunction(num, den)
    
    # Construct the transfer function of the phiPID controller
    kp_phi = controller.roll_pid['kp']
    ki_phi = controller.roll_pid['ki']
    kd_phi = controller.roll_pid['kd']
    
    # Create phiPID transfer function
    pid_phi_num = [kd_phi, kp_phi, ki_phi]
    pid_phi_den = [1, 0]  # Denominator is s
    pid_phi_tf = ctrl.tf(pid_phi_num, pid_phi_den)
    print("phiPID transfer function:")
    print(pid_phi_tf)

    # ------------------------psi----------------------
    # Specify the indices of the input (elevator differential) and psi
    input_idx = 0  # delta_e_diff input
    output_idx = 3  # psi output
    
    # Extract A, B, C, D for psi
    A_sub = sys_state_space.A  # State matrix remains unchanged
    B_sub = sys_state_space.B[:, input_idx].reshape(-1, 1)
    C_sub = sys_state_space.C[output_idx, :].reshape(1, -1)  # Only keep the row for psi
    D_sub = sys_state_space.D[output_idx, input_idx]  # Direct transmission term (usually 0)
    # Create subsystem (SISO)
    sys_sub = ctrl.ss(A_sub, B_sub, C_sub, D_sub)
    
    # Extract transfer function (ctrl.ss2tf)
    sys_tf = ctrl.ss2tf(sys_sub)
    print("Original UAV psi model transfer function:")
    print(sys_tf)
    
    kp_psi = 2.8
    ki_psi = 0.01
    kd_psi = 2.2
    
    # Create psiPID transfer function
    pid_psi_num = [kd_psi, kp_psi, ki_psi]
    pid_psi_den = [1, 0]  # Denominator is s
    pid_psi_tf = ctrl.tf(pid_psi_num, pid_psi_den)
    
    T_d = 1.9  # Time constant
    alpha = 0.1
    lead_comp = ctrl.tf([T_d, 1], [alpha * T_d, 1])
    
    open_loop_tf = pid_phi_tf * pid_psi_tf * 1 * sys_tf

    # Calculate gain margin and phase margin
    gm, pm, wg, wp = ctrl.margin(open_loop_tf)
    gain_margin_db = 20 * np.log10(gm)
    
    print(f"Gain Margin: {gain_margin_db:.2f} dB (Gain Margin: {gm:.2f})")
    print(f"Phase Margin: {pm:.2f} degrees (Phase Margin: {pm:.2f} degrees)")
    
    num = [0.0, 0.0, 1.26, 6.279, 28.72, 162.9, 22.71]
    den = [1.0, 5.506, 60.35, 225.4, 51.97, 0, 0]
    open_loop_tf_hand = ctrl.TransferFunction(num, den)
    print('Open-loop Transfer Function:')
    print(open_loop_tf_hand)
    
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
    #plt.semilogx(omega, 20*np.log10(mag))
    plt.ylabel('Magnitude (dB)')
    plt.xlim([1e-1, 1e3]) 
    plt.ylim([-150, 50])
    plt.legend()
    plt.axhline(0, color='r', linestyle='--', alpha=0.5)  # 0dB line
    plt.axhline(-gain_margin_db, color='r', linestyle='--')  # gain dB line
    plt.axvline(wp, color='r', linestyle='--', alpha=0.5)  # wp
    plt.axvline(wg, color='r', linestyle='--',
              label=f'GM={gain_margin_db:.1f} dB')  # wp
    #plt.legend(loc='lower right', frameon=True, framealpha=1)
    # Add text annotation to the right of the vertical line
    plt.annotate(f'GM={gain_margin_db:.1f} dB @ {wg:.1f} rad/s',
             xy=(wg, 0.5),                  # Position pointed by the arrow
             xycoords=('data', 'axes fraction'),  # X: data coordinates, Y: axis fraction
             xytext=(10, 0),                # Text offset (pixels)
             textcoords='offset points',
             ha='left', va='center',        # Text alignment
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
             arrowprops=dict(arrowstyle='->'))

    plt.subplot(2, 1, 2)
    #plt.semilogx(omega, phase)
    plt.ylabel('Phase (deg)')
    plt.xlabel('Frequency (rad/s)')
    plt.xlim([1e-1, 1e3]) 
    plt.ylim([-500, -90])
    plt.legend()
    plt.axhline(-180+pm, color='r', linestyle='--', alpha=0.5)  # -180° line
    plt.axhline(-180, color='r', linestyle='--')  # -180° line
    plt.axvline(wp, color='r', linestyle='--', alpha=0.5)  # wp
    plt.axvline(wg, color='r', linestyle='--')  # wg
    # Add text annotation to the left of the vertical line
    plt.annotate(f'PM={pm:.1f}°\n@ {wp:.3f} rad/s',
             xy=(wp, 0.73),                  # Position pointed by the arrow
             xycoords=('data', 'axes fraction'),  # X: data coordinates, Y: axis fraction
             xytext=(-10, 0),                # Text offset (pixels)
             textcoords='offset points',
             ha='right', va='center',        # Text alignment
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
             arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.savefig('bode_yaw.png', dpi=600,
            bbox_inches='tight', 
            pad_inches=0.05,
            transparent=True)
    #plt.show()
    plt.close()
    
    # Return the system object for further analysis
    return open_loop_tf