import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from trim_vertical import FixedWingDynamics, TrimCalculator, WindField    

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
        # Avoid division by zero. If target value is 0, adjustments may be needed based on specific scenarios
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
    
    # Find the first time points when 10% and 90% are reached
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
        # Return None if thresholds are not reached
        return None

def calculate_settling_time(errors, threshold=0.05, consecutive_steps=200):
    """
    Calculate the settling time of the system response (the number of consecutive steps where the error 
    enters and remains within the range of ±threshold of the target value)
    
    Parameters:
        errors: Error array
        threshold: Stability threshold (default 5%)
        consecutive_steps: Number of consecutive steps required to meet the condition (default 200 steps)
    
    Returns:
        settling_time: Settling time (in steps)
    """
    normalized_errors = [abs(e) for e in errors]
    consecutive_count = 0
    
    for i in range(len(normalized_errors)):
        if normalized_errors[i] <= threshold:
            consecutive_count += 1
            if consecutive_count >= consecutive_steps:
                return i - consecutive_steps + 1  # Return the time step when stability starts
        else:
            consecutive_count = 0
    
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

def plot_pid_controlled_bode(trim_calc, h_pid, theta_pid, q_pid, target_h=1, dt=0.01):
    """
    Plot the Bode diagram of the open-loop transfer function after PID control
    
    Parameters:
    trim_calc: TrimCalculator instance
    theta_pid: theta PID controller instance
    q_pid: q PID controller instance
    target_h: Target height (default 1)
    dt: Time step (seconds, default 0.01)
    """
    # Linearize the model
    A_long, B_long = trim_calc.linearize_trim()
    print('A_long:', A_long)
    print('B_long:', B_long)
    
    # Define output matrix C and direct transmission term D
    # Here we choose to output pitch rate (q) and pitch angle (theta)
    # Assume state vector x = [V, alpha, q, theta, H]
    # Therefore, q is the 3rd state (index 2), theta is the 4th state (index 3)
    C_long = np.array([
        [0, 0, 1, 0, 0],  # 1st output is q
        [0, 0, 0, 1, 0],   # 2nd output is theta
        [0, 0, 0, 0, 1]    # 3rd output is H
    ])
    D_long = np.zeros((C_long.shape[0], B_long.shape[1]))  # Create all-zero matrix
    
    # Create state space model
    sys_state_space = ctrl.ss(A_long, B_long, C_long, D_long)
    
    # Specify indices of input (elevator) and output
    input_idx = 0  # delta_e_sync input
    output_idx = 1  # H output
    
    # Extract A, B, C, D of the subsystem
    A_sub = sys_state_space.A  # State matrix remains unchanged
    B_sub = sys_state_space.B[:, input_idx].reshape(-1, 1)
    C_sub = sys_state_space.C[output_idx, :].reshape(1, -1)  # Only keep the row of H
    D_sub = sys_state_space.D[output_idx, input_idx]  # Direct transmission term (usually 0)
    # Create subsystem (SISO)
    sys_sub = ctrl.ss(A_sub, B_sub, C_sub, D_sub)
    # Extract transfer function (ctrl.ss2tf)
    sys_tf = ctrl.ss2tf(sys_sub)
    print("Original UAV model transfer function:")
    print(sys_tf)
        
    # Construct the transfer function of hPID
    # The control law of hPID is: target_q = kp*error + ki*integral(error) + kd*derivative(error)
    # This can be approximated as a PID transfer function: G_pid_h(s) = kp + ki/s + kd*s
    kp_h = h_pid.kp
    ki_h = h_pid.ki
    kd_h = h_pid.kd
    
    # Create hPID transfer function
    pid_h_num = [kd_h, kp_h, ki_h]
    pid_h_den = [1, 0]  # Denominator is s
    pid_h_tf = ctrl.tf(pid_h_num, pid_h_den)
    print("hPID transfer function:")
    print(pid_h_tf)
    
    num = [9.078, 246.5, 1320, 2643, 2000, 316, 0.1503, 0]
    den = [1.8, 45.82, 209.9, 477.9, 313.8, 82.43, 33.06, 0, 0]
    theta_q_tf = ctrl.TransferFunction(num, den)
    closed_theta_q_tf = ctrl.feedback(theta_q_tf, 1)
    print('Pitch angle inner loop closed-loop function:')
    print(closed_theta_q_tf)
    
    s = ctrl.tf('s')
    Kvz = h_pid.kvz
    
    L_forward = sys_tf * closed_theta_q_tf * pid_h_tf
    L_feedback = (Kvz*s) * closed_theta_q_tf * sys_tf
    open_loop_tf = L_forward / (1 + L_feedback)
    #open_loop_tf = L_forward * (Kvz*s)
    print("Height open-loop transfer function (including cascade PID):")
    print(open_loop_tf)

    # Calculate gain margin and phase margin
    gm, pm, wg, wp = ctrl.margin(open_loop_tf)
    gain_margin_db = 20 * np.log10(gm)
    
    print(f"Gain margin: {gain_margin_db:.2f} dB")
    print(f"Phase margin: {pm:.2f} degrees")
    print(f"Gain crossover frequency (ωg): {wg:.4f} rad/s")  # Frequency corresponding to gain margin
    print(f"Phase crossover frequency (ωp): {wp:.4f} rad/s")  # Frequency corresponding to phase margin
    
    # Configure scientific plotting style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',       # Use serif font
        'font.size': 12,              # Base font size
        'axes.grid': True,            # Show grid
        'grid.alpha': 0.3,            # Grid transparency
        'axes.linewidth': 0.8         # Axis line width
        })
    
    # Create Bode diagram (size adjusted for journal typesetting)
    plt.figure(figsize=(10, 8), dpi=600)
    
    mag, phase, omega = ctrl.bode(open_loop_tf, dB=True, Hz=False, deg=True, plot=True)

    plt.subplot(2, 1, 1)
    #plt.semilogx(omega, 20*np.log10(mag))
    plt.ylabel('Magnitude (dB)')
    plt.xlim([1e-1, 1e3]) 
    plt.ylim([-250, 100])
    plt.legend()
    plt.axhline(0, color='r', linestyle='--', alpha=0.5)  # 0dB line
    plt.axhline(-gain_margin_db, color='r', linestyle='--')  # gain dB line
    plt.axvline(wp, color='r', linestyle='--', alpha=0.5)  # wp
    plt.axvline(wg, color='r', linestyle='--',
              label=f'GM={gain_margin_db:.1f} dB')  # wp
    #plt.legend(loc='lower right', frameon=True, framealpha=1)
    # Add text annotation to the right of the vertical line
    plt.annotate(f'GM={gain_margin_db:.1f} dB @ {wg:.1f} rad/s',
             xy=(wg, 0.7),                  # Position pointed by the arrow
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
    plt.legend()
    plt.axhline(-180+pm, color='r', linestyle='--', alpha=0.5)  # -180° line
    plt.axhline(-180, color='r', linestyle='--')  # -180° line
    plt.axvline(wp, color='r', linestyle='--', alpha=0.5)  # wp
    plt.axvline(wg, color='r', linestyle='--')  # wg
    # Add text annotation to the right of the vertical line
    plt.annotate(f'PM={pm:.1f}°\n@ {wp:.3f} rad/s',
             xy=(wp, 0.71),                  # Position pointed by the arrow
             xycoords=('data', 'axes fraction'),  # X: data coordinates, Y: axis fraction
             xytext=(10, 0),                # Text offset (pixels)
             textcoords='offset points',
             ha='left', va='center',        # Text alignment
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
             arrowprops=dict(arrowstyle='->'))

    # Save vector image (PDF format is recommended)
    plt.savefig('bode_h.png', 
            bbox_inches='tight', 
            pad_inches=0.05,
            transparent=True)
    #plt.show()
    plt.close()
    
    # Return system object for further analysis
    return open_loop_tf