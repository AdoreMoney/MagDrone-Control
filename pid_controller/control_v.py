import numpy as np
import time
from trim_vertical import FixedWingDynamics, TrimCalculator, WindField
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import control as ctrl
import utils.utils_h as utils_h

class thetaPID:
    def __init__(self, kp, ki, kd, target_theta=0.0, output_q_limits=(-np.deg2rad(20.0), np.deg2rad(20.0))):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.target_theta = target_theta  # Desired value (radians)
        self.output_q_limits = output_q_limits  # Output limits (radians)
        self._integral = 0  # Integral term
        self._prev_error = 0  # Previous error

    def __call__(self, theta, dt):
        # Calculate error
        error = self.target_theta - theta

        # Proportional term
        p_term = self.kp * error

        # Integral term (with anti-windup)
        self._integral += error * dt
        if self.output_q_limits[0] is not None and self.output_q_limits[1] is not None:
            self._integral = np.clip(self._integral, self.output_q_limits[0], self.output_q_limits[1])
        i_term = self.ki * self._integral

        # Derivative term (with low-pass filter)
        if dt > 0:
            d_term = self.kd * (error - self._prev_error) / dt
        else:
            d_term = 0
        self._prev_error = error

        # Calculate output
        output_target_q = p_term + i_term + d_term

        # Output limiting
        if self.output_q_limits[0] is not None and self.output_q_limits[1] is not None:
            output_target_q = np.clip(output_target_q, self.output_q_limits[0], self.output_q_limits[1])

        return output_target_q
    
class qPID:
    def __init__(self, kp, ki, kd, target_q=0.0, output_e_sync_limits=(-np.deg2rad(10.0), np.deg2rad(10.0))):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.target_q = target_q  # Desired value (radians/second)
        self.output_e_sync_limits = output_e_sync_limits  # Output limits (radians)
        self._integral = 0  # Integral term
        self._prev_error = 0  # Previous error

    def __call__(self, q, dt, target_q=None):
        if target_q is not None:
            self.target_q = target_q  # Dynamically update target value
        
        # Calculate error
        error = self.target_q - q

        # Proportional term
        p_term = self.kp * error

        # Integral term (with anti-windup)
        self._integral += error * dt
        if self.output_e_sync_limits[0] is not None and self.output_e_sync_limits[1] is not None:
            self._integral = np.clip(self._integral, self.output_e_sync_limits[0], self.output_e_sync_limits[1])
        i_term = self.ki * self._integral

        # Derivative term (with low-pass filter)
        if dt > 0:
            d_term = self.kd * (error - self._prev_error) / dt
        else:
            d_term = 0
        self._prev_error = error

        # Calculate output
        output_e_sync = p_term + i_term + d_term

        # Output limiting
        if self.output_e_sync_limits[0] is not None and self.output_e_sync_limits[1] is not None:
            output_e_sync = np.clip(output_e_sync, self.output_e_sync_limits[0], self.output_e_sync_limits[1])

        return output_e_sync
    
def calculate_overshoot(data, target_value):
    max_value = max(data)
    if target_value == 0:
        return ((max_value - target_value) / (max_value + 1e-6)) * 100
    else:
        return ((max_value - target_value) / target_value) * 100

def calculate_rise_time(data, target_value, dt, threshold=0.9):
    lower_threshold = 0.1 * target_value
    upper_threshold = threshold * target_value
    
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
    else:
        return None

def plot_pid_controlled_bode(trim_calc, theta_pid, q_pid, target_theta_deg=5, dt=0.01):
    # Linearized model
    A_long, B_long = trim_calc.linearize_trim()
    print('A_long:', A_long)
    print('B_long:', B_long)
    
    # Define output matrix C and direct transmission term D
    C_long = np.array([
        [0, 0, 1, 0, 0],  # First output: q
        [0, 0, 0, 1, 0]   # Second output: theta
    ])
    D_long = np.zeros((C_long.shape[0], B_long.shape[1]))  # Create zero matrix
    
    # Create state space model
    sys_state_space = ctrl.ss(A_long, B_long, C_long, D_long)
    
    # Specify input (elevator) and output (pitch angle) indices
    input_idx = 0  # delta_e_sync input
    output_idx = 1  # theta output
    
    # Extract subsystem A, B, C, D
    A_sub = sys_state_space.A
    B_sub = sys_state_space.B[:, input_idx].reshape(-1, 1)
    C_sub = sys_state_space.C[output_idx, :].reshape(1, -1)
    D_sub = sys_state_space.D[output_idx, input_idx]
    # Create SISO subsystem
    sys_sub = ctrl.ss(A_sub, B_sub, C_sub, D_sub)
    # Extract transfer function
    sys_tf = ctrl.ss2tf(sys_sub)
    print("Original UAV model transfer function:")
    print(sys_tf)
    
    # Build thetaPID transfer function
    kp_theta = theta_pid.kp
    ki_theta = theta_pid.ki
    kd_theta = theta_pid.kd
    
    # Create thetaPID transfer function
    pid_theta_num = [kd_theta, kp_theta, ki_theta]
    pid_theta_den = [1, 0]  # Denominator: s
    pid_theta_tf = ctrl.tf(pid_theta_num, pid_theta_den)
    print("thetaPID transfer function:")
    print(pid_theta_tf)
    
    # Build qPID transfer function
    kp_q = q_pid.kp
    ki_q = q_pid.ki
    kd_q = q_pid.kd
    
    num = [20.0]
    den = [1, 20.0]
    actuator_tf = ctrl.TransferFunction(num, den)
    
    # Create qPID transfer function
    pid_q_num = [kd_q, kp_q, ki_q]
    pid_q_den = [1, 0]  # Denominator: s
    pid_q_tf = ctrl.tf(pid_q_num, pid_q_den)
    closed_pid_q_tf = ctrl.feedback(pid_q_tf * actuator_tf, 1)
    print("qPID transfer function:")
    print(closed_pid_q_tf)
    
    open_loop_tf = closed_pid_q_tf  * sys_tf * pid_theta_tf
    print("Open-loop transfer function (with cascaded PID):")
    print(open_loop_tf)
    
    # Calculate gain and phase margins
    gm, pm, wg, wp = ctrl.margin(open_loop_tf)
    gain_margin_db = 20 * np.log10(gm)
    
    print(f"Gain Margin: {gain_margin_db:.2f} dB (Gain Margin: {gm:.2f})")
    print(f"Phase Margin: {pm:.2f} degrees (Phase Margin: {pm:.2f} degrees)")
    
    # Configure scientific plotting style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.linewidth': 0.8
        })
    
    # Create Bode plot
    plt.figure(figsize=(10, 8), dpi=600)
    
    mag, phase, omega = ctrl.bode(open_loop_tf, dB=True, Hz=False, deg=True, plot=True)

    plt.subplot(2, 1, 1)
    plt.ylabel('Magnitude (dB)')
    plt.xlim([1e-1, 1e3]) 
    plt.legend()
    plt.axhline(0, color='r', linestyle='--')  # 0dB line
    plt.axvline(wp, color='r', linestyle='--',
              label=f'GM={gain_margin_db:.1f} dB')

    plt.annotate(f'GM={gain_margin_db:.1f} dB',
             xy=(wp, 0.78),
             xycoords=('data', 'axes fraction'),
             xytext=(10, 0),
             textcoords='offset points',
             ha='left', va='center',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
             arrowprops=dict(arrowstyle='->'))

    plt.subplot(2, 1, 2)
    plt.ylabel('Phase (deg)')
    plt.xlabel('Frequency (rad/s)')
    plt.xlim([1e-1, 1e3]) 
    plt.legend()
    plt.axhline(-180+pm, color='r', linestyle='--')  # -180° line
    plt.axvline(wp, color='r', linestyle='--',
                label=f'PM={pm:.1f}° @ {wp:.1f} rad/s')

    plt.annotate(f'PM={pm:.1f}°\n@ {wp:.1f} rad/s',
             xy=(wp, 0.22),
             xycoords=('data', 'axes fraction'),
             xytext=(10, 0),
             textcoords='offset points',
             ha='left', va='center',
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
             arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.savefig('bode_pitch.png', 
            bbox_inches='tight', 
            pad_inches=0.05,
            transparent=True)
    plt.close()
    
    # Return system object for further analysis
    return open_loop_tf

def control():
    wind_field = WindField()
    model = FixedWingDynamics(wind_field) 
    trim_calc = TrimCalculator(model, wind_field)  
    u_rel_trim, w_rel_trim, sync_trim, diff_trim, T_trim = trim_calc.find_trim()
    
    # Simulation parameters
    dt = 0.01  # Time step (seconds)
    total_time = 20.0  # Total simulation time (seconds)
    steps = int(total_time / dt)  # Number of steps
    
    # Initial states
    theta_deg = 0.0  # Initial pitch angle (deg)
    q_deg_s = 0.0    # Initial pitch rate (deg/s)
    delta_e_sync_deg = 0  # Initial elevator deflection (deg)
    
    # Linearized model
    A_long, B_long = trim_calc.linearize_trim()
    
    # Define output matrix C and direct transmission term D
    C_long = np.array([
        [0, 0, 1, 0, 0],  # First output: q
        [0, 0, 0, 1, 0]   # Second output: theta
    ])
    D_long = np.zeros((C_long.shape[0], B_long.shape[1]))
    sys = ctrl.ss(A_long, B_long, C_long, D_long)
    
    # Desired pitch angle (step input, degrees)
    target_theta_deg = 1.0  # Desired pitch angle (deg)
    target_theta = np.deg2rad(target_theta_deg)  # Convert to radians
    
    # Initialize PID controllers (initial parameters)
    initial_kp_theta = 2.1
    initial_ki_theta = 0.001
    initial_kd_theta = 1.2
    initial_kp_q = 0.9
    initial_ki_q = 1.4
    initial_kd_q = 0.04
    
    theta_pid = thetaPID(kp=initial_kp_theta, ki=initial_ki_theta, kd=initial_kd_theta, target_theta=target_theta)
    q_pid = qPID(kp=initial_kp_q, ki=initial_ki_q, kd=initial_kd_q)
    
    # Plot Bode diagram with PID control
    open_loop_tf = plot_pid_controlled_bode(trim_calc, theta_pid, q_pid, target_theta_deg=target_theta_deg, dt=dt)
    
    # Data storage
    theta_history_deg = []         # Pitch angle history (deg)
    target_q_history_deg = []      # Desired pitch rate history (deg/s)
    q_history_deg = []             # Pitch rate history (deg/s)
    delta_e_sync_history_deg = []  # Elevator deflection history (deg)
    
    # SCI style settings
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.6
    
    # Create figure and axes
    fig = plt.figure(figsize=(16, 12), dpi=600)
    
    # Create 3 subplots
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)
    
    # Initialize plotting
    time_axis = np.arange(0, total_time, dt)
    # SCI style color scheme
    color_blue = '#1f77b4'     # Primary blue
    color_red = '#d62728'      # Primary red
    color_green = '#2ca02c'    # Primary green
    color_orange = '#ff7f0e'   # Secondary orange
    color_gray = '#7f7f7f'     # Gray
    
    line_theta, = ax1.plot([], [], color=color_blue, label='Theta (deg)')
    line_target_theta, = ax1.plot([], [], color=color_red, linestyle='--', label='Target Theta (deg)')
    ax1.set_ylabel('theta (deg)')
    ax1.set_ylim(0.0, 1.2)
    ax1.legend()
    ax1.grid()
    
    line_q, = ax2.plot([], [], color=color_blue, label='q (deg/s)')
    line_target_q, = ax2.plot([], [], color=color_red, linestyle='--', label='Target q (deg/s)')
    ax2.set_ylabel('q (deg/s)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylim(-1, 5)
    ax2.legend()
    ax2.grid()
    
    line_delta_e, = ax3.plot([], [], label='Delta e sync (deg)')
    ax3.set_ylabel('delta_e_sync (deg)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylim(-11, 11)
    ax3.legend()
    ax3.grid()
    
    # Optimize borders
    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
        
    # Current simulation state
    current_step = 0
    theta = np.deg2rad(theta_deg)
    q = np.deg2rad(q_deg_s)
    delta_e_sync = np.deg2rad(delta_e_sync_deg)
    x = np.array([
        0.0,
        0.0,
        q,                                  # q
        theta,                              # θ
        0.0                                 # H
    ])
    
    # Update function - Run simulation and update plots directly
    def update_simulation():
        nonlocal current_step, theta, q, delta_e_sync, x, theta_history_deg, target_q_history_deg, q_history_deg, delta_e_sync_history_deg
        nonlocal target_theta
        
        # Reset simulation state
        current_step = 0
        theta = np.deg2rad(theta_deg)
        q = np.deg2rad(q_deg_s)
        delta_e_sync = np.deg2rad(0.0)
        x = np.array([
        0.0,
        0.0,
        q,                                  # q
        theta,                              # θ
        0.0                                 # H
        ])
    
        # Clear history data
        theta_history_deg = []         
        target_q_history_deg = []      
        q_history_deg = []             
        delta_e_sync_history_deg = []  
        target_theta_history_deg = []
    
        # Define delay steps (50ms delay, dt=0.01s)
        delay_steps = int(0.05 / dt)
        target_sync_buffer = [np.deg2rad(0)] * delay_steps  # Initial value: 0 rad/s
    
        # Re-run entire simulation (fixed 10 seconds)
        for k in range(steps):        
            # Get current states
            y = C_long @ x  # Output q and theta
            q_curr = y[0]
            theta_curr = y[1]
        
            # Convert states to degrees
            q_deg = np.rad2deg(q_curr)
            theta_deg_curr = np.rad2deg(theta_curr)
        
            # Outer loop PID (angle loop)
            target_q = theta_pid(theta_curr, dt)

            # Inner loop PID (rate loop) uses delayed target_q
            delta_e_sync = q_pid(q_curr, dt, target_q=target_q)
        
            # Delay module: Update buffer
            target_sync_buffer.pop(0) 
            target_sync_buffer.append(delta_e_sync)  # Add latest value
            
            # Convert delta_e_sync to degrees
            delayed_e_sync = target_sync_buffer[0]
            delta_e_sync_deg = np.rad2deg(delayed_e_sync)
        
            # Control input vector [delta_e_sync, T]
            u = np.array([
                delayed_e_sync,
                0.0   # Thrust remains constant
            ])
        
            # Calculate state derivative dx = A @ x + B @ u
            x_dot = A_long @ x + B_long @ u
        
            # Update state (Euler integration)
            x = x + x_dot * dt
        
            # Store data
            theta_history_deg.append(theta_deg_curr)
            target_q_history_deg.append(np.rad2deg(target_q))  # Store desired q
            q_history_deg.append(q_deg)
            delta_e_sync_history_deg.append(delta_e_sync_deg)
            target_theta_history_deg.append(np.rad2deg(theta_pid.target_theta))
        
            # Update current step
            current_step += 1
    
        # Update plots
        line_theta.set_data(time_axis[:len(theta_history_deg)], theta_history_deg)
        line_target_theta.set_data(time_axis[:len(theta_history_deg)], target_theta_history_deg)
    
        line_q.set_data(time_axis[:len(q_history_deg)], q_history_deg)
        line_target_q.set_data(time_axis[:len(q_history_deg)], target_q_history_deg)
    
        line_delta_e.set_data(time_axis[:len(delta_e_sync_history_deg)], delta_e_sync_history_deg)
    
        # Adjust axes limits
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        ax3.relim()
        ax3.autoscale_view()
    
        # Calculate overshoot and response time
        overshoot_percent = calculate_overshoot(theta_history_deg, target_theta_deg)
        rise_time_sec = calculate_rise_time(theta_history_deg, target_theta_deg, dt, threshold=0.9)
        errors = [(target_theta_deg - theta) for theta in theta_history_deg]
        settling_time = utils_h.calculate_settling_time(errors)
        steady_error = utils_h.calculate_steady_state_error(theta_history_deg, np.rad2deg(theta_pid.target_theta), settling_time)
        print('Settling time:', settling_time)
        print('Steady state error:', steady_error)
        
        # Add text annotation on plot
        for artist in ax1.texts:
            artist.remove()
    
        if overshoot_percent is not None and rise_time_sec is not None:
            text_str = f'Overshoot: {np.abs(overshoot_percent):.2f}%\nRise time: {rise_time_sec:.2f}s\nSettling time: {settling_time*0.01:.2f}s\nSteady state error: {0.0:.2f}%'
        elif overshoot_percent is not None:
            text_str = f'Overshoot: {overshoot_percent:.2f}%\nRise time: -'
        elif rise_time_sec is not None:
            text_str = f'Overshoot: -\nRise time: {rise_time_sec:.2f}s'
        else:
            text_str = 'Overshoot and Rise time: -'
        
        ax1.text(0.17, 0.73, text_str, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        ax1.axhline(y=target_theta_deg*0.95, xmin=0.0, xmax=20, color=color_gray, linestyle='--')
        ax1.axhline(target_theta_deg*1.05, xmin=0.0, xmax=20, color=color_gray, linestyle='--')
        ax1.plot([0.0, 0.0], [0.0, 1.0], color=color_red, linestyle='--')
        fig.canvas.draw_idle()

    # Initial simulation run
    update_simulation()

    # Show plot
    save_path = 'pitch_response_temp.png'
    fig.savefig(save_path, 
                dpi=600,
                bbox_inches='tight',
                format='png')
    plt.show()

if __name__ == "__main__":
    control()