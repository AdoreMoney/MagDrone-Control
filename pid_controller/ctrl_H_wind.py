import numpy as np
import time
from trim_vertical import FixedWingDynamics, TrimCalculator, WindField
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.ticker import MaxNLocator
import control as ctrl
import utils.utils_h as utils_h
from scipy import signal

class hPID:
    def __init__(self, kp, ki, kd, target_h=0.0, output_theta_limits=(-np.deg2rad(10.0), np.deg2rad(10.0))):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.target_h = target_h  # Desired height
        self.output_theta_limits = output_theta_limits  # Output limits (radians)
        self._integral = 0  # Integral term
        self._prev_error = 0  # Previous error
        self._prev_prev_error = 0
        # New vertical velocity related attributes (maintain original structure)
        self._prev_vel_z = 0  # New: Store previous vertical velocity
        self.kvz = 0.05 * kp  # New: Vertical velocity feedback gain (default 0.5*kp)

    def __call__(self, h, dt, vel_z=None):
        # Calculate error
        error = self.target_h - h

        # Proportional term
        p_term = self.kp * error

        # Integral term (with anti-windup)
        self._integral += error * dt
        if self.output_theta_limits[0] is not None and self.output_theta_limits[1] is not None:
            self._integral = np.clip(self._integral, self.output_theta_limits[0], self.output_theta_limits[1])
        i_term = self.ki * self._integral

        # Derivative term (with low-pass filter)
        if dt > 0:
            d_term = self.kd * (error - self._prev_error) / dt
        else:
            d_term = 0
        self._prev_error = error
        
        # New: Vertical velocity feedback term (effective when vel_z is provided)
        vz_term = 0
        if vel_z is not None:
            # Simple smoothing
            smoothed_vel_z = 0.7 * self._prev_vel_z + 0.3 * vel_z
            vz_term = -self.kvz * smoothed_vel_z  # Damping term is opposite to velocity direction
            self._prev_vel_z = smoothed_vel_z
            
        # Calculate output
        output_target_theta = p_term + i_term + d_term + vz_term
        
        self._prev_prev_error = self._prev_error
        self._prev_error = error

        # Output limiting
        if self.output_theta_limits[0] is not None and self.output_theta_limits[1] is not None:
            output_target_theta = np.clip(output_target_theta, self.output_theta_limits[0], self.output_theta_limits[1])

        return output_target_theta
    
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

def control():
    wind_field = WindField()
    model = FixedWingDynamics(wind_field) 
    trim_calc = TrimCalculator(model, wind_field)  
    u_rel_trim, w_rel_trim, sync_trim, diff_trim, T_trim = trim_calc.find_trim()
    V_trim = np.sqrt(u_rel_trim**2 + w_rel_trim**2)
    Xu = model.rho * V_trim * model.S * model.CD0
    
    # Simulation parameters
    dt = 0.01  # Time step (seconds)
    total_time = 20.0  # Total simulation time (seconds)
    steps = int(total_time / dt)  # Number of steps
    
    # Initial states
    theta_deg = 0.0  # Initial pitch angle (deg)
    q_deg_s = 0.0    # Initial pitch rate (deg/s)
    h_m = 0.0        # Initial height (meters)
    delta_e_sync_deg = 0  # Initial elevator deflection (deg)
    
    # Linearized model
    A_long, B_long = trim_calc.linearize_trim()
    
    # Define output matrix C and direct transmission term D
    # Select pitch rate (q) and pitch angle (theta) as outputs
    # Assume state vector x = [V, alpha, q, theta, H]
    # q is the 3rd state (index 2), theta is the 4th state (index 3)
    C_long = np.array([
        [0, 0, 1, 0, 0],  # 1st output: q
        [0, 0, 0, 1, 0],   # 2nd output: theta
        [0, 0, 0, 0, 1]   # 3rd output: H
    ])
    D_long = np.zeros((C_long.shape[0], B_long.shape[1]))  # Create zero matrix
    sys = ctrl.ss(A_long, B_long, C_long, D_long)
    
    # New H PID controller
    initial_kp_h = 0.033  # Initial proportional gain
    initial_ki_h = 0.01  # Initial integral gain
    initial_kd_h = 0.0  # Initial derivative gain
    target_h_m = 1.0  # Target height (meters)
    
    # Initialize PID controllers (initial parameters)
    initial_kp_theta = 2.1
    initial_ki_theta = 0.001
    initial_kd_theta = 1.2
    initial_kp_q = 0.9
    initial_ki_q = 1.4
    initial_kd_q = 0.04
    
    h_pid = hPID(kp=initial_kp_h, ki=initial_ki_h, kd=initial_kd_h, target_h=target_h_m)
    theta_pid = thetaPID(kp=initial_kp_theta, ki=initial_ki_theta, kd=initial_kd_theta)
    q_pid = qPID(kp=initial_kp_q, ki=initial_ki_q, kd=initial_kd_q)
    
    # Plot open-loop Bode diagram with PID control
    open_loop_tf = utils_h.plot_pid_controlled_bode(trim_calc, h_pid, theta_pid, q_pid, target_h=target_h_m, dt=dt)

    # Data storage
    theta_history_deg = []         # Pitch angle history (deg)
    target_q_history_deg = []      # Desired pitch rate history (deg/s)
    q_history_deg = []             # Pitch rate history (deg/s)
    delta_e_sync_history_deg = []  # Elevator deflection history (deg)
    h_history_m = []               # Height history (meters)
    target_theta_history_deg = []  # Target pitch angle history (deg)
    
    # Set SCI style parameters
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.6
    
    # Create figure and sliders
    fig = plt.figure(figsize=(8, 6))
    # Use GridSpec to split figure into left and right parts
    # Left 70% width for subplots, right 30% width for sliders
    gs_left = fig.add_gridspec(2, 1, left=0.15, right=0.90, hspace=0.4)

    # Create left subplots (add height h subplot)
    ax1 = fig.add_subplot(gs_left[0, 0])  # Height
    ax2 = fig.add_subplot(gs_left[1, 0])  # Alpha

    # Initialize plotting
    time_axis = np.arange(0, total_time, dt)
    # Define SCI style color scheme
    color_blue = '#1f77b4'     # Primary blue
    color_red = '#d62728'      # Primary red
    color_green = '#2ca02c'    # Primary green
    color_orange = '#ff7f0e'   # Secondary orange
    color_gray = '#7f7f7f'     # Gray
    
    line_h, = ax1.plot([], [], color=color_blue, label='Height (m)')
    line_target_h, = ax1.plot([], [], color=color_red, linestyle='--', label='Target Height (m)')
    ax1.set_ylabel('Height')
    ax1.set_ylim(100.0, 105.5)  # Set appropriate y-axis range
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))  # Set y-axis tick count
    ax1.legend(loc='lower right')
    ax1.grid()

    line_alpha_wind, = ax2.plot([], [], color=color_blue, label='Wind u (m/s)', alpha=0.8)
    ax2.set_ylabel('u')
    ax2.set_ylim(-1.0, 1.8)  # Set appropriate y-axis range
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax2.grid()

    # Optimize borders
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
    
    # Current simulation state
    current_step = 0
    V_trim = np.sqrt(u_rel_trim**2 + w_rel_trim**2)
    theta = np.deg2rad(theta_deg)
    q = np.deg2rad(q_deg_s)
    delta_e_sync = np.deg2rad(delta_e_sync_deg)
    h = h_m
    x = np.array([
        0.0,
        0.0,
        q,                                  # q
        theta,                              # θ
        h                                   # H
    ])
    
    # Update function - Called when slider values change
    def update_simulation(val=None):
        nonlocal current_step, theta, q, delta_e_sync, h, x, theta_history_deg, target_q_history_deg, q_history_deg, delta_e_sync_history_deg, h_history_m, target_theta_history_deg
        nonlocal V_trim
        
        h_pid.kp = initial_kp_h
        h_pid.ki = initial_ki_h
        h_pid.kd = initial_kd_h
        
        # Reset simulation state
        current_step = 0
        theta = np.deg2rad(theta_deg)
        q = np.deg2rad(q_deg_s)
        delta_e_sync = np.deg2rad(0.0)
        h = h_m
        # Add vertical velocity related variables before simulation loop
        prev_h = h  # Initial height
        vel_z = 0.0     # Vertical velocity initialization
        vel_z_smooth = 0.0  # Smoothed vertical velocity
        alpha_vel_z = 0.7   # Vertical velocity smoothing coefficient
        x = np.array([
            0.0,
            0.0,
            q,                                  # q
            theta,                              # θ
            h                                   # H
        ])
        A = 0.5            # Amplitude (height variation range)
        T = 1000           # Period (number of steps)
        
        # Clear history data
        theta_history_deg = []         
        target_q_history_deg = []      
        q_history_deg = []             
        delta_e_sync_history_deg = []  
        h_history_m = []               
        target_theta_history_deg = []  
        alpha_history = []
        alpha_disturbance_history = []
        target_h_history = []
        wind_u_history, wind_w_history = [], []
        
        # Define delay steps (50ms delay, dt=0.01s)
        delay_steps = int(0.05 / dt)
        target_sync_buffer = [np.deg2rad(0)] * delay_steps  # Initial value: 0 rad/s
        
        # Wind speed parameters
        np.random.seed(42)
        h_ref = 100                # Reference height (m)
        mean_wind_speed = 10.41    # Mean wind speed at reference height (m/s)

        # Dryden model parameters
        L_u = 200                  # Longitudinal turbulence scale (m)
        L_w = 50                   # Vertical turbulence scale (m)

        # Pre-generate random noise sequence
        white_noise_u = np.random.randn(steps)  # Longitudinal wind noise
        white_noise_w = np.random.randn(steps)  # Vertical wind noise

        # Design Dryden transfer function (continuous time)
        omega_u = 2 * np.pi * mean_wind_speed / L_u
        omega_w = 2 * np.pi * mean_wind_speed / L_w

        # Longitudinal gust transfer function
        num_u = [np.sqrt(2 * omega_u)]
        den_u = [1, omega_u]
        zi_u = np.zeros(1)

        # Vertical gust transfer function
        num_w = [np.sqrt(3 * omega_w), np.sqrt(3 * omega_w**3)]
        den_w = [1, 2 * omega_w, omega_w**2]
        zi_w = np.zeros(2)

        # Convert to discrete transfer function
        dt_disc = dt
        num_d_u, den_d_u, _ = signal.cont2discrete((num_u, den_u), dt_disc)
        num_d_u = num_d_u.flatten()
        num_d_w, den_d_w, _ = signal.cont2discrete((num_w, den_w), dt_disc)
        num_d_w = num_d_w.flatten()
        
        wind_u_sequence, zi_u = signal.lfilter(num_d_u, den_d_u, white_noise_u, zi=zi_u)
        wind_w_sequence, zi_w = signal.lfilter(num_d_w, den_d_w, white_noise_w, zi=zi_w)
        
        # Re-run entire simulation
        for k in range(steps):            
            # Get current states
            y = C_long @ x  # Output q, theta, h
            q_curr = y[0]
            theta_curr = y[1]
            h_curr = y[2]
            alpha_curr = x[1]  # Get current angle of attack (rad)
            V_curr = x[0]   # Current airspeed
            
            # Calculate vertical velocity (via height difference)
            if k > 0:
                vel_z = (h_curr - prev_h) / dt
                # Simple smoothing
                vel_z_smooth = alpha_vel_z * vel_z_smooth + (1 - alpha_vel_z) * vel_z
            prev_h = h_curr
            
            # Calculate turbulence intensity based on current height
            h_true = h_curr + h_ref
            sigma_u = 1.0 * mean_wind_speed * (h_true / 10) ** (-0.25)  # Longitudinal turbulence intensity
            sigma_w = 0.7 * mean_wind_speed * (h_true / 10) ** (-0.25)  # Vertical turbulence intensity
            
            # Scale to actual wind speed disturbance
            wind_u = sigma_u * wind_u_sequence[k]
            wind_w = sigma_w * wind_w_sequence[k]
            wind_u = 0.0
            wind_w = 0.0
            
            alpha_history.append(np.rad2deg(alpha_curr))
            wind_u_history.append(wind_u)
            
            alpha_disturbance_history.append(np.rad2deg(alpha_curr))
            
            # Convert states to degrees
            q_deg = np.rad2deg(q_curr)
            theta_deg_curr = np.rad2deg(theta_curr)
            h_m_curr = h_curr
            
            # H PID control (outermost loop)
            target_theta_h = h_pid(h_curr, dt, vel_z=vel_z_smooth)
            
            # Clip H PID output to target pitch angle range
            target_theta_h_clipped = np.clip(target_theta_h, -np.deg2rad(10.0), np.deg2rad(10.0))
            
            # Update theta PID target value
            theta_pid.target_theta = target_theta_h_clipped
            
            # Outer loop PID (angle loop)
            target_q = theta_pid(theta_curr, dt)
            
            delta_e_sync = q_pid(q_curr, dt, target_q=target_q)
            
            # Delay module: Update buffer
            target_sync_buffer.pop(0) 
            target_sync_buffer.append(delta_e_sync)  # Add latest value
            
            # Convert delta_e_sync to degrees
            delayed_e_sync = target_sync_buffer[0]
            delta_e_sync_deg = np.rad2deg(delayed_e_sync)
            
            Dw = np.array([-Xu*wind_u, -w_rel_trim*wind_u/(V_trim**2), 0, 0, 0])
            # Control input vector [delta_e_sync, T]
            u = np.array([
                delayed_e_sync,
                0.0   # Thrust remains constant
            ])
            
            # Calculate state derivative dx = A @ x + B @ u
            x_dot = A_long @ x + B_long @ u + Dw
            
            # Update state (Euler integration)
            x = x + x_dot * dt
            
            # Store data
            theta_history_deg.append(theta_deg_curr)
            target_q_history_deg.append(np.rad2deg(target_q))  # Store desired q
            q_history_deg.append(q_deg)
            delta_e_sync_history_deg.append(delta_e_sync_deg)
            h_history_m.append(h_m_curr+100.0)
            target_theta_history_deg.append(np.rad2deg(target_theta_h_clipped))
            target_h_history.append(h_pid.target_h + 100.0)
        
        # Update plots
        line_h.set_data(time_axis[:len(h_history_m)], h_history_m)
        line_target_h.set_data(time_axis[:len(h_history_m)], target_h_history)
        
        line_alpha_wind.set_data(time_axis[:len(wind_u_history)], wind_u_history)
        
        # Adjust axes limits
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        
        # Calculate overshoot and response time
        overshoot_percent = utils_h.calculate_overshoot(h_history_m, target_h_m)
        rise_time_sec = utils_h.calculate_rise_time(h_history_m, target_h_m, dt, threshold=0.9)
        errors = [(target_h_m - h) for h in h_history_m]
        settling_time = utils_h.calculate_settling_time(errors)
        steady_error = utils_h.calculate_steady_state_error(h_history_m, target_h_m, settling_time)
        
        if settling_time is not None:
            print('Settling time:', settling_time*0.01)
            print('Steady state error:', steady_error)
        else:
            print('Unable to reach steady state')
            
        fig.canvas.draw_idle()
    
    # Initial simulation run
    update_simulation(None)
    
    # Save high-resolution image (600dpi)
    save_path = 'h_wind_repo_vari.png'
    fig.savefig(save_path, 
                dpi=600,
                bbox_inches='tight',
                format='png')
    
    print(f"High-resolution image saved to: {save_path}")
    # Show figure
    plt.show()

if __name__ == "__main__":
    control()