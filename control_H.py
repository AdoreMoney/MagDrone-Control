import numpy as np
import time
from trim_vertical import FixedWingDynamics, TrimCalculator
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.ticker import MaxNLocator
import control as ctrl
import utils_h
from scipy import signal

class hPID:
    def __init__(self, kp, ki, kd, target_h=0.0, output_theta_limits=(-np.deg2rad(10.0), np.deg2rad(10.0))):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.target_h = target_h  # Target altitude
        self.output_theta_limits = output_theta_limits  # Output limits (radians)
        self._integral = 0  # Integral term
        self._prev_error = 0  # Previous error
        self._prev_prev_error = 0
        # New vertical velocity related attributes (maintain original structure)
        self._prev_vel_z = 0  # Store previous vertical velocity
        self.kvz = 0.05 * kp  # Vertical velocity feedback gain (default 0.5*kp)

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
        
        vz_term = 0
        if vel_z is not None:
            # Simple smoothing
            smoothed_vel_z = 0.7 * self._prev_vel_z + 0.3 * vel_z
            vz_term = -self.kvz * smoothed_vel_z  # Damping term opposite to velocity
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
        self.target_theta = target_theta  # Target value (radians)
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
        self.target_q = target_q  # Target value (radians/second)
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
    wind_field = None  # Wind field instance
    model = FixedWingDynamics(wind_field) 
    trim_calc = TrimCalculator(model, wind_field)  
    u_rel_trim, w_rel_trim, sync_trim, diff_trim, T_trim = trim_calc.find_trim()
    V_trim = np.sqrt(u_rel_trim**2 + w_rel_trim**2)
    Xu = model.rho * V_trim * model.S * model.CD0
    
    # Simulation parameters - fixed duration of 10 seconds
    dt = 0.01  # Time step (seconds)
    total_time = 20.0  # Fixed total simulation time of 10 seconds
    steps = int(total_time / dt)  # Calculate number of steps
    
    # Initial states
    theta_deg = 0.0  # Initial pitch angle (degrees)
    q_deg_s = 0.0    # Initial pitch rate (degrees/second)
    h_m = 0.0        # Initial altitude (meters)
    delta_e_sync_deg = 0  # Initial elevator deflection (degrees)
    
    # Linearized model
    A_long, B_long = trim_calc.linearize_trim()

    # Assuming state vector x = [V, alpha, q, theta, H]
    C_long = np.array([
        [0, 0, 1, 0, 0],  # First output is q
        [0, 0, 0, 1, 0],   # Second output is theta
        [0, 0, 0, 0, 1]   # Third output is H
    ])
    D_long = np.zeros((C_long.shape[0], B_long.shape[1]))  # Create zero matrix
    sys = ctrl.ss(A_long, B_long, C_long, D_long)
    
    # New H PID controller
    initial_kp_h = 0.033  # Initial proportional gain
    initial_ki_h = 0.01  # Initial integral gain
    initial_kd_h = 0.0  # Initial derivative gain
    target_h_m = 1.0  # Target altitude (meters)
    
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
    theta_history_deg = []         # Store pitch angle history (degrees)
    target_q_history_deg = []      # Store desired pitch rate history (degrees/second)
    q_history_deg = []             # Store pitch rate history (degrees/second)
    delta_e_sync_history_deg = []  # Store elevator deflection history (degrees)
    h_history_m = []               # Store altitude history (meters)
    target_theta_history_deg = []  # Store target pitch angle history (degrees)
    
    # Set SCI style parameters
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16  
    plt.rcParams['ytick.labelsize'] = 16  
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.0
    
    # Create figure and sliders
    fig = plt.figure(figsize=(8, 6), dpi=600)
    gs_left = fig.add_gridspec(2, 1)  

    # Create two subplots on the left (including altitude subplot)
    ax1 = fig.add_subplot(gs_left[0, 0])  # Altitude
    ax2 = fig.add_subplot(gs_left[1, 0])  # Theta

    # Initialize plots
    time_axis = np.arange(0, total_time, dt)
    # Define SCI style color scheme
    color_blue = '#1f77b4'     # Primary blue
    color_red = '#d62728'      # Primary red
    color_green = '#2ca02c'    # Primary green
    color_orange = '#ff7f0e'   # Secondary orange
    color_gray = '#7f7f7f'     # Gray
    
    line_h, = ax1.plot([], [], color=color_blue, label='Height (m)')
    line_target_h, = ax1.plot([], [], color=color_red, linestyle='--', label='Target Height (m)')
    ax1.set_ylabel('height (m)')
    ax1.set_ylim(0.0, 1.2)  
    ax1.legend(loc='lower right')
    ax1.grid()

    line_theta, = ax2.plot([], [], label='Theta (deg)')
    line_target_theta, = ax2.plot([], [], 'r--', label='Target Theta (deg)')
    ax2.set_ylabel('theta (deg)')
    ax2.set_ylim(-0.5, 2.0)  
    ax2.set_xlabel('Time (s)')
    ax2.legend(loc='upper right')
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
        h                              # H
    ])
    
    # Update function - called whenever slider values change
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
        prev_h = h  # Initial altitude
        vel_z = 0.0     # Initialize vertical velocity
        vel_z_smooth = 0.0  # Smoothed vertical velocity
        alpha_vel_z = 0.7   # Vertical velocity smoothing coefficient
        x = np.array([
            0.0,
            0.0,
            q,                                  # q
            theta,                              # θ
            h                                   # H
        ])
        
        # Clear historical data
        theta_history_deg = []         # Store pitch angle history (degrees)
        target_q_history_deg = []      # Store desired pitch rate history (degrees/second)
        q_history_deg = []             # Store pitch rate history (degrees/second)
        delta_e_sync_history_deg = []  # Store elevator deflection history (degrees)
        h_history_m = []               # Store altitude history (meters)
        target_theta_history_deg = []  # Store target pitch angle history (degrees)
        alpha_history = []
        alpha_disturbance_history = []
        target_h_history = []
        wind_u_history, wind_w_history = [], []
        
        # Define delay steps (50ms delay, dt=0.01s)
        delay_steps = int(0.05 / dt)
        target_sync_buffer = [np.deg2rad(0)] * delay_steps  # Initial value set to 0 rad/s
        
        # Wind speed parameters
        np.random.seed(42)
        h_ref = 0                # Reference altitude (m)
        mean_wind_speed = 10.41    # Mean wind speed at reference altitude (m/s)

        # Dryden model parameters
        L_u = 200                  # Longitudinal turbulence scale (m)
        L_w = 50                   # Vertical turbulence scale (m)

        # Pre-generate random noise sequence
        white_noise_u = np.random.randn(steps)  # Longitudinal wind noise
        white_noise_w = np.random.randn(steps)  # Vertical wind noise

        # Design Dryden transfer functions (continuous time)
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

        # Convert to discrete transfer functions
        dt_disc = dt
        num_d_u, den_d_u, _ = signal.cont2discrete((num_u, den_u), dt_disc)
        num_d_u = num_d_u.flatten()
        num_d_w, den_d_w, _ = signal.cont2discrete((num_w, den_w), dt_disc)
        num_d_w = num_d_w.flatten()
        
        wind_u_sequence, zi_u = signal.lfilter(num_d_u, den_d_u, white_noise_u, zi=zi_u)
        wind_w_sequence, zi_w = signal.lfilter(num_d_w, den_d_w, white_noise_w, zi=zi_w)
        
        # Re-run entire simulation (fixed 10 seconds)
        for k in range(steps):
            '''
            if k==500:
                h_pid.target_h += 2
            elif k==1000:
                h_pid.target_h -= 1
            elif k==1500:
                h_pid.target_h += 3
            elif k==1800:
                pass
            else:
                pass
            '''
            # Get current state
            y = C_long @ x  # Output q, theta, h
            q_curr = y[0]
            theta_curr = y[1]
            h_curr = y[2]
            alpha_curr = x[1]  # Get current angle of attack (rad)
            #if k==700:
                #x[0] += 10
            V_curr = x[0]   # Current airspeed
            
            # Calculate vertical velocity (via altitude difference)
            if k > 0:
                vel_z = (h_curr - prev_h) / dt
                # Simple smoothing
                vel_z_smooth = alpha_vel_z * vel_z_smooth + (1 - alpha_vel_z) * vel_z
            prev_h = h_curr
            
            # Angle of attack random disturbance simulating wind field changes
            # Calculate turbulence intensity based on current altitude
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
            
            # Limit H PID output within target pitch angle range
            target_theta_h_clipped = np.clip(target_theta_h, -np.deg2rad(10.0), np.deg2rad(10.0))
            
            # Update theta PID target value
            theta_pid.target_theta = target_theta_h_clipped
            
            # Outer loop PID (angle loop)
            target_q = theta_pid(theta_curr, dt)
            
            delta_e_sync = q_pid(q_curr, dt, target_q=target_q)
            
            # Delay module: update buffer
            target_sync_buffer.pop(0) 
            target_sync_buffer.append(delta_e_sync)  
            
            # Convert delta_e_sync to degrees if needed
            delayed_e_sync = target_sync_buffer[0]
            delta_e_sync_deg = np.rad2deg(delayed_e_sync)
            
            Dw = np.array([-Xu*wind_u, -w_rel_trim*wind_u/(V_trim**2), 0, 0, 0])
            # Control input vector [delta_e_sync, T]
            u = np.array([
                delayed_e_sync,
                0.0   # Thrust remains unchanged
            ])
            
            # Calculate state derivative dx = A @ x + B @ u
            x_dot = A_long @ x + B_long @ u + Dw
            
            # Update state (Euler integration)
            x = x + x_dot * dt
            
            # Store data
            theta_history_deg.append(theta_deg_curr)
            target_q_history_deg.append(np.rad2deg(target_q))  # Store q_desired (target q)
            q_history_deg.append(q_deg)
            delta_e_sync_history_deg.append(delta_e_sync_deg)
            h_history_m.append(h_m_curr + h_ref)
            target_theta_history_deg.append(np.rad2deg(target_theta_h_clipped))
            target_h_history.append(h_pid.target_h + h_ref)
        
        line_h.set_data(time_axis[:len(h_history_m)], h_history_m)
        line_target_h.set_data(time_axis[:len(h_history_m)], target_h_history)

        line_theta.set_data(time_axis[:len(theta_history_deg)], theta_history_deg)
        line_target_theta.set_data(time_axis[:len(theta_history_deg)], target_theta_history_deg)
        
        # Adjust axis ranges
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
            
        # Add text annotation to the figure
        # First clear previous text (if any)
        for artist in ax1.texts:
            artist.remove()
        
        if overshoot_percent is not None and rise_time_sec is not None:
            text_str = f'Overshooting: {overshoot_percent:.2f}%\nRise time: {rise_time_sec:.2f}s\nSettling time: {settling_time:.2f}s\nSteady state error: {steady_error:.2f}%'
        elif overshoot_percent is not None:
            text_str = f'Overshooting: {overshoot_percent:.2f}%\nRise time: -1'
        elif rise_time_sec is not None:
            text_str = f'Overshooting: -1\nRise time: {rise_time_sec:.2f}s'
        else:
            text_str = 'Overshooting and Rise time: -1'
        
        ax1.text(0.2, 0.05, text_str, transform=ax1.transAxes,
                 verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        ax1.axhline(y=target_h_m*0.95, xmin=0.0, xmax=20, color=color_gray, linestyle='--')
        ax1.axhline(target_h_m*1.05, xmin=0.0, xmax=20, color=color_gray, linestyle='--')
        ax1.plot([0.0, 0.0], [0.0, 1.0], color=color_red, linestyle='--')
        
        #ax1.plot([0.0, 0.0], [100.0, 101.0], color=color_red, linestyle='--')
        fig.canvas.draw_idle()

    update_simulation(None)
    
    save_path = 'h-pitch_pid/13_Step response of altitude control loop.png'
    #save_path = 'h-pitch_pid/15_Response curve of altitude PID control loop under step command input.png'
    fig.savefig(save_path, 
                dpi=600,              
                bbox_inches='tight',   
                format='png')          

    # 显示图形
    #plt.show()

if __name__ == "__main__":
    control()