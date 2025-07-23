import numpy as np
import time
from trim_hrz import FixedWingDynamics, TrimCalculator
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.ticker import MaxNLocator
import control as ctrl
import pandas as pd
import utils_ry
from scipy import signal

class CoordinatedTurnController:
    def __init__(self, V_trim, pure_roll):
        # Roll angle PID controller parameters
        self.roll_pid = {
            'kp': -1.2,
            'ki': -0.2,
            'kd': -0.0,
            'target': 0.0,  # Target roll angle (radians)
            'output_limits': (-np.deg2rad(10), np.deg2rad(10)),  # Output limits (radians)
            'integral': 0,
            'prev_error': 0
        }
        
        # Heading angle PID controller parameters
        self.heading_pid = {
            'kp': 2.8,     # Proportional gain, initial value needs adjustment
            'ki': 0.01,
            'kd': 2.2,    # Derivative gain, initial value needs adjustment
            'prev_prev_error': 0.0,
            'output_limits': (-np.deg2rad(20), np.deg2rad(20)),  # Output limits (radians)
            'integral': 0,
            'prev_error': 0
        }
        
        self.V_trim = V_trim
        
        self._pure_roll = pure_roll
        
    def update_pid_parameters(self, kp_roll, ki_roll, kd_roll, kp_beta, ki_beta, kd_beta):
        self.roll_pid['kp'] = kp_roll
        self.roll_pid['ki'] = ki_roll
        self.roll_pid['kd'] = kd_roll
        
    def compute_roll_command(self, current_yaw, target_heading, current_beta, dt):
        """
        Calculate desired roll angle using PID control law
        
        :param current_yaw: Current yaw angle (radians)
        :param target_yaw: Target yaw angle (radians)
        :param current_p: Current roll rate (radians/second)
        :param dt: Time step (seconds)
        :return: Desired roll angle (radians)
        """
        # Calculate heading error
        current_heading = current_yaw - current_beta
        heading_error = self.normalize_angle(target_heading - current_heading)
        
        # Heading PID to calculate desired roll angle
        p_term = self.heading_pid['kp'] * heading_error
        self.heading_pid['integral'] += heading_error * dt
        self.heading_pid['integral'] = np.clip(self.heading_pid['integral'], -np.deg2rad(30), np.deg2rad(30))
        # Integral limit (prevent integral windup)
        i_term = self.heading_pid['ki'] * self.heading_pid['integral']
        d_term = self.heading_pid['kd'] * (heading_error - self.heading_pid['prev_error']) / dt if dt > 0 else 0
        self.heading_pid['prev_error'] = heading_error
        
        heading_pid_control = p_term + i_term + d_term
        roll_command = np.clip(heading_pid_control, self.heading_pid['output_limits'][0], self.heading_pid['output_limits'][1]) 
        
        return roll_command
    
    def normalize_angle(self, angle):
        """
        Normalize angle to the range [-pi, pi]
        
        :param angle: Input angle (radians)
        :return: Normalized angle (radians)
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    def compute_control(self, current_roll, current_yaw, current_beta, current_p, dt, 
                       target_roll=None, target_heading=None):
        if target_heading is not None:
            desired_roll = self.compute_roll_command(current_yaw, target_heading, current_beta, dt)      
        else:
            # If target yaw is not provided, use target roll angle
            desired_roll = target_roll if target_roll is not None else self.roll_pid['target']
        
        # Update target value for roll PID
        if self._pure_roll == 0:  # Use yaw control
            #print('Yaw control mode:')
            self.roll_pid['target'] = desired_roll
        else:   # Pure roll command
            self.roll_pid['target'] = np.deg2rad(self._pure_roll)

        # Calculate roll angle PID control (inner loop)
        roll_error = self.roll_pid['target'] - current_roll
        p_term = self.roll_pid['kp'] * roll_error
        self.roll_pid['integral'] += roll_error * dt
        self.roll_pid['integral'] = np.clip(self.roll_pid['integral'], -np.deg2rad(30), np.deg2rad(30))
        # Integral limit (prevent integral windup)
        i_term = self.roll_pid['ki'] * self.roll_pid['integral']
        d_term = self.roll_pid['kd'] * (roll_error - self.roll_pid['prev_error']) / dt if dt > 0 else 0
        self.roll_pid['prev_error'] = roll_error
        
        roll_pid_control = p_term + i_term + d_term
        roll_pid_control = np.clip(roll_pid_control, self.roll_pid['output_limits'][0], self.roll_pid['output_limits'][1])

        combined_control = roll_pid_control
        
        combined_control = np.clip(combined_control, -np.deg2rad(10), np.deg2rad(10))  # Final output limit
        
        return combined_control, self.roll_pid['target']

def control():
    wind_field = None  # Wind field instance
    model = FixedWingDynamics(wind_field) 
    trim_calc = TrimCalculator(model, wind_field)  
    u_rel_trim, w_rel_trim, diff_trim, diff_trim, T_trim = trim_calc.find_trim()
    V_trim = np.sqrt(u_rel_trim**2 + w_rel_trim**2)
    CY_beta, m, Cl_beta, Cn_beta = model.CY_beta, model.m, model.Cl_beta, model.Cn_beta
    qSb = 0.5 * model.rho * V_trim**2 * model.S * model.b
    qS = 0.5 * model.rho * V_trim**2 * model.S
    Ixx, Izz = model.Ixx, model.Izz
    
    # Simulation parameters - fixed duration of 20 seconds
    dt = 0.01  # Time step (seconds)
    total_time = 20.0  # Fixed total simulation time of 20 seconds
    steps = int(total_time / dt)  # Calculate number of steps
    
    # Initial states
    phi_deg = 0.0  # Initial roll angle (degrees)
    p_deg_s = 0.0    # Initial roll rate (degrees/second)
    delta_e_diff_deg = 0  # Initial elevator deflection (degrees)
    
    # Linearized model
    A_lat, B_lat = trim_calc.linearize_trim()

    C_lat = np.array([
        [0, 1, 0, 0, 0],  # First output is p
        [0, 0, 1, 0, 0],  # Second output is r
        [0, 0, 0, 1, 0],  # Third output is phi
        [0, 0, 0, 0, 1]   # Fourth output is psi
    ])
    D_lat = np.zeros((C_lat.shape[0], B_lat.shape[1]))  
    
    # Desired roll angle (step input, unit: degrees)
    target_phi_deg = 0.0  # Desired roll angle (degrees)
    target_phi = np.deg2rad(target_phi_deg)  # Convert to radians
    
    # Desired yaw angle (step input, unit: degrees)
    target_heading_deg = 1.0  # Desired heading angle (degrees)
    target_heading = np.deg2rad(target_heading_deg)  # Convert to radians
    
    # Initialize coordinated turn controller
    coordinated_controller = CoordinatedTurnController(V_trim=V_trim, pure_roll=0.0)
    
    #open_loop_tf = utils_ry.plot_pid_controlled_bode(trim_calc, coordinated_controller, target_phi_deg=target_phi_deg, dt=dt)
    
    # Data storage
    phi_history_deg = []         # Store roll angle history (degrees)
    p_history_deg = []           # Store roll rate history (degrees/second)
    delta_e_diff_history_deg = []  # Store elevator deflection history (degrees)
    yaw_history_deg = []         # Store yaw angle history (degrees)
    wind_v_history = []
    
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
    
    fig = plt.figure(figsize=(8, 6))
    gs_left = fig.add_gridspec(2, 1)  
    
    # Create 3 subplots
    ax1 = fig.add_subplot(gs_left[0, 0])
    ax3 = fig.add_subplot(gs_left[1, 0])
    
    # Initialize plots
    time_axis = np.arange(0, total_time, dt)
    # Define SCI style color scheme
    color_blue = '#1f77b4'     # Primary blue
    color_red = '#d62728'      # Primary red
    color_green = '#2ca02c'    # Primary green
    color_orange = '#ff7f0e'   # Secondary orange
    color_gray = '#7f7f7f'     # Gray
    
    line_heading, = ax1.plot([], [], color=color_blue, label='Heading (deg)')
    line_yaw, = ax1.plot([], [], color=color_green, linestyle='--', label='Yaw psi (deg)', alpha=0.5)
    line_target_heading, = ax1.plot([], [], color=color_red, linestyle='--', label='Target Heading (deg)')
    #line_target_yaw, = ax1.plot([], [], color=color_orange, linestyle='--', label='Target Yaw psi (deg)')
    ax1.set_ylabel('Heading (deg)')
    ax1.set_ylim(0, 1.2)  # Set appropriate y-axis range
    #ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))  # Set number of y-axis ticks
    ax1.legend(loc='lower right')
    ax1.grid()
    
    line_phi, = ax3.plot([], [], color=color_blue, label='Roll phi (deg)')
    line_target_phi, = ax3.plot([], [], color=color_red, linestyle='--', label='Target Roll phi (deg)')
    ax3.set_ylabel('Roll (deg)')
    ax3.set_ylim(-0.5, 3.6)  # Set appropriate y-axis range
    ax3.set_xlabel('Time (s)')
    ax3.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax3.grid()
    
    # Optimize borders
    for ax in [ax1, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
    
    # Current simulation state
    current_step = 0
    phi = np.deg2rad(phi_deg)          # Current roll angle (radians)
    p = np.deg2rad(p_deg_s)            # Current roll rate (radians/second)
    delta_e_diff = np.deg2rad(delta_e_diff_deg)  # Current elevator deflection (radians)
    x = np.array([
        0.0,  # beta (sideslip angle)
        p,   # p (roll rate)
        0.0,   # r (yaw rate) - initially 0
        phi,   # phi (roll angle)
        0.0    # psi (yaw angle) - initially 0
    ])
    
    # Update function - called whenever slider values change
    def update_simulation(val=None):
        nonlocal current_step, phi, p, delta_e_diff, x, phi_history_deg, p_history_deg, delta_e_diff_history_deg, yaw_history_deg
        nonlocal target_heading
        
        # Reset simulation state
        x = np.array([
            0.0,  # beta
            p,   # p
            0.0,   # r
            phi,   # phi
            0.0    # psi
        ])
        
        # Clear historical data
        phi_history_deg = []         # Store roll angle history (degrees)
        delta_e_diff_history_deg = []  # Store elevator deflection history (degrees)
        yaw_history_deg = []         # Store yaw angle history (degrees)
        heading_history_deg = []
        beta_history_deg = []
        beta_disturbance_history = [] # Store disturbance values
        target_roll_history = []
        target_heading_history = []
        
        # Define delay steps (50ms delay, dt=0.01s)
        delay_steps = int(0.05 / dt)
        target_diff_buffer = [np.deg2rad(0)] * delay_steps  # Initial value set to 0 rad/s
        
        # Wind speed parameters
        np.random.seed(42)
        h_ref = 100                # Reference altitude (m)
        mean_wind_speed = 10.41    # Mean wind speed at reference altitude (m/s)

        # Dryden model parameters
        L_v = 200                 # Lateral turbulence scale (m)
        zi_v = np.zeros(2)

        # Pre-generate random noise sequence
        white_noise_v = np.random.randn(steps)  # Lateral wind noise

        # Design Dryden transfer function (continuous time)
        omega_v = 2 * np.pi * mean_wind_speed / L_v

        # Lateral gust transfer function
        num_v = [np.sqrt(3 * omega_v), np.sqrt(3 * omega_v**3)]
        den_v = [1, 2 * omega_v, omega_v**2]

        # Convert to discrete transfer function
        dt_disc = dt
        num_d_v, den_d_v, _ = signal.cont2discrete((num_v, den_v), dt_disc)
        num_d_v = num_d_v.flatten()
        
        wind_v_sequence, zi_v = signal.lfilter(num_d_v, den_d_v, white_noise_v, zi=zi_v)
        
        # Re-run entire simulation (fixed 20 seconds)
        for k in range(steps):
            '''
            if k>0 and k%500==0 and k<1500:
                target_heading -= np.deg2rad(5.0)
            elif k==1500:
                target_heading += np.deg2rad(5.0)
            '''
            '''
            if k==0:
                target_heading = np.deg2rad(5.0)
            elif k==300:
                target_heading -= np.deg2rad(5.0)
            elif k==800:
                target_heading += np.deg2rad(3.0)
            elif k==1300:
                target_heading += np.deg2rad(3.0)
            elif k==1700:
                target_heading -= np.deg2rad(3.0)
            '''
            '''
            if k==0:
                target_heading = np.deg2rad(-5.0)
            elif k==500:
                target_heading += np.deg2rad(5.0)
            elif k==1000:
                target_heading -= np.deg2rad(2.0)
            elif k==1300:
                target_heading += np.deg2rad(5.0)
            elif k==1700:
                target_heading -= np.deg2rad(0.0)
            '''
            p_curr = x[1]
            phi_curr = x[3]
            psi_curr = x[4]  # Yaw angle is the 4th element of the state vector
            beta_curr = x[0]  # Sideslip angle
            
            # Beta random disturbance simulating wind speed changes
            sigma_v = 1.0 * mean_wind_speed * (h_ref / 10) ** (-0.25)  # Lateral turbulence intensity
            
            # Scale to actual wind speed disturbance
            wind_v = sigma_v * wind_v_sequence[k]
            wind_v = 0.0
            
            Dw = np.array([
                wind_v * CY_beta * qS * np.cos(psi_curr) / (m * V_trim**2),  # beta_with_wind
                #CY_beta * qS * (wind_v*np.cos(psi_curr) / V_trim - beta_curr) / (m * V_trim**2),  # beta_with_wind
                wind_v * qSb * Cl_beta * np.cos(psi_curr)/ (Ixx * V_trim),   # p_with_wind
                wind_v * qSb * Cn_beta * np.cos(psi_curr) / (Izz * V_trim),   # r_with_wind
                0.0,   # phi
                0.0    # psi
            ])
            
            beta_history_deg.append(np.rad2deg(beta_curr))
            #beta_curr += np.arcsin(wind_v*np.cos(psi_curr) / V_trim)
            #x[0] = beta_curr
            #print('Beta after wind disturbance:', np.rad2deg(beta_curr))
            beta_disturbance_history.append(np.rad2deg(beta_curr + Dw[0]))
            
            # Convert states to degrees
            p_deg = np.rad2deg(p_curr)
            phi_deg_curr = np.rad2deg(phi_curr)
            psi_deg_curr = np.rad2deg(psi_curr)
            
            # Calculate control output
            delta_e_diff, target_roll = coordinated_controller.compute_control(
                current_roll=phi_curr,
                current_yaw=psi_curr,
                current_beta=beta_curr,
                current_p=p_curr,
                dt=dt,
                target_roll=target_phi,
                target_heading=target_heading
            )
            
            # Delay module: update buffer
            target_diff_buffer.pop(0) 
            target_diff_buffer.append(delta_e_diff)  # Add latest value
            
            # Convert delta_e_diff to degrees if needed
            delta_e_diff_deg = np.rad2deg(delta_e_diff)
            delayed_e_diff = target_diff_buffer[0]
            
            # Control input vector [delta_e_diff, T]
            u = np.array([
                delayed_e_diff,          # δe_diff 
                0.0   # Thrust remains unchanged
            ])

            # Calculate state derivative dx = A @ x + B @ u
            x_dot = A_lat @ x + B_lat @ u + Dw 
            
            # Update state (Euler integration)
            x = x + x_dot * dt
            
            # Get updated state
            #y = C_lat @ x
            p_updated = x[1]
            phi_updated = x[3]
            psi_updated = x[4]     # Yaw angle
            beta_updated = x[0]
            '''
            if k == 500:
                target_heading = np.deg2rad(10)
            elif k == 800:
                target_heading = np.deg2rad(15)
            elif k == 1500:
                target_heading = np.deg2rad(5)
            else:
                pass
            '''
            # Convert states to degrees
            heading_deg = np.rad2deg(psi_updated) - np.rad2deg(beta_updated)
            phi_deg_curr = np.rad2deg(phi_updated)
            p_deg = np.rad2deg(p_updated)
            psi_deg = np.rad2deg(psi_updated)
            target_roll_deg = np.rad2deg(target_roll)
            
            # Store data
            phi_history_deg.append(phi_deg_curr)
            p_history_deg.append(p_deg)
            delta_e_diff_history_deg.append(delta_e_diff_deg)
            heading_history_deg.append(heading_deg)
            yaw_history_deg.append(psi_deg)
            target_heading_history.append(np.rad2deg(target_heading))
            target_roll_history.append(target_roll_deg)
            
            # Update current step
            current_step += 1
        
        data = {
            'heading_deg': heading_history_deg,
            'beta_deg': beta_history_deg,
            'phi_deg': phi_history_deg,
            'psi_deg': yaw_history_deg,
            'control_deg': delta_e_diff_history_deg,
            'target_heading_deg': target_heading_deg
            }
        '''
        # Save as CSV (easy for Excel viewing)
        df = pd.DataFrame(data)
        csv_path = f"comparison/pid_data_03.csv"
        df.to_csv(csv_path, index_label="step")
        print(f"CSV saved to {csv_path}")

        # Save as NPZ (efficient binary format, suitable for Python reading)
        npz_path = f"comparison/pid_data_03.npz"
        np.savez(npz_path, **data)
        print(f"NPZ saved to {npz_path}")
        
        window_size = 100  # 2s
        heading_sw_var = utils_ry.sliding_window_variance(heading_history_deg, window_size)
        beta_sw_var = utils_ry.sliding_window_variance(beta_history_deg, window_size)
        phi_sw_var = utils_ry.sliding_window_variance(phi_history_deg, window_size)
        p_sw_var = utils_ry.sliding_window_variance(p_history_deg, window_size)
        psi_sw_var = utils_ry.sliding_window_variance(yaw_history_deg, window_size)
        control_sw_var = utils_ry.sliding_window_variance(delta_e_diff_history_deg, window_size)
        print('Heading variance:', heading_sw_var, 'Sideslip angle variance:', beta_sw_var)
        #print('Roll rate variance:', p_sw_var)
        print('Roll variance:', phi_sw_var, 'Yaw variance:', psi_sw_var, 'Differential control variance:', control_sw_var)
        
        # Calculate RMS of yaw angle
        yaw_rms = np.sqrt(np.mean(np.array(yaw_history_deg)**2))
        print(f'Yaw angle RMS (deg): {yaw_rms:.4f}')
    
        # Calculate RMS of roll angle
        # Note: According to your code, phi_history_deg stores the actual roll angle
        roll_rms = np.sqrt(np.mean(np.array(phi_history_deg)**2))
        print(f'Roll angle RMS (deg): {roll_rms:.4f}')
        
        e_diff_rms = np.sqrt(np.mean(np.array(delta_e_diff_history_deg)**2))
        print(f'Differential control RMS (deg): {e_diff_rms:.4f}')
        '''
        # Update the plot
        line_yaw.set_data(time_axis[:len(yaw_history_deg)], yaw_history_deg)
        #line_target_yaw.set_data(time_axis[:len(yaw_history_deg)], [target_heading_deg] * len(yaw_history_deg))
        line_heading.set_data(time_axis[:len(heading_history_deg)], heading_history_deg)
        line_target_heading.set_data(time_axis[:len(heading_history_deg)], target_heading_history)
        
        line_phi.set_data(time_axis[:len(phi_history_deg)], phi_history_deg)
        line_target_phi.set_data(time_axis[:len(phi_history_deg)], target_roll_history)
        
        # Adjust axis limits
        ax1.relim()
        ax1.autoscale_view()
        ax3.relim()
        ax3.autoscale_view()
        
        # Calculate overshoot and rise time
        #overshoot_percent = utils_ry.calculate_overshoot(phi_history_deg, np.rad2deg(coordinated_controller.roll_pid['target']))
        #rise_time_sec = utils_ry.calculate_rise_time(phi_history_deg, np.rad2deg(coordinated_controller.roll_pid['target']), dt, threshold=0.9)
        overshoot_percent = utils_ry.calculate_overshoot(yaw_history_deg, target_heading_deg)
        rise_time_sec = utils_ry.calculate_rise_time(yaw_history_deg, target_heading_deg, dt, threshold=0.9)
        errors = [(target_heading - np.deg2rad(heading)) for heading in yaw_history_deg]
        #errors = [np.deg2rad(np.rad2deg(coordinated_controller.roll_pid['target']) - roll) for roll in phi_history_deg]
        settling_time = utils_ry.calculate_settling_time(errors, target_heading)
        #settling_time = utils_ry.calculate_settling_time(errors)
        steady_error = utils_ry.calculate_steady_state_error(yaw_history_deg, target_heading_deg, 750)
        #steady_error = utils_ry.calculate_steady_state_error(phi_history_deg, np.rad2deg(coordinated_controller.roll_pid['target']), settling_time)

        for artist in ax1.texts:
            artist.remove()
        
        if overshoot_percent is not None and rise_time_sec is not None:
            text_str = f'Overshoot: {overshoot_percent:.2f}%\nRise time: {rise_time_sec:.2f}s\nSettling time: {settling_time:.2f}s\nSteady state error: {steady_error:.2f}%'
            #text_str = f'Overshoot: {overshoot_percent:.2f}%\nRise time: {rise_time_sec:.2f}s\nSettling time: {settling_time:.2f}s\nSteady state error: {steady_error:.2f}%'
        elif overshoot_percent is not None:
            text_str = f'Overshoot: {overshoot_percent:.2f}%\nRise time: -1'
        elif rise_time_sec is not None:
            text_str = f'Overshoot: -1\nRise time: {rise_time_sec:.2f}s'
        else:
            text_str = 'Overshoot and Rise time: -1'
        
        #ax3.text(0.15, 0.55, text_str, transform=ax3.transAxes,
                 #verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        ax1.text(0.17, 0.69, text_str, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        #ax3.axhline(y=np.rad2deg(coordinated_controller.roll_pid['target'])*0.95, xmin=0.0, xmax=20, color=color_gray, linestyle='--')
        #ax3.axhline(y=np.rad2deg(coordinated_controller.roll_pid['target'])*1.05, xmin=0.0, xmax=20, color=color_gray, linestyle='--')
        ax1.axhline(y=np.rad2deg(target_heading*0.95), xmin=0.0, xmax=20, color=color_gray, linestyle='--')
        ax1.axhline(y=np.rad2deg(target_heading*1.05), xmin=0.0, xmax=20, color=color_gray, linestyle='--')
        ax1.plot([0.0, 0.0], [0.0, 1.0], color=color_red, linestyle='--')
        
        fig.canvas.draw_idle()
        
    # Run the simulation once initially
    update_simulation(None)
    
    # Save high resolution image (600dpi)
    #save_path = 'heading-roll_pid/17_Step response of roll angle control loop.png'
    save_path = 'heading-roll_pid/19_Step response of heading control loop.png'
    fig.savefig(save_path, 
                dpi=600,               # Manually set higher dpi
                bbox_inches='tight',   # Crop extra white space
                format='png')          # Supported formats: 'png', 'pdf', 'svg', 'eps', etc.
    
    #plt.show()

if __name__ == "__main__":
    control()