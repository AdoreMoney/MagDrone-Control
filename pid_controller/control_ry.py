import numpy as np
import time
from trim_hrz import FixedWingDynamics, TrimCalculator, WindField
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.ticker import MaxNLocator
import control as ctrl
import pandas as pd
import utils.utils_ry as utils_ry
from scipy import signal

class CoordinatedTurnController:
    def __init__(self, V_trim, pure_roll):
        # Roll angle PID controller parameters
        self.roll_pid = {
            'kp': -1.2,
            'ki': -0.2,
            'kd': -0.0,
            'target': 0.0,
            'output_limits': (-np.deg2rad(10), np.deg2rad(10)),
            'integral': 0,
            'prev_error': 0
        }
        
        # Heading angle PID controller parameters
        self.heading_pid = {
            'kp': 2.8,
            'ki': 0.01,
            'kd': 2.2,
            'lead_alpha': 0.1,
            'lead_Td': 1.9,
            'prev_lead_out': 0,
            'prev_prev_error': 0.0,
            'output_limits': (-np.deg2rad(20), np.deg2rad(20)),
            'integral': 0,
            'prev_error': 0
        }
        
        # Guidance parameters
        self.V_trim = V_trim
        self._pure_roll = pure_roll
        
    def update_pid_parameters(self, kp_roll, ki_roll, kd_roll, kp_beta, ki_beta, kd_beta):
        self.roll_pid['kp'] = kp_roll
        self.roll_pid['ki'] = ki_roll
        self.roll_pid['kd'] = kd_roll
        
        self.beta_pid['kp'] = kp_beta
        self.beta_pid['ki'] = ki_beta
        self.beta_pid['kd'] = kd_beta
        
    def compute_roll_command(self, current_yaw, target_heading, current_beta, dt):
        current_heading = current_yaw - current_beta
        heading_error = self.normalize_angle(target_heading - current_heading)
        
        # Heading PID calculation for desired roll angle
        p_term = self.heading_pid['kp'] * heading_error
        self.heading_pid['integral'] += heading_error * dt
        self.heading_pid['integral'] = np.clip(self.heading_pid['integral'], -np.deg2rad(30), np.deg2rad(30))
        i_term = self.heading_pid['ki'] * self.heading_pid['integral']
        d_term = self.heading_pid['kd'] * (heading_error - self.heading_pid['prev_error']) / dt if dt > 0 else 0
        self.heading_pid['prev_error'] = heading_error
        
        heading_pid_control = p_term + i_term + d_term

        # Lead compensation (discrete implementation)
        lead_out = (
            (self.heading_pid['lead_Td'] * (heading_error - self.heading_pid['prev_error']) / dt + heading_error) -
            self.heading_pid['lead_alpha'] * self.heading_pid['lead_Td'] * 
            (heading_error - 2*self.heading_pid['prev_error'] + self.heading_pid['prev_prev_error']) / dt
            ) / (1 + self.heading_pid['lead_alpha'] * self.heading_pid['lead_Td'] / dt)
        
        self.heading_pid['prev_prev_error'] = self.heading_pid['prev_error']
        self.heading_pid['prev_lead_out'] = lead_out
        
        roll_command = np.clip(heading_pid_control, self.heading_pid['output_limits'][0], self.heading_pid['output_limits'][1])
        
        return roll_command
    
    def normalize_angle(self, angle):
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
            desired_roll = self.compute_roll_command(current_yaw, target_heading, current_beta, dt)
        
        # Update roll PID target
        if self._pure_roll == 0:
            self.roll_pid['target'] = desired_roll
        else:
            self.roll_pid['target'] = np.deg2rad(self._pure_roll)
        
        # Calculate roll PID control (inner loop)
        roll_error = self.roll_pid['target'] - current_roll
        p_term = self.roll_pid['kp'] * roll_error
        self.roll_pid['integral'] += roll_error * dt
        self.roll_pid['integral'] = np.clip(self.roll_pid['integral'], -np.deg2rad(30), np.deg2rad(30))
        i_term = self.roll_pid['ki'] * self.roll_pid['integral']
        d_term = self.roll_pid['kd'] * (roll_error - self.roll_pid['prev_error']) / dt if dt > 0 else 0
        self.roll_pid['prev_error'] = roll_error
        
        roll_pid_control = p_term + i_term + d_term
        roll_pid_control = np.clip(roll_pid_control, self.roll_pid['output_limits'][0], self.roll_pid['output_limits'][1])
        
        combined_control = roll_pid_control
        combined_control = np.clip(combined_control, -np.deg2rad(10), np.deg2rad(10))
        
        return combined_control, self.roll_pid['target']

def control():
    wind_field = WindField()
    model = FixedWingDynamics(wind_field) 
    trim_calc = TrimCalculator(model, wind_field)  
    u_rel_trim, w_rel_trim, diff_trim, diff_trim, T_trim = trim_calc.find_trim()
    V_trim = np.sqrt(u_rel_trim**2 + w_rel_trim**2)
    CY_beta, m, Cl_beta, Cn_beta = model.CY_beta, model.m, model.Cl_beta, model.Cn_beta
    qSb = 0.5 * model.rho * V_trim**2 * model.S * model.b
    qS = 0.5 * model.rho * V_trim**2 * model.S
    Ixx, Izz = model.Ixx, model.Izz
    
    # Simulation parameters
    dt = 0.01
    total_time = 20.0
    steps = int(total_time / dt)
    
    # Initial states
    phi_deg = 0.0
    p_deg_s = 0.0
    delta_e_diff_deg = 0
    
    # Linearized model
    A_long, B_long = trim_calc.linearize_trim()

    # Output matrix C and direct transmission D
    C_long = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ])
    D_long = np.zeros((C_long.shape[0], B_long.shape[1]))
    sys = ctrl.ss(A_long, B_long, C_long, D_long)
    
    # Desired angles
    target_phi_deg = 0.0
    target_phi = np.deg2rad(target_phi_deg)
    
    target_heading_deg = 10.0
    target_heading = np.deg2rad(target_heading_deg)
    
    # Initialize controller
    coordinated_controller = CoordinatedTurnController(V_trim=V_trim, pure_roll=0.0)
    
    # Plot Bode diagram
    open_loop_tf = utils_ry.plot_pid_controlled_bode(trim_calc, coordinated_controller, target_phi_deg=target_phi_deg, dt=dt)
    
    # Data storage
    phi_history_deg = []
    p_history_deg = []
    delta_e_diff_history_deg = []
    yaw_history_deg = []
    wind_v_history = []
    
    # SCI style settings
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.6
    
    # Create figure
    fig = plt.figure(figsize=(8, 6))
    gs_left = fig.add_gridspec(2, 1, left=0.15, right=0.90, hspace=0.2)
    
    # Create subplots
    ax1 = fig.add_subplot(gs_left[0, 0])
    ax3 = fig.add_subplot(gs_left[1, 0])
    
    # Initialize plotting
    time_axis = np.arange(0, total_time, dt)
    color_blue = '#1f77b4'
    color_red = '#d62728'
    color_green = '#2ca02c'
    color_orange = '#ff7f0e'
    color_gray = '#7f7f7f'
    
    line_heading, = ax1.plot([], [], color=color_blue, label='Heading psi-beta (deg)')
    line_yaw, = ax1.plot([], [], color=color_green, linestyle='--', label='Yaw psi (deg)', alpha=0.5)
    line_target_heading, = ax1.plot([], [], color=color_red, linestyle='--', label='Target Heading psi-beta (deg)')
    ax1.set_ylabel('Heading')
    ax1.set_ylim(-30.0, 30.0)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax1.legend(loc='lower right')
    ax1.grid()
    
    line_phi, = ax3.plot([], [], color=color_blue, label='Roll phi (deg)')
    line_target_phi, = ax3.plot([], [], color=color_red, linestyle='--', label='Target Roll phi (deg)', alpha=0.3)
    ax3.set_ylabel('Roll')
    ax3.set_ylim(-30.0, 30.0)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax3.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax3.grid()
    
    # Optimize borders
    for ax in [ax1, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
    
    # Current simulation state
    current_step = 0
    phi = np.deg2rad(phi_deg)
    p = np.deg2rad(p_deg_s)
    delta_e_diff = np.deg2rad(delta_e_diff_deg)
    x = np.array([
        0.0,
        p,
        0.0,
        phi,
        0.0
    ])
    
    def update_simulation(val=None):
        nonlocal current_step, phi, p, delta_e_diff, x, phi_history_deg, p_history_deg, delta_e_diff_history_deg, yaw_history_deg
        nonlocal target_heading
        
        # Reset simulation state
        x = np.array([
            0.0,
            p,
            0.0,
            phi,
            0.0
        ])
        
        # Clear history data
        phi_history_deg = []
        delta_e_diff_history_deg = []
        yaw_history_deg = []
        heading_history_deg = []
        beta_history_deg = []
        beta_disturbance_history = []
        target_roll_history = []
        target_heading_history = []
        X_traj, Y_traj = [], []
        target_X_traj, target_Y_traj = [], []
        
        # Delay settings
        delay_steps = int(0.05 / dt)
        target_diff_buffer = [np.deg2rad(0)] * delay_steps
        
        # Wind parameters
        np.random.seed(42)
        h_ref = 100
        mean_wind_speed = 10.41
        V_real = V_trim - mean_wind_speed
        X_curr, Y_curr = 0.0, 0.0

        # Dryden model parameters
        L_v = 200
        zi_v = np.zeros(2)

        # Pre-generate noise
        white_noise_v = np.random.randn(steps)

        # Dryden transfer function
        omega_v = 2 * np.pi * mean_wind_speed / L_v
        num_v = [np.sqrt(3 * omega_v), np.sqrt(3 * omega_v**3)]
        den_v = [1, 2 * omega_v, omega_v**2]

        # Convert to discrete
        dt_disc = dt
        num_d_v, den_d_v, _ = signal.cont2discrete((num_v, den_v), dt_disc)
        num_d_v = num_d_v.flatten()
        
        wind_v_sequence, zi_v = signal.lfilter(num_d_v, den_d_v, white_noise_v, zi=zi_v)
        
        # Run simulation
        for k in range(steps):
            X_des, Y_des = 300, 200
            target_heading = np.arctan2(Y_des - Y_curr, X_des - X_curr)
            
            # Get current states
            p_curr = x[1]
            phi_curr = x[3]
            psi_curr = x[4]
            beta_curr = x[0]
            
            # Wind disturbance
            sigma_v = 1.0 * mean_wind_speed * (h_ref / 10) ** (-0.25)
            wind_v = sigma_v * wind_v_sequence[k]
            wind_v = 0.0
            
            # Disturbance vector
            Dw = np.array([
                wind_v * CY_beta*30 * qS * np.cos(psi_curr) / (m * V_trim**2),
                wind_v * qSb * Cl_beta * np.cos(psi_curr)/ (Ixx * V_trim),
                wind_v * qSb * Cn_beta * np.cos(psi_curr) / (Izz * V_trim),
                0.0,
                0.0
            ])
            
            beta_history_deg.append(np.rad2deg(beta_curr))
            beta_disturbance_history.append(np.rad2deg(beta_curr + Dw[0]))
            
            # Convert to degrees
            p_deg = np.rad2deg(p_curr)
            phi_deg_curr = np.rad2deg(phi_curr)
            psi_deg_curr = np.rad2deg(psi_curr)
            
            # Calculate control
            delta_e_diff, target_roll = coordinated_controller.compute_control(
                current_roll=phi_curr,
                current_yaw=psi_curr,
                current_beta=beta_curr,
                current_p=p_curr,
                dt=dt,
                target_roll=target_phi,
                target_heading=target_heading
            )
            
            # Update delay buffer
            target_diff_buffer.pop(0) 
            target_diff_buffer.append(delta_e_diff)
            
            # Convert to degrees
            delta_e_diff_deg = np.rad2deg(delta_e_diff)
            delayed_e_diff = target_diff_buffer[0]
            
            # Control input
            u = np.array([
                delayed_e_diff,
                0.0
            ])
            
            # Update state
            x_dot = A_long @ x + B_long @ u + Dw 
            x = x + x_dot * dt
            X_curr += V_real * np.cos(psi_curr - beta_curr) * dt
            Y_curr += V_real * np.sin(psi_curr - beta_curr) * dt
            
            # Get updated states
            p_updated = x[1]
            phi_updated = x[3]
            psi_updated = x[4]
            beta_updated = x[0]
            
            # Convert to degrees
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
            X_traj.append(X_curr)
            Y_traj.append(Y_curr)
            target_X_traj.append(X_des)
            target_Y_traj.append(Y_des)
            
            current_step += 1
        
        # Plot trajectory
        plt.figure(figsize=(12, 6))
        plt.plot(X_traj, Y_traj, 'b-', label='real_trajectory')
        plt.plot(target_X_traj, target_Y_traj, 'r--', label='target_trajectory')
        plt.xlabel('X (m)'); plt.ylabel('Y (m)')
        plt.axis('equal'); plt.legend()
        
        # Data saving
        data = {
            'heading_deg': heading_history_deg,
            'beta_deg': beta_history_deg,
            'phi_deg': phi_history_deg,
            'psi_deg': yaw_history_deg,
            'control_deg': delta_e_diff_history_deg,
            'target_heading_deg': target_heading_deg
            }
    
        df = pd.DataFrame(data)
        csv_path = f"comparison/pid_data_03.csv"
        npz_path = f"comparison/pid_data_03.npz"
        
        # Calculate sliding window variance
        window_size = 100
        heading_sw_var = utils_ry.sliding_window_variance(heading_history_deg, window_size)
        beta_sw_var = utils_ry.sliding_window_variance(beta_history_deg, window_size)
        phi_sw_var = utils_ry.sliding_window_variance(phi_history_deg, window_size)
        p_sw_var = utils_ry.sliding_window_variance(p_history_deg, window_size)
        psi_sw_var = utils_ry.sliding_window_variance(yaw_history_deg, window_size)
        control_sw_var = utils_ry.sliding_window_variance(delta_e_diff_history_deg, window_size)
        print('Heading Variance:', heading_sw_var, 'Sideslip Angle Variance:', beta_sw_var)
        print('Roll Variance:', phi_sw_var, 'Yaw Variance:', psi_sw_var, 'Differential Control Variance:', control_sw_var)
        
        # Calculate RMS
        yaw_rms = np.sqrt(np.mean(np.array(yaw_history_deg)**2))
        print(f'Yaw RMS (deg): {yaw_rms:.4f}')
    
        roll_rms = np.sqrt(np.mean(np.array(phi_history_deg)**2))
        print(f'Roll RMS (deg): {roll_rms:.4f}')
        
        e_diff_rms = np.sqrt(np.mean(np.array(delta_e_diff_history_deg)**2))
        print(f'Differential Control RMS (deg): {e_diff_rms:.4f}')
        
        # Update plots
        line_yaw.set_data(time_axis[:len(yaw_history_deg)], yaw_history_deg)
        line_heading.set_data(time_axis[:len(heading_history_deg)], heading_history_deg)
        line_target_heading.set_data(time_axis[:len(heading_history_deg)], target_heading_history)
        
        line_phi.set_data(time_axis[:len(phi_history_deg)], phi_history_deg)
        line_target_phi.set_data(time_axis[:len(phi_history_deg)], target_roll_history)
        
        # Adjust axes
        ax1.relim()
        ax1.autoscale_view()
        ax3.relim()
        ax3.autoscale_view()
        fig.canvas.draw_idle()
    
    def update_pid_params(val):
        update_simulation(val)
        
    # Initial simulation run
    update_simulation(None)
    
    # Save high-resolution image
    save_path = 'heading_response_vari.png'
    fig.savefig(save_path, 
                dpi=600,
                bbox_inches='tight',
                format='png')
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    control()