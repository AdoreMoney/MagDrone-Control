import numpy as np
import time
from trim_vertical import FixedWingDynamics, TrimCalculator, WindField
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import control as ctrl
import utils.utils_h as utils_h

class hPID:
    def __init__(self, kp, ki, kd, target_h=0.0, output_theta_limits=(-np.deg2rad(10.0), np.deg2rad(10.0))):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_h = target_h
        self.output_theta_limits = output_theta_limits
        self._integral = 0
        self._prev_error = 0

    def __call__(self, h, dt):
        error = self.target_h - h

        p_term = self.kp * error

        self._integral += error * dt
        if self.output_theta_limits[0] is not None and self.output_theta_limits[1] is not None:
            self._integral = np.clip(self._integral, self.output_theta_limits[0], self.output_theta_limits[1])
        i_term = self.ki * self._integral

        if dt > 0:
            d_term = self.kd * (error - self._prev_error) / dt
        else:
            d_term = 0
        self._prev_error = error

        output_target_theta = p_term + i_term + d_term

        if self.output_theta_limits[0] is not None and self.output_theta_limits[1] is not None:
            output_target_theta = np.clip(output_target_theta, self.output_theta_limits[0], self.output_theta_limits[1])

        return output_target_theta
    
class thetaPID:
    def __init__(self, kp, ki, kd, target_theta=0.0, output_q_limits=(-np.deg2rad(20.0), np.deg2rad(20.0))):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_theta = target_theta
        self.output_q_limits = output_q_limits
        self._integral = 0
        self._prev_error = 0

    def __call__(self, theta, dt):
        error = self.target_theta - theta

        p_term = self.kp * error

        self._integral += error * dt
        if self.output_q_limits[0] is not None and self.output_q_limits[1] is not None:
            self._integral = np.clip(self._integral, self.output_q_limits[0], self.output_q_limits[1])
        i_term = self.ki * self._integral

        if dt > 0:
            d_term = self.kd * (error - self._prev_error) / dt
        else:
            d_term = 0
        self._prev_error = error

        output_target_q = p_term + i_term + d_term

        if self.output_q_limits[0] is not None and self.output_q_limits[1] is not None:
            output_target_q = np.clip(output_target_q, self.output_q_limits[0], self.output_q_limits[1])

        return output_target_q
    
class qPID:
    def __init__(self, kp, ki, kd, target_q=0.0, output_e_sync_limits=(-np.deg2rad(10.0), np.deg2rad(10.0))):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_q = target_q
        self.output_e_sync_limits = output_e_sync_limits
        self._integral = 0
        self._prev_error = 0

    def __call__(self, q, dt, target_q=None):
        if target_q is not None:
            self.target_q = target_q
        
        error = self.target_q - q

        p_term = self.kp * error

        self._integral += error * dt
        if self.output_e_sync_limits[0] is not None and self.output_e_sync_limits[1] is not None:
            self._integral = np.clip(self._integral, self.output_e_sync_limits[0], self.output_e_sync_limits[1])
        i_term = self.ki * self._integral

        if dt > 0:
            d_term = self.kd * (error - self._prev_error) / dt
        else:
            d_term = 0
        self._prev_error = error

        output_e_sync = p_term + i_term + d_term

        if self.output_e_sync_limits[0] is not None and self.output_e_sync_limits[1] is not None:
            output_e_sync = np.clip(output_e_sync, self.output_e_sync_limits[0], self.output_e_sync_limits[1])

        return output_e_sync

def control():
    wind_field = WindField()
    model = FixedWingDynamics(wind_field) 
    trim_calc = TrimCalculator(model, wind_field)  
    u_rel_trim, w_rel_trim, sync_trim, diff_trim, T_trim = trim_calc.find_trim()
    
    dt = 0.01
    total_time = 20.0
    steps = int(total_time / dt)
    
    theta_deg = 0.0
    q_deg_s = 0.0
    h_m = 0.0
    delta_e_sync_deg = 0
    
    A_long, B_long = trim_calc.linearize_trim()
    
    C_long = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ])
    D_long = np.zeros((C_long.shape[0], B_long.shape[1]))
    sys = ctrl.ss(A_long, B_long, C_long, D_long)
    
    initial_kp_h = 0.038
    initial_ki_h = 0.0
    initial_kd_h = 0.007
    target_h_m = 1.0
    
    initial_kp_theta = 1.571
    initial_ki_theta = 0.0
    initial_kd_theta = 0.344
    initial_kp_q = 12.08
    initial_ki_q = 0.85
    initial_kd_q = 0.0
    
    h_pid = hPID(kp=initial_kp_h, ki=initial_ki_h, kd=initial_kd_h, target_h=target_h_m)
    theta_pid = thetaPID(kp=initial_kp_theta, ki=initial_ki_theta, kd=initial_kd_theta)
    q_pid = qPID(kp=initial_kp_q, ki=initial_ki_q, kd=initial_kd_q)
    
    open_loop_tf = utils_h.plot_pid_controlled_bode(trim_calc, h_pid, theta_pid, q_pid, target_h=target_h_m, dt=dt)
    
    theta_history_deg = []
    target_q_history_deg = []
    q_history_deg = []
    delta_e_sync_history_deg = []
    h_history_m = []
    target_theta_history_deg = []
    
    fig = plt.figure(figsize=(16, 12))
    gs_left = fig.add_gridspec(3, 1, left=0.15, right=0.70, hspace=0.4)
    gs_right = fig.add_gridspec(3, 1, left=0.72, right=0.85, hspace=0.5)

    ax1 = fig.add_subplot(gs_left[0, 0])
    ax2 = fig.add_subplot(gs_left[1, 0])
    ax3 = fig.add_subplot(gs_left[2, 0])
    
    ax_kp_h = fig.add_subplot(gs_right[0, 0])
    ax_ki_h = fig.add_subplot(gs_right[1, 0])
    ax_kd_h = fig.add_subplot(gs_right[2, 0])

    slider_kp_h = Slider(ax_kp_h, 'kp_h', 0.0, 0.08, valinit=initial_kp_h, valstep=0.0001)
    slider_ki_h = Slider(ax_ki_h, 'ki_h', -0.8, 0.8, valinit=initial_ki_h, valstep=0.0001)
    slider_kd_h = Slider(ax_kd_h, 'kd_h', -0.05, 0.05, valinit=initial_kd_h, valstep=0.0001)

    for ax in [ax_kp_h, ax_ki_h, ax_kd_h]:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width, 0.02])

    time_axis = np.arange(0, total_time, dt)
    line_theta, = ax1.plot([], [], label='Theta (deg)')
    line_target_theta, = ax1.plot([], [], 'r--', label='Target Theta (deg)')
    ax1.set_ylabel('theta (deg)')
    ax1.set_ylim(-11.0, 11.0)
    ax1.legend(loc='upper right')
    ax1.grid()

    line_delta_e, = ax2.plot([], [], label='Delta e sync (deg)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('delta_e_sync (deg)')
    ax2.set_ylim(-11, 11)
    ax2.legend(loc='upper right')
    ax2.grid()

    line_h, = ax3.plot([], [], label='Height (m)')
    line_target_h, = ax3.plot([], [], 'r--', label='Target Height (m)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('H (m)')
    ax3.set_ylim(0, 2)
    ax3.legend(loc='upper right')
    ax3.grid()
    
    current_step = 0
    theta = np.deg2rad(theta_deg)
    q = np.deg2rad(q_deg_s)
    delta_e_sync = np.deg2rad(delta_e_sync_deg)
    h = h_m
    x = np.array([
        0.0,
        0.0,
        q,
        theta,
        h
    ])
    alpha_val = 0.3
    T_val = 0.1
    
    def update_simulation(val=None):
        nonlocal current_step, theta, q, delta_e_sync, h, x, theta_history_deg, target_q_history_deg, q_history_deg, delta_e_sync_history_deg, h_history_m, target_theta_history_deg
        
        kp_h = slider_kp_h.val
        ki_h = slider_ki_h.val
        kd_h = slider_kd_h.val
        
        h_pid.kp = kp_h
        h_pid.ki = ki_h
        h_pid.kd = kd_h
        
        current_step = 0
        theta = np.deg2rad(theta_deg)
        q = np.deg2rad(q_deg_s)
        delta_e_sync = np.deg2rad(0.0)
        h = h_m
        x = np.array([
            0.0,
            0.0,
            q,
            theta,
            h
        ])
        
        theta_history_deg = []
        target_q_history_deg = []
        q_history_deg = []
        delta_e_sync_history_deg = []
        h_history_m = []
        target_theta_history_deg = []
        
        delay_steps = int(0.05 / dt)
        target_q_buffer = [np.deg2rad(0)] * delay_steps
        
        for k in range(steps):
            current_time = k * dt
            
            y = C_long @ x
            q_curr = y[0]
            theta_curr = y[1]
            h_curr = y[2]
            
            q_deg = np.rad2deg(q_curr)
            theta_deg_curr = np.rad2deg(theta_curr)
            h_m_curr = h_curr
            
            target_theta_h = h_pid(h_curr, dt)
            
            target_theta_h_clipped = np.clip(target_theta_h, -np.deg2rad(10.0), np.deg2rad(10.0))
            
            theta_pid.target_theta = target_theta_h_clipped
            
            target_q = theta_pid(theta_curr, dt)
            
            target_q_buffer.pop(0) 
            target_q_buffer.append(target_q)

            delayed_target_q = target_q_buffer[0]
            delta_e_sync = q_pid(q_curr, dt, target_q=delayed_target_q)
            
            delta_e_sync_deg = np.rad2deg(delta_e_sync)
            
            u = np.array([
                delta_e_sync,
                0.0
            ])
            
            x_dot = A_long @ x + B_long @ u
            
            x = x + x_dot * dt
            
            theta_history_deg.append(theta_deg_curr)
            target_q_history_deg.append(np.rad2deg(delayed_target_q))
            q_history_deg.append(q_deg)
            delta_e_sync_history_deg.append(delta_e_sync_deg)
            h_history_m.append(h_m_curr)
            target_theta_history_deg.append(np.rad2deg(target_theta_h_clipped))
        
        line_theta.set_data(time_axis[:len(theta_history_deg)], theta_history_deg)
        line_target_theta.set_data(time_axis[:len(theta_history_deg)], target_theta_history_deg)
        
        line_delta_e.set_data(time_axis[:len(delta_e_sync_history_deg)], delta_e_sync_history_deg)
        
        line_h.set_data(time_axis[:len(h_history_m)], h_history_m)
        line_target_h.set_data(time_axis[:len(h_history_m)], [target_h_m] * len(h_history_m))
        
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        ax3.relim()
        ax3.autoscale_view()
        
        overshoot_percent = utils_h.calculate_overshoot(h_history_m, target_h_m)
        rise_time_sec = utils_h.calculate_rise_time(h_history_m, target_h_m, dt, threshold=0.9)
        
        for artist in ax3.texts:
            artist.remove()
        
        if overshoot_percent is not None and rise_time_sec is not None:
            text_str = f'Overshooting: {overshoot_percent:.2f}%\nRise time: {rise_time_sec:.2f}s'
        elif overshoot_percent is not None:
            text_str = f'Overshooting: {overshoot_percent:.2f}%\nRise time: -1'
        elif rise_time_sec is not None:
            text_str = f'Overshooting: -1\nRise time: {rise_time_sec:.2f}s'
        else:
            text_str = 'Overshooting and Rise time: -1'
        
        ax3.text(0.2, 0.05, text_str, transform=ax3.transAxes,
                 verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        fig.canvas.draw_idle()
    
    def update_pid_params(val):
        update_simulation(val)
    
    slider_kp_h.on_changed(update_pid_params)
    slider_ki_h.on_changed(update_pid_params)
    slider_kd_h.on_changed(update_pid_params)
    
    update_simulation(None)
    
    plt.show()

if __name__ == "__main__":
    control()