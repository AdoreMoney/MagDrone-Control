import numpy as np
import os
import sys
sys.path.append(r'C:\Users\a1fla\Desktop\MagDrone')
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import utils.utils_ry as utils_ry
from trim.trim_hrz import FixedWingDynamics, TrimCalculator, WindField
from stable_baselines3 import PPO, SAC, TD3  # Added TD3 support (optional)
from RollPIDcontroller import DualPIDController
from adrc_controller import ADRCController
from env.WindCompEnv import WindCompEnv

# ====================== Multi-Type Wind Disturbance Generation Function ======================
def generate_wind_sequence(steps, dt, wind_type="turbulence", base_wind=10.0, 
                           custom_step_periods=None):
    """
    Generate different types of lateral wind disturbance sequences (support manual specification of step periods)
    :param steps: Number of simulation steps
    :param dt: Simulation time step
    :param wind_type: Wind disturbance type - "constant"(steady wind), "step"(step wind), "sinusoidal"(sinusoidal wind), 
                      "turbulence"(turbulence wind), "combination"(combined wind)
    :param base_wind: Base wind speed (steady component)
    :param custom_step_periods: Manually specified list of step periods, format: [(start1, end1), (start2, end2), ...], unit: seconds
                                e.g., [(2,4), (7,9), (15,17)] means applying steps during 2-4s, 7-9s, 15-17s respectively
    :return: Wind speed sequence (steps,) + step wind key information (returned only for step type)
    """
    # Time axis
    time = np.arange(steps) * dt
    step_wind_info = {}  # New: Store step wind key information
    
    # 1. Steady wind
    if wind_type == "constant":
        wind_seq = np.full(steps, base_wind)
    
    # 2. Step wind (support manual specification of specific step periods, core modification)
    elif wind_type == "step":
        wind_seq = np.full(steps, base_wind)
        step_delta = 6.0  # Step increment (m/s), adjustable as needed
        
        # Manually specified step periods (highest priority, use custom periods if provided)
        if custom_step_periods is not None and len(custom_step_periods) > 0:
            step_intervals = custom_step_periods
        else:
            # Backup: Default periodic steps (can be commented out, keep only manual mode)
            step_interval = 5.0
            step_duration = 2.0
            step_start_time = 2.0
            step_start_times = np.arange(step_start_time, time[-1], step_interval)
            step_intervals = [(t, t+step_duration) for t in step_start_times]
        
        # Iterate through all specified step periods and apply wind speed increment
        valid_step_intervals = []
        for (t_start, t_end) in step_intervals:
            # Validate period validity
            if t_start >= t_end or t_end > time[-1]:
                print(f"Warning: Invalid step period ({t_start}, {t_end}), skipped")
                continue
            # Calculate start and end indices of the step
            idx_start = int(t_start / dt)
            idx_end = int(t_end / dt)
            idx_end = min(idx_end, steps)  # Prevent out of bounds
            # Apply step increment
            wind_seq[idx_start:idx_end] += step_delta
            # Record valid step periods
            valid_step_intervals.append((t_start, t_end))
        
        # Store step wind key information
        step_wind_info = {
            "type": "custom_step",
            "step_count": len(valid_step_intervals),
            "step_delta": step_delta,
            "step_intervals": valid_step_intervals,  # All valid step periods (start, end)
            "base_wind": base_wind
        }
    
    # 3. Sinusoidal wind (periodically varying wind)
    elif wind_type == "sinusoidal":
        freq = 0.1  # Sinusoidal frequency (Hz)
        amplitude = 14.0  # Sinusoidal amplitude (m/s)
        wind_seq = base_wind + amplitude * np.sin(2 * np.pi * freq * time)
    
    # 4. Dryden turbulence wind (original logic)
    elif wind_type == "turbulence":
        np.random.seed(40)
        h_ref = 100                # Reference height (m)
        mean_wind_speed = 10.41    # Mean wind speed at reference height (m/s)
        L_v = 200                  # Lateral turbulence scale (m)
        zi_v = np.zeros(2)
        white_noise_v = np.random.randn(steps)
        omega_v = 2 * np.pi * mean_wind_speed / L_v
        
        # Lateral gust transfer function
        num_v = [np.sqrt(3 * omega_v), np.sqrt(3 * omega_v**3)]
        den_v = [1, 2 * omega_v, omega_v**2]
        dt_disc = dt
        num_d_v, den_d_v, _ = signal.cont2discrete((num_v, den_v), dt_disc)
        num_d_v = num_d_v.flatten()
        wind_turbulence, _ = signal.lfilter(num_d_v, den_d_v, white_noise_v, zi=zi_v)
        
        # Turbulence wind superimposed on base wind
        sigma_v = 1.0 * mean_wind_speed * (h_ref / 10) ** (-0.25)
        wind_seq = base_wind + sigma_v * wind_turbulence
    
    # 5. Combined wind (steady + step + turbulence)
    elif wind_type == "combination":
        # Steady wind + manually specified step wind
        step_wind = np.full(steps, base_wind)
        step_delta = 5.0
        if custom_step_periods is not None and len(custom_step_periods) > 0:
            step_intervals = custom_step_periods
        else:
            step_interval = 8.0
            step_duration = 2.0
            step_start_times = np.arange(5.0, time[-1], step_interval)
            step_intervals = [(t, t+step_duration) for t in step_start_times]
        
        for (t_start, t_end) in step_intervals:
            if t_start >= t_end or t_end > time[-1]:
                continue
            idx_start = int(t_start / dt)
            idx_end = int(t_end / dt)
            idx_end = min(idx_end, steps)
            step_wind[idx_start:idx_end] += step_delta
        
        # Superimpose turbulence wind
        np.random.seed(43)
        h_ref = 100
        mean_wind_speed = 8.0
        L_v = 300
        zi_v = np.zeros(2)
        white_noise_v = np.random.randn(steps)
        omega_v = 2 * np.pi * mean_wind_speed / L_v
        num_v = [np.sqrt(3 * omega_v), np.sqrt(3 * omega_v**3)]
        den_v = [1, 2 * omega_v, omega_v**2]
        dt_disc = dt
        num_d_v, den_d_v, _ = signal.cont2discrete((num_v, den_v), dt_disc)
        num_d_v = num_d_v.flatten()
        wind_turbulence, _ = signal.lfilter(num_d_v, den_d_v, white_noise_v, zi=zi_v)
        sigma_v = 0.8 * mean_wind_speed * (h_ref / 10) ** (-0.25)
        
        # Final combined wind
        wind_seq = step_wind + sigma_v * wind_turbulence
    
    else:
        raise ValueError(f"Unsupported wind disturbance type: {wind_type}, options: constant/step/sinusoidal/turbulence/combination")
    
    return wind_seq, step_wind_info

# Simulate optimized control for verification
def simulate_optimized_control(agent, agent_type, model, trim_calc, wind_field, target_heading, dt=0.01, 
                               wind_type="turbulence", custom_step_periods=None):
    # Get trim state
    u_rel_trim, w_rel_trim, _, _, T_trim = trim_calc.find_trim()
    V_trim = np.sqrt(u_rel_trim**2 + w_rel_trim**2)  # Fix typo in original code
    A_hrz, B_hrz = trim_calc.linearize_trim()
    CY_beta, m, Cl_beta, Cn_beta = model.CY_beta, model.m, model.Cl_beta, model.Cn_beta
    qSb = 0.5 * model.rho * V_trim**2 * model.S * model.b
    Ixx, Izz = model.Ixx, model.Izz
    speed = 6
    
    # Initialize environment
    env = WindCompEnv(model, trim_calc, wind_field, target_heading, dt)
    # Initialize controller (compatible with PID/ADRC)
    if agent_type == "ADRC":
        adrc_controller = ADRCController(V_trim)
    else:
        pid_controller = DualPIDController(V_trim)
    
    # Store data
    heading_history_deg = []
    beta_history_deg = []
    phi_history_deg = []
    psi_history_deg = []
    control_history_deg = []
    compensation_history = []
    target_heading_deg = []
    wind_v_history = []
    
    # Initial state
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    target_roll = 0.0
    
    # Generate specified type of wind disturbance sequence (pass manual step periods)
    wind_v_sequence, step_wind_info = generate_wind_sequence(
        env.steps, dt, wind_type=wind_type, base_wind=0.0,
        custom_step_periods=custom_step_periods
    )
    
    # Run simulation
    for k in range(env.steps):
        # Sinusoidal time-varying target heading
        sine_amplitude_deg = 20.0
        sine_frequency_hz = 0.05
        sine_offset_deg = 0.0
        current_time = k * dt
        target_heading = np.deg2rad(
            sine_offset_deg + sine_amplitude_deg * np.sin(2 * np.pi * sine_frequency_hz * current_time)
        )

        # Construct state vector
        phi_curr = x[3]
        psi_curr = x[4]
        beta_curr = x[0]
        p_curr, r_curr = x[1], x[2]
        
        # Get current wind speed
        wind_v = wind_v_sequence[k]
        wind_v_history.append(wind_v)
        
        Dw = np.array([
            wind_v * CY_beta * np.cos(psi_curr)/(m * V_trim**2),
            wind_v * qSb * Cl_beta * np.cos(psi_curr) / (Ixx * V_trim),
            wind_v * qSb * Cn_beta * np.cos(psi_curr) / (Izz * V_trim),
            0.0,
            0.0
        ])
        
        # Calculate heading and error
        heading = psi_curr - beta_curr
        heading_error = env._normalize_angle(target_heading - heading)
        
        state = np.array([
            heading_error, psi_curr, beta_curr, phi_curr, target_roll, p_curr, r_curr
        ])
        
        # Get compensation amount
        if agent_type in ["PPO", "SAC", "TD3"]:
            control_compensation_tensor, _ = agent.predict(state, deterministic=True)
            control_compensation = control_compensation_tensor[0]
        else:
            control_compensation = 0.0
        
        # Calculate control command
        if agent_type == "ADRC":
            control_action, target_roll = adrc_controller.compute_control(
                current_roll=phi_curr,
                current_yaw=psi_curr,
                current_beta=beta_curr,
                dt=dt,
                target_heading=target_heading
            )
        else:
            control_action, target_roll = pid_controller.compute_control(
                current_roll=phi_curr,
                current_yaw=psi_curr,
                current_beta=beta_curr,
                dt=dt,
                target_heading=target_heading
            )
            control_action += control_compensation
        control_action = np.clip(control_action, -np.deg2rad(10), np.deg2rad(10))
        
        # Apply control input
        u = np.array([control_action, 0.0])
        x_dot = A_hrz @ x + B_hrz @ u + Dw
        x = x + x_dot * dt
        beta, phi, psi = x[0], x[3], x[4]

        # Record data
        heading_history_deg.append(np.rad2deg(env._normalize_angle(psi-beta)))
        beta_history_deg.append(np.rad2deg(env._normalize_angle(beta)))
        phi_history_deg.append(np.rad2deg(env._normalize_angle(phi)))
        psi_history_deg.append(np.rad2deg(env._normalize_angle(psi)))
        control_history_deg.append(np.rad2deg(control_action))
        target_heading_deg.append(np.rad2deg(target_heading))
        compensation_history.append(np.rad2deg(control_compensation))
        
        # Print periodically
        if k % 500 == 0:
            print(f"Step {k}, Wind Speed: {wind_v:.2f} m/s, Control Compensation: {control_compensation:.4f}")
    
    # Save data
    data = {
        'heading_deg': heading_history_deg,
        'beta_deg': beta_history_deg,
        'phi_deg': phi_history_deg,
        'psi_deg': psi_history_deg,
        'control_deg': control_history_deg,
        'target_heading_deg': target_heading_deg,
        'compensation_deg': compensation_history,
        'wind_v_mps': wind_v_history
    }
    
    csv_path = f"data/csv/sin/{wind_type}_{speed}ms/{agent_type}_data_{wind_type}.csv"
    npz_path = f"data/npz/sin/{wind_type}_{speed}ms/{agent_type}_data_{wind_type}.npz"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index_label="step")
    print(f"CSV saved to {csv_path}")
    np.savez(npz_path,** data)
    print(f"NPZ saved to {npz_path}")
    
    # Calculate performance metrics
    window_size = 100
    heading_sw_var = utils_ry.sliding_window_variance(heading_history_deg, window_size)
    beta_sw_var = utils_ry.sliding_window_variance(beta_history_deg, window_size)
    phi_sw_var = utils_ry.sliding_window_variance(phi_history_deg, window_size)
    psi_sw_var = utils_ry.sliding_window_variance(psi_history_deg, window_size)
    control_sw_var = utils_ry.sliding_window_variance(control_history_deg, window_size)
    print('\n===== Performance Metrics =====')
    print(f'Heading Variance: {heading_sw_var:.4f}, Sideslip Angle Variance: {beta_sw_var:.4f}')
    print(f'Roll Variance: {phi_sw_var:.4f}, Yaw Variance: {psi_sw_var:.4f}, Control Action Variance: {control_sw_var:.4f}')
    
    # Calculate RMS
    yaw_rms = np.sqrt(np.mean(np.array(psi_history_deg)**2))
    roll_rms = np.sqrt(np.mean(np.array(phi_history_deg)**2))
    e_diff_rms = np.sqrt(np.mean(np.array(control_history_deg)**2))
    print(f'Yaw RMS (deg): {yaw_rms:.4f}')
    print(f'Roll RMS (deg): {roll_rms:.4f}')
    print(f'Control Action RMS (deg): {e_diff_rms:.4f}')
        
    # Calculate time-domain performance metrics
    overshoot = utils_ry.calculate_overshoot(heading_history_deg, np.rad2deg(target_heading))
    rise_time = utils_ry.calculate_rise_time(heading_history_deg, np.rad2deg(target_heading), dt, threshold=0.9)
    errors = [(target_heading - np.deg2rad(heading)) for heading in heading_history_deg]
    settling_time = utils_ry.calculate_settling_time(errors)
    steady_error = utils_ry.calculate_steady_state_error(heading_history_deg, np.rad2deg(target_heading), settling_time)
    
    if rise_time is None: rise_time = -1.0
    if settling_time is None: settling_time = -1.0
    if steady_error is None: steady_error = -1.0
       
    print(f"\n===== Time-Domain Metrics ===== ")
    print(f"Overshoot: {overshoot:.2f}%")
    print(f"Rise Time: {rise_time:.2f}s")
    print(f'Settling Time: {settling_time*0.01:.2f}s')
    print(f'Steady-State Error: {steady_error:.2f}')
    
    # ====================== Heading Recovery Time Calculation for Multiple Step Winds ======================
    heading_recovery_info = []
    heading_threshold = 0.5  # Heading recovery threshold
    target_heading_deg_val = np.rad2deg(target_heading)
    time_arr = np.arange(len(heading_history_deg)) * dt
    
    if wind_type == "step" and step_wind_info.get("type") in ["custom_step", "multiple_step"]:
        step_count = step_wind_info["step_count"]
        step_intervals = step_wind_info["step_intervals"]
        
        print(f"\n===== Custom Step Wind Recovery Time Statistics (Total {step_count} Steps) ===== ")
        for i, (step_start, step_end) in enumerate(step_intervals):
            # Heading data after step ends
            post_step_mask = time_arr >= step_end
            if not np.any(post_step_mask):
                heading_recovery_info.append({
                    "step_index": i+1, "step_start": step_start, "step_end": step_end,
                    "recovery_time": -1, "recovery_delay": -1
                })
                print(f"Step {i+1} ({step_start:.1f}~{step_end:.1f}s): No subsequent data, cannot calculate recovery time")
                continue
            
            post_step_headings = np.array(heading_history_deg)[post_step_mask]
            post_step_times = time_arr[post_step_mask]
            # Find the first time when error is less than threshold
            errors = np.abs(post_step_headings - target_heading_deg_val)
            recovery_idx = np.where(errors < heading_threshold)[0]
            
            if len(recovery_idx) > 0:
                recovery_time = post_step_times[recovery_idx[0]]
                recovery_delay = recovery_time - step_end
                heading_recovery_info.append({
                    "step_index": i+1, "step_start": step_start, "step_end": step_end,
                    "recovery_time": recovery_time, "recovery_delay": recovery_delay
                })
                print(f"Step {i+1} ({step_start:.1f}~{step_end:.1f}s): Recovery Time={recovery_time:.2f}s (Delay {recovery_delay:.2f}s)")
            else:
                heading_recovery_info.append({
                    "step_index": i+1, "step_start": step_start, "step_end": step_end,
                    "recovery_time": -1, "recovery_delay": -1
                })
                print(f"Step {i+1} ({step_start:.1f}~{step_end:.1f}s): Not Recovered (Error > 0.5deg)")
    
    # ====================== Plotting: Annotate All Custom Step Periods ======================
    plt.figure(figsize=(10, 8))
    time_plot = np.arange(len(heading_history_deg)) * dt
    
    # Subplot 1: Heading tracking
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(time_plot, heading_history_deg, 'b-', label='Actual heading')
    ax1.plot(time_plot, target_heading_deg, 'r--', label='Target heading')
    # Annotate all custom step periods
    if wind_type == "step" and step_wind_info.get("type") in ["custom_step", "multiple_step"]:
        for i, (step_start, step_end) in enumerate(step_intervals):
            # Show legend for first step, hide for subsequent steps
            if i == 0:
                ax1.axvspan(step_start, step_end, alpha=0.2, color='orange', label='Step wind period')
            else:
                ax1.axvspan(step_start, step_end, alpha=0.2, color='orange')
            # Annotate recovery time
            if len(heading_recovery_info) > i and heading_recovery_info[i]["recovery_time"] > 0:
                ax1.axvline(x=heading_recovery_info[i]["recovery_time"], color='green', linestyle=':')
        # Recovery time legend
        if len(heading_recovery_info) > 0 and heading_recovery_info[0]["recovery_time"] > 0:
            ax1.axvline(x=heading_recovery_info[0]["recovery_time"], color='green', linestyle=':', label='Recovery time')
    ax1.set_ylabel('Heading (deg)')
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: Roll angle
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(time_plot, phi_history_deg, 'm-', label='Roll angle')
    # Annotate all custom step periods
    if wind_type == "step" and step_wind_info.get("type") in ["custom_step", "multiple_step"]:
        for step_start, step_end in step_intervals:
            ax2.axvspan(step_start, step_end, alpha=0.2, color='orange')
    ax2.set_ylabel('Roll (deg)')
    ax2.tick_params(axis='x', labelbottom=False)
    ax2.legend()
    ax2.grid(True)
    
    # Subplot 3: Control action
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.plot(time_plot, control_history_deg, 'g-', label='Control action')
    # Annotate all custom step periods
    if wind_type == "step" and step_wind_info.get("type") in ["custom_step", "multiple_step"]:
        for step_start, step_end in step_intervals:
            ax3.axvspan(step_start, step_end, alpha=0.2, color='orange')
    ax3.set_ylabel('Control action (deg)')
    ax3.tick_params(axis='x', labelbottom=False)
    ax3.legend()
    ax3.grid(True)
    
    # Subplot 4: Wind speed variation
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    ax4.plot(time_plot, wind_v_history, 'k-', label='Wind speed')
    # Annotate all custom step periods
    if wind_type == "step" and step_wind_info.get("type") in ["custom_step", "multiple_step"]:
        for i, (step_start, step_end) in enumerate(step_intervals):
            if i == 0:
                ax4.axvspan(step_start, step_end, alpha=0.2, color='orange', label='Step wind period')
            else:
                ax4.axvspan(step_start, step_end, alpha=0.2, color='orange')
    ax4.set_ylabel('Wind speed (m/s)')
    ax4.set_xlabel('Time (s)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    # plt.savefig(f'comparison/{agent_type}_performance_{wind_type}.png', dpi=600)
    plt.show()
    
    return overshoot, rise_time, settling_time, steady_error

def main():
    # 1. Configuration parameters
    model_type = "PPO"  # Options: PPO/SAC/TD3/PID/ADRC
    wind_type = "step"  # Select step wind disturbance type
    target_heading_deg = 15.0
    target_heading = np.deg2rad(target_heading_deg)
    dt = 0.01
    
    # ====================== Core: Manually Specify Step Wind Periods (Modify as Needed) ======================
    # Format: [(start_time1, end_time1), (start_time2, end_time2), ...], unit: seconds
    # Example: Apply step winds during 2-4s, 7-9s, 15-17s, 22-25s respectively
    custom_step_periods = [
        (2, 4),
        (7, 9),
        (15, 17),
        (22, 25),
        (30, 32),
        (35, 37),
        (40, 42),
        (45, 47),
        (50, 52),
        (55, 57),
        (60, 62),
        (65, 67),
        (70, 72),
        (75, 77),
        (80, 82),
        (85, 87),
        (90, 92),
        (95, 97),
        (100, 102),
        (105, 107),
        (110, 112),
        (115, 117),
        (120, 122),
        (125, 127),
        (130, 132),
        (135, 137),
        (140, 142),
        (145, 147)
    ]
    
    # 2. Load models
    model_paths = {
        "PPO": "../logs/ppo/MagDrone_models/ppo_MagDrone_20251202_2216.zip",
        "SAC": "../logs/sac/MagDrone_models/sac_MagDrone_20251203_2138.zip",
        "TD3": "../logs/td3/MagDrone_models/td3_MagDrone_20251209_1550.zip"  
    }
    
    if model_type not in ["PID", "ADRC"] and not os.path.exists(model_paths[model_type]):
        print(f"Model file does not exist: {model_paths[model_type]}")
        return
    
    # 3. Initialize dynamics model
    wind_field = WindField()
    model = FixedWingDynamics(wind_field)
    trim_calc = TrimCalculator(model, wind_field)
    
    # 4. Load agent
    if model_type == "PPO":
        agent = PPO.load(model_paths["PPO"])
    elif model_type == "SAC":
        agent = SAC.load(model_paths["SAC"])
    elif model_type == "TD3":
        agent = TD3.load(model_paths["TD3"])
    else:
        agent = None
    
    # 5. Run simulation (pass custom step periods)
    print(f"Starting verification of {model_type} optimized control, wind disturbance type: {wind_type} (custom step periods)...")
    overshoot, rise_time, settling_time, steady_error = simulate_optimized_control(
        agent, model_type, model, trim_calc, wind_field, target_heading, dt,
        wind_type=wind_type, custom_step_periods=custom_step_periods
    )
    
    # Print final results
    print(f"\n===== Final Performance Summary ===== ")
    print(f"{model_type} ({wind_type} wind disturbance): Overshoot={overshoot:.2f}%, Rise Time={rise_time:.2f}s, Settling Time={settling_time*0.01:.2f}s, Steady-State Error={steady_error:.2f}")

if __name__ == "__main__":
    main()