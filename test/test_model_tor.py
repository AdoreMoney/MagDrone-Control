import numpy as np
import os
import sys
sys.path.append(r'C:\Users\a1fla\Desktop\MagDrone')
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import utils.utils_ry as utils_ry
from trim.trim_hrz import FixedWingDynamics, TrimCalculator, WindField
from stable_baselines3 import PPO,SAC,TD3
from RollPIDcontroller import DualPIDController
from adrc_controller import ADRCController
from env.WindCompEnv import WindCompEnv

def simulate_optimized_control(agent, agent_type, model, trim_calc, target_heading, dt=0.01):
    """
    Control simulation in disturbance-free environment (All wind disturbance related logic removed)
    """
    u_rel_trim, w_rel_trim, _, _, T_trim = trim_calc.find_trim()
    V_trim = np.sqrt(u_rel_trim**2 + w_rel_trim**2)
    A_hrz, B_hrz = trim_calc.linearize_trim()
    CY_beta, m, Cl_beta, Cn_beta = model.CY_beta, model.m, model.Cl_beta, model.Cn_beta
    qSb = 0.5 * model.rho * V_trim**2 * model.S * model.b
    Ixx, Izz = model.Ixx, model.Izz
    value = -0.007
    
    # Wind field is only a placeholder and has no actual effect
    wind_field = WindField()
    env = WindCompEnv(model, trim_calc, wind_field, target_heading, dt)
    
    if agent_type == "ADRC":
        adrc_controller = ADRCController(V_trim)
    else:
        pid_controller = DualPIDController(V_trim)
    
    heading_history_deg = []
    beta_history_deg = []
    phi_history_deg = []
    psi_history_deg = []
    control_history_deg = []
    target_heading_deg = []
    compensation_history = []
    
    # Disturbance-free initial states
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    target_roll = 0.0
    
    # Run simulation without wind disturbance
    for k in range(env.steps):
        phi_curr = x[3]
        psi_curr = x[4]
        beta_curr = x[0]
        p_curr, r_curr = x[1], x[2]
        
        # No wind disturbance: Dw is all-zero vector
        Dw = np.zeros(5)
        
        heading = psi_curr - beta_curr
        heading_error = env._normalize_angle(target_heading - heading)
        
        state = np.array([
            heading_error,
            psi_curr,
            beta_curr,
            phi_curr,
            target_roll,
            p_curr, r_curr
        ])
        
        if agent_type in ["PPO", "SAC", "TD3"]:
            control_compensation_tensor, _ = agent.predict(state, deterministic=True)
            control_compensation = control_compensation_tensor[0]
        else:
            control_compensation = 0.0
        
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
        
        u = np.array([control_action, 0.0])
        
        # Calculate state derivative without disturbance
        x_dot = A_hrz @ x + B_hrz @ u + Dw
        
        x = x + x_dot * dt
        beta, phi, psi = x[0], x[3], x[4]

        heading_history_deg.append(np.rad2deg(env._normalize_angle(psi-beta)))
        beta_history_deg.append(np.rad2deg(env._normalize_angle(beta)))
        phi_history_deg.append(np.rad2deg(env._normalize_angle(phi)))
        psi_history_deg.append(np.rad2deg(env._normalize_angle(psi)))
        control_history_deg.append(np.rad2deg(control_action))
        target_heading_deg.append(np.rad2deg(target_heading))
        compensation_history.append(np.rad2deg(control_compensation))
        
        if k % 500 == 0:
            print(f"Step {k}, Control Compensation: {control_compensation:.4f} rad")
    
    data = {
        'heading_deg': heading_history_deg,
        'beta_deg': beta_history_deg,
        'phi_deg': phi_history_deg,
        'psi_deg': psi_history_deg,
        'control_deg': control_history_deg,
        'target_heading_deg': target_heading_deg,
        'compensation_deg': compensation_history
    }
    
    csv_path = f"data/csv/10deg/Cl_delta_e_diff_{value}/{agent_type}_data_Cl_delta_e_diff_{value}.xlsx"
    npz_path = f"data/npz/10deg/Cl_delta_e_diff_{value}/{agent_type}_data_Cl_delta_e_diff_{value}.npz"

    csv_dir = os.path.dirname(csv_path)
    os.makedirs(csv_dir, exist_ok=True)
    npz_dir = os.path.dirname(npz_path)
    os.makedirs(npz_dir, exist_ok=True)
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index_label="step")
    print(f"CSV saved to {csv_path}")
    np.savez(npz_path,** data)
    print(f"NPZ saved to {npz_path}")
    
    window_size = 100
    heading_sw_var = utils_ry.sliding_window_variance(heading_history_deg, window_size)
    beta_sw_var = utils_ry.sliding_window_variance(beta_history_deg, window_size)
    phi_sw_var = utils_ry.sliding_window_variance(phi_history_deg, window_size)
    psi_sw_var = utils_ry.sliding_window_variance(psi_history_deg, window_size)
    control_sw_var = utils_ry.sliding_window_variance(control_history_deg, window_size)
    
    print('\n===== Disturbance-Free Environment Performance Metrics =====')
    print(f'Heading Variance: {heading_sw_var:.4f}, Sideslip Angle Variance: {beta_sw_var:.4f}')
    print(f'Roll Variance: {phi_sw_var:.4f}, Yaw Variance: {psi_sw_var:.4f}, Control Action Variance: {control_sw_var:.4f}')
    
    yaw_rms = np.sqrt(np.mean(np.array(psi_history_deg)**2))
    roll_rms = np.sqrt(np.mean(np.array(phi_history_deg)**2))
    control_rms = np.sqrt(np.mean(np.array(control_history_deg)**2))
    print(f'Yaw RMS (deg): {yaw_rms:.4f}')
    print(f'Roll RMS (deg): {roll_rms:.4f}')
    print(f'Control Action RMS (deg): {control_rms:.4f}')
        
    overshoot = utils_ry.calculate_overshoot(heading_history_deg, np.rad2deg(target_heading))
    rise_time = utils_ry.calculate_rise_time(heading_history_deg, np.rad2deg(target_heading), dt, threshold=0.9)
    errors = [(target_heading - np.deg2rad(heading)) for heading in heading_history_deg]
    settling_time = utils_ry.calculate_settling_time(errors)
    steady_error = utils_ry.calculate_steady_state_error(heading_history_deg, np.rad2deg(target_heading), settling_time)
    
    if rise_time is None:
       rise_time = -1.0  
       print("Warning: Rise time not detected")
    if settling_time is None:
       settling_time = -1.0  
       print("Warning: Settling time not detected")
    if steady_error is None:
       steady_error = -1.0   
       print("Warning: Steady-state error not detected")
       
    print(f"\n===== Disturbance-Free Environment Time-Domain Metrics ===== ")
    print(f"Overshoot: {overshoot:.2f}%")
    print(f"Rise Time: {rise_time:.2f}s")
    print(f'Settling Time: {settling_time*0.01:.2f}s')
    print(f'Steady-State Error: {steady_error:.2f}deg')
    
    plt.figure(figsize=(10, 6))
    
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(np.arange(0, len(heading_history_deg)) * dt, heading_history_deg, 'b-', label='Actual heading')
    ax1.plot(np.arange(0, len(heading_history_deg)) * dt, target_heading_deg, 'r--', label='Target heading')
    ax1.set_ylabel('Heading (deg)')
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.legend()
    ax1.grid(True)

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(np.arange(0, len(phi_history_deg)) * dt, phi_history_deg, 'm-', label='Roll angle')
    ax2.set_ylabel('Roll (deg)')
    ax2.tick_params(axis='x', labelbottom=False)
    ax2.legend()
    ax2.grid(True)
    
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(np.arange(0, len(control_history_deg)) * dt, control_history_deg, 'g-', label='Control action')
    ax3.set_ylabel('Control action (deg)')
    ax3.set_xlabel('Time (s)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return overshoot, rise_time, settling_time, steady_error

def main():
    # Configuration parameters for disturbance-free environment
    model_type = "PPO"
    target_heading_deg = 10.0
    target_heading = np.deg2rad(target_heading_deg)
    dt = 0.01
    
    model_paths = {
        "PPO": "../logs/ppo/MagDrone_models/ppo_MagDrone_20251202_2216.zip",
        "SAC": "../logs/sac/MagDrone_models/sac_MagDrone_20251203_2138.zip",
        "TD3": "../logs/td3/MagDrone_models/td3_MagDrone_20251209_1550.zip"  
    }
    
    if model_type not in ["PID", "ADRC"] and not os.path.exists(model_paths[model_type]):
        print(f"Model file does not exist: {model_paths[model_type]}")
        return
    
    # Wind field is only a placeholder
    wind_field = WindField()
    model = FixedWingDynamics(wind_field)
    trim_calc = TrimCalculator(model, wind_field)
    
    if model_type == "PPO":
        agent = PPO.load(model_paths["PPO"])
    elif model_type == "SAC":
        agent = SAC.load(model_paths["SAC"])
    elif model_type == "TD3":
        agent = TD3.load(model_paths["TD3"])
    else:
        agent = None
    
    # Run disturbance-free simulation
    print(f"Starting verification of {model_type} control performance in disturbance-free environment...")
    overshoot, rise_time, settling_time, steady_error = simulate_optimized_control(
        agent, model_type, model, trim_calc, target_heading, dt
    )
    
    # Final performance summary for disturbance-free environment
    print(f"\n===== Disturbance-Free Environment Final Performance Summary ===== ")
    print(f"{model_type}: Overshoot={overshoot:.2f}%, Rise Time={rise_time:.2f}s, Settling Time={settling_time*0.01:.2f}s, Steady-State Error={steady_error:.2f}deg")

if __name__ == "__main__":
    main()