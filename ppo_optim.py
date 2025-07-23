import numpy as np
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import utils_ry
from scipy import signal
import os
from trim_hrz import FixedWingDynamics, TrimCalculator, WindField
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from RollPIDcontroller import DualPIDController
from WindCompEnv import WindCompEnv

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
    
# Train PPO to optimize dual PID parameters
def train_ppo_for_compensation(env, offline=False, total_timesteps=10_0000,
                             use_behavior_clone=True, pretrained_model_path=None):
    # Create SB3-compatible environment
    vec_env = make_vec_env(lambda: env, n_envs=1)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    if offline:
        pass
    else:
        print("Starting online training: (PPO)")
        
        policy_kwargs = {
            'net_arch': [256, 128, 64, 32],
            'activation_fn': nn.ReLU,
            'log_std_init': -1.0,
        }
        
        tensorboard_log_dir = "./ppo_tensorboard_logs/"
        
        # =============== Key modification section ===============
        if pretrained_model_path:
            print(f"Loading pretrained model: {pretrained_model_path}")
            ppo_agent = PPO.load(
                pretrained_model_path,
                env=vec_env,
                tensorboard_log=tensorboard_log_dir,
                device='auto',  # Automatically select GPU/CPU
                # Override some training parameters
                learning_rate=1e-4,  
                n_steps=2048,
                batch_size=128,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                policy_kwargs=policy_kwargs
            )
            print(f"Model parameters loaded, will continue training for {total_timesteps} steps from checkpoint")
        else:
            ppo_agent = PPO(
                "MlpPolicy", 
                vec_env,
                verbose=1,
                learning_rate=1e-4,
                n_steps=2048,
                batch_size=128,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.05,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tensorboard_log_dir
            )
        # ==========================================

        # Callback functions
        eval_callback = EvalCallback(
            vec_env,
            best_model_save_path='./ppo_logs/',
            log_path='./ppo_logs/',
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=1_0000,
            save_path="./ppo_logs/ppo_checkpoints/",
            name_prefix="rl_model"
        )

        # Train the model
        ppo_agent.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            reset_num_timesteps=False  # Key parameter: do not reset step counter
        )

        # Save the final model
        model_path = f"ppo/ppo_continued_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
        ppo_agent.save(model_path)
        print(f"Training completed, model saved to: {model_path}")
        
        return ppo_agent, [], None

# Simulate optimized control
def simulate_optimized_control(ppo_agent, model, trim_calc, wind_field, target_heading, dt=0.01):
    # Get trim state
    u_rel_trim, w_rel_trim, _, _, T_trim = trim_calc.find_trim()
    V_trim = np.sqrt(u_rel_trim**2 + w_rel_trim**2)  # Fixed previous error
    A_hrz, B_hrz = trim_calc.linearize_trim()
    CY_beta, m, Cl_beta, Cn_beta = model.CY_beta, model.m, model.Cl_beta, model.Cn_beta
    qSb = 0.5 * model.rho * V_trim**2 * model.S * model.b
    Ixx, Izz = model.Ixx, model.Izz
    
    # Initialize environment
    env = WindCompEnv(model, trim_calc, wind_field, target_heading, dt)
    # Create PID controller
    pid_controller = DualPIDController(V_trim)
    
    # Store data
    heading_history_deg = []
    beta_history_deg = []
    phi_history_deg = []
    psi_history_deg = []
    control_history_deg = []
    compensation_history = []
    target_heading_deg = []
    
    # Initial state
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    target_roll = 0.0
    wind_v = 0
    # Wind speed parameters
    np.random.seed(42)
    h_ref = 100                # Reference height (m)
    mean_wind_speed = 10.41    # Mean wind speed at reference height (m/s)

    # Dryden model parameters
    L_v = 200                 # Lateral turbulence scale (m)
    zi_v = np.zeros(2)

    # Pre-generate random noise sequence
    white_noise_v = np.random.randn(env.steps)  # Lateral wind noise

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
    
    # Run simulation
    for k in range(env.steps):
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
        
        # Construct state vector
        phi_curr = x[3]
        psi_curr = x[4]
        beta_curr = x[0]
        p_curr, r_curr = x[1], x[2]
        
        sigma_v = 1.0 * env.mean_wind_speed * (env.h_ref / 10) ** (-0.25)  # Lateral turbulence intensity
        wind_v = sigma_v * wind_v_sequence[k]
        
        Dw = np.array([
            wind_v * CY_beta * np.cos(psi_curr)/(m * V_trim**2),  # beta_with_wind
            wind_v * qSb * Cl_beta * np.cos(psi_curr) / (Ixx * V_trim),   # p_with_wind
            wind_v * qSb * Cn_beta * np.cos(psi_curr) / (Izz * V_trim),   # r_with_wind
            0.0,   # phi
            0.0    # psi
            ])
        
        # Calculate heading and error
        heading = psi_curr - beta_curr  # Current heading in rad
        heading_error = env._normalize_angle(target_heading - heading)  # Current heading error in rad
        
        state = np.array([
            heading_error,  # Heading error in rad
            psi_curr,  # Yaw angle psi in rad
            beta_curr, # Sideslip angle beta in rad
            phi_curr,  # Roll angle phi in rad
            target_roll,  # Desired roll angle in rad (from previous state)
            p_curr, r_curr
        ])
        
        # Use PPO to get differential control compensation
        control_compensation_tensor, _ = ppo_agent.predict(state, deterministic=True)
        control_compensation = control_compensation_tensor[0]
        
        # Calculate comprehensive control output
        control_action, target_roll = pid_controller.compute_control(
                current_roll=phi_curr,
                current_yaw=psi_curr,
                current_beta=beta_curr,
                dt=dt,
                target_heading=target_heading
            )
        # Add wind disturbance sideslip compensation to roll angle control
        control_action += control_compensation
        control_action = np.clip(control_action, -np.deg2rad(10), np.deg2rad(10))  # in rad
        
        # Apply control input
        u = np.array([control_action, 0.0])
        
        # Calculate state derivative
        x_dot = A_hrz @ x + B_hrz @ u + Dw
        
        # Update state
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
        
        # Print parameters periodically
        if k % 500 == 0:
            print(f"Step {k}, Control compensation: "
                  f"control_compensation={control_compensation:.4f}")
    
    data = {
    'heading_deg': heading_history_deg,
    'beta_deg': beta_history_deg,
    'phi_deg': phi_history_deg,
    'psi_deg': psi_history_deg,
    'control_deg': control_history_deg,
    'target_heading_deg': target_heading_deg,
    'compensation_deg': compensation_history
    }
    
    # Option 1: Save as CSV (easy to view in Excel)
    df = pd.DataFrame(data)
    csv_path = f"ppo_comp_pid/ppo_data_03.csv"
    df.to_csv(csv_path, index_label="step")
    print(f"CSV saved to {csv_path}")

    # Option 2: Save as NPZ (efficient binary, suitable for Python reading)
    npz_path = f"ppo_comp_pid/ppo_data_03.npz"
    np.savez(npz_path, **data)
    print(f"NPZ saved to {npz_path}")
    
    window_size = 100
    heading_sw_var = utils_ry.sliding_window_variance(heading_history_deg, window_size)
    beta_sw_var = utils_ry.sliding_window_variance(beta_history_deg, window_size)
    phi_sw_var = utils_ry.sliding_window_variance(phi_history_deg, window_size)
    psi_sw_var = utils_ry.sliding_window_variance(psi_history_deg, window_size)
    control_sw_var = utils_ry.sliding_window_variance(control_history_deg, window_size)
    print('Heading variance:', heading_sw_var, 'Sideslip angle variance:', beta_sw_var)
    print('Roll variance:', phi_sw_var, 'Yaw variance:', psi_sw_var, 'Differential variance:', control_sw_var)
    
    # Calculate RMS of yaw angle
    yaw_rms = np.sqrt(np.mean(np.array(psi_history_deg)**2))
    print(f'Yaw angle RMS (deg): {yaw_rms:.4f}')
    
    # Calculate RMS of roll angle
    # Note: According to your code, phi_history_deg stores actual roll angle
    roll_rms = np.sqrt(np.mean(np.array(phi_history_deg)**2))
    print(f'Roll angle RMS (deg): {roll_rms:.4f}')
    
    e_diff_rms = np.sqrt(np.mean(np.array(control_history_deg)** 2))
    print(f'Differential quantity RMS (deg): {e_diff_rms:.4f}')
        
    # Calculate performance metrics
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
       
    print(f"\nAfter optimization: ")
    print(f"Overshoot: {overshoot:.2f}%")
    print(f"Rise time: {rise_time:.2f}s")
    print(f'Settling time:{settling_time:.2f}s')
    print(f'Steady-state error:{steady_error:.2f}')
    
    # Plot results
    plt.figure(figsize=(8, 6))
    
    # Create first subplot as benchmark with shared x-axis, return subplot object ax1
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(np.arange(0, len(heading_history_deg)) * dt, heading_history_deg, 'b-', label='heading')
    ax1.plot(np.arange(0, len(heading_history_deg)) * dt, target_heading_deg, 'r--', label='target heading')
    ax1.set_ylabel('heading (deg)')
    ax1.tick_params(axis='x', labelbottom=False) # Hide x-axis ticks
    ax1.legend()
    ax1.grid(True)

    ax3 = plt.subplot(2, 1, 2, sharex=ax1)
    ax3.plot(np.arange(0, len(phi_history_deg)) * dt, phi_history_deg, 'm-', label='roll')
    ax3.set_ylabel('roll (deg)')
    ax3.set_xlabel('Time (s)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('ppo/ppo_optimized_control.png', dpi=600)
    plt.show()
    
    return overshoot, rise_time, settling_time, steady_error

# Main control function
def control(offline_mode=False, use_behavior_clone=True):
    # Initialize model
    wind_field = WindField()
    model = FixedWingDynamics(wind_field)  # Aerodynamic model
    trim_calc = TrimCalculator(model, wind_field)
    
    # Target heading
    target_heading_deg = 10.0
    target_heading = np.deg2rad(target_heading_deg)  # in rad
    
    # Initialize environment
    env = WindCompEnv(model, trim_calc, wind_field, target_heading)
    # Model save directory
    os.makedirs('ppo', exist_ok=True)
    
    # Define ppo model path
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
    ppo_model_path = f"ppo/ppo_compensation_{timestamp}.zip"
    
    if offline_mode:
        pass
        
    else:
        print("Starting PPO online training mode...")
        
        # Check for pretrained PPO model
        use_existing_ppo = use_behavior_clone and os.path.exists(ppo_model_path)
        if use_existing_ppo:
            print(f'Pretrained model exists, using {ppo_model_path} for training!')
            # Load model
            ppo_agent, _, _ = train_ppo_for_compensation(
                env, offline=False, total_timesteps=50_0000,
                use_behavior_clone=use_existing_ppo,
                pretrained_model_path=ppo_model_path if use_existing_ppo else None
            )
        
        else:
            # Train PPO agent (can optionally use PPO pretraining)
            ppo_agent, _, _ = train_ppo_for_compensation(
                env, offline=False, total_timesteps=100_0000,
                use_behavior_clone=use_existing_ppo,
                pretrained_model_path=ppo_model_path if use_existing_ppo else None
            )
        
        # Simulation verification (using PPO model)
        print("\nStarting verification of PPO-optimized control...")
        overshoot, rise_time, settling_time, steady_error = simulate_optimized_control(
            ppo_agent, model, trim_calc, wind_field, target_heading
        )
    
    # Print final results
    print(f"Performance after optimization: Overshoot={overshoot:.2f}%, Rise time={rise_time:.2f}s, Settling time={settling_time:.2f}, Steady-state error={steady_error:.2f}")

if __name__ == "__main__":
    # Default to online training
    control()
    
    # Run offline training mode
    #control(offline_mode=True)