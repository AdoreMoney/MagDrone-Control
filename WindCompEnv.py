import numpy as np
from collections import deque
from scipy import signal
from RollPIDcontroller import DualPIDController
import utils_ry
import gymnasium as gym
from gymnasium import spaces

class WindCompEnv(gym.Env):
    def __init__(self, model, trim_calc, wind_field, target_heading, dt=0.01, generate_offline_data=False):
        super(WindCompEnv, self).__init__()
        
        # Save original parameters
        self.model = model
        self.trim_calc = trim_calc
        self.wind_field = wind_field
        self.target_heading = target_heading
        self.dt = dt
        self.generate_offline_data = generate_offline_data
        
        # Load trim state
        u_rel_trim, w_rel_trim, _, _, T_trim = trim_calc.find_trim()
        self.V_trim = np.sqrt(u_rel_trim**2 + w_rel_trim**2)
        self.A_hrz, self.B_hrz = self.trim_calc.linearize_trim()
        self.CY_beta, self.m, self.Cl_beta, self.Cn_beta = model.CY_beta, model.m, model.Cl_beta, model.Cn_beta
        self.qSb = 0.5 * model.rho * self.V_trim**2 * model.S * model.b
        self.Ixx, self.Izz = model.Ixx, model.Izz
        
        # Define action space (control surface differential compensation)
        self.action_space = spaces.Box(
            low=np.deg2rad(-5.0),
            high=np.deg2rad(5.0),
            dtype=np.float32
        )
        
        # Define observation space (7 state variables: heading_error, psi, beta, phi, phi_target, p, r)
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            shape=(7,),  # [heading_error, psi, beta, phi, phi_target, p, r]
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        # Set random seed if needed
        if seed is not None:
            super().reset(seed=seed)  # Call parent class reset method to set seed
            np.random.seed(seed)
            
        # Initial state
        self.beta, self.p, self.r, self.phi, self.psi, self.phi_target = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        self.heading, self.heading_error = 0.0, 0.0
        self.current_step = 0
        self.wind_v = 0
        self.first_cross_flag = False
        self.zi_v = np.zeros(2)
        
        # Dryden model parameters
        self.h_ref = 100
        self.mean_wind_speed = 10.41
        self.L_v = 200  # Lateral turbulence scale
        
        # Pre-generate random noise sequence
        np.random.seed(42)
        self.steps = int(20.0 / self.dt)  # 20-second simulation
        self.white_noise_v = np.random.randn(self.steps)  # Lateral wind noise
        
        # Design Dryden transfer function
        self.omega_v = 2 * np.pi * self.mean_wind_speed / self.L_v
        self.num_v = [np.sqrt(3 * self.omega_v), np.sqrt(3 * self.omega_v**3)]
        self.den_v = [1, 2 * self.omega_v, self.omega_v**2]
        self.num_d_v, self.den_d_v, _ = signal.cont2discrete((self.num_v, self.den_v), self.dt)
        self.num_d_v = self.num_d_v.flatten()
        self.wind_v_sequence, self.zi_v = signal.lfilter(self.num_d_v, self.den_d_v, self.white_noise_v, zi=self.zi_v)
        
        # Create PID controller
        self.pid_controller = DualPIDController(self.V_trim)
        
        self.prev_control_action = 0.0
        self.prev_phi = 0.0
        self.prev_psi = 0.0
        self.error_integral = 0.0
        self.prev_error = 0.0
        self.prev_beta = 0.0
        self.beta_integral = 0.0
        self.error_history = []
        
        # Reset performance index tracking data
        self.heading_history = []
        self.control_history = []
        self.time_history = []
        
        # Calculate initial error
        self.heading = self.psi - self.beta
        self.heading_error = self._normalize_angle(self.target_heading - self.heading)
        
        # Return observation (7 state variables) [heading_error, psi, beta, phi, phi_target, p, r]
        observation = np.array([
            self.heading_error,   # Heading error
            self.psi,        # Yaw angle
            self.beta,       # Sideslip angle
            self.phi,        # Roll angle
            self.phi_target,  # Desired roll angle
            self.p,
            self.r
        ])
        
        return observation, {}
    
    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    def _get_wind_disturbance(self):
        
        self.wind_v = 1.0 * self.mean_wind_speed * (self.h_ref / 10) ** (-0.25) * self.wind_v_sequence[self.current_step]
        
        return self.wind_v
    
    def step(self, action):
        # Unpack PID parameters (obtained from action)
        control_compensation = action[0]
        
        # Get wind disturbance
        wind_v = self._get_wind_disturbance()
        #wind_v *= 20
        #print('wind_v:', wind_v)
        
        # Calculate sideslip angle disturbance
        Dw = np.array([
            wind_v * self.CY_beta * np.cos(self.psi)/(self.m * self.V_trim**2),  # beta_with_wind
            wind_v * self.qSb * self.Cl_beta * np.cos(self.psi) / (self.Ixx * self.V_trim),   # p_with_wind
            wind_v * self.qSb * self.Cn_beta * np.cos(self.psi) / (self.Izz * self.V_trim),   # r_with_wind
            0.0,   # phi
            0.0    # psi
            ])
        #self.beta += np.arcsin(wind_v / self.V_trim)
        #self.beta = np.clip(self.beta, -np.deg2rad(5), np.deg2rad(5))  # rad
        
        # Construct state vector
        x = np.array([
            self.beta,       
            self.p,          
            self.r,          
            self.phi,        # phi (roll angle) rad
            self.psi         
        ])

        # Calculate current heading and error
        heading = self.psi - self.beta
        heading_error = np.deg2rad(np.rad2deg(self.psi) - np.rad2deg(self.beta))    # rad
        
        # States include: heading error, yaw angle, sideslip angle, roll angle, desired roll angle
        state = np.array([
            self.heading_error,       
            self.psi,          
            self.beta,          
            self.phi,        # phi (roll angle) rad
            self.phi_target, 
            self.p,
            self.r         
        ])
        
        # Calculate control output (using current PID parameters)
        control_action, target_roll = self.pid_controller.compute_control(
                current_roll=self.phi,
                current_yaw=self.psi,
                current_beta=self.beta,
                dt=self.dt,
                target_heading=self.target_heading
            )
        #print(' ')
        #print('PID control output:', np.rad2deg(control_action), 'degrees')
        
        # Add wind disturbance sideslip compensation to roll angle control output
        control_action += control_compensation
        #control_action = control_compensation
        control_action = np.clip(control_action, -np.deg2rad(10), np.deg2rad(10))  # rad
        #print('Compensation amount:', np.rad2deg(control_compensation), 'Total control output:', np.rad2deg(control_action))

        # Apply control input
        u = np.array([
            control_action,  # δe_diff 
            0.0              # Thrust remains unchanged
        ])
        
        # Get linearized model
        A_hrz, B_hrz = self.A_hrz, self.B_hrz
        
        # Calculate state derivative
        x_dot = A_hrz @ x + B_hrz @ u + Dw
        
        # Update state
        x = x + x_dot * self.dt
        self.beta = self._normalize_angle(x[0])
        self.phi = self._normalize_angle(x[3])
        self.psi = self._normalize_angle(x[4])  # rad
        self.p, self.r = self._normalize_angle(x[1]), self._normalize_angle(x[2])
        self.phi_target = target_roll  # rad (desired from previous step)
        
        # Calculate new heading and error
        new_heading = self.psi - self.beta  # rad
        new_heading_error = np.deg2rad(np.rad2deg(self.psi) - np.rad2deg(self.beta))  # rad
        
        # Record data for performance evaluation
        self.heading_history.append(np.rad2deg(new_heading))  # deg
        self.control_history.append(np.rad2deg(control_action))  # deg
        self.time_history.append(self.current_step * self.dt)
        
        # Calculate reward
        reward = self._calculate_reward(heading_error, new_heading_error, control_action, self.beta)
        #print('')
        #print('Total reward for this step:', reward)
        #print('')
        
        self.current_step += 1
        
        # Check if terminated
        done = self.current_step >= self.steps or np.abs(self.beta) >= np.deg2rad(10) or np.abs(self.psi) >= np.deg2rad(60)
        
        # New state
        next_state = np.array([
            new_heading_error,
            self.psi,
            self.beta,
            self.phi,
            self.phi_target,    # This desired roll angle is from the previous state
            self.p,
            self.r
        ])
        
        truncated = False
        
        if self.generate_offline_data:
            return next_state, reward, done, truncated, {
                'state': state,
                'action': control_action,
                'control_compensation': action,  # action here is the compensation amount
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'target_heading': np.rad2deg(self.target_heading)
            }
        else:
            return next_state, reward, done, truncated, {
                'heading_error': np.rad2deg(new_heading_error),
                'control_action': control_action,
                'pid_params': action
            }
    
    def _calculate_reward(self, error, new_error, control_action, beta):
        # Weight parameter configuration
        WEIGHTS = {
            'tracking_error': 7.0,          # Base tracking error weight
            'integral_error': 0.2,          # Cumulative error weight
            'error_change_rate': 7.5,      # Error reduction reward weight
            'beta_error': 3.0,              # Sideslip angle error penalty
            'beta_integral': 0.3,           # Sideslip angle cumulative error penalty
            'beta_reduction': 3.0,          # Sideslip angle reduction reward
            'steady_state_hold': 10.0,
            'mean_steady_state': 2.0,
            'control_smoothness': 0.1,     # Control smoothness reward
            'roll_smoothness': 0.2,          # Roll angle smoothness reward
            'yaw_smoothness': 0.1,           # Yaw angle smoothness reward
            'rise_time_bonus': {            # Rise time reward configuration
            'fast': 600,         # Reward for <3 seconds
            'medium': 400,       # Reward for 3-5 seconds
            'slow': 0          # No reward for >5 seconds
            }
        }
        
        reward = 0.0 
        
        # 1. Base tracking error penalty (Kp)
        tracking_error = np.abs(new_error)
        # Normalization
        tracking_error_normalized = tracking_error / (2 * np.pi)
        
        if tracking_error_normalized > self.target_heading * 0.05:
            reward = -WEIGHTS['tracking_error'] * tracking_error_normalized

        # 2. Cumulative error penalty (Ki)
        self.error_integral += np.abs(new_error)
        # Anti-windup
        self.error_integral = np.clip(self.error_integral, 0.0, np.deg2rad(30))
        # Normalization
        error_integral_normalized = self.error_integral / np.deg2rad(30)
        
        reward -= WEIGHTS['integral_error'] * error_integral_normalized
        
        # 3. Error reduction reward (Kd)
        error_change_rate = np.rad2deg(error - new_error)  # Error reduction rate for this step
        reward += WEIGHTS['error_change_rate'] * error_change_rate
    
        # 4. Rise time bonus
        if not self.first_cross_flag and np.abs(new_error) <= 0.1 * np.abs(self.target_heading):  # First time reaching 90% of target
            rise_time = self.current_step * self.dt
            self.first_cross_flag = True
            if rise_time <= 3.0:
                reward += WEIGHTS['rise_time_bonus']['fast']
            elif rise_time <= 5.0:
                reward += WEIGHTS['rise_time_bonus']['medium']
            else:
                reward += WEIGHTS['rise_time_bonus']['slow']
        
        # 5. Steady state hold reward (rarely triggered)
        if len(self.error_history) > 0 and len(self.error_history) % 500 == 0:
            max_error = np.max(self.error_history[-500:])  * np.pi * 2
            mean_error = np.mean(np.abs(self.error_history))* np.pi * 2
            # Calculate max error margin percentage from 5% error tolerance - larger means smaller steady-state error and less fluctuation
            max_error_margin_percent = 100 * (0.05 * self.target_heading - max_error) / (0.05 * self.target_heading) 
            reward += max(WEIGHTS['steady_state_hold'] * max_error_margin_percent, 0.0)    
            # Calculate mean error margin percentage from 20% error tolerance - larger means closer to steady state
            mean_error_margin_percent = 100 * (0.05 * self.target_heading - mean_error) / (0.05 * self.target_heading) 
            reward += max(WEIGHTS['mean_steady_state'] * mean_error_margin_percent, 0)     
                
        # 6. Control smoothness reward
        # Normalize control change
        control_change_normalized = np.abs(control_action - self.prev_control_action) / np.deg2rad(20) # Max control change 20 degrees
        if control_change_normalized <= 0.25:  # Step control change <5 degrees
            reward += WEIGHTS['control_smoothness'] * control_change_normalized
        else:
            reward -= WEIGHTS['control_smoothness'] * control_change_normalized
            
        # 7. Roll angle smoothness reward
        roll_change_normalized = np.rad2deg(np.abs(self.phi - self.prev_phi))  # No normalization needed due to small values
        reward += WEIGHTS['roll_smoothness'] * max(0.4 - roll_change_normalized, 0.0) # Change <0.4 degrees - smaller changes get more reward, no reward otherwise
        
        # 8. Yaw angle smoothness reward
        yaw_change_normalized = np.rad2deg(np.abs(self.psi - self.prev_psi))
        reward += WEIGHTS['yaw_smoothness'] * max(0.3 - yaw_change_normalized, 0.0)
        
        # 9. Sideslip angle deviation penalty (Kp)
        beta_error_normalized = np.abs(beta) / np.deg2rad(10)
        reward -= (WEIGHTS['beta_error'] * beta_error_normalized) **2 
        
        # 10. Sideslip angle error reduction reward (Kd)
        beta_reduction = np.rad2deg(np.abs(self.prev_beta) - np.abs(beta))
        reward += WEIGHTS['beta_reduction'] * beta_reduction
        
        # Save state information
        self.prev_beta = beta
        self.prev_control_action = control_action
        self.prev_phi = self.phi
        self.prev_psi = self.psi
        self.prev_error = error
        self.error_history.append(tracking_error)
        
        return reward