import numpy as np
from collections import deque
from scipy import signal
from RollPIDcontroller import DualPIDController
import utils.utils_ry as utils_ry
import gymnasium as gym
from gymnasium import spaces
    
class WindCompEnv(gym.Env):
    def __init__(self, model, trim_calc, wind_field, target_heading, dt=0.01, generate_offline_data=False, wind_intensity=1.0):
        super(WindCompEnv, self).__init__()
        
        # Save original parameters
        self.model = model
        self.trim_calc = trim_calc
        self.wind_field = wind_field
        self.target_heading = target_heading
        self.dt = dt
        self.generate_offline_data = generate_offline_data

        self.wind_intensity = wind_intensity

        self._seed_value = None
        self.np_random = None
        
        self.sine_amplitude_deg = 10  # Sine wave amplitude (degrees): ±10 degrees fluctuation around initial value
        self.sine_frequency_hz = 0.1  # Sine wave frequency (Hz): 10 seconds per cycle
        self.sine_offset_rad = np.deg2rad(target_heading)  # Initial value = sine wave offset (converted to radians)
        self.current_target_heading = self.sine_offset_rad  # Current target heading (initial = initial value)

        # Load trim state
        u_rel_trim, w_rel_trim, _, _, T_trim = trim_calc.find_trim()
        self.V_trim = np.sqrt(u_rel_trim**2 + w_rel_trim**2)
        self.A_hrz, self.B_hrz = self.trim_calc.linearize_trim()
        self.CY_beta, self.m, self.Cl_beta, self.Cn_beta = model.CY_beta, model.m, model.Cl_beta, model.Cn_beta
        self.qSb = 0.5 * model.rho * self.V_trim**2 * model.S * model.b
        self.Ixx, self.Izz = model.Ixx, model.Izz
        
        # Define action space (aileron differential compensation)
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
    
    def seed(self, seed=None):
        """
        Set random seed
        """
        self._seed_value = seed
        if seed is not None:
            np.random.seed(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        # Set random seed
        if seed is not None:
            super().reset(seed=seed)  # Call parent class reset method to set seed
            self._seed_value = seed
            np.random.seed(seed)
        elif self._seed_value is not None:
            # If seed was set before, continue using it
            np.random.seed(self._seed_value)
            
        # Initial states
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
        #np.random.seed(42)
        current_seed = self._seed_value if self._seed_value is not None else 42
        #print("Environment reset, using random seed:", current_seed)
        np.random.seed(current_seed)  # ✅ Use current environment seed
        self.steps = int(150.0 / self.dt)  # 20 seconds simulation
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
        self.target_heading_history = []
        self.heading_history = []
        self.control_history = []
        self.time_history = []
        
        self.current_target_heading = self.sine_offset_rad
        self.target_heading_history.append(self.target_heading)  # Record initial value (degrees)

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
        ], dtype=np.float32)
        
        return observation, {}
    
    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    def _update_sine_target(self):
        """
        Generate time-varying sinusoidal target heading based on current simulation time:
        current_target = initial value (offset) + amplitude * sin(2π*frequency*time)
        """
        current_time = self.current_step * self.dt  # Current simulation time (seconds)
        # 1. Calculate sinusoidal fluctuation (convert degrees to radians)
        sine_fluctuation = np.deg2rad(self.sine_amplitude_deg) * np.sin(
            2 * np.pi * self.sine_frequency_hz * current_time
        )
        # 2. Superimpose initial value (offset) to get current target heading
        self.current_target_heading = self.sine_offset_rad + sine_fluctuation
        # 3. Normalize angle (ensure it's within [-π, π] to avoid abnormal error calculation)
        self.current_target_heading = self._normalize_angle(self.current_target_heading)
        # 4. Record current target (convert to degrees for subsequent analysis)
        self.target_heading_history.append(np.rad2deg(self.current_target_heading))
    
    def _get_wind_disturbance(self):
        """
        Get lateral wind disturbance based on Dryden turbulence model
        """
        self.wind_v = self.wind_intensity * self.mean_wind_speed * (self.h_ref / 10) ** (-0.25) * self.wind_v_sequence[self.current_step]
        
        return self.wind_v
    
    def step(self, action):
        self._update_sine_target()
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
        
        # State includes: heading error, yaw angle, sideslip angle, roll angle, desired roll angle
        state = np.array([
            self.heading_error,       
            self.psi,          
            self.beta,          
            self.phi,        # phi (roll angle) rad
            self.phi_target, 
            self.p,
            self.r         
        ])
        
        # Calculate control input (using current PID parameters)
        control_action, target_roll = self.pid_controller.compute_control(
                current_roll=self.phi,
                current_yaw=self.psi,
                current_beta=self.beta,
                dt=self.dt,
                target_heading=self.target_heading
            )
        #print(' ')
        #print('PID control input:', np.rad2deg(control_action), 'degrees')
        
        # Add wind disturbance sideslip angle compensation to roll angle control input
        control_action += control_compensation
        #control_action = control_compensation
        control_action = np.clip(control_action, -np.deg2rad(10), np.deg2rad(10))  # rad
        #print('Compensation:', np.rad2deg(control_compensation), 'Total control input:', np.rad2deg(control_action))

        # Apply control input
        u = np.array([
            control_action,  # δe_diff 
            0.0              # Thrust remains constant
        ])
        
        # Get linearized model
        A_hrz, B_hrz = self.A_hrz, self.B_hrz
        
        # Calculate state derivative
        x_dot = A_hrz @ x + B_hrz @ u + Dw
        #x_dot = A_hrz @ x + B_hrz @ u
        
        # Update state
        x = x + x_dot * self.dt
        self.beta = self._normalize_angle(x[0])
        self.phi = self._normalize_angle(x[3])
        self.psi = self._normalize_angle(x[4])  # rad
        self.p, self.r = self._normalize_angle(x[1]), self._normalize_angle(x[2])
        self.phi_target = target_roll  # rad Desired value from previous step
        
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
        
        # Check if episode is done
        done = self.current_step >= self.steps or np.abs(self.beta) >= np.deg2rad(10) or np.abs(self.psi) >= np.deg2rad(60)
        
        # New state
        next_state = np.array([
            new_heading_error,
            self.psi,
            self.beta,
            self.phi,
            self.phi_target,    # This desired roll angle is from the previous time step's state
            self.p,
            self.r
        ], dtype=np.float32)
        
        truncated = False
        
        if self.generate_offline_data:
            return next_state, reward, done, truncated, {
                'state': state,
                'action': control_action,
                'control_compensation': action,  # Action here is the compensation value
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
            'beta_integral': 0.3,           # Cumulative sideslip angle error penalty
            'beta_reduction': 3.0,          # Sideslip angle reduction reward
            'steady_state_hold': 10.0,
            'mean_steady_state': 2.0,
            'control_smoothness': 0.1,     # Control smoothness reward
            'roll_smoothness': 0.2,          # Roll angle smoothness reward
            'yaw_smoothness': 0.1,           # Yaw angle smoothness reward
            'rise_time_bonus': {            # Rise time reward configuration
            'fast': 600,         # Reward 100 for <3 seconds
            'medium': 400,       # Reward 60 for 3-5 seconds
            'slow': 0          # No reward for 5-10 seconds
            }
        }
        
        reward = 0.0 
        
        # 1. Base tracking error penalty (Kp)
        tracking_error = np.abs(new_error)
        # Normalization
        tracking_error_normalized = tracking_error / (2 * np.pi)
        
        if tracking_error_normalized > self.target_heading * 0.05:
            reward = -WEIGHTS['tracking_error'] * tracking_error_normalized
        #print('Tracking error penalty:', -WEIGHTS['tracking_error'] * tracking_error_normalized)
        
        # 2. Cumulative error penalty (Ki)
        self.error_integral += np.abs(new_error)
        # Anti-windup
        self.error_integral = np.clip(self.error_integral, 0.0, np.deg2rad(30))
        # Normalization
        error_integral_normalized = self.error_integral / np.deg2rad(30)
        
        reward -= WEIGHTS['integral_error'] * error_integral_normalized
        #print('Cumulative error penalty:', -WEIGHTS['integral_error'] * error_integral_normalized)
        
        # 3. Error reduction reward (Kd)
        error_change_rate = np.rad2deg(error - new_error)  # Error reduction rate for this step
        #prev_error_change_rate = (self.prev_error - error) / self.dt  # Error reduction rate for previous step
        reward += WEIGHTS['error_change_rate'] * error_change_rate
        #print('Error reduction reward:', WEIGHTS['error_change_rate'] * error_change_rate)
        
        # 4. Rise time reward
        if not self.first_cross_flag and np.abs(new_error) <= 0.1 * np.abs(self.target_heading):  # First time reaching 90% of desired value
            rise_time = self.current_step * self.dt
            self.first_cross_flag = True
            if rise_time <= 3.0:
                reward += WEIGHTS['rise_time_bonus']['fast']
                #print('Rise time reward for <3s:', WEIGHTS['rise_time_bonus']['fast'])
            elif rise_time <= 5.0:
                reward += WEIGHTS['rise_time_bonus']['medium']
                #print('Rise time reward for 3~5s:', WEIGHTS['rise_time_bonus']['medium'])
            else:
                reward += WEIGHTS['rise_time_bonus']['slow']
        
        # 5. Steady state hold reward
        if len(self.error_history) > 0 and len(self.error_history) % 500 == 0:
            max_error = np.max(self.error_history[-500:])  * np.pi * 2
            mean_error = np.mean(np.abs(self.error_history))* np.pi * 2
            # Calculate the percentage of maximum error margin from 5% error - larger value means smaller steady state error and smaller fluctuation range
            max_error_margin_percent = 100 * (0.05 * self.target_heading - max_error) / (0.05 * self.target_heading) 
            reward += max(WEIGHTS['steady_state_hold'] * max_error_margin_percent, 0.0)   
            #print('Steady state hold reward:', max(WEIGHTS['steady_state_hold'] * max_error_margin_percent, 0.0))  
            # Calculate the percentage of mean error margin from 20% error - larger value means mean error is closer to steady state
            mean_error_margin_percent = 100 * (0.05 * self.target_heading - mean_error) / (0.05 * self.target_heading) 
            reward += max(WEIGHTS['mean_steady_state'] * mean_error_margin_percent, 0)   
            #print('Mean error reward:', max(WEIGHTS['mean_steady_state'] * mean_error_margin_percent, 0))  
                
        # 6. Control smoothness penalty
        # Control change normalization
        control_change_normalized = np.abs(control_action - self.prev_control_action) / np.deg2rad(20) # Maximum control input change is 20 degrees
        if control_change_normalized <= 0.25:  # Control change is less than 5 degrees in one step
            reward += WEIGHTS['control_smoothness'] * control_change_normalized
            #print('Control smoothness reward:', WEIGHTS['control_smoothness'] * control_change_normalized) 
        else:
            reward -= WEIGHTS['control_smoothness'] * control_change_normalized
            #print('Control non-smoothness penalty:', -WEIGHTS['control_smoothness'] * control_change_normalized) 
            
        # 7. Roll angle smoothness reward
        roll_change_normalized = np.rad2deg(np.abs(self.phi - self.prev_phi))  # No normalization needed since it's very small
        reward += WEIGHTS['roll_smoothness'] * max(0.4 - roll_change_normalized, 0.0) # Reward more for changes less than 0.4 degrees, no reward otherwise
        #print('Roll angle smoothness reward:', WEIGHTS['roll_smoothness'] * max(0.4 - roll_change_normalized, 0.0))
        
        # 8. Yaw angle smoothness reward
        yaw_change_normalized = np.rad2deg(np.abs(self.psi - self.prev_psi))
        reward += WEIGHTS['yaw_smoothness'] * max(0.3 - yaw_change_normalized, 0.0)
        #print('Yaw angle smoothness reward:', WEIGHTS['yaw_smoothness'] * max(0.3 - yaw_change_normalized, 0.0))
        
        # 9. Sideslip angle deviation penalty (Kp)
        beta_error_normalized = np.abs(beta) / np.deg2rad(10)
        reward -= (WEIGHTS['beta_error'] * beta_error_normalized) **2 
        #print('Sideslip angle deviation penalty:', -WEIGHTS['beta_error'] * beta_error_normalized)
        
        # 10. Cumulative sideslip angle error penalty (Ki)
        #self.beta_integral += np.abs(beta)
        #self.beta_integral = np.clip(self.beta_integral, 0, np.deg2rad(30))
        #beta_integral_normalized = self.beta_integral / np.rad2deg(30)
        #reward -= WEIGHTS['beta_integral'] * beta_integral_normalized
        #print('Cumulative sideslip angle error penalty:', -WEIGHTS['beta_integral'] * beta_integral_normalized)
        
        # 11. Sideslip angle error reduction reward (Kd)
        beta_reduction = np.rad2deg(np.abs(self.prev_beta) - np.abs(beta))
        reward += WEIGHTS['beta_reduction'] * beta_reduction
        #print('Sideslip angle error reduction reward:', WEIGHTS['beta_reduction'] * beta_reduction)
        
        # Save state information
        self.prev_beta = beta
        self.prev_control_action = control_action
        self.prev_phi = self.phi
        self.prev_psi = self.psi
        self.prev_error = error
        #self.error_history.append(tracking_error)
        
        return reward