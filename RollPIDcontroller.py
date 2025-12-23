import numpy as np

# PID Controller Implementation
class DualPIDController:
    def __init__(self, V_trim):
        # Roll Angle PID Controller Parameters
        self.roll_pid = {
            'kp': -1.4,
            'ki': -0.2,
            'kd': -0.0,
            'target': 0.0,  # Target roll angle (radians)
            'output_limits': (-np.deg2rad(10), np.deg2rad(10)),  # Output limits (radians)
            'integral': 0,
            'prev_error': 0
        }
        
        # Heading Angle PID Controller Parameters
        self.heading_pid = {
            'kp': 2.8,     # Proportional gain, initial value needs tuning
            'ki': 0.01,
            'kd': 0.06,    # Derivative gain, initial value needs tuning
            'output_limits': (-np.deg2rad(20), np.deg2rad(20)),  # Output limits (radians)
            'integral': 0,
            'prev_error': 0
        }
        
        # Sideslip Angle PID Controller Parameters
        self.beta_pid = {
            'kp': 0.0,     # Proportional gain, initial value needs tuning
            'ki': 0.0,
            'kd': 0.0,    # Derivative gain, initial value needs tuning
            'target': 0.0,  # Target sideslip angle (radians)
            'output_limits': (-np.deg2rad(5), np.deg2rad(5)),  # Output limits (radians)
            'integral': 0,
            'prev_error': 0
        }
        
        self.V_trim = V_trim
    
    def compute_roll_command(self, current_yaw, target_heading, current_beta, dt):
        """
        Calculate the desired roll angle using PID control
        
        :param current_yaw: Current yaw angle (radians)
        :param target_heading: Target heading angle (radians)
        :param current_beta: Current sideslip angle (radians)
        :param dt: Time step (seconds)
        :return: Desired roll angle (radians)
        """
        # Calculate heading angle error
        current_heading = current_yaw - current_beta
        heading_error = self.normalize_angle(target_heading - current_heading)
        #print('Heading angle error:', np.rad2deg(heading_error))
        
        # Calculate desired roll angle using heading angle PID
        p_term = self.heading_pid['kp'] * heading_error
        self.heading_pid['integral'] += heading_error * dt
        self.heading_pid['integral'] = np.clip(self.heading_pid['integral'], -np.deg2rad(30), np.deg2rad(30))
        # Integral clamping (prevent integral windup)
        i_term = self.heading_pid['ki'] * self.heading_pid['integral']
        d_term = self.heading_pid['kd'] * (heading_error - self.heading_pid['prev_error']) / dt if dt > 0 else 0
        self.heading_pid['prev_error'] = heading_error
        
        heading_pid_control = p_term + i_term + d_term
        #print('Unclamped desired roll angle:', np.rad2deg(heading_pid_control))
        roll_command = np.clip(heading_pid_control, self.heading_pid['output_limits'][0], self.heading_pid['output_limits'][1])
        
        #print('Desired roll angle:', np.rad2deg(roll_command))
        
        return roll_command
    
    def normalize_angle(self, angle):
        """
        Normalize the angle to the range [-pi, pi]
        
        :param angle: Input angle (radians)
        :return: Normalized angle (radians)
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle
    
    def compute_control(self, current_roll, current_yaw, current_beta, dt, target_heading=None):
        """
        Compute the combined control output for roll and sideslip angle
        
        :param current_roll: Current roll angle (radians)
        :param current_yaw: Current yaw angle (radians)
        :param current_beta: Current sideslip angle (radians)
        :param dt: Time step (seconds)
        :param target_heading: Target heading angle (radians, optional)
        :return: Combined control output (radians), desired roll angle (radians)
        """
        desired_roll = self.compute_roll_command(current_yaw, target_heading, current_beta, dt)
        self.roll_pid['target'] = desired_roll
        
        # Calculate inner loop roll angle PID control
        roll_error = self.roll_pid['target'] - current_roll
        p_term = self.roll_pid['kp'] * roll_error
        self.roll_pid['integral'] += roll_error * dt
        # Integral clamping (prevent integral windup)
        i_term = self.roll_pid['ki'] * self.roll_pid['integral']
        d_term = self.roll_pid['kd'] * (roll_error - self.roll_pid['prev_error']) / dt if dt > 0 else 0
        self.roll_pid['prev_error'] = roll_error
        
        roll_pid_control = p_term + i_term + d_term
        roll_pid_control = np.clip(roll_pid_control, self.roll_pid['output_limits'][0], self.roll_pid['output_limits'][1])

        # Calculate sideslip angle PID control
        beta_error = self.beta_pid['target'] - current_beta
        p_term_beta = self.beta_pid['kp'] * beta_error
        self.beta_pid['integral'] += beta_error * dt
        # Integral clamping (prevent integral windup)
        i_term_beta = self.beta_pid['ki'] * self.beta_pid['integral']
        d_term_beta = self.beta_pid['kd'] * (beta_error - self.beta_pid['prev_error']) / dt if dt > 0 else 0
        self.beta_pid['prev_error'] = beta_error
        
        beta_pid_control = p_term_beta + i_term_beta + d_term_beta
        beta_pid_control = np.clip(beta_pid_control, self.beta_pid['output_limits'][0], self.beta_pid['output_limits'][1])
        
        # Combined control output: simply sum the two control outputs here, 
        # more complex combination strategies can be designed as needed
        combined_control = roll_pid_control + beta_pid_control
        combined_control = np.clip(combined_control, -np.deg2rad(10), np.deg2rad(10))  # Final output clamping
        #combined_control = 0.0  # For testing
        
        return combined_control, desired_roll