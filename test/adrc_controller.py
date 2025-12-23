import numpy as np

class ADRCController:
    """
    Active Disturbance Rejection Controller (ADRC) - Adapted for Fixed-Wing UAV Lateral Heading Control
    Core Function: Output roll angle control command based on heading error, sideslip angle, roll angle and other states
    Adaptation Features:
    1. Control command amplitude limiting (-10°~10°), complying with UAV actuator constraints
    2. Extended State Observer (ESO) to estimate total disturbances (including wind disturbances and model errors)
    3. Nonlinear State Error Feedback (NLSEF) to achieve overshoot-free tracking
    """
    def __init__(self, V_trim, h0=45, w0=0.9, b0=2.0):
        """
        Initialize the ADRC Controller
        :param V_trim: UAV trim velocity (m/s) - used for dynamics adaptation
        :param h0: Tracking factor - larger value means faster tracking (default: 100)
        :param w0: Bandwidth - larger value means faster response (default: 10)
        :param b0: Control gain - adapted to control command amplitude (default: 1.0)
        """
        # Basic parameters
        self.V_trim = V_trim
        self.h0 = h0
        self.w0 = w0
        self.b0 = b0
        
        # Extended State Observer (ESO) parameters (classic parameter configuration)
        self.beta01 = 3 * self.w0
        self.beta02 = 3 * self.w0**2
        self.beta03 = self.w0**3
        
        # Nonlinear State Error Feedback (NLSEF) parameters
        self.alpha1 = 0.1   # First-order nonlinear coefficient of error
        self.alpha2 = 0.1   # Second-order nonlinear coefficient of error
        self.delta = 0.3    # Linear segment threshold (to avoid chattering)
        
        # State initialization (ESO observed states)
        self.z1 = 0.0  # Observed heading error
        self.z2 = 0.0  # Observed differential of heading error
        self.z3 = 0.0  # Observed total disturbance (wind disturbance + model error)
        
        # Control output buffer
        self.u = 0.0   # Previous time step control command
        self.e = 0.0   # Previous time step error

    def fal(self, e, alpha, delta):
        """
        Nonlinear function fal - Core nonlinear component of ADRC
        Function: Linear for small errors, nonlinear for large errors, balancing accuracy and rapidity
        :param e: Error value
        :param alpha: Nonlinear exponent (0<alpha<1)
        :param delta: Linear segment threshold
        :return: Nonlinearly processed error
        """
        if abs(e) > delta:
            return np.sign(e) * abs(e)**alpha
        else:
            return e / (delta**(1 - alpha))

    def eso(self, y, u, dt):
        """
        Extended State Observer (ESO) - Core module of ADRC
        Function: Real-time observation of system states and total disturbances
        :param y: Actual output (heading error in this case)
        :param u: Control input (previous time step roll angle control command)
        :param dt: Simulation time step (s)
        """
        # Calculate observation error
        self.e = self.z1 - y
        
        # Update observed states
        self.z1 += dt * (self.z2 - self.beta01 * self.e)
        self.z2 += dt * (self.z3 - self.beta02 * self.fal(self.e, self.alpha1, self.delta) + self.b0 * u)
        self.z3 += dt * (-self.beta03 * self.fal(self.e, self.alpha2, self.delta))

    def nlsef(self, ref_error, dt):
        """
        Linear State Error Feedback (LSEF) - Replaces the original NLSEF for ADRC simplification
        Function: Calculate control command (linear feedback) based on observed states and reference error
        :param ref_error: Reference error (target heading error, fixed at 0)
        :param dt: Simulation time step (s)
        :return: Roll angle control command (rad)
        """
        # Calculate state errors
        e1 = ref_error - self.z1  # Deviation between observed heading error and target
        e2 = 0 - self.z2          # Deviation of heading error differential (target differential is 0)

        # === Key Modification: Replace nonlinear fal function with linear feedback ===
        # Set linear feedback gains (can be manually adjusted or using pole placement)
        k1 = 6.0   # Proportional gain for e1 (corresponding to the role of original h0 and alpha1)
        k2 = 15.0  # Proportional gain for e2 (corresponding to the role of original alpha2)

        # Calculate original control command with linear feedback
        u0 = k1 * e1 + k2 * e2

        # Disturbance compensation + control command limiting (-10°~10°, converted to radians)
        self.u = (u0 - self.z3) / self.b0
        self.u = np.clip(self.u, -np.deg2rad(10), np.deg2rad(10))

        return self.u

    def compute_control(self, current_roll, current_yaw, current_beta, dt, target_heading):
        """
        External Interface: Calculate roll angle control command
        Input and output are fully compatible with PID controller, can be directly replaced
        :param current_roll: Current roll angle (rad)
        :param current_yaw: Current yaw angle (rad)
        :param current_beta: Current sideslip angle (rad)
        :param dt: Simulation time step (s)
        :param target_heading: Target heading (rad)
        :return: control_action (rad), target_roll (rad)
                 - control_action: Roll angle control command
                 - target_roll: Target roll angle (compatible with PID return format, fixed at 0)
        """
        # Calculate current heading and heading error
        current_heading = current_yaw - current_beta
        heading_error = self._normalize_angle(target_heading - current_heading)
        
        # 1. Update Extended State Observer (observe error and disturbance)
        self.eso(heading_error, self.u, dt)
        
        # 2. Calculate nonlinear state error feedback control command
        control_action = self.nlsef(0, dt)  # Target error is 0
        
        # 3. Compatible with PID return format (second value is target roll angle, set to 0 for ADRC temporarily)
        target_roll = 0.0
        
        return control_action, target_roll

    @staticmethod
    def _normalize_angle(angle):
        """
        Normalize angle to [-π, π] (adapted for heading calculation)
        :param angle: Original angle (rad)
        :return: Normalized angle (rad)
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi