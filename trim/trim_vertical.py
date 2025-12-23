import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import approx_fprime
import control as ctrl
import sympy as sp
from scipy import signal

class FixedWingDynamics:
    def __init__(self, wind_field):
        # Aircraft parameters (example for small UAV)
        self.m = 0.0  # Mass (kg)
        self.Ixx, self.Iyy, self.Izz = 0.0, 0.0, 0.0  # Moment of inertia (kg·m²)
        self.Ixy, self.Iyz, self.Ixz = 0.0, 0.0, 0.0
        self.Ix, self.Iy, self.Iz = 0.0, 0.0, 0.0
        self.S = 0.0   # Wing area (m²)
        self.b = 0.0   # Wingspan (m)
        self.c = 0.0   # Mean aerodynamic chord (m)
        self.h_ac = 0.0 # Aerodynamic center position
        self.h_cg = 0.0 # Center of gravity position
        self.rho = 0.0  # Air density (kg/m³)
        self.g = 0.0  # Gravitational acceleration (m/s²)
        self.max_throttle = 0.0  # Maximum thrust (N)
        self.max_elevator = 0.0  # Maximum deflection angle (rad)
        
        self.wind_field = wind_field if wind_field else WindField()
        
        # Longitudinal aerodynamic coefficients
        self.CL0 = 0.0 # Zero-lift coefficient
        self.CL_alpha = 0.0  # Lift curve slope (1/°)
        self.CD0 = 0.0      # Zero-drag coefficient
        self.CD_alpha = 0.0     # Drag variation with angle of attack
        self.Cm0 = 0.0         # Pitching moment at zero angle of attack
        self.Cm_alpha = 0.0 # Pitching moment derivative (deg-1)
        self.Cm_q = 0.0       # Pitch damping derivative 
        self.Cm_delta_e_sync = 0.0 # Elevator effectiveness (synchronous)
        self.Cl_delta_e_sync = 0.0 # Elevator lift effectiveness (synchronous)
        
        # Lateral-directional aerodynamic coefficients (aileron and rudder disabled, coupling still exists)
        self.CY_beta = 0.0     # Side force derivative
        self.Cl_beta = 0.0     # Roll static stability
        self.Cl_p = 0.0        # Roll damping
        # delta_e_diff = right - left, positive for right elevator down deflection
        self.Cl_delta_e_diff = 0.0
        self.Cl_r = 0.0         # Yaw damping (roll due to yaw rate)
        self.Cl_delta_r = 0.0   
        
        self.Cn_delta_e_diff = 0.0 # Yaw effectiveness of differential elevator
        self.Cn_beta = 0.0      # Directional static stability
        self.Cn_r = 0.0         # Yaw damping
        self.Cn_p = 0.0         # Weak cross-coupling yaw due to roll
        self.Cn_delta_r = 0.0
        
        self.CY_delta_r = 0.0    # Rudder effectiveness (reasonable range: 0.2~0.5/rad)
        self.CY_p = 0.0         # Side force derivative with roll rate (reasonable range: -0.05 ~ -0.2)
        self.CY_r = 0.0         # Side force derivative with yaw rate (reasonable range: 0.1~0.3)

    def compute_forces(self, x, u, wind_body):
        # Calculate aerodynamic forces and moments (differential control)
        u_b, v_b, w_b = x[0], x[1], x[2]  # Body-axis velocities
        p, q, r = x[3], x[4], x[5]
        phi, theta, psi = x[6], x[7], x[8]
        delta_e_sync, delta_e_diff, delta_T = u  # Synchronous/differential elevator + throttle
        
        # Calculate relative wind velocity (body frame)
        u_rel = u_b - wind_body[0]
        v_rel = v_b - wind_body[1]
        w_rel = w_b - wind_body[2]
        
        # Calculate airspeed, angle of attack, sideslip angle
        V = np.sqrt(u_rel**2 + v_rel**2 + w_rel**2)
        alpha = np.arctan2(w_rel, u_rel) if V > 1e-3 else 0
        alpha = np.clip(alpha, np.deg2rad(-5.0), np.deg2rad(5.0)) 
        
        beta = np.arcsin(v_rel/V) if V > 1e-3 else 0
        
        # Dynamic pressure
        q_dynamic = 0.5 * self.rho * V**2 * self.S
        
        # ------Longitudinal aerodynamic coefficients-------
        CL = self.CL0 + self.CL_alpha * np.rad2deg(alpha) + self.Cl_delta_e_sync * np.rad2deg(delta_e_sync)
        CD = self.CD0 + self.CD_alpha * np.rad2deg(alpha)**2
        Cm_sync = (self.Cm0 + self.Cm_alpha * np.rad2deg(alpha) + 
                  self.Cm_q * np.rad2deg((q * self.c / (2 * V))) + self.Cm_delta_e_sync * np.rad2deg(delta_e_sync))
        
        # Lateral aerodynamic coefficients
        CY = (self.CY_beta * np.rad2deg(beta) + 
            self.CY_delta_r * 0.0 +  # Rudder deflection
            self.CY_p * (p * self.b / (2 * V)) + 
            self.CY_r * (r * self.b / (2 * V)))
        Cl_diff = (self.Cl_beta * np.rad2deg(beta) + self.Cl_delta_r * 0.0  + 
            self.Cl_delta_e_diff * delta_e_diff +  # Differential elevator as equivalent aileron
            self.Cl_p * (p*self.b)/(2*V) + self.Cl_r * (r*self.b) /(2*V))
        Cn_diff = (self.Cn_beta * np.rad2deg(beta) + 
            self.Cn_delta_e_diff * delta_e_diff +  # Differential elevator as equivalent aileron
            self.Cn_delta_r * 0.0 + self.Cn_r * (r*self.b) /(2*V)) + self.Cn_p * (p*self.b)/(2*V)
             
        # Lift (wind frame)
        Lift = q_dynamic *  CL
        
        # Side force (wind frame)
        Y = q_dynamic * CY
        
        # Drag (wind frame)
        D = q_dynamic * CD
        
        # Total forces in body frame (X,Y,Z)
        F_x = delta_T - self.m*self.g*np.sin(theta) - D*np.cos(alpha)*np.cos(beta) + Lift*np.sin(alpha) - Y*np.cos(alpha)*np.sin(beta)
        F_y = Y*np.cos(beta) + self.m*self.g*np.sin(phi)*np.cos(theta) - D*np.sin(beta)
        F_z = self.m*self.g*np.cos(phi)*np.cos(theta) - D*np.sin(alpha)*np.cos(beta) - Lift*np.cos(alpha) - Y*np.sin(alpha)*np.sin(beta)
        
        # Pitching moment (L,M,N)
        M = q_dynamic * self.c * Cm_sync
        
        # ------Lateral aerodynamic moments-------
        # Roll and yaw moments from differential elevator (equivalent aileron effect)
        L = q_dynamic * self.b * Cl_diff
        N = q_dynamic * self.b * Cn_diff
        
        return np.array([F_x, F_y, F_z, L, M, N])  # Simplified by ignoring side force

    def dynamics(self, t, x, u):
        
        phi, theta, psi = x[6], x[7], x[8]
        
        # Rotation matrix: NED to body frame
        R_ned_to_body = np.array([
            [np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), -np.sin(theta)],
            [np.sin(phi)*np.sin(theta)*np.cos(psi)-np.cos(phi)*np.sin(psi), 
             np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi), 
             np.sin(phi)*np.cos(theta)],
            [np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi), 
             np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi), 
             np.cos(phi)*np.cos(theta)]
        ])
        
        wind_ned = self.wind_field.get_wind(t, x[9:12])
        wind_body = R_ned_to_body @ wind_ned        # Transform to body frame
        
        # 6DOF nonlinear dynamic equations
        forces = self.compute_forces(x, u, wind_body)
        u_b, v_b, w_b = x[0], x[1], x[2]
        p, q, r = x[3], x[4], x[5]
        
        L, M, N = forces[3], forces[4], forces[5]
        
        # Translational motion
        du = (v_b*r - w_b*q) + forces[0]/self.m
        dv = (w_b*p - u_b*r) + forces[1]/self.m
        dw = (u_b*q - v_b*p) + forces[2]/self.m
        
        # Rotational motion
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = self.compute_coef()
        dp = (c1*r + c2*p)*q + c3*L + c4*N
        dq = c5*p*r + c6*(p**2 - r**2) + c7*M
        dr = (c8*p - c2*r)*q + c4*L + c9*N
        
        # Euler angle kinematics
        dphi = p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)
        dtheta = q*np.cos(phi) - r*np.sin(phi)
        dpsi = (q*np.sin(phi) + r*np.cos(phi)) / np.cos(theta)
        
        # Rotation matrix: body frame to NED frame
        R = np.array([
        [np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi)-np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi)],
        [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi)],
        [-np.sin(theta),            np.sin(phi)*np.cos(theta),                                     np.cos(phi)*np.cos(theta)]
        ])
        
        # Position increment
        dX_ned = R @ np.array([u_b, v_b, w_b])
        
        # Altitude constraint: stop descending when Z_down approaches 0
        
        if x[11] >= 0:  # Z_down >= 0 means ground contact
           x[11] = 0  # Clamp altitude
           dZ_down = max(0, dX_ned[2])  # Only allow ascending
           dX_ned = np.array([dX_ned[0], dX_ned[1], dZ_down])
        
        # Combine all derivatives (9 original states + 3 position states)
        return np.concatenate(([du, dv, dw, dp, dq, dr, dphi, dtheta, dpsi], dX_ned))
    
    def compute_coef(self):
        Ix, Iy, Iz = self.Ix, self.Iy, self.Iz
        Ixz = 0.0
        
        total = Ix*Iz - Ixz**2
        c1, c2 = ((Iy - Iz)*Iz - Ixz**2) / total, (Ix - Iy + Iz)*Ixz / total
        c3, c4 = Iz / total, Ixz / total
        c5, c6 = (Iz - Ix) / Iy, Ixz / Iy
        c7, c8 = 1 / Iy, (Ix*(Ix-Iy) + Ixz**2)/total
        c9 = Ix / total
        
        return c1, c2, c3, c4, c5, c6, c7, c8, c9

# ================== Wind Field Model Class ==================
class WindField:
    def __init__(self):
        self.wind_type = "constant"  # Options: "constant", "gust", "dryden"
        self.base_wind = np.array([0.0, 0.0, 0.0])  # Base wind [u, v, w] (m/s)
        self.gust_magnitude = 0.0  # Gust amplitude (m/s)
        self.turbulence_intensity = 0.0  # Turbulence intensity
        
    def get_wind(self, t, position):
        # Return wind vector in NED frame based on time and position
        if self.wind_type == "constant":
            return self.base_wind
            
        elif self.wind_type == "gust":
            # Apply step gust at 5 seconds
            if 5.0 < t < 8.0:
                return self.base_wind + np.array([self.gust_magnitude, 0])
            else:
                return self.base_wind
                
        elif self.wind_type == "dryden":
            # Simplified Dryden turbulence model (discrete implementation)
            tau = 2.0  # Time constant
            h = position[2]  # Altitude (positive downward in NED frame)
            scale = np.exp(-h/300)  # Altitude attenuation factor
            
            # Generate colored noise
            if not hasattr(self, 'last_wind'):
                self.last_wind = self.base_wind.copy()
                self.wind_noise = np.random.randn(3)
            
            self.wind_noise = 0.9 * self.wind_noise + 0.1 * np.random.randn(3)
            turbulence = scale * self.turbulence_intensity * self.wind_noise * 10
            
            return self.base_wind + turbulence
            
        return np.zeros(3)
    
class TrimCalculator:
    def __init__(self, model, wind_field=None):
        self.model = model
        self.wind_field = wind_field if wind_field else WindField()
        
    def find_trim(self, gamma=0, t_trim=0.0):
        # Calculate trim state (steady level flight)   
        
        wind_body = [0.0, 0.0, 0.0] # No wind at initial state
        
        from scipy.optimize import root

        def equations(vars):
            # Trim solution variables: alpha(u_rel, w_rel), theta, V, T
            u_rel, w_rel, delta_e_sync, delta_e_diff, T = vars
            
            # Angle constraints: delta_e_sync ∈ [-10°, 10°]
            delta_e_sync_min, delta_e_sync_max = np.deg2rad(-10), np.deg2rad(10)
            delta_e_diff_min, delta_e_diff_max = np.deg2rad(-10), np.deg2rad(10)
            delta_e_sync_constraint, delta_e_diff_constraint = 0.0, 0.0
            if delta_e_sync < delta_e_sync_min:
                delta_e_sync_constraint = (delta_e_sync - delta_e_sync_min) * 1e6  # Large penalty coefficient
            elif delta_e_sync > delta_e_sync_max:
                delta_e_sync_constraint = (delta_e_sync - delta_e_sync_max) * 1e6
            if delta_e_diff < delta_e_diff_min:
                delta_e_diff_constraint = (delta_e_diff - delta_e_diff_min) * 1e6  # Large penalty coefficient
            elif delta_e_diff > delta_e_diff_max:
                delta_e_diff_constraint = (delta_e_diff - delta_e_diff_max) * 1e6
    
            # Thrust constraint: T ∈ [0, 13] N
            T_min, T_max = 0, 13
            T_constraint = 0.0
            if T < T_min:
                T_constraint = (T_min - T) * 1e6
            elif T > T_max:
                T_constraint = (T - T_max) * 1e6
                
            # Set initial level flight state: phi=psi=0, theta!=alpha, p=q=r=0, beta=0
            phi, psi = 0.0, 0.0
            p, q, r = 0.0, 0.0, 0.0
            beta = 0.0
            v_rel = 0.0
            
            # Solve V and alpha indirectly from u and w
            V = np.sqrt(u_rel**2 + v_rel**2 + w_rel**2)
            alpha = np.arctan2(w_rel, u_rel) if V > 1e-3 else 0
            theta = alpha + gamma
            
            # ----------------Establish force equations------------------ 
            # Force coefficients
            CL = self.model.CL0 + self.model.CL_alpha * np.rad2deg(alpha)
            CD = self.model.CD0 + self.model.CD_alpha * np.rad2deg(alpha)**2
            
            # Lateral aerodynamic coefficients
            CY = (self.model.CY_beta * np.rad2deg(beta) + 
                self.model.CY_delta_r * 0.0 +  # Rudder deflection
                self.model.CY_p * (p * self.model.b / (2 * V)) + 
                self.model.CY_r * (r * self.model.b / (2 * V)))
            Cl_diff = (self.model.Cl_beta * np.rad2deg(beta) + self.model.Cl_delta_r * 0.0  + 
                self.model.Cl_delta_e_diff * delta_e_diff +  # Differential elevator as equivalent aileron
                self.model.Cl_p * (p*self.model.b)/(2*V) + self.model.Cl_r * (r*self.model.b) /(2*V))
            Cn_diff = (self.model.Cn_beta * np.rad2deg(beta) + 
                self.model.Cn_delta_e_diff * delta_e_diff +  # Differential elevator as equivalent aileron
                self.model.Cn_delta_r * 0.0 + self.model.Cn_r * (r*self.model.b) /(2*V) + self.model.Cn_p * (p*self.model.b)/(2*V))
            
            # Dynamic pressure, lift, side force, drag (wind frame)
            q_dynamic = 0.5 * self.model.rho * V**2 * self.model.S
            Lift = q_dynamic *  CL
            Y = q_dynamic * CY
            D = q_dynamic * CD
            
            # Total forces in body frame (X,Y,Z) should be zero
            F_x = T - self.model.m*self.model.g*np.sin(theta) - D*np.cos(alpha)*np.cos(beta) + Lift*np.sin(alpha) - Y*np.cos(alpha)*np.sin(beta)
            F_y = Y*np.cos(beta) + self.model.m*self.model.g*np.sin(phi)*np.cos(theta) - D*np.sin(beta)
            F_z = self.model.m*self.model.g*np.cos(phi)*np.cos(theta) - D*np.sin(alpha)*np.cos(beta) - Lift*np.cos(alpha) - Y*np.sin(alpha)*np.sin(beta)
        
            # -----------Establish moment (L,M,N) equations----------------
            # Moment coefficients
            Cm_sync = (self.model.Cm0 + self.model.Cm_alpha * np.rad2deg(alpha) + 
                  self.model.Cm_q * np.rad2deg((q * self.model.c / (2 * V))) +
                  self.model.Cm_delta_e_sync * np.rad2deg(delta_e_sync))  # Pitching moment coefficient (synchronous elevator)
                        
            # Roll, pitch, yaw moments should be zero
            # Pitch moment (L,M,N)
            M = q_dynamic * self.model.c * Cm_sync
        
            # ------Lateral aerodynamic moments-------
            # Roll and yaw moments from differential elevator (equivalent aileron effect)
            L = q_dynamic * self.model.b * Cl_diff
            N = q_dynamic * self.model.b * Cn_diff
            
            return [F_x, F_y, F_z, L, M, N, delta_e_sync_constraint, delta_e_diff_constraint, T_constraint]
            
        x0 = [27.0, 2.0, 0.0, 0.0, 10.0]  # Initial guess: u_rel, w_rel, delta_e_sync, delta_e_diff, T
        # Call root solver
        result = root(equations, x0, method='lm')
        if result.success:
            print("Trim solution:")
            print(f"Alpha = {np.rad2deg(np.arctan2(result.x[1], result.x[0])):.2f} deg")
            print(f"Delta_e_sync = {np.rad2deg(result.x[2]):.2f} deg")
            print(f"Delta_e_diff = {np.rad2deg(result.x[3]):.2f} deg")
            print(f"Airspeed V = {np.sqrt(result.x[0]**2 + result.x[1]**2):.2f} m/s")
            print(f"Thrust T = {result.x[4]:.2f} N")
            
            residuals = equations(result.x)
            print("\nTrim residual verification:")
            print(f"Fx: {residuals[0]:.2e} N | Fy: {residuals[1]:.2e} N | Fz: {residuals[2]:.2e} N")
            print(f"L: {residuals[3]:.2e} N·m | M: {residuals[4]:.2e} N·m | N: {residuals[5]:.2e} N·m")
            
            self.trim_state = {
            'u_rel': result.x[0],
            'w_rel': result.x[1],
            'delta_e_sync': result.x[2],
            'delta_e_diff': result.x[3],
            'T': result.x[4],
            'V': np.sqrt(result.x[0]**2 + result.x[1]**2),
            'alpha': np.arctan2(result.x[1], result.x[0])  # rad
            }
        else:
            print("Trim solution failed:", result.message)
        
        return result.x
    
    def update_static_margin(self):
        """Dynamically update static margin (call when CG changes)"""
        SM = (self.model.h_ac - self.model.h_cg) / self.model.c * 100
        print(f"Current static margin: {SM:.1f}%")
    
    def linearize_trim(self, epsilon=1e-6):
        """Small perturbation linearization around trim state"""
        if not hasattr(self, 'trim_state'):
            raise RuntimeError("Please run find_trim() first to get trim state")
    
        # Extract trim state
        # Longitudinal states
        x0 = np.array([
            self.trim_state['V'],  # Airspeed V 
            self.trim_state['alpha'],  # Angle of attack α
            0.0,                        # Pitch rate q
            self.trim_state['alpha'],  # Pitch angle θ
            -100.0,                      # Altitude H
            ])

        u0 = np.array([
            self.trim_state['delta_e_sync'],  # Synchronous elevator δe_sync 
            self.trim_state['T']        # Thrust T
            ])

        # Numerically calculate Jacobian matrices
        def f(x, u):
            """Wrapper for nonlinear dynamic function"""
            return self.dynamics(t=0, x=x, u=u)  

        # Calculate A and B matrices
        n_states = len(x0)
        n_controls = len(u0)
    
        # State matrix A
        A = np.zeros((n_states, n_states))
        for i in range(n_states):
            x_plus = x0.copy()
            x_plus[i] += epsilon
            dx0 = f(x0, u0)
            dx_plus = f(x_plus, u0)
            A[:, i] = (dx_plus - dx0) / epsilon
    
        # Control matrix B
        B = np.zeros((n_states, n_controls))
        for i in range(n_controls):
            u_plus = u0.copy()
            u_plus[i] += epsilon
            # Calculate state derivatives before and after perturbation
            dx0 = f(x0, u0)
            dx_plus = f(x0, u_plus)
            # Approximate partial derivatives
            B[:, i] = (dx_plus - dx0) / epsilon
            
        return A, B
    
    def dynamics(self, t, x, u):
        """Longitudinal dynamics for linearization"""
        phi, theta, psi = 0.0, x[3], 0.0
        V, alpha, q, H = x[0], x[1], x[2], x[4]
        delta_e_sync, T = u[0], u[1]
        
        # Longitudinal aerodynamic coefficients
        CL = self.model.CL0 + self.model.CL_alpha * np.rad2deg(alpha) + self.model.Cl_delta_e_sync * np.rad2deg(delta_e_sync)
        CD = self.model.CD0 + self.model.CD_alpha * np.rad2deg(alpha)**2
        Cm_sync = (self.model.Cm0 + self.model.Cm_alpha * np.rad2deg(alpha) + 
                  self.model.Cm_q * np.rad2deg((q * self.model.c / (2 * V))) + self.model.Cm_delta_e_sync * np.rad2deg(delta_e_sync))
        
        # Dynamic pressure, lift, drag (wind frame)
        q_dynamic = 0.5 * self.model.rho * V**2 * self.model.S
        Lift = q_dynamic *  CL
        Y = 0.0
        D = q_dynamic * CD
        
        beta = 0.0 
        # Forces in body frame
        F_x = T - self.model.m*self.model.g*np.sin(theta) - D*np.cos(alpha)*np.cos(beta) + Lift*np.sin(alpha) - Y*np.cos(alpha)*np.sin(beta)
        F_z = self.model.m*self.model.g*np.cos(phi)*np.cos(theta) - D*np.sin(alpha)*np.cos(beta) - Lift*np.cos(alpha) - Y*np.sin(alpha)*np.sin(beta)
        
        # Pitch moment
        M = q_dynamic * self.model.c * Cm_sync
        
        # Body frame velocities
        u_b, v_b, w_b = V*np.cos(alpha)*np.cos(beta), 0.0, V*np.sin(alpha)*np.cos(beta)
        p, r = 0.0, 0.0
        
        # Translational accelerations
        du = (v_b*r - w_b*q) + F_x/self.model.m
        dw = (u_b*q - v_b*p) + F_z/self.model.m
        
        # Airspeed and angle of attack derivatives
        dV = (u_b*du + w_b*dw) / V
        dalpha = (u_b*dw - w_b*du)/(u_b**2+w_b**2)
        
        # Rotational acceleration (pitch rate)
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = self.model.compute_coef()
        dq = c5*p*r + c6*(p**2 + r**2) + c7*M
        
        # Euler angle kinematics
        dtheta = q*np.cos(phi) - r*np.sin(phi)
        
        # Position increment (NED frame)
        R = np.array([
        [np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi)-np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi)],
        [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi)],
        [-np.sin(theta),            np.sin(phi)*np.cos(theta),                                     np.cos(phi)*np.cos(theta)]
        ])
        
        dX_ned = R @ np.array([u_b, v_b, w_b])
        dH = -dX_ned[2]  # Altitude rate (positive upward)
        
        # Combine all derivatives
        return np.array([dV, dalpha, dq, dtheta, dH])
    
    def analyze_frequency_response(self, A, B):
        # Define output matrix C - select key states as output
        C = np.eye(A.shape[0])  # Output all states
        D = np.zeros((A.shape[0], B.shape[1]))  # Direct transmission term
    
        # Create state space system
        sys = ctrl.ss(A, B, C, D)
        
        # Define input and output names
        input_name = ['delta_e_sync']
        output_name = ['theta']
        
        # Analyze transfer function
        self.analyze_transfer_function(sys, input_name, output_name)
        
        # Analyze key transfer functions and plot
        self.analyze_transfer_functions(sys, input_name, output_name)
        
    def analyze_transfer_function(self, sys, input_name, output_name):
        # Single channel transfer function analysis
        """
        Extract and print transfer function of state space model (SISO only)
        :param sys: Control system state space model (ctrl.ss)
        :param input_name: Input name list (e.g., ['delta_e_sync'])
        :param output_name: Output name list (e.g., ['theta'])
        """
        # Check for SISO
        if len(input_name) != 1 or len(output_name) != 1:
            raise ValueError("This simplified version only supports SISO transfer function extraction!")
        
        input_name = input_name[0]
        output_name = output_name[0]
        input_index = {'delta_e_diff': 0, 'delta_e_sync': 1, 'Thrust': 2}[input_name]
        output_index = 3  # Theta index
        
        # Extract subsystem A, B, C, D
        A_sub = sys.A
        B_sub = sys.B[:, input_index].reshape(-1, 1)
        C_sub = sys.C[output_index, :].reshape(1, -1)
        D_sub = sys.D[output_index, input_index]
        
        # Create SISO subsystem
        sys_sub = ctrl.ss(A_sub, B_sub, C_sub, D_sub)
        
        # Extract transfer function
        tf = ctrl.ss2tf(sys_sub)
        num, den = tf.num[0][0], tf.den[0][0]
        
        print(f"\nTransfer function: {input_name} -> {output_name}")
        print(f"Numerator coefficients (num): {num}")
        print(f"Denominator coefficients (den): {den}")
        
        # Analyze system modes
        self.analyze_modes(sys.A)
    
    def analyze_transfer_functions(self, sys, input_names, output_names):
        # Set frequency range (rad/s)
        omega = np.logspace(-2, 2, 500)
    
        # Analyze pitch channel: delta_e_sync -> theta
        plt.figure(figsize=(10, 8))
    
        # Magnitude plot
        plt.subplot(2, 1, 1)
        mag, phase, omega = ctrl.bode(sys[output_names.index('theta'), input_names.index('delta_e_sync')], 
                                 omega, dB=True, Plot=False)
        plt.semilogx(omega, 20*np.log10(mag))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('Pitch channel: δe_sync → θ', fontsize=20)
        plt.ylabel('Magnitude (dB)', fontsize=20)
        plt.grid(True)
    
        # Phase plot
        plt.subplot(2, 1, 2)
        plt.semilogx(omega, np.rad2deg(phase))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Frequency (rad/s)', fontsize=20)
        plt.ylabel('Phase (deg)', fontsize=20)
        plt.grid(True)
    
        plt.tight_layout()
        plt.show()
    
        # Analyze system modes
        self.analyze_modes(sys.A)

    def analyze_modes(self, A):
        # Calculate characteristic polynomial
        poly_coeffs = np.poly(A)
        
        # Format polynomial output
        poly_str = "Characteristic polynomial: \n"
        n = len(poly_coeffs)-1
        for i, coeff in enumerate(poly_coeffs):
            power = n - i
            if power > 1:
                poly_str += f"{coeff:.4f}λ^{power} + "
            elif power == 1:
                poly_str += f"{coeff:.4f}λ + "
            else:
                poly_str += f"{coeff:.4f}"
        print(poly_str)
        
        # Calculate system eigenvalues and modes
        eigvals, eigvecs = np.linalg.eig(A)
    
        print("\nSystem mode analysis:")
        print("="*50)
    
        # Analyze each eigenvalue
        for i, eig in enumerate(eigvals):
            wn = np.abs(eig)  # Natural frequency
            zeta = -np.real(eig)/wn if wn != 0 else 0  # Damping ratio
        
            print(f"Mode {i+1}:")
            print(f"  Eigenvalue: {eig:.3f}")
            print(f"  Natural frequency (wn): {wn:.3f} rad/s ({wn/(2*np.pi):.3f} Hz)")
            print(f"  Damping ratio (ζ): {zeta:.3f}")
        
            # Determine mode type
            if np.iscomplex(eig):
                if zeta < 0.1:
                    print("  Type: Underdamped oscillatory mode")
                elif 0.1 <= zeta < 0.7:
                    print("  Type: Moderately damped oscillatory mode")
                else:
                    print("  Type: Overdamped oscillatory mode")
            else:
                if eig < 0:
                    print("  Type: Stable aperiodic mode")
                else:
                    print("  Type: Unstable aperiodic mode")
        
            print("-"*50)
            
    def simulate_perturbation(self, t_end=90.0):
        """Time domain simulation of small perturbations"""
        
        from scipy.integrate import odeint

        plt.style.use('seaborn-v0_8-paper')  # Use academic style

        # Get linearized model
        A, B = self.linearize_trim()
        x0 = np.zeros(A.shape[0])  
        x0[2] += 0.1  # Initial pitch rate perturbation (Δq=0.1 rad/s)

        # Time domain simulation
        t = np.linspace(0, t_end, 900)
        response = odeint(lambda x, t: A @ x, x0, t)
        
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': 'Times New Roman',
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
        })
        
        # Create figure (SCI standard size)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)

        # Plot curves (academic color scheme)
        colors = ['#66d28e', '#65d1d2', '#ffbe7a', '#fa8878', '#f76666',]
        ax.plot(t, np.rad2deg(response[:,2]), color=colors[0], linewidth=2, 
            label=r'$\Delta q$ (deg/s)')
        ax.plot(t, np.rad2deg(response[:,3]), color=colors[1], linewidth=2, 
            label=r'$\Delta \theta$ (deg)')
        ax.plot(t, np.rad2deg(response[:,1]), color=colors[2], linewidth=2, 
            label=r'$\Delta \alpha$ (deg)')
        ax.plot(t, response[:,0], color=colors[3], linewidth=2, 
            label=r'$\Delta V$ (m/s)')
        ax.plot(t, response[:,4], color=colors[4], linewidth=2, 
            label=r'$\Delta H$ (m)')

        # Axis settings
        ax.set_xlabel('Time (s)', fontsize=12, labelpad=10)
        ax.set_ylabel('State Variation', fontsize=12, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
        # Legend settings
        legend = ax.legend(loc='upper right', fontsize=10, framealpha=1, 
                      edgecolor='k', fancybox=False)
        legend.get_frame().set_linewidth(0.5)

        # Grid and borders
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        
        plt.savefig('long_small_turbulance.png', dpi=600, bbox_inches='tight')
        plt.show()
    
    def plot_trim_results(self, trim_state):
        """Plot trim results in 3D"""
        fig1 = plt.figure(figsize=(8, 8))
        
        # 3D attitude plot
        ax1 = fig1.add_subplot(111, projection='3d')
        self._plot_3d_attitude(ax1, trim_state)
        plt.show()
         
    def _plot_3d_attitude(self, ax, state):
        """3D aircraft attitude visualization"""
        u_rel, w_rel, delta_e_sync, delta_e_diff, T = state
        alpha = np.arctan2(w_rel, u_rel)
        V = np.sqrt(u_rel**2 + w_rel**2)
        
        # Simplified aircraft model (cuboid fuselage + wings)
        fuselage = np.array([[3,-2,0], [5,0,0], [3,2,0], [-3,2,0], [-3,-2,0]])
        wings = np.array([[1,-4,0], [1,4,0], [-1,4,0], [-1,-4,0]])
        left_e = np.array([[-2.8,-1.5,0], [-2.8,-0.5,0], [-3,-0.5,0], [-3,-1.5,0]])
        right_e = np.array([[-2.8,0.5,0], [-2.8,1.5,0], [-3,1.5,0], [-3,0.5,0]])
        
        # Rotation for angle of attack
        rot_a = np.array([
            [np.cos(alpha), 0, np.sin(alpha)],
            [0, 1, 0],
            [-np.sin(alpha), 0, np.cos(alpha)]
        ])
        fuselage = fuselage @ rot_a
        wings = wings @ rot_a
        
        # Rotation for elevator deflection (synchronous)
        rot = np.array([
            [np.cos(delta_e_sync), 0, np.sin(delta_e_sync)],
            [0, 1, 0],
            [-np.sin(delta_e_sync), 0, np.cos(delta_e_sync)]
        ])
        left_e = (left_e - [-2.8, 0, 0]) @ rot.T + [-2.8, 0, 0]
        right_e = (right_e - [-2.8, 0, 0]) @ rot.T + [-2.8, 0, 0]
        
        # Plot components
        ax.plot_trisurf(fuselage[:,0], fuselage[:,1], fuselage[:,2], color='#a8dadc', alpha=0.5)
        ax.plot_trisurf(wings[:,0], wings[:,1], wings[:,2], color='#1f77b4', alpha=0.6)
        ax.plot_trisurf(left_e[:,0], left_e[:,1], left_e[:,2], color='#ff9999', alpha=0.5)
        ax.plot_trisurf(right_e[:,0], right_e[:,1], right_e[:,2], color='#ff9999', alpha=0.5)
        
        # Annotations
        ax.text(0, 0, 2, 
        f"$\\alpha$={np.rad2deg(alpha):.1f}°\n"
        f"$V$={V:.2f} m/s\n"
        f"$T$={T:.2f} N",
        fontsize=12, 
        bbox=dict(facecolor='white'))
        
        ax.plot([], [], ' ', label='Trim Attitude') 
        ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=14,
        handlelength=0
        ) 
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-3, 3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('H (m)')

# Main simulation function
def simulate_differential_elevator():
    wind_field = WindField()  # Wind field instance
    model = FixedWingDynamics(wind_field) 
    trim_calc = TrimCalculator(model, wind_field)   
    
    u_rel_trim, w_rel_trim, sync_trim, diff_trim, T_trim = trim_calc.find_trim()
    
    # Linearization verification
    #trim_calc.verify_dynamic_balance()
    # Perturbation response simulation
    #trim_calc.simulate_perturbation()

simulate_differential_elevator()