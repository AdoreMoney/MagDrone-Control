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
        # Aircraft Parameters (Example for Small UAV)
        self.m = 5.5  # Mass (kg)
        self.Ixx, self.Iyy, self.Izz = 0.0, 0.0, 0.0  # Moment of Inertia (kg·m²)
        self.Ixy, self.Iyz, self.Ixz = 0.0, 0.0, 0.0
        self.Ix, self.Iy, self.Iz = 0.0, 0.0, 0.0
        self.S = 0.15   # Wing Area (m²)
        self.b = 1.47   # Wingspan (m)
        self.c = 0.1   # Mean Aerodynamic Chord (m)
        self.h_ac = 0.33 # Aerodynamic Center Position
        self.h_cg = 0.30 # Center of Gravity Position
        self.rho = 1.225  # Air Density (kg/m³)
        self.g = 9.81  # Gravitational Acceleration (m/s²)
        self.max_throttle = 0.0  # Maximum Thrust 13N
        self.max_elevator = np.deg2rad(10)  # Maximum Deflection Angle 10°
        
        self.wind_field = wind_field if wind_field else WindField()
        
        # Longitudinal Aerodynamic Coefficients (All set to 0.0)
        self.CL0 = 0.0 # Zero-Lift Angle Coefficient
        self.CL_alpha = 0.0  # Lift Curve Slope 1/°
        self.CD0 = 0.0      # Zero-Lift Drag Coefficient
        self.CD_alpha = 0.0     # Drag Variation with Angle of Attack
        self.Cm0 = 0.0         # Zero Angle of Attack Pitching Moment
        self.Cm_alpha = 0.0 # Pitching Moment Derivative deg-1
        self.Cm_q = 0.0       # Pitch Damping Derivative 
        self.Cm_delta_e_sync = 0.0 # Elevator Efficiency
        self.Cl_delta_e_sync = 0.0 # Elevator Lift Efficiency
        
        # Lateral Aerodynamic Coefficients (Aileron and Rudder Disabled but Coupling Still Exists) (All set to 0.0)
        self.CY_beta = 0.0     # Side Force Derivative
        self.Cl_beta = 0.0     # Roll Static Stability
        self.Cl_p = 0.0        # Roll Damping
        # delta_e_diff = right -left, right downward deflection is positive
        self.Cl_delta_e_diff = 0.0  # Roll Effect
        self.Cl_r = 0.0         # Yaw Damping
        self.Cl_delta_r = 0.0   
        
        self.Cn_delta_e_diff = 0.0 # Differential Elevator Yaw Efficiency
        self.Cn_beta = 0.0      # Directional Static Stability
        self.Cn_r = 0.0        # Yaw Damping
        self.Cn_p = 0.0         # Weak Cross-Coupling Yaw Caused by Roll
        self.Cn_delta_r = 0.0
        
        self.CY_delta_r = 0.0    # Rudder Efficiency (Reasonable Range: 0.2~0.5/rad)
        self.CY_p = 0.0         # Side Force Derivative with Respect to Roll Rate (Reasonable Range: -0.05 ~ -0.2)
        self.CY_r = 0.0         # Side Force Derivative with Respect to Yaw Rate (Reasonable Range: 0.1~0.3)

    def compute_forces(self, x, u, wind_body):
        #Calculate Aerodynamic Forces and Moments (Differential Control)
        u_b, v_b, w_b = x[0], x[1], x[2]  # Body Axis Velocities
        p, q, r = x[3], x[4], x[5]
        phi, theta, psi = x[6], x[7], x[8]
        #delta_e, delta_a, delta_r, delta_T = u
        delta_e_sync, delta_e_diff, delta_T = u  # Synchronous/Differential Elevator + Throttle
        #print('d_e_sync:', delta_e_sync, 'd_e_diff:', delta_e_diff, 'rad')
        
        # Calculate Relative Wind Velocity (Body Axis System)
        u_rel = u_b - wind_body[0]
        v_rel = v_b - wind_body[1]
        w_rel = w_b - wind_body[2]
        
        # Calculate Airspeed, Angle of Attack, Sideslip Angle
        #V = np.sqrt(u_b**2 + v_b**2 + w_b**2)
        #alpha = np.arctan2(w_b, u_b) if V > 1e-3 else 0
        #beta = np.arcsin(v_b/V) if V > 1e-3 else 0
        
        # Calculate Angle of Attack and Sideslip Angle Using Relative Velocity in Body Frame
        V = np.sqrt(u_rel**2 + v_rel**2 + w_rel**2)
        alpha = np.arctan2(w_rel, u_rel) if V > 1e-3 else 0
        alpha= np.clip(alpha, np.deg2rad(-5.0), np.deg2rad(5.0)) 
        
        beta = np.arcsin(v_rel/V) if V > 1e-3 else 0
        
        #print('Airspeed:', V, 'Angle of Attack:', alpha, np.rad2deg(alpha),'degrees', 'Sideslip Angle:', beta)
        
        # Dynamic Pressure
        q_dynamic = 0.5 * self.rho * V**2 * self.S
        #print('Dynamic Pressure：', q_dynamic)
        
        # ------Longitudinal Aerodynamic Coefficients-------
        CL = self.CL0 + self.CL_alpha * np.rad2deg(alpha) + self.Cl_delta_e_sync * np.rad2deg(delta_e_sync)
        #CL = self.CL0 + self.CL_alpha * np.rad2deg(alpha)
        CD = self.CD0 + self.CD_alpha * np.rad2deg(alpha)**2
        #Cm_sync = (self.Cm0 + self.Cm_alpha * np.rad2deg(alpha) + 
                  #self.Cm_q * (q * self.c / (2 * V)) + 
                  #self.Cm_delta_e_sync * np.rad2deg(delta_e_sync))
        Cm_sync = (self.Cm0 + self.Cm_alpha * np.rad2deg(alpha) + 
                  self.Cm_q * np.rad2deg((q * self.c / (2 * V))) + self.Cm_delta_e_sync * np.rad2deg(delta_e_sync))
        #print( 'self.Cm_alpha * np.rad2deg(alpha) :', self.Cm_alpha * np.rad2deg(alpha))
        #print('self.Cm_q * (q * self.c / (2 * V))', self.Cm_q * (q * self.c / (2 * V)))
        
        #print('self.Cm_delta_e_sync * np.rad2deg(delta_e_sync)', self.Cm_delta_e_sync * np.rad2deg(delta_e_sync))
        
        #print('Synchronous Elevator Deflection:', np.rad2deg(delta_e_sync), 'Differential Deflection:', np.rad2deg(delta_e_diff))
        
        # Lateral Aerodynamic Coefficients
        CY = (self.CY_beta * np.rad2deg(beta) + 
            self.CY_delta_r * 0.0 +  # Rudder Deflection
            self.CY_p * (p * self.b / (2 * V)) + 
            self.CY_r * (r * self.b / (2 * V)))
        Cl_diff = (self.Cl_beta * np.rad2deg(beta) + self.Cl_delta_r * 0.0  + 
            self.Cl_delta_e_diff * delta_e_diff +  # Elevator Differential Equivalent Aileron
            self.Cl_p * (p*self.b)/(2*V) + self.Cl_r * (r*self.b) /(2*V))
        Cn_diff = (self.Cn_beta * np.rad2deg(beta) + 
            self.Cn_delta_e_diff * delta_e_diff +  # Elevator Differential Equivalent Aileron
            self.Cn_delta_r * 0.0 + self.Cn_r * (r*self.b) /(2*V)) + self.Cn_p * (p*self.b)/(2*V)
             
        #print('CL:', CL, 'CD:', CD, 'Cm_sync:', Cm_sync, 'CY:', CY)
        
        # Lift (Wind Axis System)
        Lift = q_dynamic *  CL
        
        # Side Force (Wind Axis System)
        Y = q_dynamic * CY
        
        # Drag (Wind Axis System)
        D = q_dynamic * CD
        
        #print('Absolute Lift Value:', Lift, 'Side Force Y:', Y, 'Drag:', D)
         
        # Resultant Force in Body Frame (X,Y,Z)
        F_x = delta_T - self.m*self.g*np.sin(theta) - D*np.cos(alpha)*np.cos(beta) + Lift*np.sin(alpha) - Y*np.cos(alpha)*np.sin(beta)
        F_y = Y*np.cos(beta) + self.m*self.g*np.sin(phi)*np.cos(theta) - D*np.sin(beta)
        F_z = self.m*self.g*np.cos(phi)*np.cos(theta) - D*np.sin(alpha)*np.cos(beta) - Lift*np.cos(alpha) - Y*np.sin(alpha)*np.sin(beta)
        
        # Pitching Moment (L,M,N)
        M = q_dynamic * self.c * Cm_sync
        
        # ------Lateral Aerodynamic Forces-------
        # Roll and Yaw Moments Generated by Differential Elevator (Equivalent Aileron Effect)
        # Total Roll/Yaw Moments (Including Natural Coupling)
        L = q_dynamic * self.b * Cl_diff
        N = q_dynamic * self.b * Cn_diff
        
        #print('Fx:', F_x, 'Fy:', F_y, 'Fz:', F_z)
        #print('L, M, N:', L, M, N)
        
        return np.array([F_x, F_y, F_z, L, M, N])  # Ignore Side Force for Simplification

    def dynamics(self, t, x, u):
        
        phi, theta, psi = x[6], x[7], x[8]
        
        # Rotation Matrix: NED to Body Frame
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
        wind_body = R_ned_to_body @ wind_ned        # Convert to Body Coordinate System
        
        #6DOF Nonlinear Dynamic Equations
        forces = self.compute_forces(x, u, wind_body)
        u_b, v_b, w_b = x[0], x[1], x[2]
        p, q, r = x[3], x[4], x[5]
        #phi, theta, psi = x[6], x[7], x[8]
        
        L, M, N = forces[3], forces[4], forces[5]
        
        # Translational Motion
        du = (v_b*r - w_b*q) + forces[0]/self.m
        dv = (w_b*p - u_b*r) + forces[1]/self.m
        dw = (u_b*q - v_b*p) + forces[2]/self.m
        
        # Rotational Motion
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = self.compute_coef()
        #print('c1~c9:', c1, c2, c3, c4, c5, c6, c7, c8, c9)
        dp = (c1*r + c2*p)*q + c3*L + c4*N
        dq = c5*p*r + c6*(p**2 - r**2) + c7*M
        dr = (c8*p - c2*r)*q + c4*L + c9*N
        
        # Euler Angle Kinematics
        dphi = p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)
        dtheta = q*np.cos(phi) - r*np.sin(phi)
        dpsi = (q*np.sin(phi) + r*np.cos(phi)) / np.cos(theta)
        
        # Rotation Matrix: Body Frame to NED Frame
        R = np.array([
        [np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi)-np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi)],
        [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi)],
        [-np.sin(theta),            np.sin(phi)*np.cos(theta),                                     np.cos(phi)*np.cos(theta)]
        ])
        
        # Position Increment
        dX_ned = R @ np.array([u_b, v_b, w_b])
        
        # Altitude Constraint: Force Stop Descent When Z_down Approaches 0
        
        if x[11] >= 0:  # Z_down >= 0 indicates ground contact
           x[11] = 0  # Clamp Altitude
           dZ_down = max(0, dX_ned[2])  # Only Allow Ascent
           dX_ned = np.array([dX_ned[0], dX_ned[1], dZ_down])
        
        # Merge All Derivatives (9 Original States + 3 Position States)
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

class WindField:
    def __init__(self):
        self.constant_wind = np.array([0.0, 0.0, 0.0])  # [u, v, w] Wind Velocity Components (m/s)
        self.gust_wind = np.array([0.0, 0.0, 0.0])      # Gust Wind Velocity
        self.turbulence = np.array([0.0, 0.0, 0.0])     # Turbulence Wind Velocity
        self.time = 0.0
    
    def update(self, dt):
        """Update Wind Field State to Simulate Dynamic Wind Disturbances"""
        self.time += dt
        
        # Example: Sinusoidally Varying Gust
        gust_magnitude = 7.0  # Gust Intensity (m/s)
        gust_frequency = 0.5  # Gust Frequency (Hz)
        self.gust_wind[0] = gust_magnitude * np.sin(2 * np.pi * gust_frequency * self.time)
        
        # Example: Random Turbulence (Simplified Model)
        turbulence_magnitude = 0.5  # Turbulence Intensity (m/s)
        self.turbulence[0] = turbulence_magnitude * np.random.randn()
        
        # Total Wind Velocity = Constant Wind + Gust + Turbulence
        total_wind = self.constant_wind + self.gust_wind + self.turbulence
        return total_wind
    
    def get_wind(self):
        """Get Current Wind Velocity"""
        return self.constant_wind + self.gust_wind + self.turbulence
    
class TrimCalculator:
    def __init__(self, model, wind_field=None):
        self.model = model
        self.wind_field = wind_field if wind_field else WindField()
        
    def find_trim(self, gamma=0, t_trim=0.0):
        # Calculate Trim State (Steady Level Flight)   
        
        wind_body = [0.0, 0.0, 0.0] #Initial State with No Wind
        
        from scipy.optimize import root

        def equations(vars):
            # Trim Solution Variables: alpha(u_rel, w_rel), theta, V, T
            u_rel, w_rel, delta_e_sync, delta_e_diff, T = vars
            
            # Angle Constraint: delta_e_sync ∈ [-10°, 10°]
            delta_e_sync_min, delta_e_sync_max = np.deg2rad(-10), np.deg2rad(10)
            delta_e_diff_min, delta_e_diff_max = np.deg2rad(-10), np.deg2rad(10)
            delta_e_sync_constraint, delta_e_diff_constraint = 0.0, 0.0
            if delta_e_sync < delta_e_sync_min:
                delta_e_sync_constraint = (delta_e_sync - delta_e_sync_min) * 1e6  # Large Penalty Coefficient
            elif delta_e_sync > delta_e_sync_max:
                delta_e_sync_constraint = (delta_e_sync - delta_e_sync_max) * 1e6
            if delta_e_diff < delta_e_diff_min:
                delta_e_diff_constraint = (delta_e_diff - delta_e_diff_min) * 1e6  # Large Penalty Coefficient
            elif delta_e_diff > delta_e_diff_max:
                delta_e_diff_constraint = (delta_e_diff - delta_e_diff_max) * 1e6
    
            # Thrust Constraint: T ∈ [0, 13] N
            T_min, T_max = 0, 13
            T_constraint = 0.0
            if T < T_min:
                T_constraint = (T_min - T) * 1e6
            elif T > T_max:
                T_constraint = (T - T_max) * 1e6
                
            # Set Initial Level Flight State: phi=psi=0, theta!=alpha, p=q=r=0, beta=0, delta_diff = 0
            # Equations (1-8)
            phi, psi = 0.0, 0.0
            p, q, r = 0.0, 0.0, 0.0
            beta = 0.0
            v_rel = 0.0
            #delta_e_diff = 0.0
            
            # Solve V, alpha Indirectly via u,w (Equation 9)
            V = np.sqrt(u_rel**2 + v_rel**2 + w_rel**2)
            alpha = np.arctan2(w_rel, u_rel) if V > 1e-3 else 0
            theta = alpha + gamma
            
            # ----------------Establish Force Equations------------------ 
            # Force Coefficients
            CL = self.model.CL0 + self.model.CL_alpha * np.rad2deg(alpha)
            CD = self.model.CD0 + self.model.CD_alpha * np.rad2deg(alpha)**2
            
            # Lateral Aerodynamic Coefficients
            CY = (self.model.CY_beta * np.rad2deg(beta) + 
                self.model.CY_delta_r * 0.0 +  # Rudder Deflection
                self.model.CY_p * (p * self.model.b / (2 * V)) + 
                self.model.CY_r * (r * self.model.b / (2 * V)))
            Cl_diff = (self.model.Cl_beta * np.rad2deg(beta) + self.model.Cl_delta_r * 0.0  + 
                self.model.Cl_delta_e_diff * delta_e_diff +  # Elevator Differential Equivalent Aileron
                self.model.Cl_p * (p*self.model.b)/(2*V) + self.model.Cl_r * (r*self.model.b) /(2*V))
            Cn_diff = (self.model.Cn_beta * np.rad2deg(beta) + 
                self.model.Cn_delta_e_diff * delta_e_diff +  # Elevator Differential Equivalent Aileron
                self.model.Cn_delta_r * 0.0 + self.model.Cn_r * (r*self.model.b) /(2*V) + self.model.Cn_p * (p*self.model.b)/(2*V))
            
            # Dynamic Pressure, Lift, Side Force, Drag (Wind Axis System)
            q_dynamic = 0.5 * self.model.rho * V**2 * self.model.S
            Lift = q_dynamic *  CL
            Y = q_dynamic * CY
            D = q_dynamic * CD
            
            # Resultant Force in Body Frame (X,Y,Z) All Zero (Equations 10, 11, 12)
            F_x = T - self.model.m*self.model.g*np.sin(theta) - D*np.cos(alpha)*np.cos(beta) + Lift*np.sin(alpha) - Y*np.cos(alpha)*np.sin(beta)
            F_y = Y*np.cos(beta) + self.model.m*self.model.g*np.sin(phi)*np.cos(theta) - D*np.sin(beta)
            F_z = self.model.m*self.model.g*np.cos(phi)*np.cos(theta) - D*np.sin(alpha)*np.cos(beta) - Lift*np.cos(alpha) - Y*np.sin(alpha)*np.sin(beta)
        
            # -----------Establish Moment (L,M,N) Equations----------------
            # Moment Coefficients
            Cm_sync = (self.model.Cm0 + self.model.Cm_alpha * np.rad2deg(alpha) + 
                  self.model.Cm_q * np.rad2deg((q * self.model.c / (2 * V))) +
                  self.model.Cm_delta_e_sync * np.rad2deg(delta_e_sync))  #Pitching Moment Coefficient, Synchronous Elevator
                        
            #print("Pitching Moment Coefficient from Synchronous Deflection：", Cm_sync)
            #print("Roll Moment Coefficient from Differential Deflection：", Cl_diff)
            #print('Yaw Moment Coefficient from Differential Deflection：', Cn_diff)
            # Roll, Pitch, Yaw Moments All Zero (Equations 13, 14, 15)
            # Pitching Moment (L,M,N)
            M = q_dynamic * self.model.c * Cm_sync
        
            # ------Lateral Aerodynamic Forces-------
            # Roll and Yaw Moments Generated by Differential Elevator (Equivalent Aileron Effect)
            # Total Roll/Yaw Moments (Including Natural Coupling)
            L = q_dynamic * self.model.b * Cl_diff
            N = q_dynamic * self.model.b * Cn_diff
            
            #print('Fx:', F_x, 'Fy:', F_y, 'Fz:', F_z)
            #print('L:', L, 'M:', M, 'N:', N)
            
            return [F_x, F_y, F_z, L, M, N, delta_e_sync_constraint, delta_e_diff_constraint, T_constraint]
            
        x0 = [27.0, 2.0, 0.0, 0.0, 10.0]  # u_rel, w_rel, delta_e_sync, delta_e_diff, T
        # Call root to Solve
        result = root(equations, x0, method='lm')
        if result.success:
            
            print("Trim Solution：")
            #print(f"u_rel = {result.x[0]:.2f} m/s")
            #print(f"w_rel = {result.x[1]:.2f} m/s")
            print(f"alpha = {np.rad2deg(np.arctan2(result.x[1], result.x[0])):.2f} deg")
            print(f"delta_e_sync = {np.rad2deg(result.x[2]):.2f} deg")
            print(f"delta_e_diff = {np.rad2deg(result.x[3]):.2f} deg")
            print(f"V = {np.sqrt(result.x[0]**2 + result.x[1]**2):.2f} m/s")
            print(f"T = {result.x[4]:.2f} N")
            
            residuals = equations(result.x)
            print("\nTrim Residual Verification：")
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
            #self.plot_trim_results(result.x)
            #self.update_static_margin()
            #A, B = self.linearize_trim()
        else:
            print("Solution Failed:", result.message)
        
        return result.x
    
    def update_static_margin(self):
        """Dynamically Update Static Margin (Call When CG Changes)"""
        SM = (self.model.h_ac - self.model.h_cg) / self.model.c * 100
        print(f"Current Static Margin: {SM:.1f}%")
    
    def linearize_trim(self, epsilon=1e-6):
        """Small Disturbance Linearization Based on Trim State"""
        if not hasattr(self, 'trim_state'):
            raise RuntimeError("Please execute find_trim() first to obtain trim state")
    
        # Extract Trim State
        # Lateral States
        x0 = np.array([
            0.0,  # beta 
            0.0,  # p
            0.0,  # r
            0.0,  # phi
            0.0   # psi
            ])

        u0 = np.array([
            self.trim_state['delta_e_diff'],  # δe_diff
            #self.trim_state['delta_e_sync'],  # δe_sync 
            self.trim_state['T']        # Thrust
            ])

        # Numerically Calculate Jacobian Matrix
        def f(x, u):
            """Wrap Nonlinear Dynamic Function"""
            return self.dynamics(t=0, x=x, u=u)  

        # Calculate A,B Matrices
        n_states = len(x0)
        n_controls = len(u0)
    
        # State Matrix A
        A = np.zeros((n_states, n_states))
        for i in range(n_states):
            x_plus = x0.copy()
            x_plus[i] += epsilon
            dx0 = f(x0, u0)
            dx_plus = f(x_plus, u0)
            A[:, i] = (dx_plus - dx0) / epsilon
    
        # Control Matrix B
        B = np.zeros((n_states, n_controls))
        for i in range(n_controls):
            u_plus = u0.copy()
            u_plus[i] += epsilon
            # Calculate State Derivatives Before and After Disturbance
            dx0 = f(x0, u0)
            dx_plus = f(x0, u_plus)
            # Calculate Approximate Partial Derivatives
            B[:, i] = (dx_plus - dx0) / epsilon
            
        #print('A:', A)
        #print('B:', B)
        #self.analyze_frequency_response(A, B)
        #self.analyze_modes(A)
        return A, B
    
    def dynamics(self, t, x, u):
        
        phi, theta, psi = x[3], self.trim_state['alpha'], x[4]
        V, alpha = self.trim_state['V'], self.trim_state['alpha']
        #u, v, w = self.trim_state['u_rel'], 0.0, self.trim_state['w_rel']
        p, q, r = x[1], 0.0, x[2]
        beta = x[0]
        delta_e_diff, T = u[0], u[1]
        
        # Rotation Matrix: NED to Body Frame
        R_ned_to_body = np.array([
            [np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), -np.sin(theta)],
            [np.sin(phi)*np.sin(theta)*np.cos(psi)-np.cos(phi)*np.sin(psi), 
             np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi), 
             np.sin(phi)*np.cos(theta)],
            [np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi), 
             np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi), 
             np.cos(phi)*np.cos(theta)]
        ])
        
        #wind_ned = self.wind_field.get_wind(t, x[9:12])
        #wind_body = R_ned_to_body @ wind_ned        # Convert to Body Coordinate System
        
        #6DOF Nonlinear Dynamic Equations
        # ------Longitudinal Aerodynamic Coefficients-------
        #CL = self.model.CL0 + self.model.CL_alpha * np.rad2deg(alpha) + self.model.Cl_delta_e_sync * np.rad2deg(delta_e_sync)
        CD = self.model.CD0 + self.model.CD_alpha * np.rad2deg(alpha)**2
        CL = self.model.CL0 + self.model.CL_alpha * np.rad2deg(alpha)

        # Dynamic Pressure, Lift, Side Force, Drag (Wind Axis System)
        q_dynamic = 0.5 * self.model.rho * V**2 * self.model.S
        
        # -------Lateral Aerodynamic Coefficients--------
        CY = (self.model.CY_beta * np.rad2deg(beta) + 
            self.model.CY_delta_r * 0.0 +  # Rudder Deflection
            self.model.CY_p * (p * self.model.b / (2 * V)) + 
            self.model.CY_r * (r * self.model.b / (2 * V)))
        Cl_diff = (self.model.Cl_beta * np.rad2deg(beta) + self.model.Cl_delta_r * 0.0  + 
            self.model.Cl_delta_e_diff * delta_e_diff +  # Elevator Differential Equivalent Aileron
            self.model.Cl_p * (p*self.model.b)/(2*V) + self.model.Cl_r * (r*self.model.b) /(2*V))
        Cn_diff = (self.model.Cn_beta * np.rad2deg(beta) + 
            self.model.Cn_delta_e_diff * delta_e_diff +  # Elevator Differential Equivalent Aileron
            self.model.Cn_delta_r * 0.0 + self.model.Cn_r * (r*self.model.b) /(2*V)) + self.model.Cn_p * (p*self.model.b)/(2*V)
        
        Lift = q_dynamic *  CL
        Y = q_dynamic * CY
        D = q_dynamic * CD
        
        # Resultant Force in Body Frame (X,Y,Z)
        F_x = T - self.model.m*self.model.g*np.sin(theta) - D*np.cos(alpha)*np.cos(beta) + Lift*np.sin(alpha) - Y*np.cos(alpha)*np.sin(beta)
        F_z = self.model.m*self.model.g*np.cos(phi)*np.cos(theta) - D*np.sin(alpha)*np.cos(beta) - Lift*np.cos(alpha) - Y*np.sin(alpha)*np.sin(beta)
        F_y = Y*np.cos(beta) + self.model.m*self.model.g*np.sin(phi)*np.cos(theta) - D*np.sin(beta)
        
        # Pitching Moment (L,M,N)
        L = q_dynamic * self.model.b * Cl_diff
        N = q_dynamic * self.model.b * Cn_diff
        
        # Translational Motion
        u_b, v_b, w_b = V*np.cos(alpha)*np.cos(beta), V*np.sin(beta), V*np.sin(alpha)*np.cos(beta)
        
        du = (v_b*r - w_b*q) + F_x/self.model.m
        dv = (w_b*p - u_b*r) + F_y/self.model.m
        dw = (u_b*q - v_b*p) + F_z/self.model.m
        
        dV = (u_b*du + w_b*dw) / V
        
        dbeta = (V*dv - v_b*dV) / (V**2 - v_b**2)
        
        # Rotational Motion
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = self.model.compute_coef()
        dp = (c1*r + c2*p)*q + c3*L + c4*N
        dr = (c8*p - c2*r)*q + c4*L + c9*N
        #print('dr:', dr)
        #print('dp:', dp)
        
        # Euler Angle Kinematics
        dphi = p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)
        dpsi = (q*np.sin(phi) + r*np.cos(phi)) / np.cos(theta)
        #print('dphi:', dphi, 'phi:', phi)
        #print('dpsi:', dpsi)
        
        # Merge All Derivatives (9 Original States + 3 Position States)
        return np.array([dbeta, dp, dr, dphi, dpsi])
    
    def analyze_frequency_response(self, A, B):
        # Define System Output Matrix C - Here We Select Several Key States as Output
        # 
        C = np.eye(A.shape[0])  # Output All States
        D = np.zeros((A.shape[0], B.shape[1]))  # Direct Transmission Term
    
        # Create State Space System
        sys = ctrl.ss(A, B, C, D)
        
        # Define Input Names (Assume Elevator is the First Input)
        input_name = ['delta_e_diff']  #
        #output_name = ['phi']
        output_name = ['psi']
        
        # Analyze Transfer Function
        #self.analyze_transfer_function(sys, input_name, output_name)
    
        # Analyze Several Key Transfer Functions and Plot
        #self.analyze_transfer_functions(sys, input_name, output_name)
        
    def analyze_transfer_function(self, sys, input_name, output_name):
        # Single Channel Transfer Function Analysis
        """
        Extract and Print Transfer Function of State Space Model (Only Supports SISO)
        :param sys: Control System State Space Model (ctrl.ss)
        :param input_names: Input Name List (e.g., ['delta_e_sync'])
        :param output_names: Output Name List (e.g., ['theta'])
        """
       
        # Check if Input/Output Count is 1 (Simplified Version Only Supports SISO)
        if len(input_name) != 1 or len(output_name) != 1:
            raise ValueError("This simplified version only supports SISO transfer function extraction!")
        
        input_name = input_name[0]
        output_name = output_name[0]
        input_index = {'delta_e_diff': 0, 'delta_e_sync': 1, 'Thrust': 2}[input_name]
        output_index = 4 # phi:3 psi:4
        
        # Extract A, B, C, D of the Subsystem
        A_sub = sys.A  # State Matrix Remains Unchanged
        B_sub = sys.B[:, input_index].reshape(-1, 1)
        C_sub = sys.C[output_index, :].reshape(1, -1)  # Only Keep Row for phi/psi
        D_sub = sys.D[output_index, input_index]  # Direct Transmission Term (Usually 0)
        
        # Create Subsystem (SISO)
        sys_sub = ctrl.ss(A_sub, B_sub, C_sub, D_sub)
        
        # Extract Transfer Function (ctrl.ss2tf)
        tf = ctrl.ss2tf(sys_sub)
        
        num, den = tf.num[0][0], tf.den[0][0]  # Get Numerator and Denominator Coefficients
        
        print(f"\nTransfer Function: {input_name} -> {output_name}")
        print(f"Numerator Coefficients (num): {num}")
        print(f"Denominator Coefficients (den): {den}")
        
        # Calculate Key Modal Characteristics
        #self.analyze_modes(sys.A)
    
    def analyze_transfer_functions(self, sys, input_names, output_names):
        
        # Set Frequency Range (rad/s)
        omega = np.logspace(-2, 2, 500)
    
        # Analyze Channel: delta_e_diff -> phi Ψ
        plt.figure(figsize=(10, 8))
    
        # Channel
        plt.subplot(2, 1, 1)
        #mag, phase, omega = ctrl.bode(sys[output_names.index('phi'), input_names.index('delta_e_diff')], 
                                 #omega, dB=True, Plot=False)
        mag, phase, omega = ctrl.bode(sys[output_names.index('psi'), input_names.index('delta_e_diff')], 
                                 omega, dB=True, Plot=False)                         
        plt.semilogx(omega, 20*np.log10(mag))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        #plt.title('roll: δe_diff → Φ', fontsize=20)
        plt.title('yaw: δe_diff → Ψ', fontsize=20)
        plt.ylabel('Magnitude (dB)', fontsize=20)
        plt.grid(True)
    
        plt.subplot(2, 1, 2)
        plt.semilogx(omega, np.rad2deg(phase))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('Freq (rad/s)', fontsize=20)
        plt.ylabel('Phase (deg)', fontsize=20)
        plt.grid(True)
    
        # 2. Roll Channel: delta_e_diff -> phi (Roll Angle)
        #plt.subplot(2, 2, 2)
        #mag, phase, omega = ctrl.bode(sys[output_names.index('phi'), input_names.index('delta_e_diff')], 
                                 #omega, dB=True, Plot=False)
        #plt.semilogx(omega, 20*np.log10(mag))
        #plt.title('roll: δe_diff → φ')
        #plt.ylabel('Magnitude (dB)')
        #plt.grid(True)
    
        #plt.subplot(2, 2, 4)
        #plt.semilogx(omega, np.rad2deg(phase))
        #plt.xlabel('Freq (rad/s)')
        #plt.ylabel('Phase (deg)')
        #plt.grid(True)
    
        plt.tight_layout()
        plt.show()
    
        # Calculate Key Modal Characteristics
        self.analyze_modes(sys.A)

    def analyze_modes(self, A):
        poly_coeffs = np.poly(A)
        
        # Format Output Polynomial
        poly_str = "Characteristic Polynomial: \n"
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
        
        # Calculate System Eigenvalues and Modes
        eigvals, eigvecs = np.linalg.eig(A)
    
        print("\nSystem Modal Analysis:")
        print("="*50)
    
        # Analyze Each Eigenvalue
        for i, eig in enumerate(eigvals):
            wn = np.abs(eig)  # Natural Frequency
            zeta = -np.real(eig)/wn if wn != 0 else 0  # Damping Ratio
        
            print(f"Mode {i+1}:")
            print(f"  Eigenvalue: {eig:.3f}")
            print(f"  Natural Frequency (wn): {wn:.3f} rad/s ({wn/(2*np.pi):.3f} Hz)")
            print(f"  Damping Ratio (ζ): {zeta:.3f}")
        
            # Determine Mode Type
            if np.iscomplex(eig):
                if zeta < 0.1:
                    print("  Type: Weakly Damped Oscillatory Mode")
                elif 0.1 <= zeta < 0.7:
                    print("  Type: Moderately Damped Oscillatory Mode")
                else:
                    print("  Type: Strongly Damped Oscillatory Mode")
            else:
                if eig < 0:
                    print("  Type: Stable Aperiodic Mode")
                else:
                    print("  Type: Unstable Aperiodic Mode")
        
            print("-"*50)
            
    def simulate_perturbation(self, t_end=90.0):
        """Small Disturbance Time Domain Simulation"""
        from scipy.integrate import odeint
        
        A, B = self.linearize_trim()
        x0 = np.zeros(A.shape[0])  # Small Disturbance Near Trim Point
    
        # Define Linearized Model
        def linear_model(x, t):
            return A @ x
    
        # Simulation Time
        t = np.linspace(0, t_end, 900)
    
        # Simulate Initial Pitch Disturbance (Δ=0.01 rad/s)
        #x0[1] += 0.1  # p
        x0[2] += 0.1  # r
        response = odeint(linear_model, x0, t)
        '''
        # Plot Key States
        plt.figure(figsize=(12,8))
        plt.plot(t, np.rad2deg(response[:,2]), label='Δr (deg/s)')
        #plt.plot(t, np.rad2deg(response[:,1]), label='Δp (deg/s)')
        plt.plot(t, np.rad2deg(response[:,3]), label='ΔRoll phi: ΔΦ (deg)')
        #plt.plot(t, np.rad2deg(response[:,2]), label='Δr (deg/s)')
        plt.plot(t, np.rad2deg(response[:,1]), label='Δp (deg/s)')
        plt.plot(t, np.rad2deg(response[:,4]), label='ΔYaw psi: ΔΨ (deg)')
        plt.plot(t, np.sqrt(response[:, 0]), label='ΔBeta: Δβ (m/s)')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('time (s)', fontsize=20)
        plt.ylabel('state', fontsize=20)
        plt.legend(fontsize=20)
        plt.grid(True)
        plt.title("Dynamic response based on Small Turbulance", fontsize=20)
        plt.show()
        '''
        plt.style.use('seaborn-v0_8-paper')  # Use Academic Style
    
        # Create Figure (Set SCI Standard Size)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)

        # Plot Curves (Use Academic Style Color Scheme)
        colors = ['#66d28e', '#65d1d2', '#ffbe7a', '#fa8878', '#ebda20',]
        ax.plot(t, np.rad2deg(response[:,1]), color=colors[0], linewidth=2, 
            label=r'$\Delta p$ (deg/s)')
        ax.plot(t, np.rad2deg(response[:,3]), color=colors[1], linewidth=2, 
            label=r'$\Delta \phi$ (deg)')
        ax.plot(t, np.rad2deg(response[:,2]), color=colors[2], linewidth=2, 
            label=r'$\Delta r$ (deg/s)')
        ax.plot(t, np.rad2deg(response[:,4]), color=colors[3], linewidth=2, 
            label=r'$\Delta \psi$ (deg)')
        ax.plot(t, np.rad2deg(response[:,0]), color=colors[4], linewidth=2, 
            label=r'$\Delta \beta$ (deg)')

        # Axis and Label Settings
        ax.set_xlabel('Time (s)', fontsize=12, labelpad=10)
        ax.set_ylabel('State Variation', fontsize=12, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
        # Legend Settings (Use LaTeX Mathematical Symbols)
        legend = ax.legend(loc='upper right', fontsize=10, framealpha=1, 
                      edgecolor='k', fancybox=False)
        legend.get_frame().set_linewidth(0.5)

        # Grid and Borders
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        
        plt.savefig('r_lat_small_turbulance.png', dpi=600, bbox_inches='tight')
        plt.show()
    
    def plot_trim_results(self, trim_state):
        """Plot Combined Trim Result Figure"""
        fig1 = plt.figure(figsize=(7, 7))
        
        # 3D Attitude
        ax1 = fig1.add_subplot(111, projection='3d')
        self._plot_3d_attitude(ax1, trim_state)
        plt.savefig('aircraft_trim.png', dpi=600, bbox_inches='tight')  # Key Saving Parameters
        plt.show()
         
    def _plot_3d_attitude(self, ax, state):
        """3D Aircraft Attitude Schematic"""
        u_rel, w_rel, delta_e_sync, delta_e_diff, T = state
        alpha = np.arctan2(w_rel, u_rel)
        V = np.sqrt(u_rel**2 + w_rel**2)
        
        # Simplified Aircraft Model (Cuboid Fuselage + Wings)
        fuselage = np.array([[3,-2,0], [5,0,0], [3,2,0], [-3,2,0], [-3,-2,0]])
        wings = np.array([[2,-7,0], [2,7,0], [0,7,0], [0,-7,0]])
        tail = np.array([[-2.4,-4,0], [-2.4,4,0], [-3,4,0], [-3,-4,0]])
        left_e = np.array([[-2.8,-3.7,0], [-2.8,-2.2,0], [-3,-2.2,0], [-3,-3.7,0]])
        right_e = np.array([[-2.8,2.2,0], [-2.8,3.7,0], [-3,3.7,0], [-3,2.2,0]])
        
        # alpha
        rot_a = np.array([
            [np.cos(alpha), 0, np.sin(alpha)],
            [0, 1, 0],
            [-np.sin(alpha), 0, np.cos(alpha)]
        ])
        fuselage = fuselage @ rot_a
        tail = tail @ rot_a
        wings = wings @ rot_a
        # Apply Rotation to Tail (sync)
        rot = np.array([
            [np.cos(delta_e_sync), 0, np.sin(delta_e_sync)],
            [0, 1, 0],
            [-np.sin(delta_e_sync), 0, np.cos(delta_e_sync)]
        ])
        left_e, right_e = (left_e - [-2.8, 0, 0]) @ rot.T + [-2.8, 0, 0], (right_e - [-2.8, 0, 0]) @ rot.T + [-2.8, 0, 0]
        
        # Plot
        ax.plot_trisurf(fuselage[:,0], fuselage[:,1], fuselage[:,2], color='#a8dadc', alpha=0.5)
        ax.plot_trisurf(wings[:,0], wings[:,1], wings[:,2], color='#1f77b4', alpha=0.6)
        ax.plot_trisurf(left_e[:,0], left_e[:,1], left_e[:,2], color='#ff9999', alpha=0.5)
        ax.plot_trisurf(right_e[:,0], right_e[:,1], right_e[:,2], color='#ff9999', alpha=0.5)
        ax.plot_trisurf(tail[:, 0], tail[:, 1], tail[:, 2], color='#1f77b4', alpha=0.6)
        
        # Annotations
        ax.text(0, 0, 3, 
        #f"$\delta_{{e,\mathrm{{sync}}}}$={np.rad2deg(delta_e_sync):.1f}°\n"
        f"$\\alpha$={np.rad2deg(alpha):.1f}°\n"
        f"$V$={V:.2f} m/s\n"
        f"$T$={T:.2f} N",
        fontsize=12, 
        bbox=dict(facecolor='white'))
        
        #ax.plot([], [], ' ', label='Trim Attitude') 
        ax.legend(
        loc='upper center',          # Place at Top Center
        bbox_to_anchor=(0.5, 1.0),  # Adjust Vertical Position (1.1=Above Figure)
        frameon=False,              # No Border
        fontsize=14,
        handlelength=0              # Hide Legend Handle (Left Line Icon)
        ) 
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-3, 3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('H (m)')
        #ax.set_title('Trim Attitude', pad=0)

# Main Simulation Function
def simulate_differential_elevator():
    #controller = PositionController()
    wind_field = WindField()  # Wind Field Instance
    model = FixedWingDynamics(wind_field) 
    trim_calc = TrimCalculator(model, wind_field)   
    
    u_rel_trim, w_rel_trim, sync_trim, diff_trim, T_trim = trim_calc.find_trim()
    print('m')
    trim_calc.simulate_perturbation()
    
#simulate_differential_elevator()