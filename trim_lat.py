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
        """
        NOTE: 
            To protect intellectual property rights, 
            the MagDrone's aerodynamic parameters are replaced with 0.0 here.
            This is solely for demonstrating the experimental structure.
            In practice, the parameters can be modified to suit specific drones.
        """
        self.m = 0.0  # Mass 
        self.Ixx, self.Iyy, self.Izz = 0.0, 0.0, 0.0  # Moments of inertia
        self.Ixy, self.Iyz, self.Ixz = 0.0, 0.0, 0.0
        self.Ix, self.Iy, self.Iz = 0.0, 0.0, 0.0
        self.S = 0.0   # Wing area 
        self.b = 0.0   # Wingspan 
        self.c = 0.0   # Mean aerodynamic chord 
        self.h_ac = 0.0 # Aerodynamic center position
        self.h_cg = 0.0 # Center of gravity position
        self.rho = 1.225  # Air density (kg/m³)
        self.g = 9.81  # Gravitational acceleration (m/s²)
        self.max_throttle = 13.0  # Maximum thrust (13N)
        self.max_elevator = np.deg2rad(10)  # Maximum deflection angle (10°)
        
        self.wind_field = wind_field if wind_field else None
        
        # Longitudinal aerodynamic coefficients
        self.CL0 = 0.0 # Zero-lift coefficient
        self.CL_alpha = 0.0  # Lift curve slope (per degree)
        self.CD0 = 0.0      # Zero-lift drag coefficient
        self.CD_alpha = 0.0     # Drag variation with angle of attack
        self.Cm0 = 0.0         # Zero-angle pitching moment
        self.Cm_alpha = 0.0 # Pitching moment derivative (per degree)
        self.Cm_q = 0.0       # Pitch damping derivative 
        self.Cm_delta_e_sync = 0.0 # Elevator effectiveness
        self.Cl_delta_e_sync = 0.0 # Lift effectiveness of elevator
        
        # Lateral aerodynamic coefficients (coupling exists even with aileron/rudder disabled)
        self.CY_beta = 0.0     # Side force derivative
        self.Cl_beta = 0.0     # Roll static stability
        self.Cl_p = 0.0        # Roll damping
        self.Cl_delta_e_diff = 0.0
        self.Cl_r = 0.0         # Yaw damping
        self.Cl_delta_r = 0.0   
        
        self.Cn_delta_e_diff = 0.0 # Yaw effectiveness of differential elevator
        self.Cn_beta = 0.0      # Directional static stability
        self.Cn_r = 0.0        # Yaw damping
        self.Cn_p = 0.0         # Weak cross-coupling yaw from roll
        self.Cn_delta_r = 0.0
        
        self.CY_delta_r = 0.0    # Rudder effectiveness
        self.CY_p = 0.0         # Side force derivative with roll rate 
        self.CY_r = 0.0         # Side force derivative with yaw rate 

    def compute_forces(self, x, u, wind_body):
        # Calculate aerodynamic forces and moments - differential control
        u_b, v_b, w_b = x[0], x[1], x[2]  # Body axis velocities
        p, q, r = x[3], x[4], x[5]
        phi, theta, psi = x[6], x[7], x[8]
        delta_e_sync, delta_e_diff, delta_T = u  # Synchronous/differential elevator + throttle
        
        # Calculate relative wind speed (body axis system)
        u_rel = u_b - wind_body[0]
        v_rel = v_b - wind_body[1]
        w_rel = w_b - wind_body[2]
        
        # Calculate angle of attack and sideslip angle using relative velocity in body frame
        V = np.sqrt(u_rel**2 + v_rel**2 + w_rel**2)
        alpha = np.arctan2(w_rel, u_rel) if V > 1e-3 else 0
        alpha = np.clip(alpha, np.deg2rad(-5.0), np.deg2rad(5.0)) 
        
        beta = np.arcsin(v_rel/V) if V > 1e-3 else 0
        
        # Dynamic pressure
        q_dynamic = 0.5 * self.rho * V**2 * self.S
        
        # ------ Longitudinal aerodynamic coefficients -------
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
            self.Cl_delta_e_diff * delta_e_diff +  # Tail differential equivalent aileron
            self.Cl_p * (p*self.b)/(2*V) + self.Cl_r * (r*self.b) /(2*V))
        Cn_diff = (self.Cn_beta * np.rad2deg(beta) + 
            self.Cn_delta_e_diff * delta_e_diff +  # Tail differential equivalent aileron
            self.Cn_delta_r * 0.0 + self.Cn_r * (r*self.b) /(2*V)) + self.Cn_p * (p*self.b)/(2*V)
        
        # Lift (wind axis system)
        Lift = q_dynamic *  CL
        
        # Side force (wind axis system)
        Y = q_dynamic * CY
        
        # Drag (wind axis system)
        D = q_dynamic * CD
         
        # Total forces in body frame (X,Y,Z)
        F_x = delta_T - self.m*self.g*np.sin(theta) - D*np.cos(alpha)*np.cos(beta) + Lift*np.sin(alpha) - Y*np.cos(alpha)*np.sin(beta)
        F_y = Y*np.cos(beta) + self.m*self.g*np.sin(phi)*np.cos(theta) - D*np.sin(beta)
        F_z = self.m*self.g*np.cos(phi)*np.cos(theta) - D*np.sin(alpha)*np.cos(beta) - Lift*np.cos(alpha) - Y*np.sin(alpha)*np.sin(beta)
        
        # Pitching moments (L,M,N)
        M = q_dynamic * self.c * Cm_sync
        
        # ------ Lateral aerodynamic forces -------
        # Rolling and yawing moments generated by differential elevators (equivalent aileron effect)
        # Total rolling/yawing moments (including natural coupling)
        L = q_dynamic * self.b * Cl_diff
        N = q_dynamic * self.b * Cn_diff
        
        return np.array([F_x, F_y, F_z, L, M, N])  # Simplified by ignoring lateral forces

    def dynamics(self, t, x, u):
        
        phi, theta, psi = x[6], x[7], x[8]
        
        # Rotation matrix: NED to body
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
        wind_body = R_ned_to_body @ wind_ned        # Convert to body coordinate system
        
        # 6DOF nonlinear dynamics equations
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
        
        # Rotation matrix: body to NED frame
        R = np.array([
        [np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi)-np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi)],
        [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi)],
        [-np.sin(theta),            np.sin(phi)*np.cos(theta),                                     np.cos(phi)*np.cos(theta)]
        ])
        
        # Position increments
        dX_ned = R @ np.array([u_b, v_b, w_b])
        
        if x[11] >= 0:  # Z_down >= 0 indicates ground contact
           x[11] = 0  # Clamp height
           dZ_down = max(0, dX_ned[2])  # Only allow ascent
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

class TrimCalculator:
    def __init__(self, model, wind_field=None):
        self.model = model
        self.wind_field = wind_field if wind_field else None
        
    def find_trim(self, gamma=0, t_trim=0.0):
        # Calculate trim state (steady level flight)   
        
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
                
            # Set initial level flight state phi=psi=0, theta!=alpha, p=q=r=0, beta=0
            phi, psi = 0.0, 0.0
            p, q, r = 0.0, 0.0, 0.0
            beta = 0.0
            v_rel = 0.0
            
            # Indirectly solve for V, alpha from u,w
            V = np.sqrt(u_rel**2 + v_rel**2 + w_rel**2)
            alpha = np.arctan2(w_rel, u_rel) if V > 1e-3 else 0
            theta = alpha + gamma
            
            # ---------------- Build force equations ------------------ 
            # Force coefficients
            CL = self.model.CL0 + self.model.CL_alpha * np.rad2deg(alpha)
            CD = self.model.CD0 + self.model.CD_alpha * np.rad2deg(alpha)**2
            
            # Lateral aerodynamic coefficients
            CY = (self.model.CY_beta * np.rad2deg(beta) + 
                self.model.CY_delta_r * 0.0 +  # Rudder deflection
                self.model.CY_p * (p * self.model.b / (2 * V)) + 
                self.model.CY_r * (r * self.model.b / (2 * V)))
            Cl_diff = (self.model.Cl_beta * np.rad2deg(beta) + self.model.Cl_delta_r * 0.0  + 
                self.model.Cl_delta_e_diff * delta_e_diff +  # Tail differential equivalent aileron
                self.model.Cl_p * (p*self.model.b)/(2*V) + self.model.Cl_r * (r*self.model.b) /(2*V))
            Cn_diff = (self.model.Cn_beta * np.rad2deg(beta) + 
                self.model.Cn_delta_e_diff * delta_e_diff +  # Tail differential equivalent aileron
                self.model.Cn_delta_r * 0.0 + self.model.Cn_r * (r*self.model.b) /(2*V) + self.model.Cn_p * (p*self.model.b)/(2*V))
            
            # Dynamic pressure, lift, side force, drag (wind axes)
            q_dynamic = 0.5 * self.model.rho * V**2 * self.model.S
            Lift = q_dynamic *  CL
            Y = q_dynamic * CY
            D = q_dynamic * CD
            
            # Total forces in body frame (X,Y,Z) all zero
            F_x = T - self.model.m*self.model.g*np.sin(theta) - D*np.cos(alpha)*np.cos(beta) + Lift*np.sin(alpha) - Y*np.cos(alpha)*np.sin(beta)
            F_y = Y*np.cos(beta) + self.model.m*self.model.g*np.sin(phi)*np.cos(theta) - D*np.sin(beta)
            F_z = self.model.m*self.model.g*np.cos(phi)*np.cos(theta) - D*np.sin(alpha)*np.cos(beta) - Lift*np.cos(alpha) - Y*np.sin(alpha)*np.sin(beta)
        
            # ----------- Build moment (L,M,N) equations ----------------
            # Moment coefficients
            Cm_sync = (self.model.Cm0 + self.model.Cm_alpha * np.rad2deg(alpha) + 
                  self.model.Cm_q * np.rad2deg((q * self.model.c / (2 * V))) +
                  self.model.Cm_delta_e_sync * np.rad2deg(delta_e_sync))  # Pitch moment coefficient, tail synchronized deflection
                        
            # Roll, pitch, yaw moments all zero 
            # Pitching moments (L,M,N)
            M = q_dynamic * self.model.c * Cm_sync
        
            # ------ Lateral aerodynamics -------
            # Rolling and yawing moments generated by differential elevators (equivalent aileron effect)
            # Total rolling/yawing moments (including natural coupling)
            L = q_dynamic * self.model.b * Cl_diff
            N = q_dynamic * self.model.b * Cn_diff
            
            return [F_x, F_y, F_z, L, M, N, delta_e_sync_constraint, delta_e_diff_constraint, T_constraint]
            
        x0 = [27.0, 2.0, 0.0, 0.0, 10.0]  # u_rel, w_rel, delta_e_sync, delta_e_diff, T
        # Call root solver
        result = root(equations, x0, method='lm')
        if result.success:
            
            print("Trim solution:")
            #print(f"u_rel = {result.x[0]:.2f} m/s")
            #print(f"w_rel = {result.x[1]:.2f} m/s")
            print(f"alpha = {np.rad2deg(np.arctan2(result.x[1], result.x[0])):.2f} deg")
            print(f"delta_e_sync = {np.rad2deg(result.x[2]):.2f} deg")
            print(f"delta_e_diff = {np.rad2deg(result.x[3]):.2f} deg")
            print(f"V = {np.sqrt(result.x[0]**2 + result.x[1]**2):.2f} m/s")
            print(f"T = {result.x[4]:.2f} N")
            
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
            
            #A, B = self.linearize_trim()
        else:
            print("Solution failed:", result.message)
        
        return result.x
    
    def linearize_trim(self, epsilon=1e-6):
        """Small perturbation linearization based on trim state"""
        if not hasattr(self, 'trim_state'):
            raise RuntimeError("Please run find_trim() first to obtain the trim state")
    
        # Extract trim state
        # Lateral states
        x0 = np.array([
            0.0,  # beta 
            0.0,  # p
            0.0,  # r
            0.0,  # phi
            0.0   # psi
            ])

        u0 = np.array([
            self.trim_state['delta_e_diff'],  # δe_diff
            self.trim_state['T']        # Thrust
            ])

        # Numerically compute Jacobian matrices
        def f(x, u):
            """Wrapper for nonlinear dynamics function"""
            return self.dynamics(t=0, x=x, u=u)  

        # Calculate A,B matrices
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
            # Calculate approximate partial derivatives
            B[:, i] = (dx_plus - dx0) / epsilon

        #self.analyze_frequency_response(A, B)
        return A, B
    
    def dynamics(self, t, x, u):
        
        phi, theta, psi = x[3], self.trim_state['alpha'], x[4]
        V, alpha = self.trim_state['V'], self.trim_state['alpha']
        p, q, r = x[1], 0.0, x[2]
        beta = x[0]
        delta_e_diff, T = u[0], u[1]
        
        # Rotation matrix: NED to body
        R_ned_to_body = np.array([
            [np.cos(theta)*np.cos(psi), np.cos(theta)*np.sin(psi), -np.sin(theta)],
            [np.sin(phi)*np.sin(theta)*np.cos(psi)-np.cos(phi)*np.sin(psi), 
             np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi), 
             np.sin(phi)*np.cos(theta)],
            [np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi), 
             np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi), 
             np.cos(phi)*np.cos(theta)]
        ])
        
        # 6DOF nonlinear dynamics equations
        # ------ Longitudinal aerodynamic coefficients -------
        CD = self.model.CD0 + self.model.CD_alpha * np.rad2deg(alpha)**2
        CL = self.model.CL0 + self.model.CL_alpha * np.rad2deg(alpha)

        # Dynamic pressure, lift, side force, drag (wind axes)
        q_dynamic = 0.5 * self.model.rho * V**2 * self.model.S
        
        # ------- Lateral aerodynamic coefficients --------
        CY = (self.model.CY_beta * np.rad2deg(beta) + 
            self.model.CY_delta_r * 0.0 +  # Rudder deflection
            self.model.CY_p * (p * self.model.b / (2 * V)) + 
            self.model.CY_r * (r * self.model.b / (2 * V)))
        Cl_diff = (self.model.Cl_beta * np.rad2deg(beta) + self.model.Cl_delta_r * 0.0  + 
            self.model.Cl_delta_e_diff * delta_e_diff +  # Tail differential equivalent aileron
            self.model.Cl_p * (p*self.model.b)/(2*V) + self.model.Cl_r * (r*self.model.b) /(2*V))
        Cn_diff = (self.model.Cn_beta * np.rad2deg(beta) + 
            self.model.Cn_delta_e_diff * delta_e_diff +  # Tail differential equivalent aileron
            self.model.Cn_delta_r * 0.0 + self.model.Cn_r * (r*self.model.b) /(2*V)) + self.model.Cn_p * (p*self.model.b)/(2*V)
        
        Lift = q_dynamic *  CL
        Y = q_dynamic * CY
        D = q_dynamic * CD
        
        # Total forces in body frame (X,Y,Z)
        F_x = T - self.model.m*self.model.g*np.sin(theta) - D*np.cos(alpha)*np.cos(beta) + Lift*np.sin(alpha) - Y*np.cos(alpha)*np.sin(beta)
        F_z = self.model.m*self.model.g*np.cos(phi)*np.cos(theta) - D*np.sin(alpha)*np.cos(beta) - Lift*np.cos(alpha) - Y*np.sin(alpha)*np.sin(beta)
        F_y = Y*np.cos(beta) + self.model.m*self.model.g*np.sin(phi)*np.cos(theta) - D*np.sin(beta)
        
        # Pitching moments (L,M,N)
        L = q_dynamic * self.model.b * Cl_diff
        N = q_dynamic * self.model.b * Cn_diff
        
        # Translational motion
        u_b, v_b, w_b = V*np.cos(alpha)*np.cos(beta), V*np.sin(beta), V*np.sin(alpha)*np.cos(beta)
        
        du = (v_b*r - w_b*q) + F_x/self.model.m
        dv = (w_b*p - u_b*r) + F_y/self.model.m
        dw = (u_b*q - v_b*p) + F_z/self.model.m
        
        dV = (u_b*du + w_b*dw) / V
        
        dbeta = (V*dv - v_b*dV) / (V**2 - v_b**2)
        
        # Rotational motion
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = self.model.compute_coef()
        dp = (c1*r + c2*p)*q + c3*L + c4*N
        dr = (c8*p - c2*r)*q + c4*L + c9*N
        
        # Euler angle kinematics
        dphi = p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)
        dpsi = (q*np.sin(phi) + r*np.cos(phi)) / np.cos(theta)
        
        # Combine all derivatives (9 original states + 3 position states)
        return np.array([dbeta, dp, dr, dphi, dpsi])
    
    def analyze_frequency_response(self, A, B):
        # Define system output matrix C - select several key states as outputs here

        C = np.eye(A.shape[0])  # Output all states
        D = np.zeros((A.shape[0], B.shape[1]))  # Direct transmission term

        # Create state-space system
        sys = ctrl.ss(A, B, C, D)
    
        # Define input names (assuming elevator is the first input)
        input_name = ['delta_e_diff']  #
        output_name = ['psi']
    
        # Analyze transfer function
        self.analyze_transfer_function(sys, input_name, output_name)

        # Analyze several key transfer functions and plot
        ##self.analyze_transfer_functions(sys, input_name, output_name)
        
    def analyze_transfer_function(self, sys, input_name, output_name):
        # Transfer function analysis for a single channel
   
        # Check if there is exactly one input and one output (simplified version supports SISO only)
        if len(input_name) != 1 or len(output_name) != 1:
            raise ValueError("This simplified version only supports transfer function extraction for single-input single-output systems!")
    
        input_name = input_name[0]
        output_name = output_name[0]
        input_index = {'delta_e_diff': 0, 'delta_e_sync': 1, 'Thrust': 2}[input_name]
        output_index = 4  # phi:3 psi:4
    
        # Extract A, B, C, D of the subsystem
        A_sub = sys.A  # State matrix remains unchanged
        B_sub = sys.B[:, input_index].reshape(-1, 1)
        C_sub = sys.C[output_index, :].reshape(1, -1)  # Keep only the row for phi/psi
        D_sub = sys.D[output_index, input_index]  # Direct transmission term (usually 0)
    
        # Create subsystem (SISO)
        sys_sub = ctrl.ss(A_sub, B_sub, C_sub, D_sub)
    
        # Extract transfer function (ctrl.ss2tf)
        tf = ctrl.ss2tf(sys_sub)
    
        num, den = tf.num[0][0], tf.den[0][0]  # Get numerator and denominator coefficients
    
        print(f"\nTransfer function: {input_name} -> {output_name}")
        print(f"Numerator coefficients (num): {num}")
        print(f"Denominator coefficients (den): {den}")
    
        # Calculate key modal characteristics
        #self.analyze_modes(sys.A)

    def analyze_transfer_functions(self, sys, input_names, output_names):
    
        # Set frequency range (rad/s)
        omega = np.logspace(-2, 2, 500)

        # Analyze channel: delta_e_diff -> phi Ψ
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

        # 2. Roll channel: delta_e_diff -> phi (roll angle)
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

        # Calculate key modal characteristics
        self.analyze_modes(sys.A)

    def analyze_modes(self, A):
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

        print("\nSystem modal analysis:")
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
                    print("  Type: Weakly damped oscillatory mode")
                elif 0.1 <= zeta < 0.7:
                    print("  Type: Moderately damped oscillatory mode")
                else:
                    print("  Type: Strongly damped oscillatory mode")
            else:
                if eig < 0:
                    print("  Type: Stable aperiodic mode")
                else:
                    print("  Type: Unstable aperiodic mode")
    
            print("-"*50)
            
    def simulate_perturbation(self, t_end=90.0):
        """Small perturbation time domain simulation"""
        from scipy.integrate import odeint
        
        A, B = self.linearize_trim()
        x0 = np.zeros(A.shape[0])  # Small perturbation near trim point
    
        # Define linearized model
        def linear_model(x, t):
            return A @ x
    
        # Simulation time
        t = np.linspace(0, t_end, 900)
    
        # Simulate initial yaw perturbation (Δ=0.01 rad/s)
        x0[1] += 0.1  # p
        #x0[2] += 0.1  # r
        response = odeint(linear_model, x0, t)
        
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': 'Times New Roman',
            'axes.titlesize': 15,
            'axes.labelsize': 15,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
        })
        plt.style.use('seaborn-v0_8-paper')  

        fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
        fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)

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

        # Axis and label settings
        ax.set_xlabel('Time (s)', fontsize=15, labelpad=10)
        ax.set_ylabel('State Variation', fontsize=15, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=15)
    
        # Legend settings
        legend = ax.legend(loc='upper right', framealpha=1, fontsize=12.5,
                      edgecolor='k', fancybox=False)
        legend.get_frame().set_linewidth(0.5)

        # Grid and border
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        
        plt.savefig('trim_results/6_Lateral dynamic response to roll perturbation.png', dpi=600, bbox_inches='tight')
        #plt.savefig('trim_results/7_Lateral dynamic response to yaw perturbation.png', dpi=600, bbox_inches='tight')
        #plt.show()
    
    def plot_trim_results(self, trim_state):
        """Plot combined trim results figure (three-in-one)"""
        fig1 = plt.figure(figsize=(8, 8))
        ax = fig1.add_subplot(111, projection='3d')
        
        """3D aircraft attitude diagram"""
        u_rel, w_rel, delta_e_sync, delta_e_diff, T = trim_state
        alpha = np.arctan2(w_rel, u_rel)
        V = np.sqrt(u_rel**2 + w_rel**2)
        
        # Simplified aircraft model (cuboid fuselage + wings)
        fuselage = np.array([[3,-2,0], [5,0,0], [3,2,0], [-3,2,0], [-3,-2,0]])
        wings = np.array([[1,-4,0], [1,4,0], [-1,4,0], [-1,-4,0]])
        left_e = np.array([[-2.8,-1.5,0], [-2.8,-0.5,0], [-3,-0.5,0], [-3,-1.5,0]])
        right_e = np.array([[-2.8,0.5,0], [-2.8,1.5,0], [-3,1.5,0], [-3,0.5,0]])
        
        # Apply alpha rotation
        rot_a = np.array([
            [np.cos(alpha), 0, np.sin(alpha)],
            [0, 1, 0],
            [-np.sin(alpha), 0, np.cos(alpha)]
        ])
        fuselage = fuselage @ rot_a
        wings = wings @ rot_a
        # Apply elevator rotation (sync)
        rot = np.array([
            [np.cos(delta_e_sync), 0, np.sin(delta_e_sync)],
            [0, 1, 0],
            [-np.sin(delta_e_sync), 0, np.cos(delta_e_sync)]
        ])
        left_e, right_e = (left_e - [-2.8, 0, 0]) @ rot.T + [-2.8, 0, 0], (right_e - [-2.8, 0, 0]) @ rot.T + [-2.8, 0, 0]
        
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
        loc='upper center',          # Position at top center
        bbox_to_anchor=(0.5, 1.0),  # Adjust vertical position (1.1 = above figure)
        frameon=False,              # No border
        fontsize=14,
        handlelength=0              # Hide legend handle (line icon on left)
        ) 
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-3, 3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('H (m)')
        plt.show()


# Main simulation function
def simulate_differential_elevator():
    wind_field = None  # Wind field instance
    model = FixedWingDynamics(wind_field) 
    trim_calc = TrimCalculator(model, wind_field)   
    
    u_rel_trim, w_rel_trim, sync_trim, diff_trim, T_trim = trim_calc.find_trim()
    
    # Observe perturbation response
    trim_calc.simulate_perturbation()
    
simulate_differential_elevator()