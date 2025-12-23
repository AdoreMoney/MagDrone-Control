import numpy as np
from adrc_controller import ADRCController  # Assume your ADRC controller is in this module
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

# Define parameter ranges
param_ranges = {
    'w0': (1, 20),       # Bandwidth
    'h0': (50, 200),     # Tracking factor
    'b0': (0.5, 2.0),    # Control gain
    'beta01': (1, 50),   # ESO parameter
    'beta02': (1, 500),  # ESO parameter
    'beta03': (1, 5000), # ESO parameter
    'alpha1': (0.01, 0.1),
    'alpha2': (0.01, 0.1),
    'delta': (0.001, 0.1)
}

# Define performance metric function
def performance_metric(params, num_simulations=10, sim_duration=20, dt=0.01):
    """
    Evaluate the performance of a set of parameters and return a comprehensive performance score.
    """
    total_overshoot = 0
    total_settling_time = 0
    for _ in range(num_simulations):
        # Initialize ADRC controller
        adrc = ADRCController(V_trim=10, h0=params['h0'], w0=params['w0'], b0=params['b0'])
        adrc.beta01 = params['beta01']
        adrc.beta02 = params['beta02']
        adrc.beta03 = params['beta03']
        adrc.alpha1 = params['alpha1']
        adrc.alpha2 = params['alpha2']
        adrc.delta = params['delta']

        # Initialize states
        current_roll = 0.0
        current_yaw = 0.0
        current_beta = 0.0
        target_heading = np.deg2rad(15)  # Target heading: 15 degrees

        # Store data for performance metric calculation
        heading_errors = []
        times = np.arange(0, sim_duration, dt)

        # Simulation loop
        for t in times:
            control_action, _ = adrc.compute_control(current_roll, current_yaw, current_beta, dt, target_heading)
            # A UAV dynamics model is required here to update current_roll, current_yaw, current_beta
            # This is just an example since no specific dynamics model is available
            current_heading = current_yaw - current_beta
            heading_error = adrc._normalize_angle(target_heading - current_heading)
            heading_errors.append(np.rad2deg(heading_error))

            # Update states (assuming a simple integral model)
            current_roll += control_action * dt
            current_yaw += (control_action + np.random.normal(0, 0.1)) * dt  # Add noise
            current_beta = np.sin(current_yaw) * 0.1  # Simple sideslip angle model

        # Calculate performance metrics
        max_overshoot = max(heading_errors) - 15  # Overshoot
        settling_time = None
        threshold = 1.0  # Set error threshold
        for i in range(len(times)):
            if abs(heading_errors[i] - 15) <= threshold:
                settling_time = times[i]
                break

        total_overshoot += max_overshoot if max_overshoot > 0 else 0
        total_settling_time += settling_time if settling_time is not None else sim_duration

    avg_overshoot = total_overshoot / num_simulations
    avg_settling_time = total_settling_time / num_simulations

    # Comprehensive performance score (smaller is better)
    score = avg_overshoot + 0.1 * avg_settling_time
    return score

# Define objective function (to be minimized)
def objective_function(params_list):
    params = {
        'w0': params_list[0],
        'h0': params_list[1],
        'b0': params_list[2],
        'beta01': params_list[3],
        'beta02': params_list[4],
        'beta03': params_list[5],
        'alpha1': params_list[6],
        'alpha2': params_list[7],
        'delta': params_list[8]
    }
    return performance_metric(params)

# Random search for parameter tuning
def random_search(num_iterations=100, num_params=9):
    best_score = float('inf')
    best_params = None
    for i in range(num_iterations):
        params_list = []
        for _ in range(num_params):
            param_name = list(param_ranges.keys())[_]
            param_range = param_ranges[param_name]
            params_list.append(random.uniform(param_range[0], param_range[1]))
        current_score = objective_function(params_list)
        if current_score < best_score:
            best_score = current_score
            best_params = params_list.copy()
        print(f"Iteration {i+1}: Score = {current_score}, Best Score = {best_score}")
    return best_params, best_score

# Execute random search
best_params, best_score = random_search(num_iterations=50)
print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Convert best parameters to dictionary format
param_names = ['w0', 'h0', 'b0', 'beta01', 'beta02', 'beta03', 'alpha1', 'alpha2', 'delta']
best_params_dict = {name: val for name, val in zip(param_names, best_params)}
print("Best Parameters Dictionary:", best_params_dict)