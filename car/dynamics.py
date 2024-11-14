import numpy as np
L = 2.0  # Distance between front and rear axles (wheelbase)

def dubins_car_dynamics(x, u, dt=0.05):
    px, py, theta = x
    v, beta = u
    
    # Discrete time update using forward Euler
    px_next = px + dt * v * np.cos(theta)
    py_next = py + dt * v * np.sin(theta)
    theta_next = theta + dt * v / L * np.tan(beta)
    return np.array([px_next, py_next, theta_next])

def measurement_model(agent_position, target_position, target_heading, noise_cov):
    '''
    Applies the measurement model according to the paper
    '''
    # Range measurement (distance)
    dx = target_position[0] - agent_position[0]
    dy = target_position[1] - agent_position[1]
    range_meas = np.sqrt(dx**2 + dy**2)
    
    # Bearing measurement (angle relative to agent's heading)
    bearing_meas = np.arctan2(dy, dx) - target_heading
    
    # Apply Gaussian noise to measurements
    noise = np.random.multivariate_normal([0, 0], noise_cov)
    measurement = np.array([range_meas, bearing_meas]) + noise
    
    return measurement

