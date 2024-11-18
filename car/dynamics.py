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

def GenRef(alpha, beta):
        v_lim = [-10,10]
        delta_lim = [-0.8,0.8]
        Dim_state = 3
        Dim_ctrl = 2
        Ns = 500 
        # generate a nominal trajectory
        x_bar = np.zeros((Ns + 1, Dim_state))
        # x_bar = np.zeros((Ns, Dim_state))
        u_bar = np.zeros((Ns    , Dim_ctrl))

        for k in range(Ns):
            u_act = np.array([ - 1 * (x_bar[k, 0] - 8 + 10 * np.sin(k / 20) + np.sin(k / np.sqrt(7)) ), 
                               np.cos(k / 10 / alpha) * 0.5 + 0.5 * np.sin(k / 10 / np.sqrt(beta))])

            u_act[0] = np.clip(u_act[0],  v_lim[0], v_lim[1])
            u_act[1] = np.clip(u_act[1],  delta_lim[0], delta_lim[1])
            
            u_bar[k, :]     = np.squeeze(u_act)
            x_bar[k + 1, :] = np.squeeze(dubins_car_dynamics(x_bar[k, :],   u_act))
            
        return u_bar, x_bar

