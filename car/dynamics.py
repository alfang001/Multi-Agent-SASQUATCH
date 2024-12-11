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

def compute_measurements(agent_pos, ref_trajectory, noise = np.array([[0.1**2,0],[0,0.02**2]])):
    T = len(ref_trajectory)
    N = agent_pos.shape[0]
    measurements = np.zeros((T,N))
    
    for t in range(T):
        for j in range(N):
            measurements[t,j] = measurement_model(agent_pos[j], ref_trajectory[t,:2],ref_trajectory[t,2], noise)[0]

    return measurements

def GenRef(alpha, beta, Ns = 500):
        v_lim = [-10,10]
        delta_lim = [-0.8,0.8]
        Dim_state = 3
        Dim_ctrl = 2
        
        # generate a nominal trajectory
        x_bar = np.zeros((Ns + 1, Dim_state))
        x_bar[0, :] = np.array([-10, -15, 0])
        # x_bar = np.zeros((Ns, Dim_state))
        u_bar = np.zeros((Ns    , Dim_ctrl))

        for k in range(Ns):
            # u_act = np.array([ - 1 * (x_bar[k, 0] - 8 + 10 * np.sin(k / 20) + np.sin(k / np.sqrt(7)) ), 
            #                    np.cos(k / 10 / alpha) * 0.5 + 0.5 * np.sin(k / 10 / np.sqrt(beta))])

            v = 8.0
            amplitude = 5
            frequency = 1
            dt = 0.05
            beta = amplitude * np.sin(2 * np.pi * frequency * k * dt)
            u_act = np.array([v, beta])

            u_act[0] = np.clip(u_act[0],  v_lim[0], v_lim[1])
            u_act[1] = np.clip(u_act[1],  delta_lim[0], delta_lim[1])
            
            u_bar[k, :]     = np.squeeze(u_act)
            x_bar[k + 1, :] = np.squeeze(dubins_car_dynamics(x_bar[k, :],   u_act))
            
        return u_bar, x_bar

def GenRef2(checkpoints, Ns=500):
    """Given a set of cheeckpoints, generate a trajectory with Ns points

    Args:
        checkpoints (np.array): Array of checkpoints, each row is a checkpoint (x,y)
        Ns (int, optional): number of steps/points in trajectory. Defaults to 500.
    
    Returns:
        np.array: Generated trajectory with Ns points
    """
    v_lim = [-10,10]
    delta_lim = [-0.8,0.8]
    Dim_state = 3
    Dim_ctrl = 2
    
    x_bar = np.zeros((Ns + 1, Dim_state))
    u_bar = np.zeros((Ns, Dim_ctrl))

    # Interpolate between checkpoints and make sure x_bar has Ns + 1 points
    x_bar[:, 0] = np.interp(np.linspace(0, Ns, Ns + 1), np.arange(checkpoints.shape[0]), checkpoints[:, 0])
    x_bar[:, 1] = np.interp(np.linspace(0, Ns, Ns + 1), np.arange(checkpoints.shape[0]), checkpoints[:, 1])

    return u_bar, x_bar

