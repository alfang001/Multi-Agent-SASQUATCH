import numpy as np


def agent_objective(x_traj, observations, dynamics_func, measurement_func, P_inv, Q_inv, L_inv, mu):
    """
    Objective function for each agent to minimize based on observations and dynamics.
    
    Parameters:
    - x_traj: Trajectory estimate for the target across time [T, 3]
    - observations: List of measurements [range, bearing] over time
    - dynamics_func: Dynamics model of the target
    - measurement_func: Measurement model for the agent
    - P_inv: Inverse covariance of process noise
    - Q_inv: Inverse covariance of measurement noise
    - L_inv: Inverse covariance of initial state prior
    - mu: Prior mean of initial state
    
    Returns:
    - cost: Objective function value
    """
    T = len(x_traj) - 1
    cost = 0.0
    
    # Initial state prior term
    cost += (x_traj[0] - mu).T @ L_inv @ (x_traj[0] - mu)
    
    # Dynamics and measurement terms
    for t in range(T):
        # Dynamics term
        x_pred = dynamics_func(x_traj[t], [1.0, 0.1], 0.1)  # Example control and timestep
        cost += (x_traj[t+1] - x_pred).T @ P_inv @ (x_traj[t+1] - x_pred)
        
        # Measurement term
        y_obs = observations[t]
        y_pred = measurement_func([2.0, 2.0], x_traj[t][:2], x_traj[t][2], np.eye(2))  # Example agent position
        error = y_obs - y_pred
        cost += error.T @ Q_inv @ error
    
    return cost
