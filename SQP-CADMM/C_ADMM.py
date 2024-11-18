import numpy as np
import casadi as ca
target_pos_ca = ca.SX.sym('x', 3)  # 2x1
agent_pos_ca = ca.SX.sym('a', 2)  # 2x1
distance = ca.SX.sym('d')     
f =  distance**2 - (ca.norm_2(target_pos_ca[:2] - agent_pos_ca))**2
phi = ca.Function('phi',[target_pos_ca,agent_pos_ca,distance],[f])
df_dx = ca.gradient(f, target_pos_ca[:2])
phi_derivative = ca.Function('f_derivative', [target_pos_ca, agent_pos_ca, distance], [df_dx])
def agent_objective(x_traj, observations, dynamics_func, measurement_func,mu=np.zeros(3)):
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
    P_inv = np.diag([0.01,0.01,0.005])
    Q_inv = np.diag([0.1**2, 0.05**2])
    L_inv = np.diag([0.6**2,0.6**2,0.1**2])
    # mu = np.zeros(3)
    T = len(x_traj) - 1
    cost = 0.0
    
    # Initial state prior term
    cost += (x_traj[0] - mu).T @ L_inv @ (x_traj[0] - mu)
    
    # Dynamics and measurement terms
    for t in range(T):

        # Dynamics term
        x_pred = dynamics_func(x_traj[t], [1.0, 0.1], 0.1)  
        cost += (x_traj[t+1] - x_pred).T @ P_inv @ (x_traj[t+1] - x_pred)
        
        # Measurement term
        y_obs = observations[t]
        y_pred = measurement_func([2.0, 2.0], x_traj[t][:2], x_traj[t][2], np.eye(2))  
        error = y_obs - y_pred
        cost += error.T @ Q_inv @ error
    
    return cost

# def agent_objective():
#     cost = 0.0
#     return cost
