import numpy as np
from scipy.optimize import minimize

def bfgs_update(H, s, y):
    """
    BFGS update to approximate the Hessian matrix. Note: this warm starts the solver by using the previous H update

    
    Parameters:
    - H: Current Hessian approximation
    - s: Step taken (x_{k+1} - x_k)
    - y: Gradient change (grad f(x_{k+1}) - grad f(x_k))
    
    Returns:
    - Updated Hessian approximation
    """
    rho = 1.0 / (y.T @ s)
    # I = np.eye(len(s))
    # V = I - rho * np.outer(s, y)
    # H_next = V.T @ H @ V + rho * np.outer(y, y)
    H_s = H @ s
    term1 = (np.outer(H_s, H_s)) / (s.T @ H_s)
    term2 = (rho * np.outer(y, y))
    H_next = H - term1 + term2
    return H_next


def finite_difference_gradient(func, x, epsilon=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_forward = np.copy(x)
        x_forward[i] += epsilon
        x_backward = np.copy(x)
        x_backward[i] -= epsilon
        grad[i] = (func(x_forward) - func(x_backward)) / (2 * epsilon)
    return grad

def sqp_update(x0, objective_func, observations, dynamics_func, measurement_func, P_inv, Q_inv, L_inv, mu, max_iters=10):
    x_traj = x0
    H = np.eye(len(x0))  # Initial Hessian
    
    for _ in range(max_iters):
        # Define the local objective function with fixed observations and parameters
        local_objective = lambda x: objective_func(x.reshape(-1, 3), observations, dynamics_func, measurement_func, P_inv, Q_inv, L_inv, mu)
        
        # Compute gradient
        grad_f = finite_difference_gradient(local_objective, x_traj.flatten())
        
        # Define QP subproblem
        def quadratic_subproblem(d):
            d = d.reshape(-1, 3)
            return grad_f.T @ d.flatten() + 0.5 * d.flatten().T @ H @ d.flatten()

        # Solve QP subproblem
        result = minimize(quadratic_subproblem, np.zeros_like(x_traj.flatten()), method='trust-constr')
        d = result.x.reshape(x_traj.shape)  # Optimal step direction

        # Update trajectory
        x_next = x_traj + d

        # Calculate new gradient
        grad_f_next = finite_difference_gradient(local_objective, x_next.flatten())
        
        # BFGS update for Hessian (warm-start)
        s = x_next.flatten() - x_traj.flatten()
        y = grad_f_next - grad_f
        H = bfgs_update(H, s, y)

        # Update trajectory
        x_traj = x_next

        # Convergence check (optional, based on step size)
        if np.linalg.norm(d) < 1e-6:
            break

    return x_traj



def consensus_admm(
    agents_data,
    rho=1.0,
    alpha=1.0,
    num_iterations=100,
    tolerance=1e-4
):
    """
    Consensus ADMM for distributed optimization.
    
    Parameters:
        agents_data (list of dict): List containing the data for each agent. Each dict should have 'cost_func' (function for local cost)
                                    and 'init_x' (initial value of the variable for each agent).
        rho (float): Augmented Lagrangian parameter.
        alpha (float): Over-relaxation parameter.
        num_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.
        
    Returns:
        numpy.ndarray: The consensus value (approximated solution) after iterations.
    """

    num_agents = len(agents_data)
    x_vals = np.array([agent['init_x'] for agent in agents_data])  # Initialize variables for each agent
    z = np.mean(x_vals, axis=0)  # Initialize global consensus variable
    u = np.zeros_like(x_vals)    # Initialize dual variables (Lagrange multipliers)
    
    def update_x(agent, x, z, u, rho):
        """Local update for x using the agent's cost function."""
        return agent['cost_func'](x, z, u, rho)
    
    for iteration in range(num_iterations):
        # Step 1: Local update for each agent
        for i, agent in enumerate(agents_data):
            x_vals[i] = update_x(agent, x_vals[i], z, u[i], rho)

        # Step 2: Update the global consensus variable
        z_prev = z
        z = np.mean(x_vals + u, axis=0)

        # Step 3: Dual variable update for each agent
        u += alpha * (x_vals - z)

        # Check convergence
        if np.linalg.norm(z - z_prev) < tolerance:
            print(f"Consensus ADMM converged in {iteration + 1} iterations.")
            break

    return z





def agent_cost_func(x, z, u, rho):
    
    return (1.0 / (1 + rho)) * (rho * (z - u) + 1)

# Initialize agents with initial values and their cost functions
agents_data = [
    {'cost_func': agent_cost_func, 'init_x': np.array([0.0])},
    {'cost_func': agent_cost_func, 'init_x': np.array([2.0])},
    {'cost_func': agent_cost_func, 'init_x': np.array([3.0])}
]

# Run consensus ADMM
# result = consensus_admm(agents_data)
# print("Consensus solution:", result)
