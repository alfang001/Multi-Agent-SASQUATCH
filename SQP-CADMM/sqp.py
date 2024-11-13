import numpy as np

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


# Example Usage
# Define cost functions for each agent
def agent_cost_func(x, z, u, rho):
    # Example quadratic cost: (x - 1)^2
    return (1.0 / (1 + rho)) * (rho * (z - u) + 1)

# Initialize agents with initial values and their cost functions
agents_data = [
    {'cost_func': agent_cost_func, 'init_x': np.array([0.0])},
    {'cost_func': agent_cost_func, 'init_x': np.array([2.0])},
    {'cost_func': agent_cost_func, 'init_x': np.array([3.0])}
]

# Run consensus ADMM
result = consensus_admm(agents_data)
print("Consensus solution:", result)
