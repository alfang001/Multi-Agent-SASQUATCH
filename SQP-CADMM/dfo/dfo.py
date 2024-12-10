import numpy as np
from utils import all_points_within_threshold, f_derivative


def distributed_optimize(z, connectivity_matrix, weights, agent_pos, alpha_0 = 0.8, max_iter = 500, error=0.1, break_early=True):
    sensing_range = 50
    def get_neighbors(adj_matrix, i):
    
        # Check the ith row of the adjacency matrix. Neighbors of agent i are where the value is 1.
        neighbors = np.where(adj_matrix[i] == 1)[0]
       
        return neighbors.tolist()

    n = len(agent_pos)  
    local_estimates = np.random.randn(n, 2)

    for k in range(1, max_iter + 1):
        alpha_k = alpha_0 / np.sqrt(k)  # Decreasing step size
        new_x = np.copy(local_estimates)
        
        for i in range(n):
            neighbors = get_neighbors(connectivity_matrix, i)
            
            weighted_sum = weights[i] * local_estimates[i] #if len(neighbors)>0 else local_estimates[i]
            for j in neighbors:
                weighted_sum += weights[i] *local_estimates[j]
            for j in neighbors:
                
                # new_x[i] = weighted_sum - alpha_k * f_derivative(euclidean( robot_positions[j] , local_estimates[j]), z[j])
                new_x[i] = weighted_sum - alpha_k * f_derivative(local_estimates[j],agent_pos[j],z[j]).full().squeeze()
                # new_x[i] = weighted_sum - alpha_k * f_derivative(local_estimates[j],robot_positions[j],z_tilde[j]).full().squeeze()
            # for j in neighbors:
            #     # new_x[i] = weighted_sum - alpha_k * f_derivative(euclidean( robot_positions[j] , local_estimates[j]), z[j])
            #     new_x[i] = weighted_sum - alpha_k * f_derivative(local_estimates[j],agent_pos[j],z[j]).full().squeeze()
            #     # new_x[i] = weighted_sum - alpha_k * f_derivative(local_estimates[j],robot_positions[j],z_tilde[j]).full().squeeze()
        local_estimates = new_x  # Update the estimates for the next iteration

        # If the difference between all the estimates is less than the error, break the loop
        if break_early and bool(all_points_within_threshold(local_estimates, error)):
            break

    estimated_pos = local_estimates
    indices = np.where(z<=sensing_range)[0]
    estimated_pos = estimated_pos[indices]
    return estimated_pos
