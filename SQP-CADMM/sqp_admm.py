import sys, os
print(os.getcwd())
sys.path.append(os.getcwd())
from sqp import sqp_update, bfgs_update, finite_difference_gradient, phi, phi_derivative
from C_ADMM import agent_objective
from movie import make_movie
from car.dynamics import dubins_car_dynamics, measurement_model, GenRef
import matplotlib.pyplot as plt
import random
from graph_utils import get_neighbors, connected, construct_graph, construct_matrix
import cvxpy as cp
import numpy as np

agent_pos = np.array([(random.uniform(-20,20),random.uniform(-20,20)) for _ in range(15)])
R = 8
T = 500  # Time horizon
state_dim = 3  # [x_position, y_position, heading]
N = 15
tolerance = 1e-4
distance = 1.5
adj_matrix = construct_matrix(agent_pos,R)
def sqp_admm(adj_matrix, c_admm_max_iter = 200): #, c_admm_max_iter = 200
    rho = 1.0 #augmented laplacian parameter
    # local_estimates = np.random.randn(N, 3)
    x_traj = np.zeros((T, state_dim))  # Estimated trajectory, initialize to zeros
    local_estimates = [np.random.rand(3) for _ in range(N)]  # Local copies of x
    z = np.copy(local_estimates)  # Consensus variables
    lambda_ = [np.zeros(2) for _ in range(N)]  # Lagrange multipliers
    x_k = [np.random.rand(2) for _ in range(N)]
    H = np.eye(state_dim)
    for t in range(T):
        # new_x = np.copy(local_estimates)
        for it in range(c_admm_max_iter):
            grad_f = [finite_difference_gradient(agent_objective, x_k[i]) for i in range(N)]
            x_new = []
            for i in range(N):
                x_var = cp.Variable(2)
                neighbors = get_neighbors(adj_matrix, i)
                phi_i = phi(x_k[i],agent_pos[i],distance)
                grad_phi_i = phi_derivative(x_k[i],agent_pos[i],distance)
                # s = 
                # y = 
                # H = bfgs_update(H, (local_estimates[i] - x_k[t]),)
                obj = (
                    grad_f[i].T @ (x_var - x_k[i]) +
                    0.5 * cp.quad_form(x_var - x_k[i], H[i]) +
                    sum([lambda_[i][j] @ (x_var - z[j]) + (rho / 2) * cp.sum_squares(x_var - z[j])
                        for j in range(N) if adj_matrix[i, j] == 1])
                )
                objective = cp.Minimize(obj)

                # Define the constraints
                constraints = [
                    grad_phi_i.T @ (x_var - x_k[i]) + phi_i <= 0
                ]

                # Solve the optimization problem
                prob = cp.Problem(objective, constraints)
                prob.solve()
                x_new.append(x_var.value)

                local_estimates[i] = x_var.value
            for i in range(N):
                # Compute new gradient at x[i]
                grad_f_new = finite_difference_gradient(agent_objective, x_new[i])

                # Compute s and y for BFGS
                s = x_new[i] - x_k[i]
                y = grad_f_new - grad_f[i]

                # Update Hessian H[i] with BFGS
                H[i] = bfgs_update(H[i], s, y)
                

                # Update x_k[i] to the new x value for the next iteration
                x_k[i] = x_new[i]
            for i in range(N):
                for j in range(N):
                    if adj_matrix[i, j] == 1:
                        z[i] = 0.5 * (local_estimates[i] + local_estimates[j])

            for i in range(N):
                for j in range(N):
                    if adj_matrix[i, j] == 1:
                        lambda_[i][j] += rho * (local_estimates[i] - z[i])
            primal_residual = sum([np.linalg.norm(estimate_trajectory[i] - z[i]) for i in range(N)])
            dual_residual = sum([np.linalg.norm(z[i] - z[j]) for i in range(N) for j in range(N) if adj_matrix[i, j] == 1])
            
            if (primal_residual < tolerance and dual_residual < tolerance):
                print(f"Convergence reached at iteration {it}")
                break
            x_traj[t, :] = np.mean(z, axis=0)
        # if i > c_admm_max_iter:
        #     print("max iter reached")
        #     break
    
    return x_traj
def estimate_trajectory():
    return



u_bar, x_bar = GenRef(2,2)
estimated_trajectory = sqp_admm(adj_matrix)
# estimated_trajectory = np.zeros((x_bar.shape[0] -1, x_bar.shape[1]))

plt.figure()
plt.plot(x_bar[:, 0], x_bar[:, 1], 'k-', label = "reference")
plt.show()

make_movie(np.array(x_bar).reshape(-1,len(x_bar),len(x_bar[0])), estimated_trajectory.reshape(-1,T,state_dim), agent_pos)




# def sqp_admm(measured_pos,adj_matrix, c_admm_max_iter = 200):

#     # local_estimates = np.random.randn(N, 3)
#     x_traj = np.zeros((T, state_dim))  # Estimated trajectory, initialize to zeros
#     local_estimates = [np.random.rand(2) for _ in range(N)]  # Local copies of x
#     z = np.copy(local_estimates)  # Consensus variables
#     lambda_ = [np.zeros(2) for _ in range(N)]  # Lagrange multipliers

#     H = np.eye(state_dim)
#     for t in range(T):
#         new_x = np.copy(local_estimates)

#         for i in range(N):
#             x_var = cp.Variable(2)
#             neighbors = get_neighbors(adj_matrix, i)
#             s = 
#             y = 
#             H = bfgs_update(H, (local_estimates[i] - x_traj[t]),)

#     estimated_trajectory = local_estimates
#     return estimated_trajectory