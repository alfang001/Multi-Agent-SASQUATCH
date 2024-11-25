import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),"SQP-CADMM"))
import random

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from dfo import distributed_optimize
from graph_utils import (
    compute_weights,
    construct_graph,
    construct_matrix,
    get_neighbors,
)
from movie import make_movie

from car.dynamics import GenRef, compute_measurements

u_bar, x_bar = GenRef(2,2,Ns=50)
N=15
sensing_range = 15
R=10


T = len(x_bar)
state_dim = len(x_bar[0])
# agent_pos = np.array([(random.uniform(-20,20),random.uniform(-20,20)) for _ in range(15)])
# agent_pos = np.hstack((np.arange(-5,7,1).reshape(-1,1), -2*np.ones((12,)).reshape(-1,1)))
agent_index = np.arange(0,12,1)
agent_pos = np.hstack((20/2*np.cos(2*np.pi/N *agent_index).reshape(-1,1), 20/2*np.sin(2*np.pi/N *agent_index).reshape(-1,1)))
# estimated_trajectory = np.zeros((x_bar.shape[0] -1, x_bar.shape[1]))

def trajectory_tracking(agent_pos, ref_trajectory):
    
    connectivity_matrix = construct_matrix (agent_pos,R=R)
    weights = compute_weights(agent_pos, R=R)
    measurements = compute_measurements(agent_pos, ref_trajectory)
    estimated_trajectory = []
    for t in range(T):
        estimated_coord = distributed_optimize(measurements[t], connectivity_matrix, weights, agent_pos, max_iter=500)
        estimated_coord = np.mean(estimated_coord, axis=0)
        estimated_trajectory.append(estimated_coord)
    return estimated_trajectory
estimated_trajectory = np.array(trajectory_tracking(agent_pos, x_bar))

# Plotting results
plt.figure()

# Plotting reference and estimated trajectories
plt.plot(x_bar[:, 0], x_bar[:, 1], 'k-', color='black', label = "Reference Trajectory")
plt.plot(estimated_trajectory[:,0],estimated_trajectory[:,1],color = 'green',label = 'Estimated Trajectory')

# Plot the agents and the communication channels
adj_matrix = construct_matrix (agent_pos,R=R)
first_line = True
for i in range(adj_matrix.shape[0]):
    for j in range(adj_matrix.shape[1]):
        if adj_matrix[i,j] == 1:
            if first_line:
                first_line = False
                plt.plot([agent_pos[i,0],agent_pos[j,0]],[agent_pos[i,1],agent_pos[j,1]], color = 'blue', label = "Channel")
            else:
                plt.plot([agent_pos[i,0],agent_pos[j,0]],[agent_pos[i,1],agent_pos[j,1]], color = 'blue')
plt.plot(agent_pos[:,0], agent_pos[:,1], 'o', color = 'orange', label='Sensor (Agent) Position')

# Making the plot look nice
plt.legend()
plt.xlabel('X Coordinate (meters)')
plt.ylabel('Y Coordinate (meters)')
plt.title('Trajectory Tracking with SQP-CADMM')
plt.grid(True)
plt.show()
# import pdb; pdb.set_trace()

# Make animation of the trajectories
make_movie(np.array(x_bar).reshape(-1,len(x_bar),len(x_bar[0])), estimated_trajectory.reshape(-1,T,state_dim-1), agent_pos)
