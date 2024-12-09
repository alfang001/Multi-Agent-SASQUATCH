import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),"SQP-CADMM"))

import matplotlib.pyplot as plt
import numpy as np
from constants import *
from dfo import distributed_optimize
from graph_utils import compute_weights, construct_matrix
from movie import make_movie
from scipy.spatial.distance import directed_hausdorff
from utils import calculate_rmse

from car.dynamics import GenRef, GenRef2, compute_measurements


def trajectory_tracking(agent_pos: np.array, ref_trajectory, sensing_radius: float = 10) -> np.array:
    """Tracks trajectory of a vehicle following the reference trajectory.

    Args:
        agent_pos (np.array): Array containing the positions of the agents in (x,y) coords
        ref_trajectory (np.array): Array containing the reference trajectory

    Returns:
        np.array: Array containing the estimated trajectory at certain time steps
    """
    T = len(ref_trajectory) # Number of time steps
    connectivity_matrix = construct_matrix (agent_pos,R=sensing_radius)
    weights = compute_weights(agent_pos, R=sensing_radius)
    measurements = compute_measurements(agent_pos, ref_trajectory)
    estimated_trajectory = []
    for t in range(T):
        estimated_coord = distributed_optimize(measurements[t], connectivity_matrix, weights, agent_pos, max_iter=500)
        estimated_coord = np.mean(estimated_coord, axis=0)
        estimated_trajectory.append(estimated_coord)
    return estimated_trajectory

def main():
    # Setting reference trajectory and parameters
    # u_bar, x_bar = GenRef(2,2,Ns=50)
    checkpoints = np.array([
        [-15, 4.5],
        [-8, 7],
        [-4, 7],
        [0, 6.8],
        [4, 6.2],
        [8, 5.5],
        [8.3, 5.0],
        [8.7, 4.5],
        [11, 2.3],
        [12.5, 0.0],
        [11, -2.0],
        [10, -7],
        [7.5, -7.5],
        [6, -7.4],
        [5, -8],
        [4, -7],
        [2, -6]
    ])
    u_bar, x_bar = GenRef2(checkpoints,Ns=250)
    R=10 # Sensing radius
    N=15 # Number of agents
    T = len(x_bar) # Number of time steps
    state_dim = len(x_bar[0])
    # agent_pos = np.array([(random.uniform(-20,20),random.uniform(-20,20)) for _ in range(15)])
    # agent_pos = np.hstack((np.arange(-5,7,1).reshape(-1,1), -2*np.ones((12,)).reshape(-1,1)))
    agent_index = np.arange(0,N,1)
    agent_pos = np.hstack((20/2*np.cos(2*np.pi/N *agent_index).reshape(-1,1), 20/2*np.sin(2*np.pi/N *agent_index).reshape(-1,1)))
    # estimated_trajectory = np.zeros((x_bar.shape[0] -1, x_bar.shape[1]))
    agent_positions = [PAPER_AGENT_POS, agent_pos, RANDOM_AGENT_POS_1, RANDOM_AGENT_POS_2, RANDOM_AGENT_POS_3, RANDOM_AGENT_POS_4, RANDOM_AGENT_POS_5, RANDOM_AGENT_POS_6]

    for sim_run in range(len(agent_positions)):
        curr_agent_pos = agent_positions[sim_run]
        estimated_trajectory = np.array(trajectory_tracking(curr_agent_pos, x_bar, R))

        # Calculate the error
        rmse = calculate_rmse(estimated_trajectory, x_bar[:, :2])
        print(f"RMSE: {rmse}")

        # Calculate the Hausdorff distance, which is the furthest distance between the two trajectories
        # Note: we use the max since we want tghe furthest distance between the two trajectories, whereas
        # only taking one of the results is a directed Hausdorff distance.
        hausdorff_distance = max(directed_hausdorff(estimated_trajectory, x_bar[:, :2])[0], directed_hausdorff(x_bar[:, :2], estimated_trajectory)[0])
        print(f"General Hausdorff Distance: {hausdorff_distance}")

        # Plotting results
        plt.figure()

        # Plotting reference and estimated trajectories
        plt.plot(x_bar[:, 0], x_bar[:, 1], 'k-', label = "Reference Trajectory")
        plt.plot(estimated_trajectory[:,0],estimated_trajectory[:,1],color = 'green',label = 'Estimated Trajectory')

        # Plot the agents and the communication channels
        adj_matrix = construct_matrix(curr_agent_pos,R=R)
        first_line = True
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if adj_matrix[i,j] == 1:
                    if first_line:
                        first_line = False
                        plt.plot([curr_agent_pos[i,0],curr_agent_pos[j,0]],[curr_agent_pos[i,1],curr_agent_pos[j,1]], color = 'blue', label = "Channel", alpha = 0.25)
                    else:
                        plt.plot([curr_agent_pos[i,0],curr_agent_pos[j,0]],[curr_agent_pos[i,1],curr_agent_pos[j,1]], color = 'blue', alpha = 0.25)
        plt.plot(curr_agent_pos[:,0], curr_agent_pos[:,1], 'o', color = 'orange', label='Sensor (Agent) Position')

        # Making the plot look nice
        plt.legend()
        plt.xlabel('X Coordinate (meters)')
        plt.ylabel('Y Coordinate (meters)')
        plt.title('Trajectory Tracking with SQP-CADMM')
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        plt.grid(True)
        plt.savefig(f'data/trajectory_tracking_{sim_run + 1}.png')
        plt.show()


        # Make animation of the trajectories
        make_movie(np.array(x_bar).reshape(-1,len(x_bar),len(x_bar[0])), estimated_trajectory.reshape(-1,T,state_dim-1), curr_agent_pos)

if __name__ == "__main__":
    main()