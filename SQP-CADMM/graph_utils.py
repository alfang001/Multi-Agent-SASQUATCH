import random

import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean
import networkx as nx
def compute_weights(pos_arr,R):
    weights = np.zeros((pos_arr.shape[0],))
    for i in range( pos_arr.shape[0] ):
        wi = 0
        coord1 = pos_arr[i]
        for j in range( pos_arr.shape[0] ):
            coord2 = pos_arr[j]
            if i != j and connected(coord1, coord2,R):
                wi = wi +1
        if wi != 0:
            weights[i] = 1/(wi+1)
    return weights
# def compute_weights_sensing(measurement, pos_arr,R ,sensing_range=15):
    
#     weights = np.zeros((pos_arr.shape[0],))
#     for i in range( pos_arr.shape[0] ):
#         wi = 0
#         coord1 = pos_arr[i]
#         for j in range( pos_arr.shape[0] ):
#             coord2 = pos_arr[j]
#             if i != j and connected(coord1, coord2,R):
#                 wi = wi +1
#         if wi != 0:
#             weights[i] = 1/(wi+1)
#     return weights
def construct_graph(pos_arr):
    G = nx.graph()
    G.add_nodes_from(range(pos_arr.shape[0]))

    for i in range( pos_arr.shape[0] ):
        coord1 = pos_arr[i]
        for j in range( pos_arr.shape[0] ):
            coord2 = pos_arr[j]
            if not np.array_equal(coord1, coord2) and connected(coord1, coord2):
                G.add_edge(i,j)
    return G

def construct_matrix (pos_arr,R):
    connectivity_matrix = np.zeros((pos_arr.shape[0], pos_arr.shape[0]))
    for i in range( pos_arr.shape[0] ):
        
        coord1 = pos_arr[i]
        for j in range( pos_arr.shape[0] ):
            coord2 = pos_arr[j]
            if i != j and connected(coord1, coord2,R):
                connectivity_matrix[i,j] = 1
        
    return connectivity_matrix

def connected(coordinate1, coordinate2, R):
    return True if euclidean(coordinate1, coordinate2) < R else False

def get_neighbors(adj_matrix, i):

    # Check the ith row of the adjacency matrix. Neighbors of agent i are where the value is 1.
    neighbors = np.where(adj_matrix[i] == 1)[0]
    
    return neighbors.tolist()

def generate_agent_positions():
    agent_size = 0.25  # radius of the agent in meters
    safe_distance = 2 * agent_size  # minimum allowed distance between agents
    num_agents = 15
    area_size = 40  # since we want to place agents between -20 to 20 in both x and y directions
    agent_pos = []
    
    while len(agent_pos) < num_agents:
        new_pos = (random.uniform(-area_size/2, area_size/2), random.uniform(-area_size/2, area_size/2))
        # Check if new_pos is far enough from all existing positions
        if all(np.linalg.norm(np.array(new_pos) - np.array(pos)) >= safe_distance for pos in agent_pos):
            agent_pos.append(new_pos)
    return np.array(agent_pos)
