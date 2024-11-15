import numpy as np
from scipy.spatial.distance import euclidean
import networkx as nx

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