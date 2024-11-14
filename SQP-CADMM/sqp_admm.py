from sqp import *
from C_ADMM import *
from movie import make_movie
import sys, os
print(os.getcwd())
sys.path.append(os.getcwd())
from car.dynamics import *
import matplotlib.pyplot as plt
import random
agent_pos = np.array([(random.uniform(-20,20),random.uniform(-20,20)) for _ in range(15)])
T = 500  # Time horizon
state_dim = 3  # [x_position, y_position, heading]

n = 15
def sqp_admm():
    local_estimates = np.random.randn(n, 3)
    x_traj = np.zeros((T, state_dim))  # Estimated trajectory, initialize to zeros
    
    H = np.eye(state_dim)
    for t in range(T):
        for a in range(n):
            s = 
            y = 
            H = bfgs_update(H, (local_estimates[a] - x_traj[t]),)
    return estimated_trajectory
u_bar, x_bar = GenRef(2,2)
estimated_trajectory = sqp_admm()


plt.figure()
plt.plot(x_bar[:, 0], x_bar[:, 1], 'k-', label = "reference")
plt.show()

make_movie(np.array(x_bar).reshape(-1,len(x_bar),len(x_bar[0])), estimated_trajectory.reshape(-1,T,state_dim), agent_pos)