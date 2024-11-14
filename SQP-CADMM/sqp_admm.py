from sqp import *
from C_ADMM import *
from movie import make_movie
import sys, os
print(os.getcwd())
sys.path.append(os.getcwd())
from car.dynamics import *
import matplotlib.pyplot as plt
u_bar, x_bar = GenRef(5,5)
plt.figure()
plt.plot(x_bar[:, 0], x_bar[:, 1], 'k-', label = "reference")
plt.show()

make_movie(np.array(x_bar).reshape(-1,len(x_bar),len(x_bar[0])))