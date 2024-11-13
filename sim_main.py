import numpy as np
from matplotlib import pyplot as plt

from vehicle.dubins_car import DubinsCar


def main():
    car = DubinsCar(x=0, y=0, theta=0, v=0)
    T = 25
    dt = 0.01
    
    # Create log for car state (T/dt) x 4
    x_log = np.zeros((int(T/dt), 4))
    
    for i in range(int(T/dt)):
        # Control input (velocity, steering angle)
        u = (1, 0)
        x_log[i, :] = car.step(u, dt)
    
    plt.figure()
    plt.plot(x_log[:, 0], x_log[:, 1])
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    main()