import numpy as np


class DubinsCar:
    """Initialize the Dubins car with the given parameters
        param x: starting x position of the car
        param y: starting y position of the car
        param theta: starting orientation of the car
        param v: starting velocity of the car
        param L: wheelbase length of the car
    """
    def __init__(self, x=0.0, y=0.0, theta=0.0, v=1.0, L=2.5):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.L = L

    """Step the simulation of dynamics dt seconds forward
        param control_input: tuple of (velocity, steering angle)
        param dt: timestep to integrate
    """
    def step(self, control_input, dt):
        self.v = control_input[0] # velocity of vehicle
        beta = control_input[1] # steering angle of vehicle
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.v / self.length * np.tan(beta) * dt

        return self.x, self.y, self.theta