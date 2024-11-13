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
    
        """Returns current state of the vehicle in numpy array
        """
    def get_state(self) -> np.array:
        return np.array([self.x, self.y, self.theta, self.v])

    """Step the simulation of dynamics dt seconds forward
        param control_input: tuple of (velocity, steering angle)
        param dt: timestep to integrate

        returns current vehicle state
    """
    def step(self, control_input, dt) -> np.array:
        self.v = control_input[0] # velocity of vehicle
        beta = control_input[1] # steering angle of vehicle
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += self.v / self.L * np.tan(beta) * dt

        return self.get_state()