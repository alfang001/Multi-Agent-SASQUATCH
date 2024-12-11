from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from PIL import Image


def make_movie(trajectories, estimated_trajectories, agent_pos, filename="trajectories.mp4"):
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(filename, fourcc, 3.0, (680, 480))
    # TODO: Rename num_agents to num_vehicles since this is ref trajectory
    num_agents, num_steps, _ = trajectories.shape
    fig, ax = plt.subplots()
    # Plotting circles for the agents
    circles = []
    first_agent = True
    for i in range(agent_pos.shape[0]):
        circle = None
        if first_agent:
            circle = Circle((agent_pos[i,0], agent_pos[i,1]), 0.5, color='orange', fill=True, label='Sensor (Agent) Position')
            first_agent = False
        else:
            circle = Circle((agent_pos[i,0], agent_pos[i,1]), 0.5, color='orange', fill=True)
        circles.append(circle)

    # circles = [Circle((agent_pos[i,0], agent_pos[i,1]), 0.5, color='orange', fill=True) for i in range(agent_pos.shape[0])]
    # Set up plot limits
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_xlabel("X Coordinate (meters)")
    ax.set_ylabel("Y Coordinate (meters)")
    ax.set_title("Tracking Vehicle Trajectory with SQP-CADMM")
    ax.grid(True)
    
    # Initial plot
    lines = [ax.plot([], [], color='black', label='Reference Trajectory')[0]]
    points = [ax.plot([], [], 'ko')[0]]
    estimated_lines = [ax.plot([], [], color = 'green',label='Estimated Trajectory')[0]]
    estimated_points = [ax.plot([], [], 'ko')[0]]
    
    # Add circles to the plot
    for circle in circles:
        ax.add_patch(circle)
    
    # Initialize legend
    ax.legend()

    # Update function for animation
    def update(frame):
        for k in range(num_agents):
            # Update trajectory lines
            lines[k].set_data(trajectories[k, :frame + 1, 0], trajectories[k, :frame + 1, 1])
            # Update the current position as a black dot
            points[k].set_data([trajectories[k, frame, 0]], [trajectories[k, frame, 1]])

            estimated_lines[k].set_data(estimated_trajectories[k, :frame + 1, 0], estimated_trajectories[k, :frame + 1, 1])
            estimated_points[k].set_data([estimated_trajectories[k, frame, 0]], [estimated_trajectories[k, frame, 1]])
        return lines + points + circles+estimated_lines+estimated_points

    ani = FuncAnimation(fig, update, frames=num_steps, blit=True, repeat=True)
    
    # To save the animation, you can use the following line:
    # ani.save('trajectories.mp4', writer='ffmpeg')

    plt.show()


