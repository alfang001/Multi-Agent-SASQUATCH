from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from PIL import Image


def make_movie(trajectories, estimated_trajectories, agent_pos):
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('trajectories.mp4', fourcc, 3.0, (680, 480))
    num_agents, num_steps, _ = trajectories.shape
    fig, ax = plt.subplots()
    circles = [Circle((agent_pos[i,0], agent_pos[i,1]), 0.5, color='orange', fill=True) for i in range(agent_pos.shape[0])]
    # Set up plot limits
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Trajectories with Safe and Goal Sets")
    ax.grid(True)
    lines = [ax.plot([], [], label=f'Trajectory {k + 1}')[0] for k in range(num_agents)]
    points = [ax.plot([], [], 'ko')[0] for _ in range(num_agents)]
    estimated_lines = [ax.plot([], [], color = 'cyan',label=f'Trajectory {k + 1}')[0] for k in range(num_agents)]
    estimated_points = [ax.plot([], [], 'ko')[0] for _ in range(num_agents)]
    # circles = [Circle((0, 0), 0.5, color='blue', fill=False) for _ in range(num_agents)]
    
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
            # import pdb; pdb.set_trace()
            points[k].set_data([trajectories[k, frame, 0]], [trajectories[k, frame, 1]])
            # Update circle position
            # circles[k].center = (trajectories[k, frame, 0], trajectories[k, frame, 1])

            estimated_lines[k].set_data(estimated_trajectories[k, :frame + 1, 0], estimated_trajectories[k, :frame + 1, 1])
            estimated_points[k].set_data([estimated_trajectories[k, frame, 0]], [estimated_trajectories[k, frame, 1]])
        return lines + points + circles

    ani = FuncAnimation(fig, update, frames=num_steps, blit=True, repeat=False)
    
    # To save the animation, you can use the following line:
    # ani.save('trajectories.mp4', writer='ffmpeg')

    plt.show()


