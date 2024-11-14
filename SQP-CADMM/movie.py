import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2
from PIL import Image
from io import BytesIO
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

def make_movie(trajectories):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('trajectories.mp4', fourcc, 3.0, (680, 480))
    num_agents, num_steps, _ = trajectories.shape
    fig, ax = plt.subplots()
    
    # Set up plot limits
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Trajectories with Safe and Goal Sets")
    ax.grid(True)
    lines = [ax.plot([], [], label=f'Trajectory {k + 1}')[0] for k in range(num_agents)]
    points = [ax.plot([], [], 'ko')[0] for _ in range(num_agents)]
    circles = [Circle((0, 0), 0.5, color='blue', fill=False) for _ in range(num_agents)]
    
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
            points[k].set_data(trajectories[k, frame, 0], trajectories[k, frame, 1])
            # Update circle position
            circles[k].center = (trajectories[k, frame, 0], trajectories[k, frame, 1])
        return lines + points + circles

    ani = FuncAnimation(fig, update, frames=num_steps, blit=True, repeat=False)
    
    # To save the animation, you can use the following line:
    # ani.save('trajectories.mp4', writer='ffmpeg')

    plt.show()


