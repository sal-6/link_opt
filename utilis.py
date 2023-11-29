import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import os
from PIL import Image

from robotics import forward_kinematics


def plot_end_position(t, states, L):
    
    # get end effector position for each time step
    
    end_pos_xs = []
    end_pos_ys = []
    
    for state in states.T:
        end_pos = forward_kinematics(state[0:2], L)
        end_pos_xs.append(end_pos[0])
        end_pos_ys.append(end_pos[1])
    
    # plot xy position of end effector over time
    
    plt.plot(end_pos_xs, end_pos_ys)
    plt.xlabel("x position")
    plt.ylabel("y position")
    
    plt.show()
    
    


def plot_end_position_animation2(t, states, L, filename='end_position.gif', plot_interval=50):
    fig, ax = plt.subplots()
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    
    end_pos_xs = []
    end_pos_ys = []

    # Collect end positions
    for i in range(len(t)):
        end_pos = forward_kinematics(states[:, i][:2], L)
        end_pos_xs.append(end_pos[0])
        end_pos_ys.append(end_pos[1])

    # Calculate fixed plot bounds
    max_x = max(end_pos_xs)
    max_y = max(end_pos_ys)
    max_bound = max(max_x, max_y)
    plot_bound = np.ceil(max_bound / 0.25) * 0.25  # Round up to nearest 0.25

    # Initialize plot line
    line, = ax.plot([], [], color='blue')

    # Find indices corresponding to roughly 0.05s intervals
    interval_indices = [0]
    prev_time = t[0]
    for i in range(1, len(t)):
        if t[i] - prev_time >= plot_interval / 1000:
            interval_indices.append(i)
            prev_time = t[i]

    def update(frame):
        idx = interval_indices[frame]
        end_pos_xs_frame = end_pos_xs[:idx+1]
        end_pos_ys_frame = end_pos_ys[:idx+1]

        line.set_data(end_pos_xs_frame, end_pos_ys_frame)
        ax.set_xlim(-plot_bound, plot_bound)
        ax.set_ylim(-plot_bound, plot_bound)

        return line,

    anim = FuncAnimation(fig, update, frames=len(interval_indices), interval=plot_interval, blit=True)

    # Save animation as GIF
    anim.save(filename, writer='pillow') 
    
    

def plot_joint_ends(t, states, L, filename='end_position.gif', plot_interval=50):
    fig, ax = plt.subplots()
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    
    end_pos_xs = []
    end_pos_ys = []
    joint1_pos_xs = []
    joint1_pos_ys = []

    # Collect end positions and joint 1 positions
    for i in range(len(t)):
        end_pos = forward_kinematics(states[:, i][:2], L)
        end_pos_xs.append(end_pos[0])
        end_pos_ys.append(end_pos[1])
        
        joint1_pos_xs.append(end_pos[2])
        joint1_pos_ys.append(end_pos[3])

    # Calculate fixed plot bounds for both positions
    max_x = max(max(end_pos_xs), max(joint1_pos_xs))
    max_y = max(max(end_pos_ys), max(joint1_pos_ys))
    max_bound = max(max_x, max_y)
    plot_bound = np.ceil(max_bound / 0.25) * 0.25  # Round up to nearest 0.25

    # Initialize plot lines for both positions
    line_end, = ax.plot([], [], color='blue', label='End Effector')
    line_joint1, = ax.plot([], [], color='red', label='Joint 1')

    # Find indices corresponding to roughly 0.05s intervals
    interval_indices = [0]
    prev_time = t[0]
    for i in range(1, len(t)):
        if t[i] - prev_time >= plot_interval / 1000:
            interval_indices.append(i)
            prev_time = t[i]

    def update(frame):

        idx = interval_indices[frame]
        end_pos_xs_frame = end_pos_xs[:idx + 1]
        end_pos_ys_frame = end_pos_ys[:idx + 1]
        joint1_pos_xs_frame = joint1_pos_xs[:idx + 1]
        joint1_pos_ys_frame = joint1_pos_ys[:idx + 1]

        line_end.set_data(end_pos_xs_frame, end_pos_ys_frame)
        line_joint1.set_data(joint1_pos_xs_frame, joint1_pos_ys_frame)
        
        if frame > 0:
            for line in ax.lines:
                if line.get_color() == 'green':
                    line.remove()

        # Draw lines from origin to joint 1 and from joint 1 to end
        if frame > 0:
            ax.plot([0, joint1_pos_xs_frame[-1]], [0, joint1_pos_ys_frame[-1]], color='green', linestyle='--')
            ax.plot([joint1_pos_xs_frame[-1], end_pos_xs_frame[-1]], [joint1_pos_ys_frame[-1], end_pos_ys_frame[-1]], color='green', linestyle='--')

        ax.set_xlim(-plot_bound, plot_bound)
        ax.set_ylim(-plot_bound, plot_bound)
        ax.legend()

        return line_end, line_joint1

    anim = FuncAnimation(fig, update, frames=len(interval_indices), interval=plot_interval, blit=True)

    # Save animation as GIF
    anim.save(filename, writer='pillow') 