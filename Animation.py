import os
from unittest.mock import inplace

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np

# Change working directory if needed
os.chdir('log/')

# Read CSV file
fn = 'logfile_State_def.csv'  # Replace with your filename if needed
df = pd.read_csv(fn)


# Function to split data into episodes for Intruder and Defender
def dfhelper(df):
    intruder, defender = df[df['Agent'] == 0].copy(), df.query('Agent == 1').copy()
    #intruder.drop(columns=df.columns[0], axis=1, inplace=True)
    #defender.drop(columns=df.columns[0], axis=1, inplace=True)
    intruder.to_csv('test.csv')

    ep_intruder, ep_defender = [], []
    for i in range(5):  # Assuming 5 episodes
        ep_intruder.append(intruder[intruder['Episode'] == i])
        ep_defender.append(defender[defender['Episode'] == i])
    return ep_intruder, ep_defender


# Get data split by episode
intruder_data, defender_data = dfhelper(df)
defender_data[0].to_csv('test.csv')
# Loop through each episode and create separate animations
for i, (intruder, defender) in enumerate(zip(intruder_data, defender_data)):
    # Initialize a new figure for each episode
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust size as needed

    # Configure the plot
    ax.set_xlim(-1, 1)  # Adjust based on your data
    ax.set_ylim(-1, 1)  # Adjust based on your data
    ax.grid(True)
    ax.set_title(f"Episode {i} Animation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Add static circles
    ax.add_patch(Circle((0, 0), 0.106, fill=False, color='red', linestyle='--'))
    ax.add_patch(Circle((0, 0), 0.5, fill=False, color='blue', linestyle='--'))

    # Initialize empty lines for animation
    line1, = ax.plot([], [], color='green', label="Intruder")
    line2, = ax.plot([], [], color='orange', label="Defender")
    Intruder_sensing = Circle((0, 0), 0.3, color='green', fill=False)  # Small circle
    intruder_marker, = ax.plot([], [], 'o', color='red', markersize=8, label="Intruder")
    defender_marker, = ax.plot([], [], 'o', color='blue', markersize=8, label="Defender")
    ax.add_patch(Intruder_sensing)

    ax.legend()


    # Frame update function for this episode
    def update(frame):
        if frame < len(intruder):  # Ensure the frame index is valid
            line1.set_data(intruder['X'][:frame + 1], intruder['Y'][:frame + 1])
            line2.set_data(defender['X'][:frame + 1], defender['Y'][:frame + 1])
            Intruder_sensing.set_center((intruder['X'].iloc[frame], intruder['Y'].iloc[frame]))
            #distance_to_center = np.sqrt(intruder['X'].iloc[frame]**2 + intruder['Y'].iloc[frame]**2)
            # Split data into parts that are inside and outside the circle for line1 and line2
            intruder_pos = (intruder['X'].iloc[frame], intruder['Y'].iloc[frame])
            defender_pos = (defender['X'].iloc[frame], defender['Y'].iloc[frame])

            # Compute distances
            dist_intruder_to_hvt = np.sqrt(intruder_pos[0]**2 + intruder_pos[1]**2)
            dist_defender_to_intruder = np.sqrt((defender_pos[0] - intruder_pos[0])**2 + (defender_pos[1] - intruder_pos[1])**2)

            # Update marker colors based on sensing ranges
            if dist_intruder_to_hvt <= 0.5:  # Intruder in HVT sensing range
                intruder_marker.set_color('red')
            else:
                intruder_marker.set_color('green')

            if dist_defender_to_intruder <= 0.2:  # Defender in Intruder sensing range
                defender_marker.set_color('blue')
            else:
                defender_marker.set_color('orange')

            # Update marker positions
            intruder_marker.set_data([intruder_pos[0]], [intruder_pos[1]])
            defender_marker.set_data([defender_pos[0]], [defender_pos[1]])

        return line1, line2, Intruder_sensing, intruder_marker, defender_marker


    # Create animation for this episode
    ani = animation.FuncAnimation(
        fig, update, frames=len(intruder), interval=100, blit=True, repeat=False
    )

    # Save the animation as a GIF
    output_file = f"episode_{i}.gif"
    ani.save(output_file, writer=animation.PillowWriter(fps=10))
    print(f"Animation for Episode {i} saved as {output_file}")

    # Close the figure to free memory
    plt.close(fig)