import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import mediapipe as mp
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

data = loadmat('./Poses/Shuffled/shuf_swing_003.mat')
landmarks = data['pose']          # Variable name matches .mat key

num_frames = landmarks.shape[0]


# --- MediaPipe Pose Connections ---
mp_pose = mp.solutions.pose
connections = mp_pose.POSE_CONNECTIONS

# --- Setup the 3D plot ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter([], [], [], c='blue', s=20)
lines = []

# Create line objects for all pose connections
for _ in connections:
    line, = ax.plot([], [], [], 'k-', linewidth=1)
    lines.append(line)

# --- Configure the 3D view ---
ax.set_xlim(0, 1)
ax.set_ylim(1, 0)    # Flip Y for consistency with 2D
ax.set_zlim(-0.5, 0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Pose Animation')
ax.view_init(elev=10, azim=-90)
ax.grid(True)

# --- Animation update function ---
def update(frame_idx):
    frame = landmarks[frame_idx]  # shape: (33, 4)
    x = frame[:, 0]
    y = frame[:, 1]
    z = frame[:, 2]

    # Update scatter
    scat._offsets3d = (x, y, z)

    # Update each connection line
    for i, (start_idx, end_idx) in enumerate(connections):
        if (not np.isnan(x[start_idx]) and not np.isnan(y[start_idx]) and not np.isnan(z[start_idx]) and
            not np.isnan(x[end_idx]) and not np.isnan(y[end_idx]) and not np.isnan(z[end_idx])):
            lines[i].set_data([x[start_idx], x[end_idx]], [y[start_idx], y[end_idx]])
            lines[i].set_3d_properties([z[start_idx], z[end_idx]])
        else:
            lines[i].set_data([], [])
            lines[i].set_3d_properties([])

    return [scat] + lines

# --- Create the animation ---
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

plt.show()
