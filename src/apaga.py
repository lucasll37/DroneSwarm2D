import os
import sys
import numpy as np


current_dir: str = os.getcwd()
config_dir: str = os.path.abspath(os.path.join(current_dir, "./src/environment"))
if config_dir not in sys.path:
    sys.path.append(config_dir)

# Project-specific imports
from utils import generate_sparse_matrix, plot_individual_states_matplotlib
from settings import *

pos = np.array([
    np.random.uniform(0, SIM_WIDTH),
    np.random.uniform(0, SIM_HEIGHT)
], dtype=np.float32)

# Generate sparse matrices for intensities and directions
friend_intensity, friend_direction = generate_sparse_matrix()
enemy_intensity, enemy_direction = generate_sparse_matrix()

# Organize the state into a dictionary.
# Convert the Vector2 object to a tuple (x, y) if needed.
state = {
    'pos': np.array(pos, dtype=np.float32),
    'friend_intensity': np.array(friend_intensity, dtype=np.float32),
    'enemy_intensity': np.array(enemy_intensity, dtype=np.float32),
    'friend_direction': np.array(friend_direction, dtype=np.float32),
    'enemy_direction': np.array(enemy_direction, dtype=np.float32)
}

plot_individual_states_matplotlib(state)