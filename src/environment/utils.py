"""
utils.py

This module provides various utility functions for simulation visualization,
matrix processing, coordinate conversion, and more.
"""

# Standard libraries
import math
import random
import io

# Third-party libraries
import numpy as np
import pygame
import cairosvg
import matplotlib.pyplot as plt

# Matplotlib utilities
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, used for 3D plotting
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Project-specific imports
from settings import *
from typing import Any, Tuple, Dict

# -----------------------------------------------------------------------------
# Drawing Utilities
# -----------------------------------------------------------------------------
def draw_dashed_circle(surface: pygame.Surface,
                       color: Tuple[int, int, int],
                       center: Tuple[int, int],
                       radius: int,
                       dash_length: int = 5,
                       space_length: int = 5,
                       width: int = 1) -> None:
    """
    Draws a dashed circle on the provided Pygame surface.

    Args:
        surface (pygame.Surface): The surface to draw on.
        color (Tuple[int, int, int]): Color of the circle.
        center (Tuple[int, int]): (x, y) center of the circle.
        radius (int): Radius of the circle.
        dash_length (int): Length of each dash.
        space_length (int): Space between dashes.
        width (int): Line width.
    """
    if radius <= 0:
        return
    
    circumference = 2 * math.pi * radius
    num_dashes = int(circumference / (dash_length + space_length))
    angle_between = 2 * math.pi / num_dashes

    for i in range(num_dashes):
        start_angle = i * angle_between
        dash_angle = dash_length / radius  # Angle corresponding to dash length
        end_angle = start_angle + dash_angle

        start_pos = (center[0] + radius * math.cos(start_angle),
                     center[1] + radius * math.sin(start_angle))
        end_pos = (center[0] + radius * math.cos(end_angle),
                   center[1] + radius * math.sin(end_angle))
        pygame.draw.line(surface, color, start_pos, end_pos, width)

# -----------------------------------------------------------------------------
# Gaussian Bump Kernel Functions
# -----------------------------------------------------------------------------
def symmetrical_flat_topped_gaussian_10x10(value: float, sigma: float, flat_radius: float) -> np.ndarray:
    """
    Creates a 10x10 Gaussian bump with a flat top.
    
    The continuous coordinates range from -4.5 to +4.5 to ensure symmetry.
    The bump has a flat top (value = 1) within the given flat_radius.

    Args:
        value (float): Peak value to scale at the center.
        sigma (float): Standard deviation for the Gaussian portion.
        flat_radius (float): Radius (in continuous coordinates) where the value is constant.

    Returns:
        np.ndarray: A 10x10 array representing the bump.
    """
    kernel_size = 10
    x = np.linspace(-4.5, 4.5, kernel_size)
    xx, yy = np.meshgrid(x, x)
    r = np.sqrt(xx**2 + yy**2)

    # Gaussian bump
    bump = np.exp(-0.5 * (r / sigma)**2)
    
    # Apply flat top where distance is less than flat_radius
    bump[r < flat_radius] = 1.0
    
    return value * bump

def smooth_matrix_with_kernel_10x10(matrix: np.ndarray,
                                    direction: np.ndarray,
                                    sigma: float = 1.0,
                                    flat_radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a 10x10 Gaussian bump (with a flat top) to each positive value in the matrix.
    The bump is centered at the pixel (i, j) without changing the original dimensions.
    
    For each peak, if the bump's value is greater than the current value in the region,
    both the intensity and the corresponding direction vector are updated.

    Args:
        matrix (np.ndarray): 2D array with values between 0 and 1.
        direction (np.ndarray): Array with same dimensions as matrix plus one extra dimension
                                for the direction vector (e.g., shape (n_rows, n_cols, 2)).
        sigma (float): Standard deviation for the Gaussian portion.
        flat_radius (float): Flat top radius for the bump.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - result: Updated matrix with the applied bumps.
            - result_direction: Updated direction array.
    """
    result = np.copy(matrix)
    result_direction = np.copy(direction)
    peaks = np.argwhere(matrix > 0)
    kernel_size = 10
    anchor = kernel_size // 2

    for (i, j) in peaks:
        value = matrix[i, j]
        bump = symmetrical_flat_topped_gaussian_10x10(value, sigma, flat_radius)
        bump_direction = direction[i, j]  # Direction vector at the peak

        i_start = i - anchor
        i_end   = i_start + kernel_size
        j_start = j - anchor
        j_end   = j_start + kernel_size

        bump_i_start = 0
        bump_j_start = 0
        bump_i_end = kernel_size
        bump_j_end = kernel_size

        # Adjust indices if the region goes beyond the matrix borders
        if i_start < 0:
            bump_i_start = -i_start
            i_start = 0
        if j_start < 0:
            bump_j_start = -j_start
            j_start = 0
        if i_end > matrix.shape[0]:
            bump_i_end -= (i_end - matrix.shape[0])
            i_end = matrix.shape[0]
        if j_end > matrix.shape[1]:
            bump_j_end -= (j_end - matrix.shape[1])
            j_end = matrix.shape[1]

        region = result[i_start:i_end, j_start:j_end]
        bump_region = bump[bump_i_start:bump_i_end, bump_j_start:bump_j_end]

        # Create a mask for pixels where the bump value is greater than the current value
        mask = bump_region > region
        region = np.maximum(region, bump_region)
        result[i_start:i_end, j_start:j_end] = region

        # Update the direction vectors where the bump increased the value
        region_direction = result_direction[i_start:i_end, j_start:j_end]
        region_direction[mask] = bump_direction
        result_direction[i_start:i_end, j_start:j_end] = region_direction

    return result, result_direction

# -----------------------------------------------------------------------------
# Coordinate Conversion Functions
# -----------------------------------------------------------------------------
def sim_to_geo(pos_x: float, pos_y: float) -> Tuple[float, float]:
    """
    Converts simulation coordinates to geographic coordinates.

    Args:
        pos_x (float): X position in simulation.
        pos_y (float): Y position in simulation.
        
    Returns:
        Tuple[float, float]: (Longitude, Latitude) coordinates.
    """
    lon_left, lat_top = GEO_TOP_LEFT
    lon_right, lat_bottom = GEO_BOTTOM_RIGHT
    
    lon = lon_left + (pos_x / SIM_WIDTH) * (lon_right - lon_left)
    lat = lat_top + (pos_y / SIM_HEIGHT) * (lat_bottom - lat_top)
    
    return lon, lat

# -----------------------------------------------------------------------------
# SVG and Image Utilities
# -----------------------------------------------------------------------------
def load_svg_as_surface(svg_path: str) -> pygame.Surface:
    """
    Converts an SVG file to a Pygame Surface with alpha support.

    Args:
        svg_path (str): Path to the SVG file.

    Returns:
        pygame.Surface: Converted image as a Pygame surface.
    """
    # Convert SVG to PNG in memory
    png_data = cairosvg.svg2png(url=svg_path)
    image_data = io.BytesIO(png_data)
    surface = pygame.image.load(image_data).convert_alpha()
    return surface

# -----------------------------------------------------------------------------
# Grid and Positioning Utilities
# -----------------------------------------------------------------------------
def pos_to_cell(pos: pygame.math.Vector2) -> Tuple[int, int]:
    """
    Converts a position (Vector2) to grid cell coordinates.

    Args:
        pos (pygame.math.Vector2): The position vector.

    Returns:
        Tuple[int, int]: Cell coordinates (x, y).
    """
    x: int = min(int(pos.x // CELL_SIZE), GRID_WIDTH - 1)
    y: int = min(int(pos.y // CELL_SIZE), GRID_HEIGHT - 1)
    return (x, y)

# -----------------------------------------------------------------------------
# Interception and Pursuit Calculations
# -----------------------------------------------------------------------------
def intercept_direction(chaser_pos: pygame.math.Vector2,
                        chaser_speed: float,
                        target_pos: pygame.math.Vector2,
                        target_vel: pygame.math.Vector2) -> pygame.math.Vector2:
    """
    Calculates the optimal interception direction for a chaser to intercept a target.
    
    The method solves for the intercept time 't' from:
        |r + target_vel * t| = chaser_speed * t
    where r = target_pos - chaser_pos.
    
    If no valid solution exists (negative discriminant or non-positive t),
    the function returns the normalized vector from chaser to target.

    Args:
        chaser_pos (pygame.math.Vector2): Chaser's position.
        chaser_speed (float): Chaser's constant speed.
        target_pos (pygame.math.Vector2): Target's position.
        target_vel (pygame.math.Vector2): Target's velocity.

    Returns:
        pygame.math.Vector2: Unit vector in the direction of interception.
    """
    r = target_pos - chaser_pos
    a = target_vel.dot(target_vel) - chaser_speed ** 2
    b = 2 * r.dot(target_vel)
    c = r.dot(r)

    t = 0.0  # Fallback time

    if abs(a) < 1e-6:
        if abs(b) > 1e-6:
            t = -c / b
    else:
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            sqrt_disc = math.sqrt(discriminant)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)
            t_candidates = [t_val for t_val in (t1, t2) if t_val > 0]
            if t_candidates:
                t = min(t_candidates)

    if t <= 0:
        direction = target_pos - chaser_pos
    else:
        intercept_point = target_pos + target_vel * t
        direction = intercept_point - chaser_pos

    if direction.length() > 0:
        return direction.normalize()
    else:
        return pygame.math.Vector2(0, 0)

# -----------------------------------------------------------------------------
# Plotting Utilities
# -----------------------------------------------------------------------------
def plot_individual_states_matplotlib(state: Dict) -> None:
    """
    Generates a 3D plot of detection states for enemy and friend drones.
    
    Displays two subplots ("Enemy Detection" and "Friend Detection") with a color legend
    that maps colors to angles (in π radians). Also plots the drone's position as a red
    vertical line.

    Args:
        state (dict): Dictionary containing drone state information.
    """
    fig = plt.figure(figsize=(8, 6))
    cmap = plt.cm.hsv  # Color palette based on angles

    # Iterate over the two detection views
    for idx, plot_view in enumerate(["Enemy Detection", "Friend Detection"]):
        ax = fig.add_subplot(2, 1, idx + 1, projection='3d')
        ax.set_box_aspect((1, 1, 0.4))

        # Create meshgrid for grid dimensions
        x = np.linspace(CELL_SIZE / 2, SIM_WIDTH - CELL_SIZE / 2, GRID_WIDTH)
        y = np.linspace(CELL_SIZE / 2, SIM_HEIGHT - CELL_SIZE / 2, GRID_HEIGHT)
        X, Y = np.meshgrid(x, y)

        if plot_view == "Enemy Detection":
            to_plot = state['enemy_intensity']
            direction = state['enemy_direction']
        elif plot_view == "Friend Detection":
            to_plot = state['friend_intensity']
            direction = state['friend_direction']

        # Smooth the matrix and get resulting directions
        Z_smoothed, result_direction = smooth_matrix_with_kernel_10x10(
            to_plot, direction, sigma=2, flat_radius=1
        )

        # Adjust shape if necessary
        if Z_smoothed.shape != X.shape:
            Z_smoothed = Z_smoothed.T
            result_direction = result_direction.transpose(1, 0, 2)

        # Calculate angles from direction vectors
        dir_x = result_direction[..., 0]
        dir_y = result_direction[..., 1]
        angle = np.arctan2(dir_y, dir_x)
        norm_angle = (angle + math.pi) / (2 * math.pi)
        facecolors = cmap(norm_angle)
        facecolors[Z_smoothed < PLOT_THRESHOLD] = [1, 1, 1, 1]

        ax.plot_surface(X, Y, Z_smoothed, facecolors=facecolors,
                        linewidth=0, antialiased=True, shade=False)
        ax.contourf(X, Y, Z_smoothed, zdir='x', offset=ax.get_xlim()[0], cmap="Greys")
        ax.contourf(X, Y, Z_smoothed, zdir='y', offset=ax.get_ylim()[0], cmap="Greys")
        ax.plot_wireframe(X, Y, Z_smoothed, color='black', linewidth=0.2, rstride=1, cstride=1)

        # Plot drone's position as a red vertical line (in XY projection)
        ax.plot([state['pos'].x, state['pos'].x],
                [SIM_HEIGHT, 0],
                [0, 0], color='red', linewidth=2, label='Drone Position')
        ax.plot([state['pos'].x, state['pos'].x],
                [state['pos'].y, state['pos'].y],
                [0, 1.5], color='red', linewidth=1, zorder=10)

        x_offset = ax.get_xlim()[0]
        ax.plot([x_offset, x_offset],
                [state['pos'].y, state['pos'].y],
                [0, 1.5], color='gray', linewidth=1, zorder=10)

        y_offset = ax.get_ylim()[0]
        ax.plot([state['pos'].y, state['pos'].y],
                [y_offset, y_offset],
                [0, 1.5], color='gray', linewidth=1, zorder=10)

        ax.set_title(plot_view, fontsize=12)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Intensity', fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, SIM_WIDTH)
        ax.set_ylim(SIM_HEIGHT, 0)
        ax.set_zlim(0, 1.5)

    fig.suptitle("Detection and Position Plot", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.75, 1])

    norm = plt.Normalize(vmin=-math.pi, vmax=math.pi)
    mappable_for_colorbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable_for_colorbar.set_array([])
    cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.7])
    cbar = fig.colorbar(mappable_for_colorbar, cax=cbar_ax, label='Angle (rad)')
    cbar.set_ticks([-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi])
    cbar.set_ticklabels(['-π', '-π/2', '0', '+π/2', '+π'])
    plt.show()

# -----------------------------------------------------------------------------
# Sparse Matrix Generation
# -----------------------------------------------------------------------------
def generate_sparse_matrix(shape: Tuple[int, int] = (GRID_WIDTH, GRID_HEIGHT),
                           max_nonzero: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a sparse matrix of the given shape with up to 'max_nonzero' nonzero elements.
    
    Nonzero values are sampled from a normal distribution (mean=1, std=1) and clipped to [0, 1].
    For each nonzero cell, a random normalized direction vector is generated.

    Args:
        shape (Tuple[int, int]): The shape of the intensity matrix.
        max_nonzero (int): Maximum number of nonzero elements.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - matrix: Array containing intensity values.
            - direction: Array of shape (rows, cols, 2) with corresponding direction vectors.
                         Cells with zero intensity have a (0, 0) vector.
    """
    matrix = np.zeros(shape)
    direction = np.zeros((shape[0], shape[1], 2), dtype=float)
    total_cells = shape[0] * shape[1]
    nonzero_count = min(random.randint(0, max_nonzero), total_cells)
    chosen_indices = random.sample(range(total_cells), nonzero_count)
    
    for idx in chosen_indices:
        i = idx // shape[1]
        j = idx % shape[1]
        value = np.random.normal(1, 1)
        matrix[i, j] = np.clip(value, 0, 1)
        
        # Generate a random normalized direction vector
        vec = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        if vec.length() > 0:
            vec = vec.normalize()
        else:
            vec = pygame.math.Vector2(0, 0)
        direction[i, j] = (vec.x, vec.y)
    
    return matrix, direction


import os
import re
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import Tuple

def load_best_model(directory: str, pattern: str, custom_objects=None) -> Tuple[tf.keras.Model, Tuple[int, int]]:
    """
    Loads the best model from the given directory by selecting the file with the lowest validation loss,
    and extracts the size from the filename in a tuple (width, height).

    Args:
        directory (str): Directory containing saved model files.
        pattern (str): Regex pattern to extract the validation loss value from the filename.
        custom_objects (dict, optional): Custom objects to be passed to load_model.

    Returns:
        Tuple[tf.keras.Model, Tuple[int, int]]: The loaded model and a tuple with the size (width, height).

    Raises:
        FileNotFoundError: If no model file is found in the specified directory.
    """
    best_file: str = ""
    min_val_metric_loss: float = float("inf")

    # Iterate over model files to find the one with the lowest val_metric_loss
    for filename in os.listdir(directory):
        if filename.endswith(".keras"):
            match = re.search(pattern, filename)
            if match:
                val_metric_loss: float = float(match.group(1))
                if val_metric_loss < min_val_metric_loss:
                    min_val_metric_loss = val_metric_loss
                    best_file = filename

    if not best_file:
        raise FileNotFoundError(f"No model files found in the directory: {directory}")

    model_path: str = os.path.join(directory, best_file)
    try:
        model = load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Error loading the model: {e}")
        sys.exit(1)

    print(f"\n\nLoaded model: {best_file} with val_metric_loss={min_val_metric_loss:.4f}\n\n")

    return model