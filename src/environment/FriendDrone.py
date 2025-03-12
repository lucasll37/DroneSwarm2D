"""
FriendDrone.py

This module defines the Drone class used in the simulation. The Drone class handles
local detection (enemy and friend), communication, merging of detection matrices,
distributed elections, action execution, and rendering. It also provides planning and
debug behaviors for drone motion.
"""

# -----------------------------------------------------------------------------
# Imports and Setup
# -----------------------------------------------------------------------------
import random
import math
import os
import sys
import numpy as np
import pygame
from typing import Tuple, List, Any

from settings import (
    CELL_SIZE, GRID_WIDTH, GRID_HEIGHT, SIM_WIDTH, SIM_HEIGHT, FRIEND_SPEED, DECAY_FACTOR,
    FRIEND_DETECTION_RANGE, COMMUNICATION_RANGE, MESSAGE_LOSS_PROBABILITY, FONT_FAMILY,
    EXTERNAL_RADIUS, INITIAL_DISTANCE, AEW_RANGE, AEW_SPEED, AEW_DETECTION_RANGE,
    INTEREST_POINT_CENTER, RADAR_DETECTION_RANGE
)
from utils import draw_dashed_circle, load_svg_as_surface, pos_to_cell, intercept_direction, load_best_model

# Add configuration directory to the system path if not already present.
# current_dir: str = os.getcwd()
# config_dir: str = os.path.abspath(os.path.join(current_dir, "./src"))
# if config_dir not in sys.path:
#     sys.path.append(config_dir)

# -----------------------------------------------------------------------------
# Drone Class Definition
# -----------------------------------------------------------------------------
class FriendDrone:

    # Class variables
    friend_id_counter: int = 0
    original_drone_image: pygame.Surface = load_svg_as_surface("./assets/drone_0.svg")
    original_aew_image: pygame.Surface = load_svg_as_surface("./assets/radar_0.svg")
    original_radar_image: pygame.Surface = load_svg_as_surface("./assets/radar_0.svg")
    
    model = None

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(self, interest_point_center, position: Tuple, behavior_type: str = "planning", fixed: bool = False) -> None:
        """
        Initializes the drone with its starting position, interest point, and behavior type.
        
        Args:
            interest_point_center: The center of the interest point.
            position (Tuple): The initial (x, y) position of the drone.
            behavior_type (str): The behavior type (default "planning").
            fixed (bool): If True, the drone remains stationary.
        """
        self.pos: pygame.math.Vector2 = pygame.math.Vector2(position[0], position[1])
        self.interest_point_center = interest_point_center
        self.behavior_type = behavior_type
        self.fixed = fixed
        self.selected = False
        self.vel: pygame.math.Vector2 = pygame.math.Vector2(0, 0)
        self.orbit_radius = None  # Used for AEW behavior
        self.trajectory: List[pygame.math.Vector2] = []

        # Drone properties
        self.color: Tuple[int, int, int] = (255, 255, 255)
        self.drone_id: int = self.assign_id()
        self.in_election: bool = False
        self.is_leader: bool = False
        self.leader_id: int = self.drone_id

        # Dictionaries for detections
        self.aux_enemy_detections: dict = {}
        self.aux_friend_detections: dict = {}
        self.current_enemy_pos_detection: dict = {}
        self.current_friend_pos_detection: dict = {}

        # Detection matrices
        self.enemy_intensity: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        self.enemy_direction: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2))
        self.enemy_timestamp: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        self.friend_intensity: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        self.friend_direction: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2))
        self.friend_timestamp: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        
                    
        if self.behavior_type == "RADAR":
            desired_width = int(SIM_WIDTH * 0.03)
            aspect_ratio = self.original_radar_image.get_height() / self.original_radar_image.get_width()
            desired_height = int(desired_width * aspect_ratio)
            self.image = pygame.transform.scale(self.original_radar_image, (desired_width, desired_height))
            
        else:
            desired_width = int(SIM_WIDTH * 0.02)
            aspect_ratio = self.original_drone_image.get_height() / self.original_drone_image.get_width()
            desired_height = int(desired_width * aspect_ratio)
            self.image = pygame.transform.scale(self.original_drone_image, (desired_width, desired_height))

    # -------------------------------------------------------------------------
    # Unique ID Assignment
    # -------------------------------------------------------------------------
    def assign_id(self) -> int:
        """
        Assigns a unique ID to the drone.
        
        Returns:
            int: The unique drone ID.
        """
        current_id: int = self.__class__.friend_id_counter
        self.__class__.friend_id_counter += 1
        return current_id

    # -------------------------------------------------------------------------
    # Matrix Decay
    # -------------------------------------------------------------------------
    def decay_matrices(self) -> None:
        """
        Applies exponential decay to both enemy and friend detection intensity matrices.
        """
        self.enemy_intensity *= DECAY_FACTOR
        self.friend_intensity *= DECAY_FACTOR

    # -------------------------------------------------------------------------
    # Update Local Enemy Detection
    # -------------------------------------------------------------------------
    def update_local_enemy_detection(self, enemy_drones: List[Any]) -> None:
        """
        Updates the local detection of enemy drones.
        
        For each enemy drone within the detection range, updates the corresponding cell in
        the enemy detection matrices (intensity, direction, timestamp). Additionally, for cells
        within the detection radius that lack a detection (intensity below threshold), the intensity
        is set to zero and the timestamp is updated using np.putmask.
        
        Args:
            enemy_drones (List[Any]): List of enemy drones.
        """
        if self.behavior_type == "AEW":
            detection_range = AEW_DETECTION_RANGE
        elif self.behavior_type == "RADAR":
            detection_range = RADAR_DETECTION_RANGE
        else:
            detection_range = FRIEND_DETECTION_RANGE
                    
        current_time: int = pygame.time.get_ticks()
        for enemy in enemy_drones:
            key: int = id(enemy)
            if self.pos.distance_to(enemy.pos) >= detection_range:
                self.current_enemy_pos_detection.pop(key, None)
                self.aux_enemy_detections.pop(key, None)
                continue
            
            cell: Tuple[int, int] = pos_to_cell(enemy.pos)
            if key not in self.current_enemy_pos_detection:
                self.current_enemy_pos_detection[key] = enemy.pos.copy()
            else:
                if key in self.current_enemy_pos_detection and key in self.aux_enemy_detections:
                    prev_cell: Tuple[int, int] = self.aux_enemy_detections[key]
                    if prev_cell != cell:
                        # Zero out the values in the previous cell
                        self.enemy_intensity[prev_cell] = 0
                        self.enemy_direction[prev_cell] = [0, 0]
                        self.enemy_timestamp[prev_cell] = current_time
                self.aux_enemy_detections[key] = cell
                self.enemy_intensity[cell] = 1.0
                delta: pygame.math.Vector2 = enemy.pos - self.current_enemy_pos_detection[key]
                self.current_enemy_pos_detection[key] = enemy.pos.copy()
                if delta.length() > 0:
                    self.enemy_direction[cell] = list(delta.normalize())
                self.enemy_timestamp[cell] = current_time

        # --- Vectorized Update for Cells Without Detection ---
        # Convert detection range (in pixels) to number of cells; scale factor (0.8) can be adjusted.
        detection_range_cells = int(np.floor(detection_range / CELL_SIZE) * 0.8)
        
        # Get the central cell of the drone (its own position)
        center_x, center_y = pos_to_cell(self.pos)
        
        # Define limits of the rectangle that encloses the detection circle
        x_min = max(center_x - detection_range_cells, 0)
        x_max = min(center_x + detection_range_cells, GRID_WIDTH - 1)
        y_min = max(center_y - detection_range_cells, 0)
        y_max = min(center_y + detection_range_cells, GRID_HEIGHT - 1)
        
        # Create a grid of indices for the region
        x_indices = np.arange(x_min, x_max + 1)
        y_indices = np.arange(y_min, y_max + 1)
        xv, yv = np.meshgrid(x_indices, y_indices, indexing='ij')
        
        # Calculate the distance (in cell units) of each cell from the center
        distances = np.sqrt((xv - center_x) ** 2 + (yv - center_y) ** 2)
        
        # Extract the region of the enemy intensity and timestamp matrices
        region_intensity = self.enemy_intensity[x_min:x_max+1, y_min:y_max+1]
        region_timestamp = self.enemy_timestamp[x_min:x_max+1, y_min:y_max+1]
        
        # Create a mask for cells within the detection circle that have low intensity (i.e., no detection)
        mask_empty = (distances <= detection_range_cells) & (region_intensity < 1)
        
        # Set intensities to 0 and update timestamps to current_time for these "empty" cells
        np.putmask(region_intensity, mask_empty, 0)
        np.putmask(region_timestamp, mask_empty, current_time)

    # -------------------------------------------------------------------------
    # Update Local Friend Detection
    # -------------------------------------------------------------------------
    def update_local_friend_detection(self, friend_drones: List[Any]) -> None:
        """
        Updates the local detection of friend drones.
        
        For each friend drone (excluding AEW drones and self), updates the corresponding
        cell in the friend detection matrices (intensity, direction, timestamp).
        
        Args:
            friend_drones (List[Any]): List of friend drones.
        """
        current_time: int = pygame.time.get_ticks()
        for friend in friend_drones:
            if friend.behavior_type == "AEW":
                continue
            
            # Do not consider self or AEW drones for interception.
            if friend is self or friend.behavior_type == "AEW":
                continue
            key: int = id(friend)
            if self.pos.distance_to(friend.pos) >= COMMUNICATION_RANGE:
                self.current_friend_pos_detection.pop(key, None)
                self.aux_friend_detections.pop(key, None)
                continue
            
            cell: Tuple[int, int] = pos_to_cell(friend.pos)
            if key not in self.current_friend_pos_detection:
                self.current_friend_pos_detection[key] = friend.pos.copy()
            else:
                if key in self.current_friend_pos_detection and key in self.aux_friend_detections:
                    prev_cell: Tuple[int, int] = self.aux_friend_detections[key]
                    if prev_cell != cell:
                        self.friend_intensity[prev_cell] = 0
                        self.friend_direction[prev_cell] = [0, 0]
                        self.friend_timestamp[prev_cell] = current_time
                self.aux_friend_detections[key] = cell
                self.friend_intensity[cell] = 1.0
                delta: pygame.math.Vector2 = friend.pos - self.current_friend_pos_detection[key]
                self.current_friend_pos_detection[key] = friend.pos.copy()
                if delta.length() > 0:
                    self.friend_direction[cell] = delta.normalize()
                self.friend_timestamp[cell] = current_time
                
        # --- Vectorized Update for Cells Without Friend Detection ---
        # Convert friend detection range (in pixels) to number of cells.
        detection_range_cells = int(np.floor(FRIEND_DETECTION_RANGE / CELL_SIZE) * 0.8)
        
        # Get the central cell corresponding to the drone's own position.
        center_x, center_y = pos_to_cell(self.pos)
        
        # Define the rectangular region that covers the detection circle.
        x_min = max(center_x - detection_range_cells, 0)
        x_max = min(center_x + detection_range_cells, GRID_WIDTH - 1)
        y_min = max(center_y - detection_range_cells, 0)
        y_max = min(center_y + detection_range_cells, GRID_HEIGHT - 1)
        
        # Create a grid of indices for this region.
        x_indices = np.arange(x_min, x_max + 1)
        y_indices = np.arange(y_min, y_max + 1)
        xv, yv = np.meshgrid(x_indices, y_indices, indexing='ij')
        
        # Compute the distance (in cell units) of each cell from the drone's center.
        distances = np.sqrt((xv - center_x)**2 + (yv - center_y)**2)
        
        # Extract the corresponding sub-regions of friend intensity and timestamp matrices.
        region_intensity = self.friend_intensity[x_min:x_max+1, y_min:y_max+1]
        region_timestamp = self.friend_timestamp[x_min:x_max+1, y_min:y_max+1]
        
        # Create a mask for cells within the detection circle that have low intensity (i.e., no detection).
        # Here we assume that a cell with intensity < 1 is considered "empty".
        mask_empty = (distances <= detection_range_cells) & (region_intensity < 1)
        
        # Reset intensities to 0 and timestamps to 0 for these "empty" cells.
        np.putmask(region_intensity, mask_empty, 0)
        np.putmask(region_timestamp, mask_empty, current_time)

    # -------------------------------------------------------------------------
    # Merge Enemy Matrix
    # -------------------------------------------------------------------------
    def merge_enemy_matrix(self, neighbor: "Drone") -> None:
        """
        Merges enemy detection data from a neighbor drone into this drone's matrices,
        propagating information globally based on timestamps.
        
        Args:
            neighbor (Drone): The neighbor drone.
        """
        update_mask = neighbor.enemy_timestamp > self.enemy_timestamp
        np.putmask(self.enemy_intensity, update_mask, neighbor.enemy_intensity)
        np.putmask(
            self.enemy_direction,
            np.broadcast_to(update_mask[..., None], self.enemy_direction.shape),
            neighbor.enemy_direction
        )
        np.putmask(self.enemy_timestamp, update_mask, neighbor.enemy_timestamp)

    # -------------------------------------------------------------------------
    # Merge Friend Matrix
    # -------------------------------------------------------------------------
    def merge_friend_matrix(self, neighbor: "Drone") -> None:
        """
        Merges the friend detection matrix from a neighbor drone.
        
        Args:
            neighbor (Drone): The neighbor drone.
        """
        update_mask = neighbor.friend_timestamp > self.friend_timestamp
        np.putmask(self.friend_intensity, update_mask, neighbor.friend_intensity)
        np.putmask(
            self.friend_direction,
            np.broadcast_to(update_mask[..., None], self.friend_direction.shape),
            neighbor.friend_direction
        )
        np.putmask(self.friend_timestamp, update_mask, neighbor.friend_timestamp)

    # -------------------------------------------------------------------------
    # Communication
    # -------------------------------------------------------------------------
    def communication(self, all_drones: List[Any]) -> None:
        """
        Simulates distributed communication by merging detection matrices
        from nearby friend drones.
        
        Args:
            all_drones (List[Any]): List of all friend drones.
        """
        for other in all_drones:
            if other is not self and self.pos.distance_to(other.pos) < COMMUNICATION_RANGE:
                if random.random() > MESSAGE_LOSS_PROBABILITY:
                    self.merge_enemy_matrix(other)
                    self.merge_friend_matrix(other)

    # -------------------------------------------------------------------------
    # Election Process
    # -------------------------------------------------------------------------
    def start_election(self, all_friend_drones: List[Any]) -> None:
        """
        Initiates a distributed election among friend drones.
        
        Args:
            all_friend_drones (List[Any]): List of all friend drones.
        """
        if self.in_election:
            return
        self.in_election = True
        responses: List[bool] = []
        for other in all_friend_drones:
            if other.drone_id > self.drone_id:
                response = other.receive_election_message(self.drone_id, all_friend_drones)
                responses.append(response)
        if not any(responses):
            self.is_leader = True
            self.leader_id = self.drone_id
            for other in all_friend_drones:
                if other is not self:
                    other.set_leader(self.drone_id)
        self.in_election = False

    def receive_election_message(self, sender_id: int, all_friend_drones: List[Any]) -> bool:
        """
        Receives an election message from another drone.
        
        Args:
            sender_id (int): The sender's ID.
            all_friend_drones (List[Any]): List of all friend drones.
        
        Returns:
            bool: True if the message is received.
        """
        if sender_id < self.drone_id and not self.in_election:
            self.start_election(all_friend_drones)
        return True

    def set_leader(self, leader_id: int) -> None:
        """
        Sets the leader for this drone.
        
        Args:
            leader_id (int): The leader's ID.
        """
        self.leader_id = leader_id
        self.is_leader = (self.drone_id == leader_id)

    # -------------------------------------------------------------------------
    # Action Execution
    # -------------------------------------------------------------------------
    def take_action(self) -> None:
        """
        Executes the drone's action based on enemy detection.
        
        The drone applies its behavior, updates its position, and ensures it stays within
        the simulation bounds. It also prevents the drone from exceeding the external radius
        from the interest point.
        """
        self.apply_behavior()
        self.pos += self.vel

        # Keep the drone within simulation bounds.
        if self.pos.x < 0:
            self.pos.x = 0
            self.vel.x = abs(self.vel.x)
        elif self.pos.x > SIM_WIDTH:
            self.pos.x = SIM_WIDTH
            self.vel.x = -abs(self.vel.x)
        if self.pos.y < 0:
            self.pos.y = 0
            self.vel.y = abs(self.vel.y)
        elif self.pos.y > SIM_HEIGHT:
            self.pos.y = SIM_HEIGHT
            self.vel.y = -abs(self.vel.y)
            
        # Prevent the drone from exceeding the EXTERNAL_RADIUS from the interest point.
        if self.pos.distance_to(self.interest_point_center) > EXTERNAL_RADIUS:
            direction = (self.pos - self.interest_point_center).normalize()
            self.pos = self.interest_point_center + direction * EXTERNAL_RADIUS
            self.vel = pygame.math.Vector2(0, 0)

    # -------------------------------------------------------------------------
    # Update Drone State
    # -------------------------------------------------------------------------
    def update(self, enemy_drones: List[Any], friend_drones: List[Any]) -> None:
        """
        Updates the drone's state by applying decay, updating local detections,
        communicating with nearby drones, and executing actions.
        
        Args:
            enemy_drones (List[Any]): List of enemy drones.
            friend_drones (List[Any]): List of friend drones.
        """
        self.decay_matrices()
        self.update_local_enemy_detection(enemy_drones)
        self.update_local_friend_detection(friend_drones)
        self.communication(friend_drones)
        self.take_action()
        self.trajectory.append(self.pos.copy())

    # -------------------------------------------------------------------------
    # Planning Policy (Class Method)
    # -------------------------------------------------------------------------
    @classmethod
    def planning_policy(cls, state, activation_threshold_position: float = 0.2):
        """
        Updates the drone's state each iteration and generates the velocity (action)
        to be applied. It processes sparse enemy and friend detection matrices and
        calculates which enemy target to pursue based on proximity.
        
        Args:
            state (dict): Contains keys 'pos', 'friend_intensity', 'enemy_intensity',
                          'friend_direction', and 'enemy_direction'.
            activation_threshold_position (float): Minimum intensity to consider a cell active.
        
        Returns:
            pygame.math.Vector2: The velocity (action) scaled by FRIEND_SPEED.
        """
        
        def hold_position(pos, friend_intensity, activation_threshold_position: float = 1) -> pygame.math.Vector2:
            """
            Makes the drone hold its position based on local friend detection.
            
            If any cell in the friend intensity matrix exceeds the threshold, returns a zero velocity.
            Otherwise, the drone moves toward the interest point if it is farther than INITIAL_DISTANCE.
            
            Args:
                pos (pygame.math.Vector2): Current position.
                friend_intensity (np.ndarray): Friend detection intensity matrix.
                activation_threshold_position (float): Threshold to consider that there is active detection.
            
            Returns:
                pygame.math.Vector2: The velocity vector (action).
            """
            for cell, intensity in np.ndenumerate(friend_intensity):
                if intensity >= activation_threshold_position:
                    return pygame.math.Vector2(0, 0)

            current_distance = pos.distance_to(INTEREST_POINT_CENTER)
            if current_distance > INITIAL_DISTANCE:
                direction = (INTEREST_POINT_CENTER - pos).normalize()
                return direction
            else:
                return pygame.math.Vector2(0, 0)
            
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        friend_intensity = np.squeeze(state['friend_intensity'])
        enemy_intensity = np.squeeze(state['enemy_intensity'])
        friend_direction = np.squeeze(state['friend_direction'])
        enemy_direction = np.squeeze(state['enemy_direction'])
        
        enemy_targets = []

        # Identify enemy targets with sufficient intensity.
        for cell, intensity in np.ndenumerate(enemy_intensity):
            if intensity < activation_threshold_position:
                continue
            target_pos = pygame.math.Vector2((cell[0] + 0.5) * CELL_SIZE, (cell[1] + 0.5) * CELL_SIZE)
            distance_to_interest = target_pos.distance_to(INTEREST_POINT_CENTER)
            enemy_targets.append((cell, target_pos, distance_to_interest))
        
        if not enemy_targets:
            return hold_position(pos, friend_intensity)
        
        # Sort targets by their distance to the interest point.
        enemy_targets.sort(key=lambda t: t[2])
        
        my_cell = pos_to_cell(pos)
        my_cell_center = pygame.math.Vector2((my_cell[0] + 0.5) * CELL_SIZE, (my_cell[1] + 0.5) * CELL_SIZE)
        
        for cell, target_pos, _ in enemy_targets:
            my_distance = my_cell_center.distance_to(target_pos)
            closest_distance = my_distance

            # Compare with friend detections.
            for cell, intensity in np.ndenumerate(friend_intensity):
                if intensity < activation_threshold_position:
                    continue
                friend_pos = pygame.math.Vector2((cell[0] + 0.5) * CELL_SIZE, (cell[1] + 0.5) * CELL_SIZE)
                friend_distance = friend_pos.distance_to(target_pos)
                if friend_distance < closest_distance:
                    closest_distance = friend_distance

            if my_distance <= closest_distance:
                direction = intercept_direction(pos, FRIEND_SPEED, target_pos, enemy_direction[cell])
                if direction.length() > 0:
                    return direction.normalize()
                else:
                    return pygame.math.Vector2(0, 0)
        
        # If no enemy target is pursued, hold position.
        return hold_position(pos, friend_intensity)
    
    # -------------------------------------------------------------------------
    # Artificial Inteligence Policy (Class Method)
    # -------------------------------------------------------------------------
    @classmethod
    def ai_policy(cls, state, activation_threshold_position: float = 0.2):
        """
        Updates the drone's state each iteration and generates the velocity (action)
        to be applied. It processes sparse enemy and friend detection matrices and
        calculates which enemy target to pursue based on proximity.
        
        Args:
            state (dict): Contains keys 'pos', 'friend_intensity', 'enemy_intensity',
                          'friend_direction', and 'enemy_direction'.
            activation_threshold_position (float): Minimum intensity to consider a cell active.
        
        Returns:
            pygame.math.Vector2: The velocity (action) scaled by FRIEND_SPEED.
        """
        
        def hold_position(pos, friend_intensity) -> pygame.math.Vector2:
            if cls.model is None:
                cls.model = load_best_model(directory='./models', pattern=r"val_loss=([\d.]+)\.keras")
                
            direction = np.squeeze(cls.model.predict(state))
            direction = pygame.math.Vector2(direction[0], direction[1]).normalize()
            
            return direction
            
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        friend_intensity = np.squeeze(state['friend_intensity'])
        enemy_intensity = np.squeeze(state['enemy_intensity'])
        friend_direction = np.squeeze(state['friend_direction'])
        enemy_direction = np.squeeze(state['enemy_direction'])
        
        enemy_targets = []

        # Identify enemy targets with sufficient intensity.
        for cell, intensity in np.ndenumerate(enemy_intensity):
            if intensity < activation_threshold_position:
                continue
            target_pos = pygame.math.Vector2((cell[0] + 0.5) * CELL_SIZE, (cell[1] + 0.5) * CELL_SIZE)
            distance_to_interest = target_pos.distance_to(INTEREST_POINT_CENTER)
            enemy_targets.append((cell, target_pos, distance_to_interest))
        
        if not enemy_targets:
            return hold_position(pos, friend_intensity)
        
        # Sort targets by their distance to the interest point.
        enemy_targets.sort(key=lambda t: t[2])
        
        my_cell = pos_to_cell(pos)
        my_cell_center = pygame.math.Vector2((my_cell[0] + 0.5) * CELL_SIZE, (my_cell[1] + 0.5) * CELL_SIZE)
        
        for cell, target_pos, _ in enemy_targets:
            my_distance = my_cell_center.distance_to(target_pos)
            closest_distance = my_distance

            # Compare with friend detections.
            for cell, intensity in np.ndenumerate(friend_intensity):
                if intensity < activation_threshold_position:
                    continue
                friend_pos = pygame.math.Vector2((cell[0] + 0.5) * CELL_SIZE, (cell[1] + 0.5) * CELL_SIZE)
                friend_distance = friend_pos.distance_to(target_pos)
                if friend_distance < closest_distance:
                    closest_distance = friend_distance

            if my_distance <= closest_distance:
                direction = intercept_direction(pos, FRIEND_SPEED, target_pos, enemy_direction[cell])
                if direction.length() > 0:
                    return direction.normalize()
                else:
                    return pygame.math.Vector2(0, 0)
        
        # If no enemy target is pursued, hold position.
        return hold_position(pos, friend_intensity)
    
    # -------------------------------------------------------------------------
    # Artificial Inteligence Policy (Class Method)
    # -------------------------------------------------------------------------
    # @classmethod
    # def ai_policy(cls, state):
    #     if cls.model is None:
    #         cls.model = load_best_model(directory='./models', pattern=r"val_loss=([\d.]+)\.keras")
            
    #     direction = np.squeeze(cls.model.predict(state))
    #     direction = pygame.math.Vector2(direction[0], direction[1]).normalize()
        
    #     return direction
        
    # -------------------------------------------------------------------------
    # Apply Behavior
    # -------------------------------------------------------------------------
    def apply_behavior(self) -> None:
        """
        Updates the drone's velocity based on its behavior type.
        """
        if self.behavior_type == "planning":
            # Generate the state from the detection matrices.
            state = {
                'pos': np.array([[self.pos.x, self.pos.y]], dtype=np.float32),
                'friend_intensity': np.expand_dims(self.friend_intensity, axis=0),
                'enemy_intensity': np.expand_dims(self.enemy_intensity, axis=0),
                'friend_direction': np.expand_dims(self.friend_direction, axis=0),
                'enemy_direction': np.expand_dims(self.enemy_direction, axis=0)
            }
            direction = self.planning_policy(state)
            self.vel = direction * FRIEND_SPEED if direction.length() > 0 else pygame.math.Vector2(0, 0)
            
        elif self.behavior_type == "AI":
            # Generate the state from the detection matrices.
            state = {
                'pos': np.array([[self.pos.x, self.pos.y]], dtype=np.float32),
                'friend_intensity': np.expand_dims(self.friend_intensity, axis=0),
                'enemy_intensity': np.expand_dims(self.enemy_intensity, axis=0),
                'friend_direction': np.expand_dims(self.friend_direction, axis=0),
                'enemy_direction': np.expand_dims(self.enemy_direction, axis=0)
            }
            direction = self.ai_policy(state)
            self.vel = direction * FRIEND_SPEED if direction.length() > 0 else pygame.math.Vector2(0, 0)
            
        elif self.behavior_type == "AEW":
            # If orbit_radius is not set, initialize it.
            if self.orbit_radius is None:
                self.orbit_radius = AEW_RANGE

            # Compute the radial vector from the interest point.
            r_vec = self.pos - self.interest_point_center
            current_distance = r_vec.length()
            if current_distance == 0:
                r_vec = pygame.math.Vector2(self.orbit_radius, 0)
                current_distance = self.orbit_radius
            radial_error = self.orbit_radius - current_distance
            k_radial = 0.05  # Radial correction factor
            radial_correction = k_radial * radial_error * r_vec.normalize()
            tangent = pygame.math.Vector2(-r_vec.y, r_vec.x).normalize()
            tangential_velocity = tangent * AEW_SPEED
            desired_velocity = tangential_velocity + radial_correction
            self.vel = desired_velocity.normalize() * AEW_SPEED if desired_velocity.length() > 0 else pygame.math.Vector2(0, 0)
            
        elif self.behavior_type == "RADAR":
            self.vel = pygame.math.Vector2(0, 0)

        elif self.behavior_type == "debug":
            target_vector = self.interest_point_center - self.pos
            direction = target_vector.normalize() if target_vector.length() > 0 else pygame.math.Vector2(0, 0)
            self.vel = direction * FRIEND_SPEED if not self.fixed else pygame.math.Vector2(0, 0)
                
        elif self.behavior_type == "u-debug":
            if not hasattr(self, 'u_debug_phase'):
                self.u_debug_phase = 0  # 0: moving forward, 1: moving perpendicular, 2: moving backward
                self.u_debug_timer = 0
                self.forward_steps = 300
                self.perp_steps = 40

            target_vector = self.interest_point_center - self.pos
            target_direction = target_vector.normalize()
            
            if not self.fixed:
                if self.u_debug_phase == 0:
                    self.vel = target_direction * FRIEND_SPEED
                    self.u_debug_timer += 1
                    if self.u_debug_timer >= self.forward_steps:
                        self.u_debug_phase = 1
                        self.u_debug_timer = 0
                elif self.u_debug_phase == 1:
                    perp_direction = pygame.math.Vector2(-target_direction.y, target_direction.x)
                    self.vel = perp_direction * FRIEND_SPEED
                    self.u_debug_timer += 1
                    if self.u_debug_timer >= self.perp_steps:
                        self.u_debug_phase = 2
                        self.u_debug_timer = 0
                elif self.u_debug_phase == 2:
                    self.vel = (-target_direction) * FRIEND_SPEED
                    self.u_debug_timer += 1
                    if self.u_debug_timer >= self.forward_steps:
                        self.vel = pygame.math.Vector2(0, 0)
            else:
                self.vel = pygame.math.Vector2(0, 0)

    # -------------------------------------------------------------------------
    # Rendering: Draw the Drone
    # -------------------------------------------------------------------------
    def draw(self, surface: pygame.Surface, show_detection: bool = True, show_comm_range: bool = True, show_trajectory: bool = False) -> None:
        """
        Draws the drone on the provided surface.
        
        Args:
            surface (pygame.Surface): The surface on which to draw the drone.
            show_detection (bool): If True, displays the detection range.
            show_comm_range (bool): If True, displays the communication range.
            show_trajectory (bool): If True, draws the drone's trajectory.
        """
        # Draw the trajectory if enabled.
        if show_trajectory and len(self.trajectory) > 1:
            traj_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            decay_rate = 0.025
            n = len(self.trajectory)
            for i in range(n - 1):
                d = n - 1 - i
                alpha = int(255 * math.exp(-decay_rate * d))
                min_alpha = 30
                alpha = max(alpha, min_alpha)
                color_with_alpha = self.color + (alpha,)
                start_pos = (int(self.trajectory[i].x), int(self.trajectory[i].y))
                end_pos = (int(self.trajectory[i+1].x), int(self.trajectory[i+1].y))
                pygame.draw.line(traj_surf, color_with_alpha, start_pos, end_pos, 2)
            surface.blit(traj_surf, (0, 0))
        
        # Draw detection range.
        if show_detection:
            if self.behavior_type == "AEW":
                detection_range = AEW_DETECTION_RANGE
            elif self.behavior_type == "RADAR":
                detection_range = RADAR_DETECTION_RANGE
            else:
                detection_range = FRIEND_DETECTION_RANGE
            
            draw_dashed_circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), detection_range, 5, 5, 1)
            
        # Draw communication range.
        if show_comm_range:
            draw_dashed_circle(surface, (255, 255, 0), (int(self.pos.x), int(self.pos.y)), COMMUNICATION_RANGE, 5, 5, 1)
        
        # Draw the drone image.
        image_rect = self.image.get_rect(center=(int(self.pos.x), int(self.pos.y)))
        surface.blit(self.image, image_rect)
        
        # Render the drone's ID with semi-transparent white text.
        font: pygame.font.Font = pygame.font.SysFont(FONT_FAMILY, 12)
        label = font.render(f"ID: F{self.drone_id}", True, (255, 255, 255))
        label.set_alpha(128)
        surface.blit(label, (int(self.pos.x) + 20, int(self.pos.y) - 20))
        
        # If the drone is a leader, indicate with a "LEADER" label.
        if self.is_leader:
            leader_label = font.render("LEADER", True, (255, 215, 0))
            surface.blit(leader_label, (int(self.pos.x) + 20, int(self.pos.y) - 5))
        
        # If the drone is selected, display a "GRAPH" label.
        if self.selected:
            selected_label = font.render("GRAPH", True, (255, 215, 0))
            surface.blit(selected_label, (int(self.pos.x) + 20, int(self.pos.y) + 10))