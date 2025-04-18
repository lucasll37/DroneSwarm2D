"""
FriendDrone.py

This module defines the FriendDrone class used in the simulation. The FriendDrone class handles
local detection (enemy and friend), communication, merging of detection matrices,
triangulation of targets, action execution, and rendering. It also provides planning and
debug behaviors for drone motion.
"""

# -----------------------------------------------------------------------------
# Imports and Setup
# -----------------------------------------------------------------------------
import random
import math
import os
import sys
from typing import Optional, Tuple, List, Any, Dict
import itertools

import numpy as np
import pygame

from settings import *
from utils import (
    draw_dashed_circle, 
    load_svg_as_surface, 
    pos_to_cell, 
    intercept_direction, 
    load_best_model, 
    generate_sparse_matrix, 
    draw_dashed_line
)
from distributedDefensiveAlgorithm import planning_policy

# -----------------------------------------------------------------------------
# FriendDrone Class Definition
# -----------------------------------------------------------------------------
class FriendDrone:
    """
    FriendDrone represents a friendly drone in the simulation environment.
    
    This class handles detection of enemy and friendly drones, communication
    between drones, movement behaviors, and visualization.
    """

    # Class variables
    friend_id_counter: int = 0
    original_drone_image: pygame.Surface = load_svg_as_surface("./assets/drone_0.svg")
    original_broken_drone_image: pygame.Surface = load_svg_as_surface("./assets/drone_broken.svg")
    original_aew_image: pygame.Surface = load_svg_as_surface("./assets/radar_0.svg")
    original_radar_image: pygame.Surface = load_svg_as_surface("./assets/radar_0.svg")
    
    # AI model (loaded on demand)
    model = None
    
    # Seed da classe - inicialmente None, será gerada se não for definida explicitamente
    class_seed = None
    class_rng = None
    class_np_rng = None
    

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(
        self, 
        interest_point_center: pygame.math.Vector2, 
        position: Tuple[float, float], 
        behavior_type: str = "planning",
        fixed: bool = False, 
        broken: bool = False
    ) -> None:
        """
        Initialize the drone with its starting position, interest point, and behavior type.
        
        Args:
            interest_point_center: The center point of interest for the drone.
            position: Initial (x, y) position of the drone.
            behavior_type: The behavior strategy ("planning", "AI", "AEW", "RADAR", "debug", "u-debug").
            fixed: If True, the drone remains stationary.
            broken: If True, the drone will provide faulty detection information.
            detection_mode: Detection mode - "direct" or "triangulation"
        """
        if FriendDrone.class_seed is None:
            FriendDrone.set_class_seed()
            
        # Basic properties
        self.pos: pygame.math.Vector2 = pygame.math.Vector2(position[0], position[1])
        self.interest_point_center = interest_point_center
        self.behavior_type = behavior_type
        self.fixed = fixed
        self.selected = False
        self.vel: pygame.math.Vector2 = pygame.math.Vector2(0, 0)
        self.orbit_radius = None  # Used for AEW behavior
        self.trajectory: List[pygame.math.Vector2] = []
        self.return_to_base: bool = False
        self.info: Tuple[str, Any, Any, Any] = ("", None, None, None)
        self.detection_mode = DETECTION_MODE
        self.neighbors = []

        # Drone properties
        self.color: Tuple[int, int, int] = (255, 255, 255)
        self.drone_id: int = self.assign_id()
        self.in_election: bool = False
        self.is_leader: bool = False
        self.leader_id: int = self.drone_id
        self.broken: bool = broken
        
        # Broken drone state tracking
        self.timer_state_broken = 0
        self.update_state_broken = UPDATE_STATE_BROKEN
        self.broken_friend_intensity = None
        self.broken_friend_direction = None
        self.broken_enemy_intensity = None
        self.broken_enemy_direction = None

        # Detection dictionaries
        self.aux_enemy_detections: Dict[int, Tuple[int, int]] = {}
        self.aux_friend_detections: Dict[int, Tuple[int, int]] = {}
        self.current_enemy_pos_detection: Dict[int, pygame.math.Vector2] = {}
        self.current_friend_pos_detection: Dict[int, pygame.math.Vector2] = {}
        
        # Direction-only detection (prototype)
        self.enemy_direction_only = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2))  # Unit direction vector
        self.enemy_detection_confidence = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        
        # State tracking
        self.state_history = {}
        self.current_state = ""
        self.total_steps = 0 
        self.messages_sent = 0
        self.distance_traveled = 0
        self.last_position = self.pos.copy()
        self.active_connections = 0
        self.messages_sent_this_cycle = 0
        
        # Detection matrices
        self.enemy_intensity: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        self.enemy_direction: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2))
        self.enemy_timestamp: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        self.friend_intensity: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        self.friend_direction: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2))
        self.friend_timestamp: np.ndarray = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        
        # Set appropriate image based on drone type
        self._setup_drone_image()
                    
    @classmethod
    def set_class_seed(cls, seed=None):            
        cls.class_seed = seed if seed is not None else random.randint(0, 10000000)
        cls.class_rng = random.Random(cls.class_seed)
        cls.class_np_rng = np.random.RandomState(cls.class_seed)
            
            
    def _setup_drone_image(self) -> None:
        """
        Set up the appropriate visual representation based on drone type.
        """
        if self.behavior_type == "RADAR":
            desired_width = int(SIM_WIDTH * 0.03)
            aspect_ratio = self.original_radar_image.get_height() / self.original_radar_image.get_width()
            desired_height = int(desired_width * aspect_ratio)
            self.image = pygame.transform.scale(self.original_radar_image, (desired_width, desired_height))
        elif self.broken:
            desired_width = int(SIM_WIDTH * 0.02)
            aspect_ratio = self.original_broken_drone_image.get_height() / self.original_broken_drone_image.get_width()
            desired_height = int(desired_width * aspect_ratio)
            self.image = pygame.transform.scale(self.original_broken_drone_image, (desired_width, desired_height))
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
        Assign a unique ID to the drone.
        
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
        Apply exponential decay to both enemy and friend detection intensity matrices.
        This simulates information becoming less reliable over time.
        """
        self.enemy_intensity *= DECAY_FACTOR
        self.friend_intensity *= DECAY_FACTOR
        
    # -------------------------------------------------------------------------
    # Passive Detection and Triangulation
    # -------------------------------------------------------------------------  
    def update_passive_detection_and_triangulate(self, enemy_drones: List[Any], friend_drones: List[Any]) -> None:
        """
        Implement passive detection (direction-only) and distributed triangulation.
        Only processes triangulation data if in "triangulation" mode.
        
        Args:
            enemy_drones: List of enemy drones in the environment.
            friend_drones: List of friendly drones for communication.
        """
        # If not in triangulation mode, do nothing
        if self.detection_mode != "triangulation":
            return
            
        # Initialize data structures if not already created
        current_time = pygame.time.get_ticks()
        self.passive_detections = {}  # {direction_hash: (direction, timestamp)}
        
        if not hasattr(self, 'direction_history'):
            self.direction_history = {}  # {direction_hash: (direction, timestamp, source_drone_id)}
            
        if not hasattr(self, 'triangulated_targets'):
            self.triangulated_targets = {}  # {target_id: (position, confidence, last_update_time)}
        
        # 1. Detect directions (without distances) of enemies within range
        self._detect_enemy_directions(enemy_drones, current_time)
        
        # 2. Share detections with nearby friendly drones
        self._share_detections_with_friends(friend_drones, current_time)
        
        # 3. Attempt triangulation with available directions
        self._perform_triangulation(friend_drones, current_time)
        
        # 4. Clean up old data
        self._cleanup_old_data(current_time)
        
        # 5. Update existing detection matrices with triangulated targets
        self._update_matrices_with_triangulation(current_time)
        
        # Store for the next cycle
        if not hasattr(self, 'previous_triangulated'):
            self.previous_triangulated = {}
        self.previous_triangulated = self.triangulated_targets.copy()
        
        # Store for visualization
        self.last_triangulated = {
            target_id: (position, confidence) 
            for target_id, (position, confidence, _) in self.triangulated_targets.items()
        }

    def _detect_enemy_directions(self, enemy_drones: List[Any], current_time: int) -> None:
        """
        Detect direction vectors to enemy drones within detection range.
        
        Args:
            enemy_drones: List of enemy drones to detect.
            current_time: Current simulation time.
        """
        for enemy in enemy_drones:
            delta = enemy.pos - self.pos
            distance = delta.length()
            
            # Check if enemy is within detection range
            detection_range = self._get_detection_range()
                
            if distance <= detection_range and distance > 0:
                # Store only the normalized direction (without distance)
                direction = delta.normalize()
                
                # Create a hash for this direction (discretized angle)
                angle = math.atan2(direction.y, direction.x)
                angle_discrete = round(angle / 0.01) * 0.01  # Discretize every 0.01 radians
                direction_hash = f"{angle_discrete:.2f}"
                
                # Store in passive detections dictionary
                self.passive_detections[direction_hash] = (direction, current_time)
                
                # Update direction history
                self.direction_history[direction_hash] = (direction, current_time, id(self))

    def _get_detection_range(self) -> float:
        """
        Get the appropriate detection range based on drone type.
        
        Returns:
            float: Detection range in pixels.
        """
        if self.behavior_type == "AEW":
            return AEW_DETECTION_RANGE
        elif self.behavior_type == "RADAR":
            return RADAR_DETECTION_RANGE
        else:
            return FRIEND_DETECTION_RANGE

    def _share_detections_with_friends(self, friend_drones: List[Any], current_time: int) -> None:
        """
        Share detection data with nearby friendly drones.
        
        Args:
            friend_drones: List of friendly drones.
            current_time: Current simulation time.
        """
        for friend in friend_drones:
            if friend is self or friend.pos.distance_to(self.pos) > COMMUNICATION_RANGE:
                continue
                
            # Share passive detections
            if hasattr(friend, 'passive_detections'):
                for direction_hash, (direction, timestamp) in friend.passive_detections.items():
                    # Only accept relatively recent detections (less than 2 seconds old)
                    if current_time - timestamp < 2000:
                        # If we don't have this direction or ours is older, update
                        if direction_hash not in self.direction_history or \
                        timestamp > self.direction_history[direction_hash][1]:
                            self.direction_history[direction_hash] = (direction, timestamp, id(friend))
            
            # Share already triangulated targets
            if hasattr(friend, 'triangulated_targets'):
                for target_id, (position, confidence, last_update) in friend.triangulated_targets.items():
                    # Only accept recent triangulations with reasonable confidence
                    if current_time - last_update < 3000 and confidence > 0.5:
                        # If we don't have this target or ours is older, update
                        if target_id not in self.triangulated_targets or \
                        last_update > self.triangulated_targets[target_id][2]:
                            self.triangulated_targets[target_id] = (position, confidence, last_update)

    def _perform_triangulation(self, friend_drones: List[Any], current_time: int) -> None:
        """
        Attempt to triangulate enemy positions using collected direction data.
        
        Args:
            friend_drones: List of friendly drones.
            current_time: Current simulation time.
        """
        # Group detectors by position for each direction
        direction_detectors = {}  # {direction_hash: [(detector_id, pos, direction, timestamp)]}
        
        for direction_hash, (direction, timestamp, detector_id) in self.direction_history.items():
            # Find the detector position
            detector_pos = None
            if detector_id == id(self):
                detector_pos = self.pos
            else:
                for friend in friend_drones:
                    if id(friend) == detector_id:
                        detector_pos = friend.pos
                        break
            
            if detector_pos:
                if direction_hash not in direction_detectors:
                    direction_detectors[direction_hash] = []
                direction_detectors[direction_hash].append((detector_id, detector_pos, direction, timestamp))
        
        # Group directions into potential targets
        potential_targets = {}  # {target_id: [(detector_id, dir_hash, pos, direction, timestamp)]}
        target_counter = 0
        
        # For each direction, check compatibility with other directions
        for dir_hash1, detectors1 in direction_detectors.items():
            for detector_id1, pos1, dir1, timestamp1 in detectors1:
                # Check if already assigned to a target
                already_assigned = False
                for target_detectors in potential_targets.values():
                    if any(did == detector_id1 and dhash == dir_hash1 
                          for did, dhash, _, _, _ in target_detectors):
                        already_assigned = True
                        break
                
                if already_assigned:
                    continue
                
                # Try to form a new group of compatible detections
                compatible_detections = [(detector_id1, dir_hash1, pos1, dir1, timestamp1)]
                
                for dir_hash2, detectors2 in direction_detectors.items():
                    if dir_hash1 == dir_hash2:
                        continue
                        
                    for detector_id2, pos2, dir2, timestamp2 in detectors2:
                        # Don't include the same detector twice
                        if detector_id2 == detector_id1:
                            continue
                            
                        # Check compatibility
                        if self._directions_compatible(pos1, dir1, pos2, dir2):
                            compatible_detections.append((detector_id2, dir_hash2, pos2, dir2, timestamp2))
                
                # If we have at least 3 compatible detectors, create a new potential target
                if len(compatible_detections) >= 3:
                    # Ensure they are 3 distinct drones
                    unique_detectors = set(did for did, _, _, _, _ in compatible_detections)
                    if len(unique_detectors) >= 3:
                        target_counter += 1
                        potential_targets[target_counter] = compatible_detections
        
        # Triangulate positions for potential targets
        for target_id, compatible_detections in potential_targets.items():
            position_estimates = []
            
            # Triangulate with all possible pairs
            for i in range(len(compatible_detections)):
                for j in range(i+1, len(compatible_detections)):
                    _, _, pos1, dir1, timestamp1 = compatible_detections[i]
                    _, _, pos2, dir2, timestamp2 = compatible_detections[j]
                    
                    position = self._triangulate_position(pos1, dir1, pos2, dir2)
                    
                    if position:
                        # Calculate confidence based on recency
                        recency1 = max(0, 1 - (current_time - timestamp1) / 1000)
                        recency2 = max(0, 1 - (current_time - timestamp2) / 1000)
                        confidence = recency1 * recency2
                        
                        position_estimates.append((position, confidence))
            
            if position_estimates:
                # Calculate weighted average of estimated positions
                total_confidence = sum(conf for _, conf in position_estimates)
                if total_confidence > 0:
                    weighted_pos = pygame.math.Vector2(0, 0)
                    for pos, conf in position_estimates:
                        weighted_pos += pos * (conf / total_confidence)
                    
                    # Calculate global confidence based on number of detectors and average confidence
                    unique_detectors = set(did for did, _, _, _, _ in compatible_detections)
                    detector_factor = min(1.0, len(unique_detectors) / 5)  # Normalized up to 5 detectors
                    
                    overall_confidence = (total_confidence / len(position_estimates)) * detector_factor
                    
                    # Store the triangulated target
                    stable_target_id = f"target_{hash(str(weighted_pos))}"
                    self.triangulated_targets[stable_target_id] = (weighted_pos, overall_confidence, current_time)

    def _cleanup_old_data(self, current_time: int) -> None:
        """
        Remove outdated direction and target data.
        
        Args:
            current_time: Current simulation time.
        """
        # Remove old directions from history
        old_directions = [hash_key for hash_key, (_, timestamp, _) in self.direction_history.items() 
                         if current_time - timestamp > 3000]  # 3 seconds
        for hash_key in old_directions:
            self.direction_history.pop(hash_key, None)
        
        # Remove old targets
        old_targets = [tid for tid, (_, _, timestamp) in self.triangulated_targets.items() 
                      if current_time - timestamp > 5000]  # 5 seconds
        for tid in old_targets:
            self.triangulated_targets.pop(tid, None)

    def _update_matrices_with_triangulation(self, current_time: int) -> None:
        """
        Update detection matrices with triangulated target information.
        
        Args:
            current_time: Current simulation time.
        """
        for target_id, (position, confidence, _) in self.triangulated_targets.items():
            # Convert position to cell index
            cell = pos_to_cell(position)
            
            # Update detection structures
            if confidence > self.enemy_intensity[cell]:
                # Store position in detection dictionary for future calculations
                key = f"triangulated_{target_id}"
                
                # Calculate direction if we have history
                direction = pygame.math.Vector2(0, 0)  # Default with no direction
                
                if hasattr(self, 'previous_triangulated') and target_id in self.previous_triangulated:
                    prev_pos, _, _ = self.previous_triangulated[target_id]
                    delta = position - prev_pos
                    if delta.length() > 0:
                        direction = delta.normalize()
                
                # Update matrices
                self.enemy_intensity[cell] = 1 # confidence
                self.enemy_direction[cell] = [direction.x, direction.y]
                self.enemy_timestamp[cell] = current_time
                
                # Record for history
                if key not in self.current_enemy_pos_detection:
                    self.current_enemy_pos_detection[key] = position.copy()
                else:
                    # Already existed, update direction records
                    self.aux_enemy_detections[key] = cell
                    self.current_enemy_pos_detection[key] = position.copy()

    def _directions_compatible(self, pos1, dir1, pos2, dir2, angle_threshold=0.2):
        """
        Check if two detection directions are compatible (might point to the same target).
        
        Args:
            pos1, pos2: Detector drone positions.
            dir1, dir2: Detection directions.
            angle_threshold: Angle threshold (in radians) for considering compatibility.
            
        Returns:
            bool: True if directions are potentially compatible.
        """
        # Calculate intersection point (may be approximate)
        intersection = self._triangulate_position(pos1, dir1, pos2, dir2)
        
        if intersection is None:
            return False
            
        # Check if the actual directions from both detectors to the intersection
        # point are close to the detection directions
        real_dir1 = (intersection - pos1)
        real_dir2 = (intersection - pos2)
        
        if real_dir1.length() > 0 and real_dir2.length() > 0:
            real_dir1 = real_dir1.normalize()
            real_dir2 = real_dir2.normalize()
            
            angle1 = math.acos(max(-1, min(1, real_dir1.dot(dir1))))
            angle2 = math.acos(max(-1, min(1, real_dir2.dot(dir2))))
            
            return angle1 < angle_threshold and angle2 < angle_threshold
        
        return False

    def _triangulate_position(self, pos1, dir1, pos2, dir2):
        """
        Triangulate a position from two positions and directions.
        Returns None if lines are nearly parallel.
        
        Args:
            pos1, pos2: Positions of two detector drones.
            dir1, dir2: Normalized directions pointing to the target.
            
        Returns:
            pygame.math.Vector2: Estimated target position, or None if triangulation not possible.
        """
        # Represent the lines as:
        # pos1 + t1 * dir1
        # pos2 + t2 * dir2
        
        # Matrix to solve the linear system
        A = np.array([
            [dir1.x, -dir2.x],
            [dir1.y, -dir2.y]
        ])
        
        b = np.array([
            pos2.x - pos1.x,
            pos2.y - pos1.y
        ])
        
        # Check if the system is well-conditioned
        # (lines are not nearly parallel)
        if abs(np.linalg.det(A)) < 0.1:  # Adjust threshold as needed
            return None
            
        try:
            # Solve the system to find t1 and t2
            t1, t2 = np.linalg.solve(A, b)
            
            # If t1 < 0 or t2 < 0, the lines cross "backwards"
            if t1 < 0 or t2 < 0:
                return None
                
            # Calculate the intersection point
            intersection1 = pygame.math.Vector2(pos1.x + dir1.x * t1, pos1.y + dir1.y * t1)
            intersection2 = pygame.math.Vector2(pos2.x + dir2.x * t2, pos2.y + dir2.y * t2)
            
            # Should be the same point, but may have numerical differences
            # Return the average as a robust measure
            return pygame.math.Vector2((intersection1.x + intersection2.x) / 2, 
                                      (intersection1.y + intersection2.y) / 2)
            
        except:
            # If there are problems solving the system
            return None
                
    # -------------------------------------------------------------------------
    # Update Local Enemy Detection
    # -------------------------------------------------------------------------
    def update_local_enemy_detection(self, enemy_drones: List[Any]) -> None:
        """
        Update local detection of enemy drones based on the selected detection mode.
        
        In "direct" mode, only direct detections are processed.
        In "triangulation" mode, only triangulated detections are processed.
        
        Args:
            enemy_drones: List of enemy drones.
        """
        current_time: int = pygame.time.get_ticks()
        detection_range = self._get_detection_range()
        
        # Direct detection (only in "direct" mode)
        if self.detection_mode == "direct":
            self._perform_direct_detection(enemy_drones, detection_range, current_time)
        
        # In "triangulation" mode, detection is already done in update_passive_detection_and_triangulate()
        # and results are directly stored in detection matrices
        
        # --- Vectorized Update for Cells Without Detection ---
        # Clean up empty cells within detection radius (important in both modes)
        detection_range_cells = int(np.floor(detection_range / CELL_SIZE) * 0.8)
        
        # Get the central cell (drone's position)
        center_x, center_y = pos_to_cell(self.pos)
        
        # Define limits of rectangle enclosing detection circle
        x_min = max(center_x - detection_range_cells, 0)
        x_max = min(center_x + detection_range_cells, GRID_WIDTH - 1)
        y_min = max(center_y - detection_range_cells, 0)
        y_max = min(center_y + detection_range_cells, GRID_HEIGHT - 1)
        
        # Create grid of indices for the region
        x_indices = np.arange(x_min, x_max + 1)
        y_indices = np.arange(y_min, y_max + 1)
        xv, yv = np.meshgrid(x_indices, y_indices, indexing='ij')
        
        # Calculate distance of each cell from center
        distances = np.sqrt((xv - center_x) ** 2 + (yv - center_y) ** 2)
        
        # Extract region of matrices
        region_intensity = self.enemy_intensity[x_min:x_max+1, y_min:y_max+1]
        region_timestamp = self.enemy_timestamp[x_min:x_max+1, y_min:y_max+1]
        
        # Create mask for cells within detection circle with low intensity
        mask_empty = (distances <= detection_range_cells) & (region_intensity < 1)
        
        # Set intensities to 0 and update timestamps for empty cells
        np.putmask(region_intensity, mask_empty, 0)
        np.putmask(region_timestamp, mask_empty, current_time)
        
        # Apply broken detection behavior if drone is broken
        if self.broken:
            self.update_broken(x_min, x_max, y_min, y_max, distances, detection_range_cells)

    def _perform_direct_detection(self, enemy_drones: List[Any], detection_range: float, current_time: int) -> None:
        """
        Perform direct detection of enemy drones within detection range.
        
        Args:
            enemy_drones: List of enemy drones.
            detection_range: Maximum detection range in pixels.
            current_time: Current simulation time.
        """
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
                        # Zero out values in the previous cell
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

    # -------------------------------------------------------------------------
    # Update Local Friend Detection
    # -------------------------------------------------------------------------
    def update_local_friend_detection(self, friend_drones: List[Any]) -> None:
        """
        Update local detection of friendly drones.
        
        For each friendly drone (excluding AEW drones and self), update the 
        corresponding cell in the detection matrices.
        
        Args:
            friend_drones: List of friendly drones.
        """
        current_time: int = pygame.time.get_ticks()
        for friend in friend_drones:
            # Skip AEW drones and self
            if friend.behavior_type == "AEW" or friend is self:
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
        # Convert friend detection range to cell units
        detection_range_cells = int(np.floor(FRIEND_DETECTION_RANGE / CELL_SIZE) * 0.8)
        
        # Get the central cell (drone's position)
        center_x, center_y = pos_to_cell(self.pos)
        
        # Define rectangle covering detection circle
        x_min = max(center_x - detection_range_cells, 0)
        x_max = min(center_x + detection_range_cells, GRID_WIDTH - 1)
        y_min = max(center_y - detection_range_cells, 0)
        y_max = min(center_y + detection_range_cells, GRID_HEIGHT - 1)
        
        # Create index grid for region
        x_indices = np.arange(x_min, x_max + 1)
        y_indices = np.arange(y_min, y_max + 1)
        xv, yv = np.meshgrid(x_indices, y_indices, indexing='ij')
        
        # Calculate cell distances from center
        distances = np.sqrt((xv - center_x)**2 + (yv - center_y)**2)
        
        # Extract sub-regions of friend detection matrices
        region_intensity = self.friend_intensity[x_min:x_max+1, y_min:y_max+1]
        region_timestamp = self.friend_timestamp[x_min:x_max+1, y_min:y_max+1]
        
        # Create mask for empty cells within detection circle
        mask_empty = (distances <= detection_range_cells) & (region_intensity < 1)
        
        # Reset intensities and timestamps for empty cells
        np.putmask(region_intensity, mask_empty, 0)
        np.putmask(region_timestamp, mask_empty, current_time)
        
        # Apply broken detection behavior if drone is broken
        if self.broken:
            self.update_broken(x_min, x_max, y_min, y_max, distances, detection_range_cells)
            
    # -------------------------------------------------------------------------
    # Merge Enemy Matrix
    # -------------------------------------------------------------------------
    def merge_enemy_matrix(self, neighbor) -> None:
        """
        Merge enemy detection data from a neighbor drone into this drone's matrices,
        propagating information globally based on timestamps.
        
        Args:
            neighbor: The neighbor drone to merge data from.
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
    def merge_friend_matrix(self, neighbor: "FriendDrone") -> None:
        """
        Merge friend detection data from a neighbor drone into this drone's matrices.
        
        Args:
            neighbor: The neighbor drone to merge data from.
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
        Comunica-se apenas com os 3 drones amigos mais próximos (dentro de COMMUNICATION_RANGE),
        e ainda inclui aqueles que te têm entre os 3 mais próximos deles, garantindo bilateralidade.
        """
        # 1) candidatos dentro do alcance físico
        candidates = [
            other for other in all_drones
            if other is not self
            and self.pos.distance_to(other.pos) < COMMUNICATION_RANGE
        ]
        # 2) ordena por distância e pega os 3 mais próximos
        candidates.sort(key=lambda o: self.pos.distance_to(o.pos))
        nearest = candidates[:N_CONNECTIONS]

        # 3) identifica quem te considera entre os 3 mais próximos deles
        reverse_neighbors = []
        for other in all_drones:
            if other is self:
                continue
            # candidatos de 'other' dentro do alcance
            others_cand = [
                o for o in all_drones
                if o is not other
                and other.pos.distance_to(o.pos) < COMMUNICATION_RANGE
            ]
            others_cand.sort(key=lambda o: other.pos.distance_to(o.pos))
            if self in others_cand[:N_CONNECTIONS]:
                reverse_neighbors.append(other)

        # 4) união dos dois conjuntos
        neighbors = set(nearest + reverse_neighbors)
        self.neighbors = neighbors

        # 5) faz a fusão de matrizes só com esses vizinhos
        connections = 0
        messages = 0
        for other in neighbors:
            connections += 1
            messages += 2
            if FriendDrone.class_rng.random() > MESSAGE_LOSS_PROBABILITY:
                self.merge_enemy_matrix(other)
                self.merge_friend_matrix(other)

        # 6) atualiza métricas
        self.active_connections = connections
        self.messages_sent_this_cycle = messages

    # def communication(self, all_drones: List[Any]) -> None:
    #     """
    #     Simulate distributed communication by merging detection matrices
    #     from nearby friend drones.
        
    #     Args:
    #         all_drones: List of all friend drones.
    #     """
    #     messages_sent_this_cycle = 0
    #     connections_this_cycle = 0
        
    #     for other in all_drones:
    #         if other is not self and self.pos.distance_to(other.pos) < COMMUNICATION_RANGE:
    #             connections_this_cycle += 1
    #             messages_sent_this_cycle += 2
                
    #             # Simulate message loss with probability
    #             if random.random() > MESSAGE_LOSS_PROBABILITY:
    #                 self.merge_enemy_matrix(other)
    #                 self.merge_friend_matrix(other)
                    
    #     # Track communication metrics
    #     self.active_connections = connections_this_cycle
    #     self.messages_sent_this_cycle = messages_sent_this_cycle
                    
    # -------------------------------------------------------------------------
    # Apply Behavior Broken
    # -------------------------------------------------------------------------   
    def update_broken(self, x_min: int, x_max: int, y_min: int, y_max: int,
                     distances: np.ndarray, detection_range_cells: int) -> None:
        """
        Update detection matrices with random values for broken drones.
        
        For broken drones, this function updates detection matrices (intensity and direction)
        only for cells within detection range, using random values. It maintains these values
        for a set period before generating new random values.
        
        Args:
            x_min, x_max, y_min, y_max: Region limits (in cells) covering detection radius.
            distances: Matrix with distances (in cell units) from each cell to center.
            detection_range_cells: Detection radius in cell units.
        """
        # Determine shape of region to update
        region_shape = (x_max - x_min + 1, y_max - y_min + 1)
        
        # Generate broken states if not already generated
        if self.broken_enemy_direction is None:
            self.broken_enemy_intensity, self.broken_enemy_direction = generate_sparse_matrix(region_shape, max_nonzero=10, seed=self.seed)
            self.broken_friend_intensity, self.broken_friend_direction = generate_sparse_matrix(region_shape, max_nonzero=10, seed=self.seed)
        
        # Extract submatrices for region of interest
        region_enemy_intensity = self.enemy_intensity[x_min:x_max+1, y_min:y_max+1]
        region_enemy_direction = self.enemy_direction[x_min:x_max+1, y_min:y_max+1]
        region_friend_intensity = self.friend_intensity[x_min:x_max+1, y_min:y_max+1]
        region_friend_direction = self.friend_direction[x_min:x_max+1, y_min:y_max+1]
        
        # Create mask for cells within detection radius
        mask = distances <= detection_range_cells

        if self.timer_state_broken < self.update_state_broken:
            # Update cells with random values using mask
            np.putmask(region_enemy_intensity, mask, self.broken_enemy_intensity[mask])
            np.putmask(region_enemy_direction,
                      np.broadcast_to(mask[..., None], region_enemy_direction.shape),
                      self.broken_enemy_direction)
            
            np.putmask(region_friend_intensity, mask, self.broken_friend_intensity[mask])
            np.putmask(region_friend_direction,
                      np.broadcast_to(mask[..., None], region_friend_direction.shape),
                      self.broken_friend_direction)
            
            self.timer_state_broken += 1
            return
        else:
            # Generate new random states when timer expires
            self.broken_enemy_intensity, self.broken_enemy_direction = generate_sparse_matrix(region_shape, max_nonzero=10, seed=self.seed)
            self.broken_friend_intensity, self.broken_friend_direction = generate_sparse_matrix(region_shape, max_nonzero=10, seed=self.seed)
            
            # Update timestamps for region
            current_time: int = pygame.time.get_ticks()
            self.enemy_timestamp[x_min:x_max+1, y_min:y_max+1].fill(current_time)
            self.friend_timestamp[x_min:x_max+1, y_min:y_max+1].fill(current_time)
            
            self.timer_state_broken = 0
            
    # -------------------------------------------------------------------------
    # Action Execution
    # -------------------------------------------------------------------------
    def take_action(self) -> None:
        """
        Execute the drone's action based on enemy detection.
        
        Apply appropriate behavior, update position, and ensure drone stays within
        simulation bounds and within the maximum allowed distance from interest point.
        """
        if self.return_to_base:
            # Head back to interest point center
            self.vel = (self.interest_point_center - self.pos).normalize() * FRIEND_SPEED
        else:
            # Apply behavior-specific movement
            self.apply_behavior()
            
        # Update position
        self.pos += self.vel

        # Keep drone within simulation bounds
        self._enforce_simulation_bounds()
            
        # Prevent drone from exceeding EXTERNAL_RADIUS from interest point
        self._enforce_external_radius()

    def _enforce_simulation_bounds(self) -> None:
        """
        Ensure the drone stays within the simulation boundaries.
        If a boundary is hit, reverse the corresponding velocity component.
        """
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
            
    def _enforce_external_radius(self) -> None:
        """
        Ensure the drone stays within the maximum allowed distance from interest point.
        If the limit is exceeded, move the drone back to the boundary and stop it.
        """
        if self.pos.distance_to(self.interest_point_center) > EXTERNAL_RADIUS:
            direction = (self.pos - self.interest_point_center).normalize()
            self.pos = self.interest_point_center + direction * EXTERNAL_RADIUS
            self.vel = pygame.math.Vector2(0, 0)

    # -------------------------------------------------------------------------
    # Update Drone State
    # -------------------------------------------------------------------------
    def update(self, enemy_drones: List[Any], friend_drones: List[Any], return_to_base: bool = False) -> None:
        """
        Update the drone's state for the current simulation step.
        
        This includes decaying matrices, updating local detections, communicating with
        nearby drones, and executing actions.
        
        Args:
            enemy_drones: List of enemy drones.
            friend_drones: List of friendly drones.
            return_to_base: If True, drone will return to the interest point center.
        """
        self.total_steps += 1
        self.return_to_base = return_to_base
        
        # Apply exponential decay to detection matrices
        self.decay_matrices()
        
        # No modo triangulação, sempre executar a detecção passiva
        self.update_passive_detection_and_triangulate(enemy_drones, friend_drones)
        
        # A função update_local_enemy_detection verifica internamente o modo de detecção
        self.update_local_enemy_detection(enemy_drones)
        self.update_local_friend_detection(friend_drones)
        
        # Communicate with nearby drones
        self.communication(friend_drones)
        
        # Execute movement action
        self.take_action()
        
        # Calculate distance traveled
        self.distance_traveled += self.pos.distance_to(self.last_position)
        self.last_position = self.pos.copy()
        
        # Record trajectory for visualization
        self.trajectory.append(self.pos.copy())
        
        # Update state history
        if self.current_state:
            if self.current_state in self.state_history:
                self.state_history[self.current_state] += 1
            else:
                self.state_history[self.current_state] = 1
                
    def get_state_percentages(self) -> dict:
        """
        Calculate the percentage of steps spent in each state.
        
        Returns:
            dict: Dictionary mapping state names to percentage of time spent.
        """
        if self.total_steps == 0:
            return {}
            
        # Calculate percentages
        percentages = {}
        for state, count in self.state_history.items():
            percentages[state] = (count / self.total_steps) * 100
                
        return percentages
    
    # -------------------------------------------------------------------------
    # Artificial Intelligence Policy (Class Method)
    # -------------------------------------------------------------------------
    @classmethod
    def ai_policy(cls, state, activation_threshold_position: float = 0.2):
        """
        AI policy for determining drone movement based on perceived state.
        
        Process detection matrices and calculate which enemy target to pursue
        based on proximity and other factors.
        
        Args:
            state: Dictionary containing pos, friend_intensity, enemy_intensity,
                  friend_direction, and enemy_direction.
            activation_threshold_position: Minimum intensity to consider a cell active.
        
        Returns:
            tuple: (info, direction) where info is a string description and
                  direction is a normalized movement vector.
        """
        
        def hold_position(pos, friend_intensity) -> Tuple[str, pygame.math.Vector2]:
            """
            Fallback behavior when no enemy targets are detected.
            Uses the AI model to predict movement direction.
            
            Returns:
                tuple: (info string, direction vector)
            """
            if cls.model is None:
                cls.model = load_best_model(directory='./models', pattern=r"val_loss=([\d.]+)\.keras")
                
            direction = np.squeeze(cls.model.predict(state))
            direction = pygame.math.Vector2(direction[0], direction[1]).normalize()
            info = "Return prediction from AI model."
            
            return info, direction
            
        # Extract state components
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        friend_intensity = np.squeeze(state['friend_intensity'])
        enemy_intensity = np.squeeze(state['enemy_intensity'])
        friend_direction = np.squeeze(state['friend_direction'])
        enemy_direction = np.squeeze(state['enemy_direction'])
        
        enemy_targets = []

        # Identify enemy targets with sufficient intensity
        for cell, intensity in np.ndenumerate(enemy_intensity):
            if intensity < activation_threshold_position:
                continue
            target_pos = pygame.math.Vector2((cell[0] + 0.5) * CELL_SIZE, (cell[1] + 0.5) * CELL_SIZE)
            distance_to_interest = target_pos.distance_to(INTEREST_POINT_CENTER)
            enemy_targets.append((cell, target_pos, distance_to_interest))
        
        if not enemy_targets:
            return hold_position(pos, friend_intensity)
        
        # Sort targets by distance to interest point
        enemy_targets.sort(key=lambda t: t[2])
        
        my_cell = pos_to_cell(pos)
        my_cell_center = pygame.math.Vector2((my_cell[0] + 0.5) * CELL_SIZE, (my_cell[1] + 0.5) * CELL_SIZE)
        
        # Check if this drone is closest to any enemy target
        for cell, target_pos, _ in enemy_targets:
            my_distance = my_cell_center.distance_to(target_pos)
            closest_distance = my_distance

            # Compare with friend detections
            for cell, intensity in np.ndenumerate(friend_intensity):
                if intensity < activation_threshold_position:
                    continue
                friend_pos = pygame.math.Vector2((cell[0] + 0.5) * CELL_SIZE, (cell[1] + 0.5) * CELL_SIZE)
                friend_distance = friend_pos.distance_to(target_pos)
                if friend_distance < closest_distance:
                    closest_distance = friend_distance

            # If this drone is closest (or tied), pursue the target
            if my_distance <= closest_distance:
                direction = intercept_direction(pos, FRIEND_SPEED, target_pos, enemy_direction[cell])
                if direction.length() > 0:
                    info = "Pursuing enemy target."
                    return info, direction.normalize()
                else:
                    info = "Enemy target is stationary."
                    return info, pygame.math.Vector2(0, 0)
        
        # If no enemy target is pursued, hold position
        return hold_position(pos, friend_intensity)
        
    # -------------------------------------------------------------------------
    # Apply Behavior
    # -------------------------------------------------------------------------
    def apply_behavior(self) -> None:
        """
        Update drone velocity based on its behavior type.
        
        Different behaviors include:
        - planning: Use the planning policy for decision-making
        - AI: Use the AI model for decision-making
        - AEW: Orbit around the interest point to provide surveillance
        - RADAR: Stationary radar
        - debug: Move directly toward interest point
        - u-debug: Move in a U-shaped pattern for testing
        """        
        if self.behavior_type == "planning":
            # Use planning policy
            self._apply_planning_behavior()
            
        elif self.behavior_type == "AI":
            # Use AI model
            self._apply_ai_behavior()
            
        elif self.behavior_type == "AEW":
            # Orbit around interest point
            self._apply_aew_behavior()
            
        elif self.behavior_type == "RADAR":
            # Stay stationary
            self._apply_radar_behavior()

        elif self.behavior_type == "debug":
            # Move toward interest point
            self._apply_debug_behavior()
                
        elif self.behavior_type == "u-debug":
            # Move in U-shaped pattern
            self._apply_u_debug_behavior()
                
    def _apply_planning_behavior(self) -> None:
        """
        Apply planning policy behavior.
        """
        # Generate state from detection matrices
        state = {
            'pos': np.array([[self.pos.x, self.pos.y]], dtype=np.float32),
            'friend_intensity': np.expand_dims(self.friend_intensity, axis=0),
            'enemy_intensity': np.expand_dims(self.enemy_intensity, axis=0),
            'friend_direction': np.expand_dims(self.friend_direction, axis=0),
            'enemy_direction': np.expand_dims(self.enemy_direction, axis=0)
        }
        
        self.info, direction = planning_policy(state)
        self.vel = direction * FRIEND_SPEED if direction.length() > 0 else pygame.math.Vector2(0, 0)
        self.current_state = self.info[0] if isinstance(self.info, tuple) else self.info
        
    def _apply_ai_behavior(self) -> None:
        """
        Apply AI model behavior.
        """
        # Generate state from detection matrices
        state = {
            'pos': np.array([[self.pos.x, self.pos.y]], dtype=np.float32),
            'friend_intensity': np.expand_dims(self.friend_intensity, axis=0),
            'enemy_intensity': np.expand_dims(self.enemy_intensity, axis=0),
            'friend_direction': np.expand_dims(self.friend_direction, axis=0),
            'enemy_direction': np.expand_dims(self.enemy_direction, axis=0)
        }
        
        self.info, direction = self.ai_policy(state)
        self.vel = direction * FRIEND_SPEED if direction.length() > 0 else pygame.math.Vector2(0, 0)
        self.current_state = self.info[0] if isinstance(self.info, tuple) else self.info
        
    def _apply_aew_behavior(self) -> None:
        """
        Apply AEW (Airborne Early Warning) behavior - orbit around interest point.
        """
        self.info = ("AEW", None, None, None)
        
        # Initialize orbit radius if not set
        if self.orbit_radius is None:
            self.orbit_radius = AEW_RANGE

        # Compute radial vector from interest point
        r_vec = self.pos - self.interest_point_center
        current_distance = r_vec.length()
        
        if current_distance == 0:
            r_vec = pygame.math.Vector2(self.orbit_radius, 0)
            current_distance = self.orbit_radius
            
        # Calculate orbit correction
        radial_error = self.orbit_radius - current_distance
        k_radial = 0.05  # Radial correction factor
        radial_correction = k_radial * radial_error * r_vec.normalize()
        
        # Calculate tangential velocity (perpendicular to radial)
        tangent = pygame.math.Vector2(-r_vec.y, r_vec.x).normalize()
        tangential_velocity = tangent * AEW_SPEED
        
        # Combine tangential and radial components
        desired_velocity = tangential_velocity + radial_correction
        self.vel = desired_velocity.normalize() * AEW_SPEED if desired_velocity.length() > 0 else pygame.math.Vector2(0, 0)
        self.current_state = self.info[0] if isinstance(self.info, tuple) else self.info
        
    def _apply_radar_behavior(self) -> None:
        """
        Apply RADAR behavior - remain stationary.
        """
        self.info = ("RADAR", None, None, None)
        self.vel = pygame.math.Vector2(0, 0)
        self.current_state = self.info[0] if isinstance(self.info, tuple) else self.info

    def _apply_debug_behavior(self) -> None:
        """
        Apply debug behavior - move toward interest point.
        """
        self.info = ("DEBUG", None, None, None)
        self.pos = pygame.math.Vector2(0.25 * SIM_WIDTH, 0.5 * SIM_HEIGHT)
        self.behavior_type = "planning"
        self.current_state = self.info[0] if isinstance(self.info, tuple) else self.info
            
    def _apply_u_debug_behavior(self) -> None:
        """
        Apply U-debug behavior - move in a U-shaped pattern for testing.
        """
        self.info = ("U-DEBUG", None, None, None)
        
        # Initialize phase tracking if needed
        if not hasattr(self, 'u_debug_phase'):
            self.u_debug_phase = 0  # 0: forward, 1: perpendicular, 2: backward
            self.u_debug_timer = 0
            self.forward_steps = 300
            self.perp_steps = 40
            
        target_vector = self.interest_point_center - self.pos
        target_direction = target_vector.normalize()
        
        if not self.fixed:
            if self.u_debug_phase == 0:
                # Move forward (toward interest point)
                self.vel = target_direction * FRIEND_SPEED
                self.u_debug_timer += 1
                if self.u_debug_timer >= self.forward_steps:
                    self.u_debug_phase = 1
                    self.u_debug_timer = 0
            elif self.u_debug_phase == 1:
                # Move perpendicular
                perp_direction = pygame.math.Vector2(-target_direction.y, target_direction.x)
                self.vel = perp_direction * FRIEND_SPEED
                self.u_debug_timer += 1
                if self.u_debug_timer >= self.perp_steps:
                    self.u_debug_phase = 2
                    self.u_debug_timer = 0
            elif self.u_debug_phase == 2:
                # Move backward (away from interest point)
                self.vel = (-target_direction) * FRIEND_SPEED
                self.u_debug_timer += 1
                if self.u_debug_timer >= self.forward_steps:
                    self.vel = pygame.math.Vector2(0, 0)
        else:
            self.vel = pygame.math.Vector2(0, 0)
            
        self.current_state = self.info[0] if isinstance(self.info, tuple) else self.info
                
    def draw_passive_detection(self, surface: pygame.Surface) -> None:
        """
        Draw lines representing passive detection directions
        and circles representing triangulated positions.
        
        Args:
            surface: Surface to draw on.
        """
        if hasattr(self, 'passive_detections'):
            # Draw direction lines for passive detections
            for direction, _ in self.passive_detections.values():
                # Calculate end point (extending the direction)
                end_point = self.pos + direction * FRIEND_DETECTION_RANGE
                draw_dashed_line(
                    surface, 
                    (255, 0, 0, 128), 
                    self.pos, 
                    end_point,
                    width=1, 
                    dash_length=5, 
                    space_length=5
                )
        
        if hasattr(self, 'last_triangulated'):
            # Draw triangulated positions
            for position, confidence in self.last_triangulated.values():
                # Circle size based on confidence
                radius = int(5 + confidence * 5)
                pygame.draw.circle(
                    surface, 
                    (0, 255, 255, int(confidence * 200)),
                    (int(position.x), int(position.y)), 
                    radius, 
                    1
                )

    # -------------------------------------------------------------------------
    # Rendering: Draw the Drone
    # -------------------------------------------------------------------------
    def draw(self, surface: pygame.Surface, show_detection: bool = True, 
             show_comm_range: bool = True, show_trajectory: bool = False, 
             show_debug: bool = False) -> None:
        """
        Draw the drone and its information on the provided surface.
        
        Args:
            surface: Surface to draw on.
            show_detection: If True, display detection range.
            show_comm_range: If True, display communication range.
            show_trajectory: If True, draw drone's trajectory.
            show_debug: If True, show additional debug information.
        """
        # Draw trajectory if enabled
        if show_trajectory and len(self.trajectory) > 1:
            self._draw_trajectory(surface)
        
        # Draw detection range
        if show_detection or self.broken:
            detection_range = self._get_detection_range()
            draw_dashed_circle(
                surface, 
                (self.color[0], self.color[1], self.color[2], 64), 
                (int(self.pos.x), int(self.pos.y)), 
                detection_range, 
                5, 5, 1
            )
            
        # Draw communication range
        if show_comm_range:
            draw_dashed_circle(
                surface, 
                (255, 255, 0, 32), 
                (int(self.pos.x), int(self.pos.y)), 
                COMMUNICATION_RANGE, 
                5, 5, 1
            )
        
        # Draw drone image
        image_rect = self.image.get_rect(center=(int(self.pos.x), int(self.pos.y)))
        surface.blit(self.image, image_rect)
        
        # Draw drone ID
        self._draw_drone_id(surface)
        
        # Draw debug information if enabled
        if show_debug:
            self._draw_debug_info(surface)
            
    def _draw_trajectory(self, surface: pygame.Surface) -> None:
        """
        Draw the drone's movement trajectory with fading effect.
        
        Args:
            surface: Surface to draw on.
        """
        traj_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        decay_rate = 0.04
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
            
    def _draw_drone_id(self, surface: pygame.Surface) -> None:
        """
        Draw the drone's ID and status labels.
        
        Args:
            surface: Surface to draw on.
        """
        font: pygame.font.Font = pygame.font.SysFont(FONT_FAMILY, 10)
        
        # Draw drone ID
        label = font.render(f"ID: F{self.drone_id}", True, (255, 255, 255))
        label.set_alpha(128)
        surface.blit(label, (int(self.pos.x) + 20, int(self.pos.y) - 20))
        
        # Draw leader indicator if applicable
        if self.is_leader:
            leader_label = font.render("LEADER", True, (0, 255, 0))
            surface.blit(leader_label, (int(self.pos.x) + 20, int(self.pos.y) - 5))
        
        # Draw selection indicator if applicable
        if self.selected:
            selected_label = font.render("GRAPH", True, (0, 255, 0))
            surface.blit(selected_label, (int(self.pos.x) + 20, int(self.pos.y) + 10))
            
    def _draw_debug_info(self, surface: pygame.Surface) -> None:
        """
        Draw additional debug information.
        
        Args:
            surface: Surface to draw on.
        """
        # Draw passive detection information if in triangulation mode
        if self.detection_mode == "triangulation":
            self.draw_passive_detection(surface)
        
        # Draw state information
        font = pygame.font.SysFont(FONT_FAMILY, 10)
        if self.info and self.info[0]:
            len_info = len(self.info[0])
            debug_label = font.render(self.info[0], True, (255, 215, 0))
            surface.blit(debug_label, (int(self.pos.x) - 3.5 * len_info, int(self.pos.y) + 25))
            
            # Draw target information if available
            if self.info[1] is not None:
                pygame.draw.circle(
                    surface, 
                    (255, 215, 0), 
                    (int(self.info[1].x), int(self.info[1].y)), 
                    4
                )
                pygame.draw.line(
                    surface,
                    (255, 215, 0),
                    (int(self.pos.x), int(self.pos.y)),
                    (int(self.info[1].x), int(self.info[1].y)),
                    2
                )
                
            # Draw interest point line if available
            if self.info[2] is not None:
                pygame.draw.line(
                    surface,
                    (255, 215, 0),
                    (int(self.interest_point_center[0]), int(self.interest_point_center[1])),
                    (int(self.info[2].x), int(self.info[2].y)),
                    2
                )