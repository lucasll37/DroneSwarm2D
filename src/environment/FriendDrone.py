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
from typing import Optional, Tuple, List, Any

from settings import *
from utils import draw_dashed_circle, load_svg_as_surface, pos_to_cell, intercept_direction, load_best_model, generate_sparse_matrix

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
    original_broken_drone_image: pygame.Surface = load_svg_as_surface("./assets/drone_broken.svg")
    original_aew_image: pygame.Surface = load_svg_as_surface("./assets/radar_0.svg")
    original_radar_image: pygame.Surface = load_svg_as_surface("./assets/radar_0.svg")
    
    model = None

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(self, interest_point_center, position: Tuple, behavior_type: str = "planning", fixed: bool = False, broken: bool = False) -> None:
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
        self.return_to_base: bool = False
        self.info: Tuple[str, Any] = ("", None, None)

        # Drone properties
        self.color: Tuple[int, int, int] = (255, 255, 255)
        self.drone_id: int = self.assign_id()
        self.in_election: bool = False
        self.is_leader: bool = False
        self.leader_id: int = self.drone_id
        self.broken: bool = broken
        
        self.timer_state_broken = 0
        self.update_state_broken = UPDATE_STATE_BROKEN
        self.broken_friend_intensity = None
        self.broken_friend_direction = None
        self.broken_enemy_intensity = None
        self.broken_enemy_direction = None

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
        
        # Se o drone está quebrado, aplica o comportamento de detecção errada
        if self.broken:
            self.update_broken(x_min, x_max, y_min, y_max, distances, detection_range_cells)

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
        
        # Se o drone está quebrado, aplica o comportamento de detecção errada
        if self.broken:
            self.update_broken(x_min, x_max, y_min, y_max, distances, detection_range_cells)
            
    # -------------------------------------------------------------------------
    # Merge Enemy Matrix
    # -------------------------------------------------------------------------
    def merge_enemy_matrix(self, neighbor) -> None:
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
    


    # def merge_enemy_matrix(self, neighbor) -> None:
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     # Parâmetros de tolerância para a comparação célula a célula
    #     POSITION_INTENSITY_THRESHOLD = 0.2  # Diferença máxima permitida na intensidade para cada célula
    #     POSITION_DIRECTION_THRESHOLD = 0.3  # Diferença máxima (norma) permitida entre os vetores de direção para cada célula
    #     TOLERANCE_RATIO = 0.9  # Fração mínima de células compatíveis na área de interseção para realizar o merge

    #     """
    #     Realiza o merge dos dados de detecção de inimigos do drone vizinho para este drone,
    #     mas somente se, na área de interseção dos círculos de detecção, uma fração (definida por TOLERANCE_RATIO)
    #     das células forem compatíveis, isto é, se para cada célula a diferença de intensidade for menor que
    #     POSITION_INTENSITY_THRESHOLD e a diferença entre os vetores de direção for menor que POSITION_DIRECTION_THRESHOLD.
    #     Se a condição for satisfeita, as matrizes são mergeadas "em bloco" conforme o funcionamento antigo,
    #     atualizando as células onde o timestamp do vizinho é mais recente.
        
    #     Args:
    #         neighbor: Drone vizinho que possui matrizes enemy_intensity, enemy_direction, enemy_timestamp,
    #                 e um atributo pos (posição).
    #     """

    #     # Define o raio de detecção; assume que ambos os drones utilizam o mesmo valor
    #     detection_range = FRIEND_DETECTION_RANGE  # ajuste conforme necessário

    #     # Cria uma grade com os centros das células
    #     grid_x = np.linspace(CELL_SIZE / 2, SIM_WIDTH - CELL_SIZE / 2, GRID_WIDTH)
    #     grid_y = np.linspace(CELL_SIZE / 2, SIM_HEIGHT - CELL_SIZE / 2, GRID_HEIGHT)
    #     X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')

    #     # Obtém as posições dos drones
    #     self_x, self_y = self.pos.x, self.pos.y
    #     neighbor_x, neighbor_y = neighbor.pos.x, neighbor.pos.y

    #     # Calcula a distância dos centros das células à posição de cada drone
    #     dist_self = np.sqrt((X - self_x) ** 2 + (Y - self_y) ** 2)
    #     dist_neighbor = np.sqrt((X - neighbor_x) ** 2 + (Y - neighbor_y) ** 2)

    #     # Máscara para as células que estão dentro dos círculos de detecção de ambos os drones
    #     intersection_mask = (dist_self <= detection_range) & (dist_neighbor <= detection_range)
        
        # ----- Plot de Debug: Intersection Mask e Áreas de Detecção -----
        # plt.figure(figsize=(8,6))
        # ax = plt.gca()
        
        # # Plot da máscara de interseção
        # plt.imshow(intersection_mask.T, origin='lower',
        #         extent=[CELL_SIZE/2, SIM_WIDTH - CELL_SIZE/2, CELL_SIZE/2, SIM_HEIGHT - CELL_SIZE/2],
        #         cmap='gray', alpha=0.6)
        # plt.colorbar(label='Intersection (1=True, 0=False)')
        
        # # Desenha os círculos de detecção para cada drone
        # circle_self = plt.Circle((self_x, self_y), detection_range, color='red', fill=False, linewidth=2, label='Self Detection Area')
        # circle_neighbor = plt.Circle((neighbor_x, neighbor_y), detection_range, color='blue', fill=False, linewidth=2, label='Neighbor Detection Area')
        # ax.add_patch(circle_self)
        # ax.add_patch(circle_neighbor)
        
        # # Plota as posições dos drones
        # plt.scatter(self_x, self_y, c='red', label='Self Drone')
        # plt.scatter(neighbor_x, neighbor_y, c='blue', label='Neighbor Drone')
        
        # plt.title("Intersection Mask and Detection Areas")
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.legend()
        # plt.show()
        # ---------------------------------------------------------------

        # # Calcula, célula a célula, a diferença absoluta de intensidade e a norma da diferença dos vetores de direção
        # intensity_diff = np.abs(self.enemy_intensity - neighbor.enemy_intensity)
        # direction_diff = np.linalg.norm(self.enemy_direction - neighbor.enemy_direction, axis=2)

        # # Máscara que identifica células compatíveis na comparação (celular a celular)
        # compatible_mask = (intensity_diff <= POSITION_INTENSITY_THRESHOLD) & \
        #                 (direction_diff <= POSITION_DIRECTION_THRESHOLD)

        # # Conta o número de células na área de interseção e quantas delas são compatíveis
        # total_cells = np.sum(intersection_mask)
        # compatible_cells = np.sum(compatible_mask & intersection_mask)

        # # Se a fração de células compatíveis for maior ou igual a TOLERANCE_RATIO, realiza o merge
        # if total_cells > 0 and (compatible_cells / total_cells) >= TOLERANCE_RATIO:
        #     update_mask = (neighbor.enemy_timestamp > self.enemy_timestamp) & intersection_mask
        #     np.putmask(self.enemy_intensity, update_mask, neighbor.enemy_intensity)
        #     np.putmask(self.enemy_direction,
        #             np.broadcast_to(update_mask[..., None], self.enemy_direction.shape),
        #             neighbor.enemy_direction)
        #     np.putmask(self.enemy_timestamp, update_mask, neighbor.enemy_timestamp)



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
    # Apply Behavior Broken
    # -------------------------------------------------------------------------   
    def update_broken(self, x_min: int, x_max: int, y_min: int, y_max: int,
                    distances: np.ndarray, detection_range_cells: int) -> None:
        """
        Para drones quebrados, atualiza os valores das matrizes de detecção (intensidade e direção)
        somente para as células que estão dentro do raio de detecção, usando valores aleatórios.
        
        Essa função mantém os valores gerados por um período determinado (update_state_broken). 
        Quando esse tempo expira, novas matrizes aleatórias são geradas e os timestamps são atualizados.
        
        Args:
            x_min, x_max, y_min, y_max: Limites da região (em células) que abrange o raio de detecção.
            distances (np.ndarray): Matriz com as distâncias (em unidades de célula) de cada célula até o centro.
            detection_range_cells (int): Raio de detecção em unidades de célula.
        """
        # Determine a forma da região a ser atualizada
        region_shape = (x_max - x_min + 1, y_max - y_min + 1)
        
        # Se os estados "quebrados" ainda não foram gerados para a região, gere-os
        if self.broken_enemy_direction is None:
            self.broken_enemy_intensity, self.broken_enemy_direction = generate_sparse_matrix(region_shape, max_nonzero=10)
            self.broken_friend_intensity, self.broken_friend_direction = generate_sparse_matrix(region_shape, max_nonzero=10)
        
        # Extraia as submatrizes correspondentes à região de interesse
        region_enemy_intensity = self.enemy_intensity[x_min:x_max+1, y_min:y_max+1]
        region_enemy_direction = self.enemy_direction[x_min:x_max+1, y_min:y_max+1]
        region_friend_intensity = self.friend_intensity[x_min:x_max+1, y_min:y_max+1]
        region_friend_direction = self.friend_direction[x_min:x_max+1, y_min:y_max+1]
        
        # Cria uma máscara para as células dentro do raio de detecção
        mask = distances <= detection_range_cells

        if self.timer_state_broken < self.update_state_broken:
            # Atualiza as células com valores aleatórios usando np.putmask com np.broadcast_to para as matrizes de direção
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
            # Quando o timer expira, gere novos estados aleatórios para a região
            self.broken_enemy_intensity, self.broken_enemy_direction = generate_sparse_matrix(region_shape, max_nonzero=10)
            self.broken_friend_intensity, self.broken_friend_direction = generate_sparse_matrix(region_shape, max_nonzero=10)
            
            # Atualiza os timestamps para a região
            current_time: int = pygame.time.get_ticks()
            self.enemy_timestamp[x_min:x_max+1, y_min:y_max+1].fill(current_time)
            self.friend_timestamp[x_min:x_max+1, y_min:y_max+1].fill(current_time)
            
            self.timer_state_broken = 0

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
        if self.return_to_base:
            self.vel = (self.interest_point_center - self.pos).normalize() * FRIEND_SPEED
        
        else:
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
    def update(self, enemy_drones: List[Any], friend_drones: List[Any], return_to_base: bool = False) -> None:
        """
        Updates the drone's state by applying decay, updating local detections,
        communicating with nearby drones, and executing actions.
        
        Args:
            enemy_drones (List[Any]): List of enemy drones.
            friend_drones (List[Any]): List of friend drones.
        """
        self.return_to_base = return_to_base
        
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
    def planning_policy(cls, state, friend_activation_threshold_position: float = 0.7, enemy_activation_threshold_position: float = 0.4):
        """
        Atualiza o estado do drone e gera a velocidade (ação) a ser aplicada com base nas
        matrizes de detecção de inimigos e amigos. Nesta política, cada alvo inimigo é
        atribuído a um "amigo" (representado pela posição na matriz de detecção) e, se o
        drone self (identificado pelo índice da sua posição na matriz) for designado para um
        alvo, ele se engaja na perseguição.
        
        Args:
            state (dict): Contém as chaves 'pos', 'friend_intensity', 'enemy_intensity',
                        'friend_direction', 'enemy_direction'. A posição (pos) é um array [x, y].
            friend_activation_threshold_position (float): Limiar de intensidade para considerar uma célula ativa.
            enemy_activation_threshold_position (float): Limiar de intensidade para considerar uma célula ativa.
            
        Returns:
            Tuple[info, velocity]: info é uma string descritiva e velocity é um pygame.math.Vector2,
                                escalado por FRIEND_SPEED.
        """

        def hold_position(pos, friend_intensity, enemy_intensity, enemy_direction,
                        activation_threshold_position: float = 1, enemy_threshold: float = 0.4) -> tuple:
            """
            Determina a ação de hold position considerando as seguintes prioridades:
            1. Se o drone estiver muito próximo do ponto de interesse (dentro de EPSILON), ele permanece parado.
            2. Se houver pelo menos dois amigos ativos (células com friend_intensity >= activation_threshold_position)
                dentro do COMMUNICATION_RANGE, o drone permanece em hold.
            3. Se o drone estiver em hold e, nesse estado, for detectado um inimigo se aproximando do ponto de interesse
                (células com enemy_intensity >= enemy_threshold e cujo vetor de direção indica movimento para o interesse),
                então:
                a) Calcula-se a projeção da posição do drone sobre a reta de trajetória esperada do inimigo.
                b) Somente se o drone estiver próximo dessa projeção (distância menor que THRESHOLD_PROJECTION),
                    ele calculará um ponto defensivo, definido pela interseção entre a reta perpendicular
                    que passa por sua posição e a reta de trajetória do inimigo.
                c) Dentre os candidatos (calculados a partir dos centros das células com friend_intensity ativa),
                    somente o drone cuja distância até o ponto defensivo for a menor é recrutado para se deslocar.
            4. Se nenhuma situação prioritária ocorrer, o drone retorna a direção para o ponto de interesse.
            
            Args:
                pos (pygame.math.Vector2): Posição atual do drone.
                friend_intensity (np.ndarray): Matriz de detecção dos amigos.
                enemy_intensity (np.ndarray): Matriz de detecção dos inimigos.
                enemy_direction (np.ndarray): Matriz de direção dos inimigos.
                activation_threshold_position (float): Limiar para considerar uma célula de friend_intensity ativa.
                enemy_threshold (float): Limiar para considerar uma célula de enemy_intensity ativa.
                        
            Returns:
                Tuple[Tuple[str, Optional[pygame.math.Vector2]], pygame.math.Vector2]:
                    Um par onde o primeiro elemento é uma tupla contendo a mensagem informativa e o defensive_point,
                    e o segundo é o vetor de ação (normalizado).
            """
            
            # Caso 1: Se o drone estiver muito próximo do ponto de interesse, permanece parado.
            if pos.distance_to(INTEREST_POINT_CENTER) < EPSILON:
                info = ("HOLD", None, None)
                return info, pygame.math.Vector2(0, 0)
            
            # Constrói a grade dos centros das células com base em friend_intensity.
            # grid_x = np.linspace(CELL_SIZE/2, SIM_WIDTH - CELL_SIZE/2, GRID_WIDTH)
            # grid_y = np.linspace(CELL_SIZE/2, SIM_HEIGHT - CELL_SIZE/2, GRID_HEIGHT)
            # X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')
            # distance_matrix = np.sqrt((X - pos.x)**2 + (Y - pos.y)**2)
            # comm_mask = distance_matrix < COMMUNICATION_RANGE
            # active_friend_count = np.sum((friend_intensity >= activation_threshold_position) & comm_mask)
            
            # # Caso 2: Se conectado a pelo menos dois amigos, verifica oportunidade defensiva.
            # if active_friend_count >= 2:
            #     candidate_intercepts = []
            #     for cell, intensity in np.ndenumerate(enemy_intensity):
            #         if intensity < enemy_threshold:
            #             continue
            #         # Centro da célula correspondente
            #         cell_center = pygame.math.Vector2((cell[0] + 0.5) * CELL_SIZE,
            #                                             (cell[1] + 0.5) * CELL_SIZE)
            #         # Vetor que vai do centro da célula até o ponto de interesse
            #         vec_to_interest = INTEREST_POINT_CENTER - cell_center
            #         if vec_to_interest.length() == 0:
            #             continue
            #         vec_to_interest = vec_to_interest.normalize()
            #         # Vetor de direção detectado para o inimigo nesta célula
            #         enemy_dir = pygame.math.Vector2(*enemy_direction[cell])
            #         if enemy_dir.length() == 0:
            #             continue
            #         enemy_dir = enemy_dir.normalize()
            #         if enemy_dir.dot(vec_to_interest) >= 0.8:
            #             candidate_intercepts.append((cell_center.distance_to(INTEREST_POINT_CENTER),
            #                                         cell_center, enemy_dir))
            #     if candidate_intercepts:
            #         candidate_intercepts.sort(key=lambda t: t[0])
            #         chosen_distance, chosen_cell_center, enemy_dir = candidate_intercepts[0]
            #         # Calcula a projeção da posição do drone sobre a reta que passa por chosen_cell_center com direção enemy_dir.
            #         s = (pos - chosen_cell_center).dot(enemy_dir)
            #         projection_point = chosen_cell_center + s * enemy_dir
                    
            #         if pos.distance_to(projection_point) < EPSILON:
            #             info = ("HOLD INTCPT", None, None)
            #             direction = pygame.math.Vector2(0, 0)
            #             return info, direction
                    
            #         if chosen_cell_center.distance_to(projection_point) > chosen_cell_center.distance_to(INTEREST_POINT_CENTER):
            #             defensive_point = INTEREST_POINT_CENTER
            #         else:
            #             defensive_point = projection_point
                    
            #         # Se o drone estiver muito distante dessa projeção, permanece em hold.
            #         if (pos - defensive_point).length() > THRESHOLD_PROJECTION:
            #             info = ("HOLD", None, None)
            #             return info, pygame.math.Vector2(0, 0)
                    
            #         info = ("GO HOLD INTCPT", defensive_point, chosen_cell_center)
            #         # info = ("GO HOLD INTCPT", defensive_point, chosen_cell_center)
            #         direction = (defensive_point - pos).normalize()
            #         return info, direction
                
            #     info = ("HOLD", None, None)
            #     return info, pygame.math.Vector2(0, 0)
            
            # # Caso 3: Se não estiver conectado a pelo menos dois amigos, retorna a direção para o PI.
            # info = ("HOLD - NO ENOUGH COMM", None, None)
            # direction = (INTEREST_POINT_CENTER - pos).normalize()
            # return info, direction
            
            if pos.distance_to(INTEREST_POINT_CENTER) > INITIAL_DISTANCE:
                info = ("GO HOLD", None, None)
                direction = (INTEREST_POINT_CENTER - pos).normalize()
                return info, direction
            else:
                info = ("HOLD", None, None)
                return info, pygame.math.Vector2(0, 0)

        # Extração e preparação do estado
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        friend_intensity = np.squeeze(state['friend_intensity'])
        enemy_intensity = np.squeeze(state['enemy_intensity'])
        friend_direction = np.squeeze(state['friend_direction'])
        enemy_direction = np.squeeze(state['enemy_direction'])
        
        defensive_point = None
        enemy_targets = []
        # Identifica células da matriz de inimigos com intensidade acima do limiar
        for cell, intensity in np.ndenumerate(enemy_intensity):
            if intensity < enemy_activation_threshold_position:
                continue
            target_pos = pygame.math.Vector2((cell[0] + 0.5) * CELL_SIZE,
                                            (cell[1] + 0.5) * CELL_SIZE)
            distance_to_interest = target_pos.distance_to(INTEREST_POINT_CENTER)
            enemy_targets.append((cell, target_pos, distance_to_interest))
        
        if not enemy_targets:
            return hold_position(pos, friend_intensity, enemy_intensity, enemy_direction) 
        
        # Ordena os alvos pelo quão próximos estão do ponto de interesse
        enemy_targets.sort(key=lambda t: t[2])
        
        # Obtém a célula correspondente à posição do drone self
        my_cell = pos_to_cell(pos)
        my_cell_center = pygame.math.Vector2((my_cell[0] + 0.5) * CELL_SIZE,
                                            (my_cell[1] + 0.5) * CELL_SIZE)
        
        # Obtém os candidatos amigos a partir da matriz friend_intensity.
        # Cada candidato é identificado pela célula em que há uma detecção ativa.
        friend_candidates = []
        for cell, intensity in np.ndenumerate(friend_intensity):
            if intensity >= friend_activation_threshold_position:
                candidate_pos = pygame.math.Vector2((cell[0] + 0.5) * CELL_SIZE,
                                                    (cell[1] + 0.5) * CELL_SIZE)
                friend_candidates.append((cell, candidate_pos))
        # Inclui a própria célula self se não houver detecção ativa (para diferenciá-lo)
        if not any(cell == my_cell for cell, pos_candidate in friend_candidates):
            friend_candidates.append((my_cell, my_cell_center))
        
        # Estrutura temporária para designação: mapa de alvo (usando target_pos como tupla) para
        # o candidato amigo (célula) que será engajado.
        engagement_assignment = {}
        assigned_friend_cells = set()
        
        # Para cada alvo inimigo, em ordem, atribui o candidato amigo mais próximo que ainda não foi designado.
        for cell, target_pos, _ in enemy_targets:
            sorted_candidates = sorted(friend_candidates, key=lambda x: x[1].distance_to(target_pos))
            for candidate in sorted_candidates:
                candidate_cell, candidate_pos = candidate
                if candidate_cell not in assigned_friend_cells:
                    engagement_assignment[tuple(target_pos)] = candidate_cell
                    assigned_friend_cells.add(candidate_cell)
                    break  # Avança para o próximo alvo
        
        # Verifica se algum dos alvos designados possui self como candidato
        engaged_enemies = []
        for cell, target_pos, _ in enemy_targets:
            if engagement_assignment.get(tuple(target_pos)) == my_cell:
                distance = my_cell_center.distance_to(target_pos)
                engaged_enemies.append((distance, cell, target_pos))
        
        if engaged_enemies:
            engaged_enemies.sort(key=lambda t: t[0])
            chosen_distance, chosen_cell, chosen_target_pos = engaged_enemies[0]
            direction = intercept_direction(pos, FRIEND_SPEED, chosen_target_pos, enemy_direction[chosen_cell])
            if direction.length() > 0:
                info = ("PURSUING", None, None)
                return info, direction.normalize()
            else:
                info = ("ERROR PURSUING", None, None)
                return info, pygame.math.Vector2(0, 0)
        
        # Se nenhum alvo foi atribuído a self, mantém a posição.
        return hold_position(pos, friend_intensity, enemy_intensity, enemy_direction) 
    

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
            info = "Return prediction from AI model."
            
            return info, direction
            
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
                    info = "Pursuing enemy target."
                    return info, direction.normalize()
                else:
                    info = "Enemy target is stationary."
                    return info, pygame.math.Vector2(0, 0)
        
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
            self.info, direction = self.planning_policy(state)
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
            self.info, direction = self.ai_policy(state)
            self.vel = direction * FRIEND_SPEED if direction.length() > 0 else pygame.math.Vector2(0, 0)
            
        elif self.behavior_type == "AEW":
            self.info = ("AEW", None, None)
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
            self.info = ("RADAR", None, None)
            self.vel = pygame.math.Vector2(0, 0)

        elif self.behavior_type == "debug":
            self.info = ("DEBUG", None, None)
            target_vector = self.interest_point_center - self.pos
            direction = target_vector.normalize() if target_vector.length() > 0 else pygame.math.Vector2(0, 0)
            self.vel = direction * FRIEND_SPEED if not self.fixed else pygame.math.Vector2(0, 0)
                
        elif self.behavior_type == "u-debug":
            self.info = ("U-DEBUG", None, None)
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
    def draw(self, surface: pygame.Surface, show_detection: bool = True, show_comm_range: bool = True, show_trajectory: bool = False, show_debug: bool = False) -> None:
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
        
        # Draw detection range.
        if show_detection or self.broken:
            if self.behavior_type == "AEW":
                detection_range = AEW_DETECTION_RANGE
            elif self.behavior_type == "RADAR":
                detection_range = RADAR_DETECTION_RANGE
            else:
                detection_range = FRIEND_DETECTION_RANGE
            
            draw_dashed_circle(surface, (self.color[0], self.color[1], self.color[2], 64), (int(self.pos.x), int(self.pos.y)), detection_range, 5, 5, 1)
            
        # Draw communication range.
        if show_comm_range:
            draw_dashed_circle(surface, (255, 255, 0, 32), (int(self.pos.x), int(self.pos.y)), COMMUNICATION_RANGE, 5, 5, 1)
        
        # Draw the drone image.
        image_rect = self.image.get_rect(center=(int(self.pos.x), int(self.pos.y)))
        surface.blit(self.image, image_rect)
        
        # Render the drone's ID with semi-transparent white text.
        font: pygame.font.Font = pygame.font.SysFont(FONT_FAMILY, 10)
        label = font.render(f"ID: F{self.drone_id}", True, (255, 255, 255))
        label.set_alpha(128)
        surface.blit(label, (int(self.pos.x) + 20, int(self.pos.y) - 20))
        
        # If the drone is a leader, indicate with a "LEADER" label.
        if self.is_leader:
            leader_label = font.render("LEADER", True, (0, 255, 0))
            surface.blit(leader_label, (int(self.pos.x) + 20, int(self.pos.y) - 5))
        
        # If the drone is selected, display a "GRAPH" label.
        if self.selected:
            selected_label = font.render("GRAPH", True, (0, 255, 0))
            surface.blit(selected_label, (int(self.pos.x) + 20, int(self.pos.y) + 10))
            
        if show_debug:
            len_info = len(self.info[0])
            debug_label = font.render(self.info[0], True, (255, 215, 0))
            surface.blit(debug_label, (int(self.pos.x) - 3.5 * len_info, int(self.pos.y) + 25))
            
            if self.info[1] is not None:
                pygame.draw.circle(surface, (0, 0, 255), (int(self.info[1].x), int(self.info[1].y)), 4)
                pygame.draw.line(surface, (0, 0, 255), (int(self.pos.x), int(self.pos.y)), (int(self.info[1].x), int(self.info[1].y)), 2)
                
            if self.info[2] is not None:
                pygame.draw.line(surface, (0, 0, 255), (int(self.interest_point_center[0]), int(self.interest_point_center[1])), (int(self.info[2].x), int(self.info[2].y)), 1)