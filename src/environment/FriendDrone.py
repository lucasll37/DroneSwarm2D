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
import itertools

from settings import *
from utils import draw_dashed_circle, load_svg_as_surface, pos_to_cell, \
    intercept_direction, load_best_model, generate_sparse_matrix, draw_dashed_line
from distributedDefensiveAlgorithm import planning_policy

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
    def __init__(
        self, interest_point_center, position: Tuple, behavior_type: str = "planning",
        fixed: bool = False, broken: bool = False) -> None:
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
        
        # PROTO ###############
        self.enemy_direction_only = np.zeros((GRID_WIDTH, GRID_HEIGHT, 2))  # Vetor unitário de direção
        self.enemy_detection_confidence = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        # PROTO ###############
        
        self.distance_traveled = 0
        self.last_position = pygame.math.Vector2(position[0], position[1])
        self.messages_sent = 0
        
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
        
        
    # PROTO ########################  
    def update_passive_detection_and_triangulate(self, enemy_drones: List[Any], friend_drones: List[Any]) -> None:
        """
        Implementa detecção passiva (apenas direção) e triangulação de forma distribuída,
        sem depender de um elemento central. Cada drone executa este algoritmo independentemente,
        usando apenas as informações disponíveis através de sua rede de comunicação local.
        
        Args:
            enemy_drones (List[Any]): Lista de drones inimigos no ambiente.
            friend_drones (List[Any]): Lista de drones amigos para comunicação.
        """
        # 1. Detectar direções (sem distâncias) dos inimigos dentro do alcance
        current_time = pygame.time.get_ticks()
        self.passive_detections = {}  # {direction_hash: (direction, timestamp)}
        
        if not hasattr(self, 'direction_history'):
            self.direction_history = {}  # {direction_hash: (direction, timestamp, source_drone_id)}
            
        if not hasattr(self, 'triangulated_targets'):
            self.triangulated_targets = {}  # {target_id: (position, confidence, last_update_time)}
        
        # Atualizar o histórico com as próprias detecções
        for enemy in enemy_drones:
            delta = enemy.pos - self.pos
            distance = delta.length()
            
            # Verifica se o inimigo está dentro do alcance de detecção
            if self.behavior_type == "AEW":
                detection_range = AEW_DETECTION_RANGE
            elif self.behavior_type == "RADAR":
                detection_range = RADAR_DETECTION_RANGE
            else:
                detection_range = FRIEND_DETECTION_RANGE
                
            if distance <= detection_range and distance > 0:
                # Armazena apenas a direção normalizada (sem distância)
                direction = delta.normalize()
                
                # Criar um hash para esta direção (discretização do ângulo)
                angle = math.atan2(direction.y, direction.x)
                angle_discrete = round(angle / 0.01) * 0.01  # Discretizar a cada 0.01 radianos
                direction_hash = f"{angle_discrete:.2f}"
                
                # Armazenar no dicionário de detecções passivas
                self.passive_detections[direction_hash] = (direction, current_time)
                
                # Atualizar o histórico de direções
                self.direction_history[direction_hash] = (direction, current_time, id(self))
        
        # 2. Comunicar e receber detecções de drones amigos próximos
        for friend in friend_drones:
            if friend is self or friend.pos.distance_to(self.pos) > COMMUNICATION_RANGE:
                continue
                
            # Compartilhar detecções passivas
            if hasattr(friend, 'passive_detections'):
                for direction_hash, (direction, timestamp) in friend.passive_detections.items():
                    # Só aceitar detecções relativamente recentes (menos de 2 segundos)
                    if current_time - timestamp < 2000:
                        # Se não temos esta direção ou a nossa é mais antiga, atualizar
                        if direction_hash not in self.direction_history or \
                        timestamp > self.direction_history[direction_hash][1]:
                            self.direction_history[direction_hash] = (direction, timestamp, id(friend))
            
            # Compartilhar alvos já triangulados
            if hasattr(friend, 'triangulated_targets'):
                for target_id, (position, confidence, last_update) in friend.triangulated_targets.items():
                    # Só aceitar triangulações recentes e com confiança razoável
                    if current_time - last_update < 3000 and confidence > 0.5:
                        # Se não temos este alvo ou o nosso é mais antigo, atualizar
                        if target_id not in self.triangulated_targets or \
                        last_update > self.triangulated_targets[target_id][2]:
                            self.triangulated_targets[target_id] = (position, confidence, last_update)
        
        # 3. Tentar triangular com as direções disponíveis
        # Agrupar detectores por posição para cada direção
        direction_detectors = {}  # {direction_hash: [(detector_id, pos, direction, timestamp)]}
        
        for direction_hash, (direction, timestamp, detector_id) in self.direction_history.items():
            # Encontrar a posição do detector
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
        
        # 4. Agrupar direções em potenciais alvos
        potential_targets = {}  # {target_id: [(detector_id, pos, direction, timestamp)]}
        target_counter = 0
        
        # Para cada direção, verificar compatibilidade com outras direções
        for dir_hash1, detectors1 in direction_detectors.items():
            for detector_id1, pos1, dir1, timestamp1 in detectors1:
                # Verificar se já foi atribuído a algum alvo
                already_assigned = False
                for target_detectors in potential_targets.values():
                    if any(did == detector_id1 and dhash == dir_hash1 
                        for did, dhash, _, _, _ in target_detectors):
                        already_assigned = True
                        break
                
                if already_assigned:
                    continue
                
                # Tentar formar um novo grupo de detecções compatíveis
                compatible_detections = [(detector_id1, dir_hash1, pos1, dir1, timestamp1)]
                
                for dir_hash2, detectors2 in direction_detectors.items():
                    if dir_hash1 == dir_hash2:
                        continue
                        
                    for detector_id2, pos2, dir2, timestamp2 in detectors2:
                        # Não incluir o mesmo detector duas vezes
                        if detector_id2 == detector_id1:
                            continue
                            
                        # Verificar compatibilidade
                        if self._directions_compatible(pos1, dir1, pos2, dir2):
                            compatible_detections.append((detector_id2, dir_hash2, pos2, dir2, timestamp2))
                
                # Se temos pelo menos 3 detectores compatíveis, criar um novo alvo potencial
                if len(compatible_detections) >= 3:
                    # Assegurar que são 3 drones distintos
                    unique_detectors = set(did for did, _, _, _, _ in compatible_detections)
                    if len(unique_detectors) >= 3:
                        target_counter += 1
                        potential_targets[target_counter] = compatible_detections
        
        # 5. Triangular posições para alvos potenciais
        for target_id, compatible_detections in potential_targets.items():
            position_estimates = []
            
            # Triangular com todos os pares possíveis
            for i in range(len(compatible_detections)):
                for j in range(i+1, len(compatible_detections)):
                    _, _, pos1, dir1, timestamp1 = compatible_detections[i]
                    _, _, pos2, dir2, timestamp2 = compatible_detections[j]
                    
                    position = self._triangulate_position(pos1, dir1, pos2, dir2)
                    
                    if position:
                        # Calcular confiança baseada na recência
                        recency1 = max(0, 1 - (current_time - timestamp1) / 1000)
                        recency2 = max(0, 1 - (current_time - timestamp2) / 1000)
                        confidence = recency1 * recency2
                        
                        position_estimates.append((position, confidence))
            
            if position_estimates:
                # Calcular média ponderada das posições estimadas
                total_confidence = sum(conf for _, conf in position_estimates)
                if total_confidence > 0:
                    weighted_pos = pygame.math.Vector2(0, 0)
                    for pos, conf in position_estimates:
                        weighted_pos += pos * (conf / total_confidence)
                    
                    # Calcular confiança global baseada no número de detectores e confiança média
                    # Número de detectores únicos
                    unique_detectors = set(did for did, _, _, _, _ in compatible_detections)
                    detector_factor = min(1.0, len(unique_detectors) / 5)  # Normalizado até 5 detectores
                    
                    overall_confidence = (total_confidence / len(position_estimates)) * detector_factor
                    
                    # Armazenar o alvo triangulado
                    stable_target_id = f"target_{hash(str(weighted_pos))}"
                    self.triangulated_targets[stable_target_id] = (weighted_pos, overall_confidence, current_time)
        
        # 6. Limpar alvos e direções antigas
        # Remover direções antigas do histórico
        old_directions = [hash_key for hash_key, (_, timestamp, _) in self.direction_history.items() 
                        if current_time - timestamp > 3000]  # 3 segundos
        for hash_key in old_directions:
            self.direction_history.pop(hash_key, None)
        
        # Remover alvos antigos
        old_targets = [tid for tid, (_, _, timestamp) in self.triangulated_targets.items() 
                    if current_time - timestamp > 5000]  # 5 segundos
        for tid in old_targets:
            self.triangulated_targets.pop(tid, None)
        
        # 7. Atualizar as estruturas de dados existentes com os alvos triangulados
        for target_id, (position, confidence, _) in self.triangulated_targets.items():
            # Converter a posição para índice de célula da grade
            cell = pos_to_cell(position)
            
            # Atualizar as estruturas existentes de detecção
            self.enemy_intensity[cell] = confidence  # Usando confiança como intensidade
            
            # Estimar a direção de movimento (se tivermos histórico)
            if hasattr(self, 'previous_triangulated') and target_id in self.previous_triangulated:
                prev_pos, _, _ = self.previous_triangulated[target_id]
                movement = position - prev_pos
                if movement.length() > 0:
                    self.enemy_direction[cell] = movement.normalize()
            else:
                # Sem histórico, usar média das direções que contribuíram para esta triangulação
                # (se tivermos acesso a elas)
                # Por simplicidade, mantemos a direção atual ou definimos como zero
                self.enemy_direction[cell] = pygame.math.Vector2(0, 0)
            
            # Atualizar o timestamp
            self.enemy_timestamp[cell] = current_time
        
        # Armazenar para o próximo ciclo
        if not hasattr(self, 'previous_triangulated'):
            self.previous_triangulated = {}
        self.previous_triangulated = self.triangulated_targets.copy()

    def _directions_compatible(self, pos1, dir1, pos2, dir2, angle_threshold=0.2):
        """
        Verifica se duas direções de detecção são compatíveis (podem apontar para o mesmo alvo).
        
        Args:
            pos1, pos2: Posições dos drones detectores.
            dir1, dir2: Direções de detecção.
            angle_threshold: Limiar de ângulo (em radianos) para considerar compatibilidade.
            
        Returns:
            bool: True se as direções são potencialmente compatíveis.
        """
        # Calcular o ponto de interseção (pode ser aproximado)
        intersection = self._triangulate_position(pos1, dir1, pos2, dir2)
        
        if intersection is None:
            return False
            
        # Verificar se as direções reais de ambos os detectores para o ponto
        # de interseção são próximas às direções de detecção
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
        Triangula uma posição a partir de duas posições e direções.
        Retorna None se as linhas forem quase paralelas.
        
        Args:
            pos1, pos2: Posições dos dois drones detectores.
            dir1, dir2: Direções normalizadas apontando para o alvo.
            
        Returns:
            pygame.math.Vector2: Posição estimada do alvo, ou None se não for possível triangular.
        """
        # Representar as linhas como:
        # pos1 + t1 * dir1
        # pos2 + t2 * dir2
        
        # Matriz para resolver o sistema linear
        A = np.array([
            [dir1.x, -dir2.x],
            [dir1.y, -dir2.y]
        ])
        
        b = np.array([
            pos2.x - pos1.x,
            pos2.y - pos1.y
        ])
        
        # Verificar se o sistema é bem-condicionado
        # (linhas não são quase paralelas)
        if abs(np.linalg.det(A)) < 0.1:  # Ajuste este limiar conforme necessário
            return None
            
        try:
            # Resolver o sistema para encontrar t1 e t2
            t1, t2 = np.linalg.solve(A, b)
            
            # Se t1 < 0 ou t2 < 0, as linhas se cruzam "para trás"
            if t1 < 0 or t2 < 0:
                return None
                
            # Calcular o ponto de interseção
            intersection1 = pygame.math.Vector2(pos1.x + dir1.x * t1, pos1.y + dir1.y * t1)
            intersection2 = pygame.math.Vector2(pos2.x + dir2.x * t2, pos2.y + dir2.y * t2)
            
            # Deveria ser o mesmo ponto, mas pode haver diferenças numéricas
            # Retorne a média como uma medida robusta
            return pygame.math.Vector2((intersection1.x + intersection2.x) / 2, 
                                    (intersection1.y + intersection2.y) / 2)
            
        except:
            # Caso haja problemas na solução do sistema
            return None
    # PROTO ########################  
                
    # -------------------------------------------------------------------------
    # Update Local Enemy Detection
    # -------------------------------------------------------------------------
    
    def update_local_enemy_detection(self, enemy_drones: List[Any]) -> None:
        """
        Atualiza a detecção local de drones inimigos.
        Agora também recebe dados da triangulação passiva quando disponíveis.
        """
        current_time: int = pygame.time.get_ticks()
        
        if self.behavior_type == "AEW":
            detection_range = AEW_DETECTION_RANGE
        elif self.behavior_type == "RADAR":
            detection_range = RADAR_DETECTION_RANGE
        else:
            detection_range = FRIEND_DETECTION_RANGE
        
        # Usar detecções passivas trianguladas se disponíveis
        if hasattr(self, 'triangulated_targets') and self.triangulated_targets:
            # Para cada alvo triangulado, atualizar as matrizes como se fosse uma detecção direta
            for target_id, (position, confidence, _) in self.triangulated_targets.items():
                cell: Tuple[int, int] = pos_to_cell(position)
                
                # Atualizar as estruturas de detecção apenas se a confiança for superior ao valor atual
                if confidence > self.enemy_intensity[cell]:
                    # Armazenar a posição no dicionário de detecção para uso em cálculos futuros
                    key = f"triangulated_{target_id}"
                    
                    # Se já temos um histórico para este alvo, podemos calcular a direção
                    direction = pygame.math.Vector2(0, 0)  # Padrão sem direção
                    
                    if hasattr(self, 'previous_triangulated') and target_id in self.previous_triangulated:
                        prev_pos, _, _ = self.previous_triangulated[target_id]
                        delta = position - prev_pos
                        if delta.length() > 0:
                            direction = delta.normalize()
                    
                    # Atualizar matrizes
                    self.enemy_intensity[cell] = confidence
                    self.enemy_direction[cell] = [direction.x, direction.y]
                    self.enemy_timestamp[cell] = current_time
                    
                    # Registrar para histórico
                    if key not in self.current_enemy_pos_detection:
                        self.current_enemy_pos_detection[key] = position.copy()
                    else:
                        # Já existia, atualizar registros de direção
                        self.aux_enemy_detections[key] = cell
                        self.current_enemy_pos_detection[key] = position.copy()
        
        # Continuar com o processo normal de detecção para inimigos em alcance direto
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

    
    # def update_local_enemy_detection(self, enemy_drones: List[Any]) -> None:
    #     """
    #     Updates the local detection of enemy drones.
        
    #     For each enemy drone within the detection range, updates the corresponding cell in
    #     the enemy detection matrices (intensity, direction, timestamp). Additionally, for cells
    #     within the detection radius that lack a detection (intensity below threshold), the intensity
    #     is set to zero and the timestamp is updated using np.putmask.
        
    #     Args:
    #         enemy_drones (List[Any]): List of enemy drones.
    #     """
    #     if self.behavior_type == "AEW":
    #         detection_range = AEW_DETECTION_RANGE
    #     elif self.behavior_type == "RADAR":
    #         detection_range = RADAR_DETECTION_RANGE
    #     else:
    #         detection_range = FRIEND_DETECTION_RANGE
                    
    #     current_time: int = pygame.time.get_ticks()
    #     for enemy in enemy_drones:
    #         key: int = id(enemy)
    #         if self.pos.distance_to(enemy.pos) >= detection_range:
    #             self.current_enemy_pos_detection.pop(key, None)
    #             self.aux_enemy_detections.pop(key, None)
    #             continue
            
    #         cell: Tuple[int, int] = pos_to_cell(enemy.pos)
    #         if key not in self.current_enemy_pos_detection:
    #             self.current_enemy_pos_detection[key] = enemy.pos.copy()
    #         else:
    #             if key in self.current_enemy_pos_detection and key in self.aux_enemy_detections:
    #                 prev_cell: Tuple[int, int] = self.aux_enemy_detections[key]
    #                 if prev_cell != cell:
    #                     # Zero out the values in the previous cell
    #                     self.enemy_intensity[prev_cell] = 0
    #                     self.enemy_direction[prev_cell] = [0, 0]
    #                     self.enemy_timestamp[prev_cell] = current_time
    #             self.aux_enemy_detections[key] = cell
    #             self.enemy_intensity[cell] = 1.0
    #             delta: pygame.math.Vector2 = enemy.pos - self.current_enemy_pos_detection[key]
    #             self.current_enemy_pos_detection[key] = enemy.pos.copy()
    #             if delta.length() > 0:
    #                 self.enemy_direction[cell] = list(delta.normalize())
    #             self.enemy_timestamp[cell] = current_time

    #     # --- Vectorized Update for Cells Without Detection ---
    #     # Convert detection range (in pixels) to number of cells; scale factor (0.8) can be adjusted.
    #     detection_range_cells = int(np.floor(detection_range / CELL_SIZE) * 0.8)
        
    #     # Get the central cell of the drone (its own position)
    #     center_x, center_y = pos_to_cell(self.pos)
        
    #     # Define limits of the rectangle that encloses the detection circle
    #     x_min = max(center_x - detection_range_cells, 0)
    #     x_max = min(center_x + detection_range_cells, GRID_WIDTH - 1)
    #     y_min = max(center_y - detection_range_cells, 0)
    #     y_max = min(center_y + detection_range_cells, GRID_HEIGHT - 1)
        
    #     # Create a grid of indices for the region
    #     x_indices = np.arange(x_min, x_max + 1)
    #     y_indices = np.arange(y_min, y_max + 1)
    #     xv, yv = np.meshgrid(x_indices, y_indices, indexing='ij')
        
    #     # Calculate the distance (in cell units) of each cell from the center
    #     distances = np.sqrt((xv - center_x) ** 2 + (yv - center_y) ** 2)
        
    #     # Extract the region of the enemy intensity and timestamp matrices
    #     region_intensity = self.enemy_intensity[x_min:x_max+1, y_min:y_max+1]
    #     region_timestamp = self.enemy_timestamp[x_min:x_max+1, y_min:y_max+1]
        
    #     # Create a mask for cells within the detection circle that have low intensity (i.e., no detection)
    #     mask_empty = (distances <= detection_range_cells) & (region_intensity < 1)
        
    #     # Set intensities to 0 and update timestamps to current_time for these "empty" cells
    #     np.putmask(region_intensity, mask_empty, 0)
    #     np.putmask(region_timestamp, mask_empty, current_time)
        
    #     # Se o drone está quebrado, aplica o comportamento de detecção errada
    #     if self.broken:
    #         self.update_broken(x_min, x_max, y_min, y_max, distances, detection_range_cells)

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
        
        messages_sent_this_cycle = 0
        connections_this_cycle = 0
        
        for other in all_drones:
            if other is not self and self.pos.distance_to(other.pos) < COMMUNICATION_RANGE:
                connections_this_cycle += 1
                messages_sent_this_cycle += 2
                if random.random() > MESSAGE_LOSS_PROBABILITY:
                    self.merge_enemy_matrix(other)
                    self.merge_friend_matrix(other)
                    
        # Count the number of connections made this cycle
        self.active_connections = connections_this_cycle
        self.messages_sent_this_cycle = messages_sent_this_cycle
                    
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
        # PROTO #################
        self.update_passive_detection_and_triangulate(enemy_drones, friend_drones)
        # PROTO #################
        self.update_local_enemy_detection(enemy_drones)
        self.update_local_friend_detection(friend_drones)
        self.communication(friend_drones)
        self.take_action()
        
        # Calcular distância percorrida
        new_distance = self.pos.distance_to(self.last_position)
        self.distance_traveled += new_distance
        self.last_position = self.pos.copy()
        
        self.trajectory.append(self.pos.copy())
    
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
            self.info, direction = planning_policy(state)
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
            self.info = ("AEW", None, None, None)
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
            self.info = ("RADAR", None, None, None)
            self.vel = pygame.math.Vector2(0, 0)

        elif self.behavior_type == "debug":
            self.info = ("DEBUG", None, None, None)
            target_vector = self.interest_point_center - self.pos
            direction = target_vector.normalize() if target_vector.length() > 0 else pygame.math.Vector2(0, 0)
            self.vel = direction * FRIEND_SPEED if not self.fixed else pygame.math.Vector2(0, 0)
                
        elif self.behavior_type == "u-debug":
            self.info = ("U-DEBUG", None, None, None)
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
                
    def draw_passive_detection(self, surface: pygame.Surface) -> None:
        """
        Desenha linhas representando as direções de detecção passiva
        e círculos representando as posições trianguladas.
        """
        if hasattr(self, 'passive_detections'):
            # Desenhar linhas de direção das detecções passivas
            for direction, _ in self.passive_detections.values():
                # Calcular ponto final da linha (estendendo a direção)
                end_point = self.pos + direction * FRIEND_DETECTION_RANGE  # Comprimento da linha de visualização
                draw_dashed_line(surface, (255, 255, 0, 64), self.pos, end_point,
                    width=1, dash_length=10, space_length=5)
                # pygame.draw.line(surface, (255, 255, 0), 
                #                 (int(self.pos.x), int(self.pos.y)), 
                #                 (int(end_point.x), int(end_point.y)), 1)
        
        if hasattr(self, 'last_triangulated'):
            # Desenhar posições trianguladas
            for position, confidence in self.last_triangulated.values():
                # Tamanho do círculo baseado na confiança
                radius = int(5 + confidence * 5)
                pygame.draw.circle(surface, (0, 255, 255, int(confidence * 200)),
                                (int(position.x), int(position.y)), radius, 1)

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
            self.draw_passive_detection(surface)
            
            len_info = len(self.info[0])
            debug_label = font.render(self.info[0], True, (255, 215, 0))
            surface.blit(debug_label, (int(self.pos.x) - 3.5 * len_info, int(self.pos.y) + 25))
            
            if self.info[1] is not None:
                pygame.draw.circle(surface, (255, 215, 0), (int(self.info[1].x), int(self.info[1].y)), 4)
                pygame.draw.line(surface, (255, 215, 0), (int(self.pos.x), int(self.pos.y)), (int(self.info[1].x), int(self.info[1].y)), 2)
                
            if self.info[2] is not None:
                pygame.draw.line(surface, (255, 215, 0), (int(self.interest_point_center[0]), int(self.interest_point_center[1])), (int(self.info[2].x), int(self.info[2].y)), 2)
                
            # if self.info[3] is not None and self.is_leader:
            #     for hold_friends in self.info[3]:
            #         pygame.draw.line(surface, (255, 215, 0), (int(self.pos.x), int(self.pos.y)), (int(hold_friends[1].x), int(hold_friends[1].y)), 1)