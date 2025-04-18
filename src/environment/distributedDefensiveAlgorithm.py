# -----------------------------------------------------------------------------
# Imports and Setup
# -----------------------------------------------------------------------------

import numpy as np
import pygame
from utils import pos_to_cell, intercept_direction, can_intercept
from settings import *

# -------------------------------------------------------------------------
# Planning Policy (Class Method)
# -------------------------------------------------------------------------   
def planning_policy(state, friend_activation_threshold_position: float = 0.7,
                    enemy_activation_threshold_position: float = 0.4):

    def hold_position(pos, friend_intensity, enemy_intensity, enemy_direction, friends_hold=None,
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
            info = ("HOLD", None, None, friends_hold)
            return info, pygame.math.Vector2(0, 0)
        
        # Constrói a grade dos centros das células com base em friend_intensity.
        grid_x = np.linspace(CELL_SIZE/2, SIM_WIDTH - CELL_SIZE/2, GRID_WIDTH)
        grid_y = np.linspace(CELL_SIZE/2, SIM_HEIGHT - CELL_SIZE/2, GRID_HEIGHT)
        X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')
        distance_matrix = np.sqrt((X - pos.x)**2 + (Y - pos.y)**2)
        comm_mask = distance_matrix < COMMUNICATION_RANGE
        active_friend_count = np.sum((friend_intensity >= activation_threshold_position) & comm_mask)
        
        # Caso 2: Se o drone não se comunica, retorna para a distância INITIAL_DISTANCE do PI.
        if MIN_COMMUNICATION_HOLD == 0 and pos.distance_to(INTEREST_POINT_CENTER) > INITIAL_DISTANCE:
            info = ("HOLD - RETURN", None, None, friends_hold)
            direction = (INTEREST_POINT_CENTER - pos).normalize()
            return info, direction
            
        # Caso 3: Se conectado a pelo menos MIN_COMMUNICATION_HOLD amigos, verifica oportunidade defensiva.
        elif active_friend_count >= MIN_COMMUNICATION_HOLD:
            candidate_intercepts = []
            for cell, intensity in np.ndenumerate(enemy_intensity):
                if intensity < enemy_threshold:
                    continue
                # Centro da célula correspondente
                cell_center = pygame.math.Vector2((cell[0] + 0.5) * CELL_SIZE,
                                                    (cell[1] + 0.5) * CELL_SIZE)
                # Vetor que vai do centro da célula até o ponto de interesse
                vec_to_interest = INTEREST_POINT_CENTER - cell_center
                if vec_to_interest.length() == 0:
                    continue
                vec_to_interest = vec_to_interest.normalize()
                # Vetor de direção detectado para o inimigo nesta célula
                enemy_dir = pygame.math.Vector2(*enemy_direction[cell])
                if enemy_dir.length() == 0:
                    continue
                enemy_dir = enemy_dir.normalize()
                if enemy_dir.dot(vec_to_interest) >= 0.9:
                    candidate_intercepts.append((cell_center.distance_to(INTEREST_POINT_CENTER),
                                                cell_center, enemy_dir))
                    
            if candidate_intercepts:
                candidate_intercepts.sort(key=lambda t: t[0])
                _, chosen_cell_center, enemy_dir = candidate_intercepts[0]
                # Calcula a projeção da posição do drone sobre a reta que passa por chosen_cell_center com direção enemy_dir.
                s = (pos - chosen_cell_center).dot(enemy_dir)
                projection_point = chosen_cell_center + s * enemy_dir
                                
                feasible = projection_point.distance_to(INTEREST_POINT_CENTER) < cell_center.distance_to(INTEREST_POINT_CENTER)
                    
                if pos.distance_to(projection_point) <= EPSILON and feasible:
                    info = ("HOLD INTCPT", None, None, friends_hold)
                    direction = pygame.math.Vector2(0, 0)
                    return info, direction
                
                if chosen_cell_center.distance_to(projection_point) > chosen_cell_center.distance_to(INTEREST_POINT_CENTER):
                    defensive_point = INTEREST_POINT_CENTER
                else:
                    defensive_point = projection_point
                
                # Se o drone estiver muito distante dessa projeção, permanece em hold.
                if (pos - defensive_point).length() < THRESHOLD_PROJECTION and feasible:
                    info = ("GO HOLD INTCPT", defensive_point, chosen_cell_center, friends_hold)
                    direction = (defensive_point - pos).normalize()
                    
                    return info, direction
            
            if HOLD_SPREAD and len(friends_hold) > 2:    
            # if len(friends_hold) > 2 and pos.distance_to(INTEREST_POINT_CENTER) <= 1.1 * INITIAL_DISTANCE:    
                # Se o drone estiver em hold e houver pelo menos dois amigos ativos, verifica a possibilidade de espalhamento.
                # Calcula a força de repulsão entre os amigos.
                repulsion = pygame.math.Vector2(0, 0)
                
                my_cell = pos_to_cell(pos)
                for cell, candidate_pos in friends_hold:
                    # Exclui self (assumindo que sua célula é única)
                    if cell == my_cell:
                        continue
                    
                    delta = pos - candidate_pos
                    repulsion += delta.normalize()
                
                # Para manter a formação em torno do ponto de interesse, extraímos somente a componente
                # tangencial do vetor repulsivo em relação ao vetor radial (do ponto de interesse até pos).
                radial = pos - INTEREST_POINT_CENTER
                if radial.length() == 0:
                    radial = pygame.math.Vector2(1, 0)
                else:
                    radial = radial.normalize()
                # Projeção da força repulsiva na direção radial.
                radial_component = repulsion.dot(radial) * radial
                # Componente tangencial: subtrai a parte radial.
                tangential = repulsion - radial_component
                direction = tangential.copy()
                
                if direction.length() > 1:
                    direction = direction.normalize()
                    
                info = ("HOLD SPREAD", None, None, friends_hold)
                    
                return info, direction

            info = ("HOLD", None, None, friends_hold)
            return info, pygame.math.Vector2(0, 0)
        
        # Caso 3: Se não estiver conectado a pelo menos dois amigos, retorna a direção para o PI.
        info = ("HOLD - NO ENOUGH COMM", None, None, friends_hold)
        direction = (INTEREST_POINT_CENTER - pos).normalize()
        return info, direction
        
    # Extração e preparação do estado
    pos = np.squeeze(state['pos'])
    pos = pygame.math.Vector2(pos[0], pos[1])
    friend_intensity = np.squeeze(state['friend_intensity'])
    enemy_intensity = np.squeeze(state['enemy_intensity'])
    friend_direction = np.squeeze(state['friend_direction'])
    enemy_direction = np.squeeze(state['enemy_direction'])
    
    enemy_targets = []
    # Identifica células da matriz de inimigos com intensidade acima do limiar
    for cell, intensity in np.ndenumerate(enemy_intensity):
        if intensity < enemy_activation_threshold_position:
            continue
        
        target_pos = pygame.math.Vector2((cell[0] + 0.5) * CELL_SIZE,
                                        (cell[1] + 0.5) * CELL_SIZE)
        distance_to_interest = target_pos.distance_to(INTEREST_POINT_CENTER)
        
        enemy_targets.append((cell, target_pos, distance_to_interest))
        
        
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
            
    if not enemy_targets:
        # Se nenhum alvo foi atribuído a self, mantém a posição.
        friends_hold = [(cell, candidate_pos) for (cell, candidate_pos) in friend_candidates]# if 1.1 * INITIAL_DISTANCE - candidate_pos.distance_to(INTEREST_POINT_CENTER) > 0]
        
        return hold_position(pos, friend_intensity, enemy_intensity, enemy_direction, friends_hold)
    
    # Ordena os alvos pelo quão próximos estão do ponto de interesse
    enemy_targets.sort(key=lambda t: t[2])
    
    engagement_assignment = {}
    assigned_friend_cells = set()
    
    # Para cada alvo inimigo, em ordem, atribui o candidato amigo mais próximo que ainda não foi designado.
    for cell, target_pos, _ in enemy_targets:
        sorted_candidates = sorted(friend_candidates, key=lambda x: x[1].distance_to(target_pos))
        enemy_dir_vec = pygame.math.Vector2(enemy_direction[cell][0], enemy_direction[cell][1])  * ENEMY_SPEED
        
        for candidate in sorted_candidates:
            candidate_cell, candidate_pos = candidate
            
            if candidate_cell not in assigned_friend_cells and \
            can_intercept(candidate_pos, FRIEND_SPEED, target_pos, enemy_dir_vec, INTEREST_POINT_CENTER):
                
                engagement_assignment[tuple(target_pos)] = candidate_cell
                assigned_friend_cells.add(candidate_cell)
                break  # Avança para o próximo alvo
            
    # friend_candidates é uma lista de tuplas (cell, candidate_pos) derivada da matriz friend_intensity
            
    # Verifica se algum dos alvos designados possui self como candidato
    engaged_enemies = []
    for cell, target_pos, _ in enemy_targets:
        if engagement_assignment.get(tuple(target_pos)) == my_cell:
            distance = my_cell_center.distance_to(target_pos)
            engaged_enemies.append((distance, cell, target_pos))
    
    if engaged_enemies:
        engaged_enemies.sort(key=lambda t: t[0])
        _, chosen_cell, chosen_target_pos = engaged_enemies[0]
        direction = intercept_direction(pos, FRIEND_SPEED, chosen_target_pos, enemy_direction[chosen_cell])
        if direction.length() > 0:
            info = ("PURSUING", None, None, None)
            return info, direction.normalize()
        else:
            info = ("ERROR PURSUING", None, None, None)
            return info, pygame.math.Vector2(0, 0)
    
    # Se nenhum alvo foi atribuído a self, mantém a posição.
    friends_hold = [(cell, candidate_pos) for (cell, candidate_pos) in friend_candidates
        if cell not in assigned_friend_cells] # and 1.1 * INITIAL_DISTANCE - candidate_pos.distance_to(INTEREST_POINT_CENTER) > 0]
    
    return hold_position(pos, friend_intensity, enemy_intensity, enemy_direction, friends_hold)