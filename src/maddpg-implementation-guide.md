# Guia de Implementação do MADDPG para Otimização da Fase HOLD_SPREAD

Este guia explica como implementar o algoritmo MADDPG (Multi-Agent Deep Deterministic Policy Gradient) para otimizar o comportamento dos drones durante a fase HOLD_SPREAD. A implementação usa uma recompensa global compartilhada que se alinha com o design atual do seu sistema.

## 1. Visão Geral

O MADDPG é uma extensão do algoritmo DDPG para cenários multi-agente, ideal para seu enxame de drones. Características principais:

- **Treinamento centralizado, execução descentralizada**: Durante o treinamento, o crítico tem acesso a informações globais, mas durante a execução, cada agente age apenas com base em suas observações locais.
- **Recompensa global compartilhada**: Todos os agentes recebem a mesma recompensa global, promovendo comportamento cooperativo.
- **Ações contínuas**: Controle otimizado dos parâmetros de repulsão e exploração.

## 2. Componentes Principais

### 2.1. Integração com seu código existente

O MADDPG será integrado em três pontos principais do seu código:

1. **Extração de estado**: Obtenção de informações relevantes de cada drone.
2. **Aplicação de ações**: Controle dos hiperparâmetros de repulsão durante HOLD_SPREAD.
3. **Cálculo de recompensa global**: Medição do desempenho coletivo do enxame.

### 2.2. Buffer de Replay

```python
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, states, actions, global_reward, next_states, done):
        experience = (states, actions, global_reward, next_states, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        # Amostragem aleatória de experiências
        samples = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        return states, actions, rewards, next_states, dones
```

### 2.3. Redes Neurais do MADDPG

```python
def _build_actor(self, state_dim, action_dim, name):
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(action_dim, activation='tanh')(x)
    # Escala as saídas para os limites de ação
    scaled_outputs = layers.Lambda(
        lambda x: x * (high - low) / 2 + (high + low) / 2
    )(outputs)
    return models.Model(inputs=inputs, outputs=scaled_outputs, name=name)

def _build_critic(self, state_dim, action_dim, name):
    state_input = layers.Input(shape=(state_dim,))
    action_input = layers.Input(shape=(action_dim,))
    
    state_x = layers.Dense(256, activation='relu')(state_input)
    state_x = layers.Dense(128, activation='relu')(state_x)
    
    concat = layers.Concatenate()([state_x, action_input])
    x = layers.Dense(128, activation='relu')(concat)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1)(x)
    
    return models.Model(inputs=[state_input, action_input], outputs=output, name=name)
```

## 3. Implementação dos Componentes de Integração

### 3.1. Extração de Estado do Drone

```python
def extract_state_for_drone(drone, all_drones, enemies, grid_coverage):
    """Extrai um vetor de estado para um drone"""
    # 1. Posição e velocidade normalizadas
    pos_x = drone.pos.x / (GRID_WIDTH * CELL_SIZE)
    pos_y = drone.pos.y / (GRID_HEIGHT * CELL_SIZE)
    vel_x = drone.vel.x / MAX_VELOCITY
    vel_y = drone.vel.y / MAX_VELOCITY
    
    # 2. Estado dos vizinhos (posição relativa, distância)
    neighbor_states = []
    for neighbor in drone.neighbors[:5]:  # Limitar a 5 vizinhos mais próximos
        rel_x = (neighbor.pos.x - drone.pos.x) / (GRID_WIDTH * CELL_SIZE)
        rel_y = (neighbor.pos.y - drone.pos.y) / (GRID_HEIGHT * CELL_SIZE)
        distance = drone.pos.distance_to(neighbor.pos) / (GRID_WIDTH * CELL_SIZE)
        neighbor_states.extend([rel_x, rel_y, distance])
    
    # Preencher com zeros se houver menos de 5 vizinhos
    while len(neighbor_states) < 15:  # 3 valores por vizinho * 5 vizinhos
        neighbor_states.append(0.0)
    
    # 3. Informações de cobertura local do grid
    coverage_info = []
    cell_x, cell_y = pos_to_cell(drone.pos, CELL_SIZE)
    for i in range(-1, 2):
        for j in range(-1, 2):
            cx, cy = cell_x + i, cell_y + j
            if 0 <= cx < GRID_WIDTH and 0 <= cy < GRID_HEIGHT:
                coverage_info.append(grid_coverage[cx, cy])
            else:
                coverage_info.append(0.0)
    
    # 4. Informações de detecção de inimigos
    enemy_detection = []
    detection_count = sum(sum(drone.passive_detection_matrix))
    normalized_detection = min(1.0, detection_count / (GRID_WIDTH * GRID_HEIGHT * 0.1))
    enemy_detection.append(normalized_detection)
    
    # 5. Parâmetros atuais (para behavior cloning)
    repulsion_params = [
        drone.repulsion_params['strength'] / 5.0,
        drone.repulsion_params['radius'] / 20.0,
        drone.repulsion_params['damping'],
        drone.repulsion_params['exploration_bias']
    ]
    
    # Concatenar todos os componentes
    state = [pos_x, pos_y, vel_x, vel_y] + neighbor_states + coverage_info + enemy_detection + repulsion_params
    
    return np.array(state, dtype=np.float32)
```

### 3.2. Cálculo da Recompensa Global

```python
def calculate_global_reward(drones, prev_global_coverage, current_global_coverage, 
                           prev_global_detection, current_global_detection):
    """Calcula a recompensa global para o enxame de drones"""
    # 1. Recompensa por aumento de cobertura
    coverage_diff = current_global_coverage - prev_global_coverage
    coverage_reward = coverage_diff * 10.0
    
    # 2. Recompensa por detecção de inimigos
    detection_diff = current_global_detection - prev_global_detection
    detection_reward = detection_diff * 5.0
    
    # 3. Penalidade por sobreposição entre drones
    overlap_penalty = 0.0
    for i, drone_i in enumerate(drones):
        for j, drone_j in enumerate(drones):
            if i < j:  # Evita contar duas vezes
                distance = drone_i.pos.distance_to(drone_j.pos)
                min_desired_distance = (drone_i.repulsion_radius + drone_j.repulsion_radius) * 0.5
                if distance < min_desired_distance:
                    overlap_penalty -= (1.0 - distance / min_desired_distance) * 0.1
    
    # 4. Recompensa por manter conectividade
    connectivity_reward = 0.0
    avg_neighbors = sum(len(drone.neighbors) for drone in drones) / len(drones)
    if avg_neighbors > 1:
        connectivity_reward = 0.2 * min(1.0, avg_neighbors / 3.0)
    
    # 5. Penalidade por consumo de energia
    energy_penalty = 0.0
    for drone in drones:
        energy_penalty -= 0.001 * (drone.vel.x**2 + drone.vel.y**2)
    
    return coverage_reward + detection_reward + overlap_penalty + connectivity_reward + energy_penalty
```

### 3.3. Aplicação de Ações do MADDPG

```python
def apply_maddpg_actions(drone, action_values):
    """Aplica as ações produzidas pelo MADDPG ao drone"""
    # Atualizar parâmetros de repulsão com base nas ações do MADDPG
    drone.repulsion_params['strength'] = max(0.1, min(5.0, action_values[0]))
    drone.repulsion_params['radius'] = max(1.0, min(20.0, action_values[1]))
    drone.repulsion_params['damping'] = max(0.1, min(0.9, action_values[2]))
    drone.repulsion_params['exploration_bias'] = max(0.0, min(1.0, action_values[3]))
    
    # Atualizar os parâmetros no drone
    drone.repulsion_strength = drone.repulsion_params['strength']
    drone.repulsion_radius = drone.repulsion_params['radius']