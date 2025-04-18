from typing import Tuple
import pygame

# Initialize pygame if not already initialized
if not pygame.get_init():
    pygame.init()

# -------- Global Display Configuration --------
display_info: pygame.display.Info = pygame.display.Info()
FULL_WIDTH: int = display_info.current_w
FULL_HEIGHT: int = display_info.current_h
screen = pygame.display.set_mode((FULL_WIDTH, FULL_HEIGHT), pygame.FULLSCREEN)

# -------- Frame Rate Control --------
clock: pygame.time.Clock = pygame.time.Clock()
DT_STEP: int = 3/5  # Time step for simulation

# -------- Simulation Parameters --------
EPISODES: int = 2  # Number of episodes to run
EPSILON: int = 2

# -------- Drone Counts --------
FRIEND_COUNT: int = 50 # 10
ENEMY_COUNT: int = 50 # 10

# -------- Aggressiveness and Escape Settings --------
# 0: flee, 1: head towards the point of interest when detected
INITIAL_AGGRESSIVENESS: float = 0.5 # 0.5 # 1
ESCAPE_STEPS: int = 40  # Number of steps to escape

# -------- Screen Layout --------
# Left portion for simulation, right portion for graphs
SIM_WIDTH: int = int(FULL_WIDTH * 0.7)  # 70% of screen width for simulation
SIM_HEIGHT: int = FULL_HEIGHT
GRAPH_WIDTH: int = FULL_WIDTH - SIM_WIDTH  # Remaining width for graphs
GRAPH_HEIGHT: int = FULL_HEIGHT

# -------- Grid and Simulation Parameters --------
CELL_SIZE: int = 20
GRID_WIDTH: int = SIM_WIDTH // CELL_SIZE
GRID_HEIGHT: int = SIM_HEIGHT // CELL_SIZE

DECAY_FACTOR: float = 0.99 # Factor for exponential decay in detection matrices
FRIEND_DETECTION_RANGE: int = 100 # 0 # 20 # 100 # Range (in pixels) for friend detection
ENEMY_DETECTION_RANGE: int = 100 # Range (in pixels) for enemy detection
COMMUNICATION_RANGE: int = 250 # 0 # 250  # Communication range between drones
N_CONNECTIONS: int = 4 # Number of connections for each drone
MESSAGE_LOSS_PROBABILITY: float = 0.1
TARGET_INFLUENCE: float = 0.05
BASE_SPEED: float = 2.0
ENEMY_SPEED: float = BASE_SPEED
FRIEND_SPEED: float = 1 * BASE_SPEED # 1.1
PLOT_THRESHOLD: float = 0.05

# -------- Interest Point Constants --------
INTEREST_POINT_CENTER = pygame.math.Vector2(SIM_WIDTH / 2, SIM_HEIGHT / 2) # debug apaga!
CENTER = INTEREST_POINT_CENTER.copy()
INTERNAL_RADIUS: int = min(SIM_WIDTH, SIM_HEIGHT) / 10
EXTERNAL_RADIUS: int = INTERNAL_RADIUS * 4
INTEREST_POINT_ATTACK_RANGE: int = EPSILON
INTEREST_POINT_INITIAL_HEALTH: int = 100
INTEREST_POINT_DAMAGE: int = INTEREST_POINT_INITIAL_HEALTH // ENEMY_COUNT


# -------- Drone Constants --------
NEUTRALIZATION_RANGE: int = 20  # Capture distance for neutralization
NEUTRALIZATION_PROB_FRIEND_ALIVE = 0.5 # 0 # 0.5  # Probabilidade de o amigo sobreviver (inimigo removido)
NEUTRALIZATION_PROB_ENEMY_ALIVE = 0.2 # 0 # 0.2   # Probabilidade de o inimigo sobreviver (amigo removido)
NEUTRALIZATION_PROB_BOTH_DEAD = 1 - (NEUTRALIZATION_PROB_FRIEND_ALIVE + NEUTRALIZATION_PROB_ENEMY_ALIVE)
INITIAL_DISTANCE = INTERNAL_RADIUS * 1.4
THRESHOLD_PROJECTION = INTERNAL_RADIUS * 0.5  # Máxima distância permitida entre o drone e sua projeção na reta do inimigo
MIN_COMMUNICATION_HOLD: int = 2 # 2 # 0 # Minimum friend communication hold time
HOLD_SPREAD: bool = True
DETECTION_MODE: str = "direct" # "triangulation"  # "direct"

# -------- AEW --------
AEW_COUNT: int = 0 # 0 # 5
AEW_RANGE: int = 350
AEW_SPEED: float = FRIEND_SPEED
AEW_DETECTION_RANGE: int = 200

# -------- RADAR --------
RADAR_COUNT = 0 # 0 # 1
RADAR_RANGE = 0
RADAR_DETECTION_RANGE = 350

# -------- BROKEN --------
BROKEN_COUNT = 0 # 0 # 1
UPDATE_STATE_BROKEN = 100

# DMZ
DMZ = [
    (SIM_WIDTH * 0.35, SIM_HEIGHT * 0.30, 60),
    (SIM_WIDTH * 0.65, SIM_HEIGHT * 0.35, 40),
    (SIM_WIDTH * 0.55, SIM_HEIGHT * 0.75, 80)
]


# -------- Geographic Coordinates --------
# Top-left corner (Longitude, Latitude)
GEO_TOP_LEFT: Tuple[float, float] = (-74.0, 40.8)
# Bottom-right corner (Longitude, Latitude)
GEO_BOTTOM_RIGHT: Tuple[float, float] = (-73.9, 40.7)

# -------- Font Setup --------
FONT_FAMILY: str = "Courier New" # "Consolas"
