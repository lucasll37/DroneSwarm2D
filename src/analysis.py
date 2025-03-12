"""
run_simulation.py

This module sets up the simulation environment and runs episodes for the Air Traffic
environment. It saves environment variables to a text file and persists the results
of each episode to a CSV file using pandas.
"""

# -----------------------------------------------------------------------------
# Imports and Configuration
# -----------------------------------------------------------------------------
import os
import sys
import random
import pygame
import datetime
import pandas as pd
from typing import Tuple, List, Any

# Add configuration directory to the system path (if necessary)
current_dir: str = os.getcwd()
config_dir: str = os.path.abspath(os.path.join(current_dir, "./src/environment"))
if config_dir not in sys.path:
    sys.path.append(config_dir)
    
# Project-specific imports
from AirTrafficEnv import AirTrafficEnv
from settings import *

# -----------------------------------------------------------------------------
# Environment Variables Saving Function
# -----------------------------------------------------------------------------
def save_variables(save_folder: str) -> None:
    """
    Saves environment variables to a text file in the given folder.
    
    Args:
        save_folder (str): The folder where the file "env_variables.txt" will be created.
    """
    txt_path = os.path.join(save_folder, "env_variables.txt")
    variables_to_save = {
        "FULL_WIDTH": FULL_WIDTH,
        "FULL_HEIGHT": FULL_HEIGHT,
        "EPISODES": EPISODES,
        "FRIEND_COUNT": FRIEND_COUNT,
        "ENEMY_COUNT": ENEMY_COUNT,
        "INITIAL_AGGRESSIVENESS": INITIAL_AGGRESSIVENESS,
        "ESCAPE_STEPS": ESCAPE_STEPS,
        "SIM_WIDTH": SIM_WIDTH,
        "SIM_HEIGHT": SIM_HEIGHT,
        "GRAPH_WIDTH": GRAPH_WIDTH,
        "GRAPH_HEIGHT": GRAPH_HEIGHT,
        "CELL_SIZE": CELL_SIZE,
        "GRID_WIDTH": GRID_WIDTH,
        "GRID_HEIGHT": GRID_HEIGHT,
        "DECAY_FACTOR": DECAY_FACTOR,
        "ENEMY_DETECTION_RANGE": ENEMY_DETECTION_RANGE,
        "FRIEND_DETECTION_RANGE": FRIEND_DETECTION_RANGE,
        "COMMUNICATION_RANGE": COMMUNICATION_RANGE,
        "MESSAGE_LOSS_PROBABILITY": MESSAGE_LOSS_PROBABILITY,
        "TARGET_INFLUENCE": TARGET_INFLUENCE,
        "FRIEND_SPEED": FRIEND_SPEED,
        "ENEMY_SPEED": ENEMY_SPEED,
        "PLOT_THRESHOLD": PLOT_THRESHOLD,
        "INTEREST_POINT_ATTACK_RANGE": INTEREST_POINT_ATTACK_RANGE,
        "INTEREST_POINT_INITIAL_HEALTH": INTEREST_POINT_INITIAL_HEALTH,
        "INTEREST_POINT_DAMAGE": INTEREST_POINT_DAMAGE,
        "NEUTRALIZATION_RANGE": NEUTRALIZATION_RANGE,
        "NEUTRALIZATION_PROB_FRIEND_ALIVE": NEUTRALIZATION_PROB_FRIEND_ALIVE,
        "NEUTRALIZATION_PROB_ENEMY_ALIVE": NEUTRALIZATION_PROB_ENEMY_ALIVE,
        "GEO_TOP_LEFT": GEO_TOP_LEFT,
        "GEO_BOTTOM_RIGHT": GEO_BOTTOM_RIGHT,
        "FONT_FAMILY": FONT_FAMILY
    }
    with open(txt_path, "w") as f:
        for var, value in variables_to_save.items():
            f.write(f"{var} = {value}\n")
    print(f"Environment variables saved in: {txt_path}")

# -----------------------------------------------------------------------------
# Episode Result Persistence Function
# -----------------------------------------------------------------------------
def persist_episode_result(result: dict, csv_path: str) -> None:
    """
    Persists the result of an episode to a CSV file using pandas.
    
    Args:
        result (dict): A dictionary containing the episode results.
        csv_path (str): Path to the CSV file.
    """
    df = pd.DataFrame([result])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False, mode='w')
    else:
        df.to_csv(csv_path, index=False, mode='a', header=False)

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Main entry point for running the Air Traffic simulation.
    
    This function creates a folder for saving environment variables and episode
    results, initializes the simulation environment, runs episodes, and saves
    the results after each episode.
    """
    # Record start time and create a timestamped save folder.
    start_time = datetime.datetime.now()
    folder_timestamp = start_time.strftime("%Y_%m_%d_%Hh%Mm%Ss")
    save_folder = os.path.join("data", folder_timestamp)
    os.makedirs(save_folder, exist_ok=True)
    
    # Save the environment variables to a text file.
    save_variables(save_folder)
    
    # Define the CSV file path for saving episode results.
    csv_path = os.path.join(save_folder, "results.csv")
    
    # Create an instance of the Air Traffic Environment.
    env: AirTrafficEnv = AirTrafficEnv(mode=None, friend_behavior='planning', enemy_behavior='direct')
    
    # Run episodes and persist results.
    for episode in range(EPISODES):
        obs, done = env.reset()
        total_reward: float = 0.0
        
        while not done:
            action = None  # Implement your action logic here if needed.
            obs, reward, done, info = env.step(action)
            total_reward += reward
        
        # Gather episode statistics.
        n_steps: int = info['current_step']
        accum_reward: float = info['accum_reward']
        enemies_shotdown: int = info['enemies_shotdown']
        friends_shotdown: int = info['friends_shotdown']
        sucessful_attacks: int = info['sucessful_attacks']
        interest_point_health: int = info['interest_point_health']
        
        result = {
            "episode": episode,
            "steps": n_steps,
            "accumulated_reward": accum_reward,
            "enemies_shotdown": enemies_shotdown,
            "friends_shotdown": friends_shotdown,
            "sucessful_attacks": sucessful_attacks,
            "interest_point_health": interest_point_health
        }
        
        # Persist the episode result immediately.
        persist_episode_result(result, csv_path)
        
        print(f"FINAL: Air Traffic Env - Episode: {episode:3d} | Steps: {n_steps:4d} | Acc. Reward: {accum_reward:7.3f} | Enemies Shotdown: {enemies_shotdown} | Friends Shotdown: {friends_shotdown} | Sucessful Attacks: {sucessful_attacks:3d} | IP Health: {interest_point_health}")
    
    # Close the environment.
    env.close()

# -----------------------------------------------------------------------------
# Run the Simulation if Executed as Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()