# type: ignore
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
import gc
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
    timestamp = start_time.strftime("%Y_%m_%d_%Hh%Mm%Ss")
    save_folder = os.path.join("data", TYPE_OF_SCENARIO)
    os.makedirs(save_folder, exist_ok=True)
    
    # Define the CSV file path for saving episode results.
    csv_path = "./data/proposal_aew_spread/results_2025_04_25_09h22m50s.csv"
    # csv_path = os.path.join(save_folder, F"results_{timestamp}.csv")
    
    # Create an instance of the Air Traffic Environment.
    env: AirTrafficEnv = AirTrafficEnv(mode=None, friend_behavior='planning', enemy_behavior=None, demilitarized_zones=DMZ, seed=42)
    
    # Run episodes and persist results.
    for episode in range(12): # range(ANALYSIS_EPISODES):
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
        state_percentages = info['state_percentages']
        total_distance = info['total_distance_traveled']

        result = {
            "episode": episode,
            "steps": n_steps,
            "accumulated_reward": accum_reward,
            "enemies_shotdown": enemies_shotdown,
            "friends_shotdown": friends_shotdown,
            "sucessful_attacks": sucessful_attacks,
            "interest_point_health": interest_point_health,
            "total_distance_traveled": total_distance
        }
        
        for state, percentage in state_percentages.items():
            result[f"state_{state.replace(' ', '_')}"] = percentage

        # Persist the episode result immediately.
        persist_episode_result(result, csv_path)
        
        print("-" * 50)
        print(f"FINAL: Air Traffic Env Episode {episode+1:3d}\n")
        print(f"\tSteps: {n_steps:4d}")
        print(f"\tAccumulated Reward: {accum_reward:7.3f}")
        print(f"\tEnemies Shotdown: {enemies_shotdown}")
        print(f"\tFriends Shotdown: {friends_shotdown}")
        print(f"\tSuccessful Attacks: {sucessful_attacks:3d}")
        print(f"\tInterest Point Health: {interest_point_health}")
        print(f"\tTotal Distance Traveled: {total_distance:.2f} px")
        print("\tState Percentages:")
        
        for state, percentage in state_percentages.items():
            print(f"\t\t{state}: {percentage:.2f}%")
            
        gc.collect()
        
    print("-" * 50)
        
    # Close the environment.
    env.close()

# -----------------------------------------------------------------------------
# Run the Simulation if Executed as Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()