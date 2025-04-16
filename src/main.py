"""
run_air_traffic_env.py

This module demonstrates running an example of the AirTrafficEnv reinforcement learning
environment. The environment is run for a specified number of episodes, and statistics are
printed for each episode.
"""

# -----------------------------------------------------------------------------
# Imports and Configuration
# -----------------------------------------------------------------------------
import os
import sys
import random
import pygame
from typing import List, Any

# Get current working directory and add configuration directory to the system path.
current_dir: str = os.getcwd()
config_dir: str = os.path.abspath(os.path.join(current_dir, "./src/environment"))
if config_dir not in sys.path:
    sys.path.append(config_dir)

# Project-specific imports
from AirTrafficEnv import AirTrafficEnv
from settings import *

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Main function to run an example of the AirTrafficEnv RL environment.
    
    This example initializes the environment in 'human' mode with a planning friend behavior
    and no specific enemy behavior. The environment is run for one episode, and key metrics
    are printed at each step.
    """
    # Create an instance of the AirTrafficEnv.
    # Friend behavior examples: "direct", "zigzag", "spiral", etc.
    # env: AirTrafficEnv = AirTrafficEnv(mode='human', friend_behavior='AI', enemy_behavior=None, demilitarized_zones=DMZ)
    # env: AirTrafficEnv = AirTrafficEnv(mode='human', friend_behavior='planning', enemy_behavior="formation", demilitarized_zones=DMZ)
    env: AirTrafficEnv = AirTrafficEnv(mode='human', friend_behavior='planning', enemy_behavior=None, demilitarized_zones=DMZ)
    
    episodes: int = 1  # Set number of episodes to run
    
    # Main loop: run for the specified number of episodes.
    for episode in range(episodes):
        # Reset the environment to start a new episode.
        obs, done = env.reset()
        total_reward: float = 0.0
        
        # Run the episode until the environment signals 'done'.
        while not done:
            action = None  # Replace with your action logic if needed.
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            # Extract episode metrics from the environment info.
            n_steps: int = info['current_step']
            accum_reward: float = info['accum_reward']
            enemies_shotdown: int = info['enemies_shotdown']
            friends_shotdown: int = info['friends_shotdown']
            sucessful_attacks: int = info['sucessful_attacks']
            interest_point_health: int = info['interest_point_health']
            
            # Print current episode metrics.
            print(f"Air Traffic Env - Episode: {episode:3d} | Steps: {n_steps:4d} | Acc. Reward: {accum_reward:7.3f} | "
                  f"Enemies Shotdown: {enemies_shotdown} | Friends Shotdown: {friends_shotdown} | "
                  f"Sucessful Attacks: {sucessful_attacks:3d} | IP Health: {interest_point_health}")
        
        # Print final statistics for the episode.
        print(f"\n\nFINAL: Air Traffic Env - Episode: {episode:3d} | Steps: {n_steps:4d} | Acc. Reward: {accum_reward:7.3f} | "
              f"Enemies Shotdown: {enemies_shotdown} | Friends Shotdown: {friends_shotdown} | "
              f"Sucessful Attacks: {sucessful_attacks:3d} | IP Health: {interest_point_health}\n\n")
        
    # Close the environment after all episodes have completed.
    env.close()

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()

# """
# run_air_traffic_env.py

# This module demonstrates running an example of the AirTrafficEnv reinforcement learning
# environment. The environment is run for a specified number of episodes, and statistics are
# printed for each episode.
# """

# # -----------------------------------------------------------------------------
# # Imports and Configuration
# # -----------------------------------------------------------------------------
# import os
# import sys
# import random
# import pygame
# from typing import List, Any

# # Get current working directory and add configuration directory to the system path.
# current_dir: str = os.getcwd()
# config_dir: str = os.path.abspath(os.path.join(current_dir, "./src/environment"))
# if config_dir not in sys.path:
#     sys.path.append(config_dir)

# # Project-specific imports
# from AirTrafficEnv import AirTrafficEnv
# from settings import EPISODES

# # -----------------------------------------------------------------------------
# # Main Function
# # -----------------------------------------------------------------------------
# def main() -> None:
#     """
#     Main function to run an example of the AirTrafficEnv RL environment.
    
#     This example initializes the environment in 'human' mode with a planning friend behavior
#     and no specific enemy behavior. The environment is run for one episode, and key metrics
#     are printed at each step.
#     """
#     # Create an instance of the AirTrafficEnv.
#     # Friend behavior examples: "direct", "zigzag", "spiral", etc.
#     # env: AirTrafficEnv = AirTrafficEnv(mode='human', friend_behavior='AI', enemy_behavior=None)
#     env: AirTrafficEnv = AirTrafficEnv(mode='human', friend_behavior='planning', enemy_behavior=None)
    
#     episodes: int = 5  # Set number of episodes to run
    
#     # Main loop: run for the specified number of episodes.
#     for episode in range(episodes):
#         # Reset the environment to start a new episode.
#         obs, done = env.reset()
#         total_reward: float = 0.0
        
#         # Run the episode until the environment signals 'done'.
#         while not done:
#             action = None  # Replace with your action logic if needed.
#             obs, reward, done, info = env.step(action)
#             total_reward += reward
            
#             # Extract episode metrics from the environment info.
#             n_steps: int = info['current_step']
#             accum_reward: float = info['accum_reward']
#             enemies_shotdown: int = info['enemies_shotdown']
#             friends_shotdown: int = info['friends_shotdown']
#             sucessful_attacks: int = info['sucessful_attacks']
#             interest_point_health: int = info['interest_point_health']
            
#             # Print current episode metrics.
#             print(f"Air Traffic Env - Episode: {episode:3d} | Steps: {n_steps:4d} | Acc. Reward: {accum_reward:7.3f} | "
#                   f"Enemies Shotdown: {enemies_shotdown} | Friends Shotdown: {friends_shotdown} | "
#                   f"Sucessful Attacks: {sucessful_attacks:3d} | IP Health: {interest_point_health}")
        
#         # Print final statistics for the episode.
#         print(f"\n\nFINAL: Air Traffic Env - Episode: {episode:3d} | Steps: {n_steps:4d} | Acc. Reward: {accum_reward:7.3f} | "
#               f"Enemies Shotdown: {enemies_shotdown} | Friends Shotdown: {friends_shotdown} | "
#               f"Sucessful Attacks: {sucessful_attacks:3d} | IP Health: {interest_point_health}\n\n")
        
#     # Close the environment after all episodes have completed.
#     env.close()

# # -----------------------------------------------------------------------------
# # Entry Point
# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     main()