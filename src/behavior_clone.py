# type: ignore
import os
import sys
import numpy as np
import pygame
import tensorflow as tf

from typing import Dict
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# Initialize pygame if not already initialized
if not pygame.get_init():
    pygame.init()

# Add configuration directory if necessary
current_dir = os.getcwd()
config_dir = os.path.abspath(os.path.join(current_dir, "./src/environment"))
if config_dir not in sys.path:
    sys.path.append(config_dir)

# Project-specific imports
from settings import SIM_WIDTH, SIM_HEIGHT, GRID_HEIGHT, GRID_WIDTH, INTEREST_POINT_CENTER  # Simulation parameters
from utils import generate_sparse_matrix  # Utility function for generating sparse matrices
from FriendDrone import FriendDrone  # FriendDrone class with planning policy

def create_behavior_dataset():
    
    # Generator function that produces state-action pairs
    def generator():
        # For each sample, generate a state and compute the action using the class method
        # Generate a random position within the simulation boundaries
        while True:
            pos = np.array([
                np.random.uniform(0, SIM_WIDTH),
                np.random.uniform(0, SIM_HEIGHT)
            ], dtype=np.float32)
            
            # Generate sparse matrices for intensities and directions
            friend_intensity, friend_direction = generate_sparse_matrix()
            enemy_intensity, enemy_direction = generate_sparse_matrix()
            
            # Organize the state into a dictionary.
            # Convert the Vector2 object to a tuple (x, y) if needed.
            state: Dict = {
                'pos': np.array(pos, dtype=np.float32),
                'friend_intensity': np.array(friend_intensity, dtype=np.float32),
                'enemy_intensity': np.array(enemy_intensity, dtype=np.float32),
                'friend_direction': np.array(friend_direction, dtype=np.float32),
                'enemy_direction': np.array(enemy_direction, dtype=np.float32)
            }
            
            # Compute the action using the class method of planning policy
            action = FriendDrone.planning_policy(state)
            # Convert the action to a NumPy array with float32 type if necessary
            action = np.array(action, dtype=np.float32)
            
            yield state, action
        
    # Create a tf.data.Dataset from the generator function.
    # The output_signature defines the expected shapes and data types.
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            {
                'pos': tf.TensorSpec(shape=(2,), dtype=tf.float32),
                'friend_intensity': tf.TensorSpec(shape=(GRID_WIDTH, GRID_HEIGHT), dtype=tf.float32),
                'enemy_intensity': tf.TensorSpec(shape=(GRID_WIDTH, GRID_HEIGHT), dtype=tf.float32),
                'friend_direction': tf.TensorSpec(shape=(GRID_WIDTH, GRID_HEIGHT, 2), dtype=tf.float32),
                'enemy_direction': tf.TensorSpec(shape=(GRID_WIDTH, GRID_HEIGHT, 2), dtype=tf.float32)
            },
            tf.TensorSpec(shape=(2,), dtype=tf.float32)
        )
    )

    # Configure batching and prefetching for performance
    dataset = dataset.batch(128).prefetch(tf.data.AUTOTUNE)
    
    return dataset

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Concatenate, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_model():
    # Input para posição (vetor de 2 elementos)
    pos_input = Input(shape=(2,), name='pos')

    # Inputs para os dados de grid
    # friend_intensity e enemy_intensity são originalmente 2D, vamos expandir para incluir canal único.
    friend_intensity_input = Input(shape=(GRID_WIDTH, GRID_HEIGHT), name='friend_intensity')
    enemy_intensity_input = Input(shape=(GRID_WIDTH, GRID_HEIGHT), name='enemy_intensity')
    # Os inputs de direção já possuem canal (2) e são 3D
    friend_direction_input = Input(shape=(GRID_WIDTH, GRID_HEIGHT, 2), name='friend_direction')
    enemy_direction_input = Input(shape=(GRID_WIDTH, GRID_HEIGHT, 2), name='enemy_direction')
    
    # Adiciona dimensão de canal para os inputs de intensidade
    friend_intensity_reshaped = Reshape((GRID_WIDTH, GRID_HEIGHT, 1))(friend_intensity_input)
    enemy_intensity_reshaped = Reshape((GRID_WIDTH, GRID_HEIGHT, 1))(enemy_intensity_input)
    
    # Branch para friend_intensity
    fi = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(friend_intensity_reshaped)
    fi = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(fi)
    fi = MaxPooling2D(pool_size=(2,2))(fi)
    
    fi = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(fi)
    fi = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(fi)
    fi = MaxPooling2D(pool_size=(2,2))(fi)
    
    fi = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(fi)
    fi = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(fi)
    fi = MaxPooling2D(pool_size=(2,2))(fi)
    
    fi = Flatten()(fi)
    
    # Branch para enemy_intensity
    ei = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(enemy_intensity_reshaped)
    ei = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(ei)
    ei = MaxPooling2D(pool_size=(2,2))(ei)
    
    ei = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(ei)
    ei = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(ei)
    ei = MaxPooling2D(pool_size=(2,2))(ei)
    
    ei = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(ei)
    ei = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(ei)
    ei = MaxPooling2D(pool_size=(2,2))(ei)
    
    ei = Flatten()(ei)
    
    # Branch para friend_direction
    fd = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(friend_direction_input)
    fd = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(fd)
    fd = MaxPooling2D(pool_size=(2,2))(fd)

    fd = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(fd)
    fd = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(fd)
    fd = MaxPooling2D(pool_size=(2,2))(fd)
    
    fd = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(fd)
    fd = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(fd)
    fd = MaxPooling2D(pool_size=(2,2))(fd)

    fd = Flatten()(fd)
    
    # Branch para enemy_direction
    ed = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(enemy_direction_input)
    ed = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(ed)
    ed = MaxPooling2D(pool_size=(2,2))(ed)

    ed = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(ed)
    ed = Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(ed)
    ed = MaxPooling2D(pool_size=(2,2))(ed)
    
    ed = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(ed)
    ed = Conv2D(128, kernel_size=(3,3), activation='relu', padding='same')(ed)
    ed = MaxPooling2D(pool_size=(2,2))(ed)
    
    ed = Flatten()(ed)
    # ed = GlobalAveragePooling2D()(ed)
    
    # Concatena todos os features processados com o input posicional
    concatenated = Concatenate()([pos_input, fi, ei, fd, ed])
    
    # Camadas densas
    x = Dense(2048, activation='relu')(concatenated)
    x = BatchNormalization()(x)
    
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Camada de saída para a ação (vetor de 2 elementos)
    action_output = Dense(2, activation='tanh', name='action')(x)
    
    # Cria e compila o modelo
    model = Model(
        inputs=[
            pos_input, 
            friend_intensity_input, 
            enemy_intensity_input, 
            friend_direction_input, 
            enemy_direction_input
        ],
        outputs=action_output
    )
    
    model.compile(optimizer=Adam(), loss='mse')
    
    return model



import os

from typing import Any, Dict, List, Optional, Tuple
from tensorflow.keras.callbacks import Callback


class CustomModelCheckpoint(Callback):
    """
    Custom callback to save the model when the monitored metric improves.
    It deletes the previous best model file to save disk space.
    """
    def __init__(self, base_path: str, monitor: str = 'val_loss') -> None:
        """
        Args:
            base_path (str): Base file path for saving the model.
            monitor (str): Metric name to monitor.
        """
        super(CustomModelCheckpoint, self).__init__()
        self.base_path: str = base_path
        self.monitor: str = monitor
        self.best: float = float('inf')
        self.last_saved_model: Optional[str] = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        current_loss: Optional[float] = logs.get(self.monitor)
        if current_loss is not None and current_loss < self.best:
            self.best = current_loss
            filepath: str = f"{self.base_path}/best_model_epoch={epoch+1:02d}_{self.monitor}={current_loss:.4f}.keras"
            
            # Remove the previous saved model if it exists
            if self.last_saved_model and os.path.exists(self.last_saved_model):
                os.remove(self.last_saved_model)
            
            # Save the current model
            self.model.save(filepath)
            self.last_saved_model = filepath
            print(f"\n\n\nCheckpoint: New best model saved | {self.monitor} = {self.best}\n\n")
            

    
checkpoint: CustomModelCheckpoint = CustomModelCheckpoint(base_path="./models/", monitor='val_loss')
    
model = build_model()

os.makedirs('./images', exist_ok=True)
plot_model(model, to_file='./images/model_architecture.png', show_shapes=True, show_layer_names=True)

dataset = create_behavior_dataset()

# Display the model summary
model.summary()



# Train the neural network
# Adjust the number of epochs as needed
model.fit(
    dataset,                         # Training dataset
    steps_per_epoch=100,             # Number of steps per epoch
    validation_data=dataset,         # Validation dataset (pode ser diferente do training)
    validation_steps=100,            # Number of validation steps per epoch
    validation_freq=10,              # Number of epochs between validation steps
    epochs=1_000_000,                # Total number of epochs
    callbacks=[checkpoint]           # Custom callback for saving the best model
    )