import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.preprocessing import is_image_space

class ObservationReshapeWrapper:
    def __init__(self, model):
        self.model = model
        
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        # Ensure observation is a numpy array
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
            
        # Handle NaN values
        observation = np.nan_to_num(observation, nan=0.0)
        
        # Normalize the data to prevent extreme values
        observation = np.clip(observation, -10, 10)
        
        # Reshape observation if needed
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)
            
        # Convert to the expected shape (1, 20, 16)
        if observation.shape[1] != 320:  # 20 * 16 = 320
            # Pad or truncate to get 320 elements
            flat_obs = observation.flatten()
            if flat_obs.size < 320:
                pad_size = 320 - flat_obs.size
                flat_obs = np.pad(flat_obs, (0, pad_size), 'constant', constant_values=0)
            else:
                flat_obs = flat_obs[:320]
            observation = flat_obs.reshape(1, 20, 16)
            
        # Double-check for NaN values after reshaping
        observation = np.nan_to_num(observation, nan=0.0)
        
        try:
            return self.model.predict(observation, deterministic=deterministic)
        except Exception as e:
            print(f"Error in prediction: {e}")
            print(f"Observation shape: {observation.shape}")
            print(f"Observation contains NaN: {np.isnan(observation).any()}")
            print(f"Observation min/max: {np.min(observation)}/{np.max(observation)}")
            raise 