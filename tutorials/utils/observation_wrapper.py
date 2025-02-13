import numpy as np
from gym import ObservationWrapper
import gym

class ObservationReshapeWrapper:
    def __init__(self, model, model_type="portfolio"):
        """
        Initialize wrapper with model type specification
        model_type: "portfolio" or "fundamental"
        """
        self.model = model
        self.model_type = model_type
        
        # Define observation spaces for each model type
        if model_type == "fundamental":
            self.observation_space = gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(511,), 
                dtype=np.float32
            )
        else:  # portfolio model
            self.observation_space = gym.spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(20, 16), 
                dtype=np.float32
            )
        
    def predict(self, observation, *args, **kwargs):
        """Reshape observation based on model type."""
        if self.model_type == "fundamental":
            # For fundamental trading model, reshape from (47,) to (511,)
            if isinstance(observation, np.ndarray) and observation.shape == (47,):
                padded_obs = np.pad(observation, (0, 511 - 47), 'constant')
                return self.model.predict(padded_obs, *args, **kwargs)
        else:  # portfolio model
            # For portfolio model, reshape from (47,) to (20, 16)
            if isinstance(observation, np.ndarray) and observation.shape == (47,):
                padded_obs = np.pad(observation, (0, 320 - 47), 'constant')
                reshaped_obs = padded_obs.reshape(20, 16)
                return self.model.predict(reshaped_obs, *args, **kwargs)
        return self.model.predict(observation, *args, **kwargs) 