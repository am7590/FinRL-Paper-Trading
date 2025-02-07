import numpy as np
from gym import ObservationWrapper

class ObservationReshapeWrapper:
    def __init__(self, model):
        self.model = model
        
    def predict(self, observation, *args, **kwargs):
        """Reshape observation to match expected model input shape."""
        # Reshape from (47,) to (20, 16)
        if isinstance(observation, np.ndarray) and observation.shape == (47,):
            # Pad with zeros if needed
            padded_obs = np.pad(observation, (0, 320 - 47), 'constant')
            reshaped_obs = padded_obs.reshape(20, 16)
            return self.model.predict(reshaped_obs, *args, **kwargs)
        return self.model.predict(observation, *args, **kwargs) 