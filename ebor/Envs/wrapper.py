import gym
import numpy as np
from ebor.Envs.utils import load_dataset

# load expert data for RCE
class LoadDataset(gym.Wrapper):
    def __init__(self, env, exp_data):
        super(LoadDataset, self).__init__(env)
        # load expert data for RCE
        if exp_data:
            self.exp_data =load_dataset(f'../ExpertDatasets/{exp_data}.pth')
        else:
            self.exp_data = None

    def get_dataset(self, num_obs=256):  # optional
        """
        to fit the need of RCE baseline
        """
        # sample a batch of random actions
        action_vec = [self.sample_action() for _ in range(num_obs)]
        # sample a batch of example states
        ind = np.random.randint(0, len(self.exp_data), size=(num_obs,))
        obs_vec = self.exp_data[ind]
        dataset = {
            'observations': np.array(obs_vec, dtype=np.float32),
            'actions': np.array(action_vec, dtype=np.float32),
            'rewards': np.zeros(num_obs, dtype=np.float32),
        }
        return dataset