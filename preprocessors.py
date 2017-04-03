
import numpy as np
from PIL import Image


class AtariPreprocessor(object):

    def __init__(self, new_size):
        self.new_size = new_size

    def obs_to_obs_mem(self, state):
        img = Image.fromarray(state)
        img = img.resize(self.new_size)
        img = img.convert('L')
        return np.asarray(img)

    def state_mem_to_state(self, state_mem):
        return state_mem.astype(np.float32)

    def clip_reward(self, reward):
        if reward > 0.0:
            return 1.0;
        elif reward < 0.0:
            return -1.0
        else:
            return 0.0
