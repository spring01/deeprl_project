
import numpy as np

class History(object):

    def __init__(self, state, timesteps):
        self.seq_state = [state for _ in xrange(timesteps)]

    def append(self, state):
        self.seq_state.append(state)
        self.seq_state.pop(0)

    def get_seq_state(self):
        return np.stack([np.stack(self.seq_state)])
