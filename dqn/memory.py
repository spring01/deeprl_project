
import random
import cPickle as pickle

class ReplayMemory(object):

    def __init__(self, max_size, window):
        self.maxlen = int(max_size / window)
        self.clear()

    def append(self, mem):
        self.ring_buffer.append(mem)
        if len(self.ring_buffer) > self.maxlen:
            self.ring_buffer.pop(0)

    def sample(self, batch_size):
        return random.sample(self.ring_buffer, batch_size)

    def clear(self):
        self.ring_buffer = []

    def length(self):
        return len(self.ring_buffer)

    def save(self, filepath):
        with open(filepath, 'wb') as save:
            pickle.dump(self, save, protocol=pickle.HIGHEST_PROTOCOL)
