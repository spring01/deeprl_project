
import random
import cPickle as pickle

class ReplayMemory(object):

    def __init__(self, max_size, window):
        self.maxlen = int(max_size / window)
        self.clear()

    def append(self, mem):
        self.ring_buffer[self.index] = mem
        self.index = (self.index + 1) % self.maxlen
        self.length = min(self.length + 1, self.maxlen)

    def sample(self, batch_size):
        idx = random.sample(xrange(self.length), batch_size)
        return [self.ring_buffer[i] for i in idx]

    def clear(self):
        self.ring_buffer = [None for _ in xrange(self.maxlen)]
        self.index = 0
        self.length = 0

    def __len__(self):
        return self.length

    def save(self, filepath):
        with open(filepath, 'wb') as save:
            pickle.dump(self, save, protocol=pickle.HIGHEST_PROTOCOL)
