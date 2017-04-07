
import random
import numpy as np
import cPickle as pickle


class Episode(object):

    def __init__(self, init_state_mem):
        self.steps = [(init_state_mem,)]

    def append(self, state_mem_next, act, reward, done):
        step = state_mem_next, act, reward, done
        self.steps.append(step)

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, key):
        return self.steps[key]


class ReplayMemoryEpisode(object):

    def __init__(self, max_size, window):
        self.maxlen = int(max_size / window)
        self.clear()

    def append(self, episode):
        self.episodes.append(episode)
        while len(self) > self.maxlen:
            self.episodes.pop(0)

    def sample_batch(self, timesteps, batch_size):
        mini_batch = []
        for _ in xrange(batch_size):
            mini_batch.append(self.sample_seq(timesteps))
        return mini_batch

    def sample_seq(self, timesteps):
        episode = random.sample(self.episodes, 1)[0]
        pivot = random.randint(0, len(episode) - timesteps - 1)
        seq_state_mem = []
        seq_state_mem_next = []
        seq_reward = 0.0
        seq_done = False
        state_mem = episode[pivot][0]
        for i in xrange(timesteps):
            state_mem_next, last_act, reward, done = episode[pivot + i + 1]
            seq_state_mem.append(state_mem)
            seq_state_mem_next.append(state_mem_next)
            seq_reward += reward
            seq_done = seq_done or done
            state_mem = state_mem_next
        seq_state_mem = np.stack(seq_state_mem)
        seq_state_mem_next = np.stack(seq_state_mem_next)
        return seq_state_mem, last_act, seq_reward, seq_state_mem_next, seq_done

    def clear(self):
        self.episodes = []

    def __len__(self):
        return sum(len(ep) for ep in self.episodes)

    def save(self, filepath):
        with open(filepath, 'wb') as save:
            pickle.dump(self, save, protocol=pickle.HIGHEST_PROTOCOL)

