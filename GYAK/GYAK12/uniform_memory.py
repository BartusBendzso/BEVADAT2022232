import random
import numpy as np
from collections import namedtuple


class MemoryBuffer:
    transition = namedtuple("t", ("state", "action", "reward", "next_state", "done"))

    def __init__(self, memory_size=10000):
        self.memory_size = memory_size
        self.memory = []

    def push(self, state, action, reward, next_state, done):
        experience = self.transition(state, action, reward, next_state, done)
        if len(self.memory) > self.memory_size:
            delete_index = np.random.randint(0, len(self.memory), 1)[0]
            self.memory.pop(delete_index)
        self.memory.append(experience)

    def sample(self, batch_size):
        if len(self.memory) > batch_size:
            return random.sample(list(self.memory), batch_size)
        else:
            return self.memory
