import random
from collections import namedtuple, deque

# 경험을 저장할 namedtuple 정의
Transition = namedtuple('Transition', ('heatmap', 'depthmap', 'action', 'reward', 
                                       'next_heatmap', 'next_depthmap', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
