from typing import List

import numpy as np
import torch

from gflownet.config import Config
import random

class ReplayBuffer(object):
    def __init__(self, cfg: Config, rng: np.random.Generator = None):
        self.capacity = cfg.replay.capacity
        self.warmup = cfg.replay.warmup
        assert self.warmup <= self.capacity, "ReplayBuffer warmup must be smaller than capacity"

        self.buffer: List[tuple] = []
        self.position = 0
        self.rng = rng
        self.priorities = []

    def push(self, *args):
        if len(self.buffer) == 0:
            self._input_size = len(args)
        else:
            assert self._input_size == len(args), "ReplayBuffer input size must be constant"
        # if len(self.buffer) < self.capacity:
        #     self.buffer.append(None)
        # self.buffer[self.position] = args
        # self.position = (self.position + 1) % self.capacity
        traj = args 
        """Stores a transition in the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(args)
            # Assign a default priority for new experiences
            # You might want to adjust the default priority based on your specific needs
            self.priorities.append(0)  
        else:
            # Find the index of the experience with the lowest priority
            min_priority_index = self.priorities.index(min(self.priorities))
            
            # Remove the lowest priority experience
            self.buffer.pop(min_priority_index)
            self.priorities.pop(min_priority_index)
            
            # Add the new experience
            self.buffer.append(args)
            self.priorities.append(0)
            # self.buffer = sorted(self.buffer, key = lambda rew: rew[2])[-self.capacity:] #Adding the traj to the buffer and sorting the buffer based on the traj reward

    def sample(self, batch_size):
        # Sampling based on priorities
        total_priority = sum(self.priorities)
        if total_priority != 0: 
            probs = [p / total_priority for p in self.priorities]
            indices = random.choices(range(len(self.buffer)), weights=probs, k=batch_size)
            out = list(zip(*[self.buffer[idx] for idx in indices]))
        else:
            indices = self.rng.choice(len(self.buffer), batch_size)
            out = list(zip(*[self.buffer[idx] for idx in indices]))
            
        for i in range(len(out)):
            # stack if all elements are numpy arrays or torch tensors
            # (this is much more efficient to send arrays through multiprocessing queues)
            if all([isinstance(x, np.ndarray) for x in out[i]]):
                out[i] = np.stack(out[i], axis=0)
            elif all([isinstance(x, torch.Tensor) for x in out[i]]):
                out[i] = torch.stack(out[i], dim=0)
        return tuple(out), indices
    
    def update_priorities(self, indices, priorities):
        """Updates the priorities of specific experiences."""
        for index, priority in zip(indices, priorities):
            if 0 <= index < len(self.priorities):
                self.priorities[index] = priority
            else:
                print(f"Index {index} is out of bounds for the priority list.")

    def __len__(self):
        return len(self.buffer)
