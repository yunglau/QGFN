from typing import List

import numpy as np
import torch
from threading import Lock

from gflownet.config import Config
import random
from gflownet.data.tree import SumTree

class ReplayBuffer(object):
    def __init__(self, cfg: Config, rng: np.random.Generator = None):
        self.capacity = cfg.replay.capacity
        self.warmup = cfg.replay.warmup
        assert self.warmup <= self.capacity, "ReplayBuffer warmup must be smaller than capacity"

        self.buffer: List[tuple] = [(None, None, None, None, None, None)] * self.capacity
        self.position = 0
        self.rng = rng
        # self.priorities = []
        self.eps = 1e-2 # minimal priority, prevents zero probabilities
        self.alpha = 0.9
        self.tree = SumTree(size=self.capacity)
        self.max_priority = self.eps  # priority for new samples, init as eps
        self.count = 0
        
        self.real_size = 0
        self.size = self.capacity
        self.insertion_lock = Lock()
        
        
    def push(self, *args):
        with self.insertion_lock:
            traj = args 
            """Stores a transition in the buffer."""
            # store transition index with maximum priority in sum tree
            self.tree.add(self.max_priority, self.count)

            # store transition in the buffer
            self.buffer[self.count] = args
            # self.priorities[self.count] = 0

            # update counters
            self.count = (self.count + 1) % self.size
            self.real_size = min(self.size, self.real_size + 1)
            
    def sample(self, batch_size): 
        # print(self.real_size)
        # assert self.real_size >= batch_size, "buffer contains less samples than batch size"
        if self.real_size < batch_size:
            batch_size = self.real_size
        # Sampling based on priorities
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        with self.insertion_lock:  # While we're not modifying the SumTree here, if the total is modified while we're 
                                   # sampling, the `cumsum` value won't make sense anymore.
            segment = self.tree.total / batch_size
            for i in range(batch_size):
                a, b = segment * i, segment * (i + 1)

                cumsum = random.uniform(a, b)
                # sample_idx is a sample index in buffer, needed further to sample actual transitions
                # tree_idx is a index of a sample in the tree, needed further to update priorities
                tree_idx, priority, sample_idx = self.tree.get(cumsum)

                priorities[i] = priority
                tree_idxs.append(tree_idx)
                sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total
        
        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        # weights = (self.real_size * probs) ** -self.beta

        # # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        # weights = weights / weights.max()
        
        # print(sample_idxs)
        
        out = list(zip(*[self.buffer[idx] for idx in sample_idxs]))
            
        for i in range(len(out)):
            # stack if all elements are numpy arrays or torch tensors
            # (this is much more efficient to send arrays through multiprocessing queues)
            if all([isinstance(x, np.ndarray) for x in out[i]]):
                out[i] = np.stack(out[i], axis=0)
            elif all([isinstance(x, torch.Tensor) for x in out[i]]):
                out[i] = torch.stack(out[i], dim=0)

        return tuple(out), tree_idxs
    
    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        with self.insertion_lock:
            for data_idx, priority in zip(data_idxs, priorities):
                # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
                # where eps is a small positive constant that prevents the edge-case of transitions not being
                # revisited once their error is zero. (Section 3.3)
                priority = (priority + self.eps) ** self.alpha

                self.tree.update(data_idx, priority)
                self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        # print(self.real_size)
        return self.real_size

# from typing import List

# import numpy as np
# import torch
# import random

# from gflownet.config import Config


# class ReplayBuffer(object):
#     def __init__(self, cfg: Config, rng: np.random.Generator = None):
#         self.capacity = cfg.replay.capacity
#         self.warmup = cfg.replay.warmup
#         assert self.warmup <= self.capacity, "ReplayBuffer warmup must be smaller than capacity"
#         self.method = 'Random'
#         self.priorities = []

#         self.buffer: List[tuple] = []
#         self.position = 0
#         self.rng = rng

#     def push(self, *args):
#         if len(self.buffer) == 0:
#             self._input_size = len(args)
#         else:
#             assert self._input_size == len(args), "ReplayBuffer input size must be constant"
#         if self.method == 'Random':
#             traj = args 
#             if len(self.buffer) < self.capacity:
#                 self.buffer.append(None)
#             self.buffer[self.position] = (traj)
#             self.position = (self.position + 1) % self.capacity
#         elif self.method == 'Prioritized':
#             traj, reward = args 
#             if len(self.buffer) < self.capacity or args[1] > self.buffer[0][1]: #Checking if the current traj reward is greater than the least traj reward in the buffer 
#                 self.buffer.append((traj, reward))
#                 self.buffer = sorted(self.buffer, key = lambda rew: rew[1])[-self.capacity:] #Adding the traj to the buffer and sorting the buffer based on the traj reward
#         elif self.method == 'PER':
#             traj, reward, priority = args 
#             if len(self.buffer) < self.capacity:
#                 self.buffer.append((traj, reward))
#                 self.priorities.append(priority)
#             else:
#                 # Replace the experience with the lowest priority
#                 min_priority_index = min(range(len(self.priorities)), key=lambda idx: self.priorities[idx])
#                 self.buffer[min_priority_index] = (traj, reward)
#                 self.priorities[min_priority_index] = priority


#     def sample(self, batch_size):
#         if self.method == 'Random':
#             idxs = self.rng.choice(len(self.buffer), batch_size)
#             out = list(zip(*[self.buffer[idx] for idx in idxs]))
#             # print("Sample shape: ",np.array(out).shape)
#         elif self.method == 'Prioritized':
#             idxs = self.rng.choice(len(self.buffer), batch_size)
#             out = list(zip(*[self.buffer[idx] for idx in idxs]))
#         elif self.method == 'PER':
#             # Sampling based on priorities
#             total_priority = sum(self.priorities)
#             probs = [p / total_priority for p in self.priorities]
#             indices = random.choices(range(len(self.buffer)), weights=probs, k=batch_size)
#             out = list(zip(*[self.buffer[idx] for idx in indices]))
            
#         for i in range(len(out)):
#             # stack if all elements are numpy arrays or torch tensors
#             # (this is much more efficient to send arrays through multiprocessing queues)
#             if all([isinstance(x, np.ndarray) for x in out[i]]):
#                 out[i] = np.stack(out[i], axis=0)
#             elif all([isinstance(x, torch.Tensor) for x in out[i]]):
#                 out[i] = torch.stack(out[i], dim=0)
#         return tuple(out), idxs

#     def __len__(self):
#         return len(self.buffer)