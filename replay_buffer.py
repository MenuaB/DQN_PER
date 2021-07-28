import numpy as np
import random
from sumtree import create_tree, update, retrieve


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0

    def store(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if self.index >= len(self.buffer):
            self.buffer.append(data)
        else:
            self.buffer[self.index] = data
        self.index = (self.index + 1) % self.buffer_size

    def sample(self, *args):
        sample_indices = [random.randint(0, len(self.buffer) - 1) for _ in range(args[0])]
        return tuple(list(self._encode_sample(indices=sample_indices)) + [np.ones_like(sample_indices), 0])

    def _encode_sample(self, indices):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            data = self.buffer[i]
            state, action, reward, next_state, done = data
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, alpha):
        super().__init__(buffer_size=buffer_size)
        self.alpha = alpha
        self.max_priority = 1.0
        self.base_node, self.leaves = create_tree(np.zeros(shape=self.buffer_size))

    def store(self, *args, **kwargs):
        index = self.index
        super().store(*args, **kwargs)
        update(self.leaves[index], self.max_priority ** self.alpha)

    def _proportional_sample(self, batch_size):
        step = self.base_node.value / batch_size
        indices = []
        priorities = []
        for i in range(batch_size):
            node = retrieve(np.random.random() * step + i * step, self.base_node)
            priorities.append(node.value / self.base_node.value)
            indices.append(node.idx)
        return indices, priorities

    def sample(self, *args):
        batch_size = args[0]
        beta = args[1]

        indices, priorities = self._proportional_sample(batch_size)
        weights = (np.array(priorities) * len(self.buffer)) ** (-beta)
        weights = weights / np.max(weights)
        # print(indices)
        # print(priorities)
        # print(weights)
        # for i in indices:
        # print(self.buffer[i])
        return tuple(list(self._encode_sample(indices=indices)) + [weights, indices])

    def update_priorities(self, new_priorities, indices):
        for p, i in zip(new_priorities, indices):
            update(self.leaves[i], p ** self.alpha)

        self.max_priority = max(self.max_priority, max(new_priorities))


if __name__ == "__main__":
    b = PrioritizedReplayBuffer(128, 1)
    for i in range(15):
        b.store(1, 2, 3, 4, i)
    b.sample(5, 1)
