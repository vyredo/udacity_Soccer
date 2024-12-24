import numpy as np
import torch


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size, alpha):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.position = 0

    def add(self, experience, td_error):
        priority = (abs(td_error) + 1e-5) ** self.alpha
        if len(self.memory) < self.buffer_size:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, beta=0.4):
        if len(self.memory) == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        priorities = self.priorities[:len(self.memory)]
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(
            len(self.memory), self.batch_size, p=probabilities)
        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        experiences = [self.memory[i] for i in indices]
        return experiences, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-5) ** self.alpha

    def __len__(self):
        return len(self.memory)
