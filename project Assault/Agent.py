from Network import NeuralNetwork

import torch

from functorch import vmap
import torchvision.transforms as transforms 
import numpy as np

class Agent:
    
    def __init__(self):
        pass
    
    def build_model(self, num_channels, height, width, n_actions):
        model = NeuralNetwork(num_channels=num_channels, height=height, width=width, n_actions=n_actions)
        self.n_actions = n_actions
        # self.buffer = PrioritizedReplayBuffer(50000)
        return model
    
    def __init__(self, num_channels, height, width, n_actions):
        self.model = self.build_model(num_channels=num_channels, height=height, width=width, n_actions=n_actions)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)