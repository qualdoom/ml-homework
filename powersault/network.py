import torch

import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, num_channels, height, width, n_actions):
        super(NeuralNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=(8, 8), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(4, 4), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1))
        self.pool = nn.MaxPool2d((2, 2))

        self.fc_size = self.compute_fc_size(num_channels, height, width)

        self.fc1 = nn.Linear(self.fc_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, n_actions)
        
    def compute_fc_size(self, num_channels, height, width):
        # Применение сверточных и пулинг слоев для вычисления размера входа
        x = torch.rand(1, num_channels, height, width)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))

        # Вычисление размера входа для полносвязанного слоя
        fc_size = x.view(1, -1).size(1)
        print("Size of layer after convolution layers", fc_size)
        return fc_size
    
    def forward(self, x):

        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))

        x = x.view(-1, self.fc_size)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x