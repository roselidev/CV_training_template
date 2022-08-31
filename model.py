import torch
import torch.nn as nn
from torch.nn import functional as F

class CustomModel(nn.Module):
    def __init__(self, image_size, hidden_size_1, hidden_size_2, latent_size):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(image_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)

    def forward(self, x):
        x = self.fc1(x)
        out = self.fc2(x)
        return out