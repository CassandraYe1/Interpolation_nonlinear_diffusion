import torch
import torch.nn as nn


class DeepNN(nn.Module):
    """
    Set the neural network model
    """
    
    def __init__(self):
        super().__init__()
        self.InpLayer = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU())
        self.HiddenLayer1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU())
        self.HiddenLayer2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU())
        self.OutLayer = nn.Sequential(
            nn.Linear(512,2),
        )
        
    def forward(self, X, Z):
        H = self.InpLayer(X)
        H = self.HiddenLayer1(H)
        H = self.HiddenLayer2(H)
        out = self.OutLayer(H)
        out = out[:,[0]] * Z + out[:,[1]] * (~Z)
        return out