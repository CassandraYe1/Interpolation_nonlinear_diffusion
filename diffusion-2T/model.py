import torch
import torch.nn as nn


class DeepNN(nn.Module):
    """
    Deep Neural Network model definition.
    
    Architecture:
        Input layer   : 2D   -> 512D + ReLU activation
        Hidden layer 1: 512D -> 512D + ReLU activation
        Hidden layer 2: 512D -> 512D + ReLU activation 
        Output layer  : 512D -> 2D
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
        """
        Forward propagation process.
        
        Args:
            X: [N, 2] Input feature tensor
            Z: [N, 1] Boolean mask tensor used for channel selection
            
        Returns:
            out: [N, 1] Output tensor
        """
        
        H = self.InpLayer(X)
        H = self.HiddenLayer1(H)
        H = self.HiddenLayer2(H)
        out = self.OutLayer(H)
        out = out[:,[0]] * Z + out[:,[1]] * (~Z)
        return out
        
