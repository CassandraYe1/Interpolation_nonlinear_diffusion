import torch
import torch.nn as nn
from config import Config


class DeepNN(nn.Module):
    """
    深度神经网络模型定义
    Deep Neural Network model definition
    
    网络架构 | Architecture:
        [2, cfg.width, cfg.width, cfg.width, 2], with ReLU activation functions
    """
    
    def __init__(self, cfg: Config):
        """
        初始化神经网络模型
        Initialize the neural network model
        
        Args:
            cfg: 包含以下属性的配置对象 | Configuration object with attributes:
                - width: int 隐藏层单元数 | Number of units in each hidden layer
        """
        super().__init__()

        # 输入层定义 | Input layer definition
        self.InpLayer = nn.Sequential(
            nn.Linear(2, cfg.width),
            nn.ReLU())
        # 第一个隐藏层 | First hidden layer
        self.HiddenLayer1 = nn.Sequential(
            nn.Linear(cfg.width, cfg.width),
            nn.ReLU())
        # 第二个隐藏层 | Second hidden layer
        self.HiddenLayer2 = nn.Sequential(
            nn.Linear(cfg.width, cfg.width),
            nn.ReLU())
        # 输出层 (双通道) | Output layer (dual-channel)
        self.OutLayer = nn.Sequential(
            nn.Linear(cfg.width, 2),
        )
        
    def forward(self, X, Z):
        """
        前向传播过程
        Forward propagation process.
        
        Args:
            X: [cfg.Nx*cfg.Ny, 2] 输入特征张量 (通常是XY坐标)
               [cfg.Nx*cfg.Ny, 2] Input feature tensor (typically XY coordinates)
            Z: [cfg.Nx*cfg.Ny, 1] 布尔掩码张量，用于通道选择
               [cfg.Nx*cfg.Ny, 1] Boolean mask tensor for channel selection
               
        Returns:
            out: [cfg.Nx*cfg.Ny, 1] 输出张量
                 [cfg.Nx*cfg.Ny, 1] Output tensor
        """
        H = self.InpLayer(X)
        H = self.HiddenLayer1(H)
        H = self.HiddenLayer2(H)
        out = self.OutLayer(H)

        # 通道选择: 根据Z的值选择输出通道
        # Channel selection: choose output channel based on Z values
        out = out[:,[0]] * Z + out[:,[1]] * (~Z)
        return out
        