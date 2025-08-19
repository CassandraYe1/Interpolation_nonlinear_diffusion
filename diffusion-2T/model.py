import torch
import torch.nn as nn
from config import Config


class DeepNN(nn.Module):
    """
    深度神经网络模型定义
    Deep Neural Network model definition
    
    网络架构 | Architecture:
        [2, cfg.width, ..., cfg.width, 2], with ReLU activation functions
    """
    
    def __init__(self, cfg: Config):
        """
        初始化神经网络模型
        Initialize the neural network model
        
        Args:
            cfg: 包含以下属性的配置对象 | Configuration object with attributes:
                - depth: int 隐藏层层数 | Number of hidden layers
                - width: int 隐藏层单元数 | Number of units in each hidden layer
        """
        super().__init__()

        # 输入层定义 | Input layer definition
        self.InpLayer = nn.Sequential(
            nn.Linear(2, cfg.width),
            nn.ReLU())
        # 隐藏层定义 | Hidden layer defination
        self.HiddenLayers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.width, cfg.width),
                nn.ReLU()
            ) for _ in range(cfg.depth)
        ])
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
        for layer in self.HiddenLayers:
            H = layer(H)
        out = self.OutLayer(H)

        # 通道选择: 根据Z的值选择输出通道
        # Channel selection: choose output channel based on Z values
        out = out[:,[0]] * Z + out[:,[1]] * (~Z)
        return out


class MultiRegionModel(nn.Module):
    """
    管理四个区域的模型
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.regions = ["top_left", "top_right", "bottom_left", "bottom_right"]
        self.models = nn.ModuleDict({
            region: DeepNN(cfg) for region in self.regions
        })
        self.cfgs = {
            region: Config(region) for region in self.regions
        }
        
        # 初始化各区域配置
        for region in self.regions:
            self.cfgs[region].init_config()
    
    def forward(self, inputs):
        outputs = {}
        for region in self.regions:
            # 获取对应区域的输入
            x, z = inputs[region]
            # 通过对应区域的模型
            outputs[region] = self.models[region](x, z)
        return outputs
    
    def train_step(self, optimizer, closure):
        """
        执行一个训练步骤
        """
        total_loss = 0.0
        optimizer.zero_grad()
        
        # 计算所有区域的损失
        for region in self.regions:
            loss = closure(region)
            total_loss += loss
        
        # 反向传播和优化
        total_loss.backward()
        optimizer.step()
        
        return total_loss
    
    def get_results(self):
        """
        获取合并后的结果
        """
        full_size = 257
        full_E_reg = torch.zeros((full_size, full_size), device=self.cfgs["top_left"].device_name)
        full_E_pinn = torch.zeros_like(full_E_reg)
        count = torch.zeros_like(full_E_reg)
        
        for region in self.regions:
            cfg = self.cfgs[region]
            x_start, x_end, y_start, y_end = cfg.region_boundaries[region]
            
            # 获取预测结果
            E_reg = self.models[region](cfg.inp_fine, cfg.Z_fine_bool).detach().reshape(cfg.Nx, cfg.Ny)
            E_pinn = E_reg.clone()  # 简化为相同，实际中应为第二阶段结果
            
            # 调整尺寸以匹配完整网格
            if region in ["top_right", "bottom_right"]:
                E_reg = E_reg[:, 1:]
                E_pinn = E_pinn[:, 1:]
            if region in ["bottom_left", "bottom_right"]:
                E_reg = E_reg[1:, :]
                E_pinn = E_pinn[1:, :]
            
            # 累加到完整网格
            x_len, y_len = E_reg.shape
            full_E_reg[x_start:x_start+x_len, y_start:y_start+y_len] += E_reg
            full_E_pinn[x_start:x_start+x_len, y_start:y_start+y_len] += E_pinn
            count[x_start:x_start+x_len, y_start:y_start+y_len] += 1
        
        # 计算平均值
        full_E_reg /= count
        full_E_pinn /= count
        
        return full_E_reg, full_E_pinn
        