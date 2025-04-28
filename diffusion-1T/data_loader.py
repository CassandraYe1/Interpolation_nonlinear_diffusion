import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from typing import Tuple
import os


def load_data(config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    加载精细网格参考解数据
    Load the fine-grid reference solution.

    Args:
        config: 包含以下属性的配置对象 | Configuration object with attributes:
            - model_name: str 模型名称 | Model name
            - device_name: str ('cuda' or 'cpu') 计算设备 | Computing device
            - Nx: int X轴网格点数 | Grid points on X-axis
            - Ny: int Y轴网格点数 | Grid points on Y-axis

    Returns:
        D_ref  : D^n     [cfg.Nx, cfg.Ny] 当前扩散系数 | Current diffusion coefficient
        E_prev : E^{n-1} [cfg.Nx, cfg.Ny] 前一时刻电场 | Previous electric field
        E_ref  : E^n     [cfg.Nx, cfg.Ny] 当前参考电场 | Current reference electric field
        X      : [cfg.Nx, cfg.Ny] X轴网格坐标 | Grid points on X-axis
        Y      : [cfg.Nx, cfg.Ny] Y轴网格坐标 | Grid points on Y-axis
    """
    # 获取项目根目录和数据路径 | Get project root and data path
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, 'data', config.model_name)

    # 加载numpy格式的参考解数据 | Load reference solution data in numpy format
    sol = np.load(os.path.join(data_path, 'sol.npy'))
    kappa = np.load(os.path.join(data_path, 'kappa.npy'))
    X = np.load(os.path.join(data_path, 'X.npy'))
    Y = np.load(os.path.join(data_path, 'Y.npy'))

    # 将数据转换为PyTorch张量并转移到GPU | Convert data to PyTorch tensors and move to GPU
    D_ref = torch.tensor(kappa[-1]).cuda()
    E_prev = torch.tensor(sol[-2]).cuda()
    E_ref = torch.tensor(sol[-1]).cuda()
    X = torch.tensor(X).requires_grad_().cuda()
    Y = torch.tensor(Y).requires_grad_().cuda()

    return D_ref, E_prev, E_ref, X, Y


def z_const(X, Y, config) -> torch.Tensor:
    """
    生成常数电离函数，并以2为阈值转换为布尔值
    Generate constant ionization function and convert to Boolean values with threshold 2

    Args:
        X: [cfg.Nx, cfg.Ny] X轴网格坐标 | Grid points on X-axis
        Y: [cfg.Nx, cfg.Ny] Y轴网格坐标 | Grid points on Y-axis
        config: 需要包含 | Configuration should contain:
            - Nx: int X轴网格点数 | Grid points in X direction
            - Ny: int Y轴网格点数 | Grid points in Y direction
            - device_name: str 计算设备 | Computing device

    Returns:
        Z: [cfg.Nx, cfg.Ny] "zconst"类型的布尔电离函数 | Boolean ionization function of type "zconst"
    """
    Z = torch.ones(config.Nx, config.Ny)
    Z = (Z>2).cuda()

    return Z


def z_line(X, Y, config) -> torch.Tensor:
    """
    生成间断线性电离函数，并以2为阈值转换为布尔值
    Generate intermittent linear ionization function and convert to Boolean values with threshold 2

    Args:
        X: [cfg.Nx, cfg.Ny] X轴网格坐标 | Grid points on X-axis
        Y: [cfg.Nx, cfg.Ny] Y轴网格坐标 | Grid points on Y-axis
        config: 配置对象 | Configuration object

    Returns:
        Z: [cfg.Nx, cfg.Ny] "zline"类型的布尔电离函数 | Boolean ionization function of type "zline"
    """
    Z = torch.zeros(config.Nx, config.Ny)
    for i in range(config.Nx):
        for j in range(config.Nx):
            Z[i,j] = (X[i,j]<1./2.)*10.0 + (X[i,j]>=1./2.)*1.0
    Z = (Z>2).cuda()

    return Z


def z_square(X, Y, config) -> torch.Tensor:
    """
    生成双方形电离函数，并以2为阈值转换为布尔值
    Generate two-squares ionization function and convert to Boolean values with threshold 2

    Args:
        X: [cfg.Nx, cfg.Ny] X轴网格坐标 | Grid points on X-axis
        Y: [cfg.Nx, cfg.Ny] Y轴网格坐标 | Grid points on Y-axis
        config: 配置对象 | Configuration object

    Returns:
        Z: [cfg.Nx, cfg.Ny] "zsquare"类型的布尔电离函数 | Boolean ionization function of type "zsquare"
    """
    ax, ay, bx, by = 3., 9., 9., 3.
    Z = torch.zeros(config.Nx, config.Ny)
    for i in range(config.Nx):
        for j in range(config.Nx):
            Z[i,j] = (X[i,j]<(ax+4.)/16.)*(X[i,j]>ax/16.0)*(Y[i,j]<(ay+4.)/16.)*(Y[i,j]>ay/16.0)*9.0 + \
                     (X[i,j]<(bx+4.)/16.)*(X[i,j]>bx/16.0)*(Y[i,j]<(by+4.)/16.)*(Y[i,j]>by/16.0)*9.0 + 1.0
    Z = (Z>2).cuda()

    return Z
