import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from typing import Tuple
import os


def load_data(config) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        D_ref, K_ref   : D^n, K^n         [cfg.Nx, cfg.Ny] 当前扩散系数 | Current diffusion coefficient
        E_prev, T_prev : E^{n-1}, T^{n-1} [cfg.Nx, cfg.Ny] 前一时刻电场 | Previous electric field
        E_ref, T_ref   : E^n, T^n         [cfg.Nx, cfg.Ny] 当前参考电场 | Current reference electric field
        sigma_ref      : [cfg.Nx, cfg.Ny] sigma^n * ((T^n)**4 - (E^n)) [cfg.Nx, cfg.Ny]
        X      : [cfg.Nx, cfg.Ny] X轴网格坐标 | Grid points on X-axis
        Y      : [cfg.Nx, cfg.Ny] Y轴网格坐标 | Grid points on Y-axis
    """
    # 获取项目根目录和数据路径 | Get project root and data path
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, 'data', config.model_name)

    # 加载numpy格式的参考解数据 | Load reference solution data in numpy format
    sol_E = np.load(os.path.join(data_path, 'sol_E.npy'))
    sol_T = np.load(os.path.join(data_path, 'sol_T.npy'))
    kappa_E = np.load(os.path.join(data_path, 'kappa_E.npy'))
    kappa_T = np.load(os.path.join(data_path, 'kappa_T.npy'))
    sigma = np.load(os.path.join(data_path, 'sigma.npy'))
    X = np.load(os.path.join(data_path, 'X.npy'))
    Y = np.load(os.path.join(data_path, 'Y.npy'))

    # 将数据转换为PyTorch张量并转移到GPU | Convert data to PyTorch tensors and move to GPU
    D_ref = torch.tensor(kappa_E[-1]).cuda()
    K_ref = torch.tensor(kappa_T[-1]).cuda()
    E_prev = torch.tensor(sol_E[-2]).cuda()
    E_ref = torch.tensor(sol_E[-1]).cuda()
    T_prev = torch.tensor(sol_T[-2]).cuda()
    T_ref = torch.tensor(sol_T[-1]).cuda()
    sigma_ref = torch.tensor(sigma[-1] * (sol_T[-1]**4 - sol_E[-1])).cuda()
    X = torch.tensor(X).requires_grad_().cuda()
    Y = torch.tensor(Y).requires_grad_().cuda()

    return D_ref, K_ref, E_prev, E_ref, T_prev, T_ref, sigma_ref, X, Y


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
