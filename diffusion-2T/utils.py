import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import random
import os
import matplotlib.pyplot as plt
import matplotlib as mpl


def set_seed(seed=42):
    """
    设置所有相关的随机数种子以确保可重复性
    Set all relevant random number seeds for reproducibility
    
    Args:
        seed: 随机种子值，默认为42
               Random seed value, default is 42
    """
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def relative_l2(Eref, Epred):
    """
    计算预测解与参考解之间的相对L2误差
    Calculate relative L2 error between predicted and reference solutions
    
    Args:
        Eref:  [cfg.Nx*cfg.Ny,] 参考解向量/张量
               [cfg.Nx*cfg.Ny,] Reference solution vector/tensor
        Epred: [cfg.Nx*cfg.Ny,] 预测解向量/张量 
               [cfg.Nx*cfg.Ny,] Predicted solution vector/tensor
    
    Returns:
        float: 相对L2误差值 | Relative L2 error value
    """
    return ((Eref - Epred)**2).mean() / (Eref**2).mean()


def pde_res(E, T, D, K, E_, T_, sigma, X, dt):
    """
    计算非线性辐射扩散方程的PDE残差
    Calculate PDE residual for nonlinear radiation diffusion equation
    
    Args:
        E, T  : [cfg.Nx*cfg.Ny, 1] 当前时刻解 | [cfg.Nx*cfg.Ny, 1] Current time step solution
        D, K  : [cfg.Nx*cfg.Ny, 1] 扩散系数 | [cfg.Nx*cfg.Ny, 1] Diffusion coefficient
        E_, T_: [cfg.Nx*cfg.Ny, 1] 前一时刻解 | [cfg.Nx*cfg.Ny, 1] Previous time step solution
        sigma : [cfg.Nx*cfg.Ny, 1] sigma^n * ((T^n)**4 - (E^n))
        X:  [cfg.Nx*cfg.Ny, 2] 输入坐标(需要梯度) | [cfg.Nx*cfg.Ny, 2] Input coordinates (requires grad)
        dt: 时间步长 | Time step size
    
    Returns:
        [float, float]: 关于E和T的PDE残差的均值 | Mean value of PDE residual E and T
    """
    ones = torch.ones_like(E)
    # 计算一阶梯度 | Calculate first-order gradients
    Egrad = torch.autograd.grad(E, X, grad_outputs=ones, create_graph=True)[0]
    Ex = Egrad[:,[0]]
    Ey = Egrad[:,[1]]
    Tgrad = torch.autograd.grad(T, X, grad_outputs=ones, create_graph=True)[0]
    Tx = Tgrad[:,[0]]
    Ty = Tgrad[:,[1]]
    # 计算二阶梯度 | Calculate second-order gradients
    Exx = torch.autograd.grad(Ex, X, grad_outputs=ones, create_graph=True)[0][:,[0]]
    Eyy = torch.autograd.grad(Ey, X, grad_outputs=ones, create_graph=True)[0][:,[1]]
    Txx = torch.autograd.grad(Tx, X, grad_outputs=ones, create_graph=True)[0][:,[0]]
    Tyy = torch.autograd.grad(Ty, X, grad_outputs=ones, create_graph=True)[0][:,[1]]

    res_E = (((Exx + Eyy) * D * dt + E_ - E)**2) + sigma * dt 
    res_T = (((Txx + Tyy) * K * dt + T_ - T)**2) - sigma * dt 

    return [torch.abs(res_E.mean()), torch.abs(res_T.mean())]


def plot_E(E_reg, E_pinn, E_ref, X, Y, data_path, vmax_E=None):
    """
    绘制关于E的回归解和PINN解的对比图
    Plot comparison of regression and PINN solutions of E
    
    Args:
        E_reg:  [cfg.Nx, cfg.Ny] 回归解 | [cfg.Nx, cfg.Ny] Regression solution
        E_pinn: [cfg.Nx, cfg.Ny] PINN解 | [cfg.Nx, cfg.Ny] PINN solution
        E_ref:  [cfg.Nx, cfg.Ny] 参考解 | [cfg.Nx, cfg.Ny] Reference solution
        X:      [cfg.Nx, cfg.Ny] X坐标网格 | [cfg.Nx, cfg.Ny] X coordinate grid
        Y:      [cfg.Nx, cfg.Ny] Y坐标网格 | [cfg.Nx, cfg.Ny] Y coordinate grid
        data_path: 图像保存路径 | Path to save figure
        vmax_E: 误差图的最大值 | Maximum value for error plots
    """
    # 设置matplotlib绘图参数 | Set matplotlib plotting parameters
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9

    # 创建2x2子图 | Create 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 7), layout='constrained', 
                            sharex=True, sharey=True)
    # 设置参考解的颜色范围 | Set color range based on reference solution
    vmin_ref = E_ref.min()
    vmax_ref = E_ref.max()
    cbar_kw = {'fraction': 0.046, 'pad': 0.04}

    # 子图1: 回归解 | Subplot 1: Regression Solution
    pcm1 = axs[0,0].pcolormesh(X, Y, E_reg, vmin=vmin_ref, vmax=vmax_ref, cmap='jet', shading='auto')
    axs[0,0].set_title("(a) Regression Solution", pad=12)
    axs[0,0].set_xlabel("x")
    axs[0,0].set_ylabel("y")
    axs[0,0].grid(True, linestyle=':', alpha=0.6)
    fig.colorbar(pcm1, ax=axs[0,0], **cbar_kw)

    # 子图2: 回归误差 | Subplot 2: Regression Error
    pcm2 = axs[0,1].pcolormesh(X, Y, np.abs(E_ref - E_reg), vmin=0, vmax=vmax_E, cmap='jet', shading='auto')
    axs[0,1].set_title("(b) Regression Error", pad=12)
    axs[0,1].set_xlabel("x")
    axs[0,1].set_ylabel("y")
    axs[0,1].grid(True, linestyle=':', alpha=0.6)
    fig.colorbar(pcm2, ax=axs[0,1], **cbar_kw)

    # 子图3: PINN解 | Subplot 3: PINN Solution
    pcm3 = axs[1,0].pcolormesh(X, Y, E_pinn, vmin=vmin_ref, vmax=vmax_ref, cmap='jet', shading='auto')
    axs[1,0].set_title("(c) PINN Solution", pad=12)
    axs[1,0].set_xlabel("x")
    axs[1,0].set_ylabel("y")
    axs[1,0].grid(True, linestyle=':', alpha=0.6)
    fig.colorbar(pcm3, ax=axs[1,0], **cbar_kw)

    # 子图4: PINN误差 | Subplot 4: PINN Error
    pcm4 = axs[1,1].pcolormesh(X, Y, np.abs(E_ref - E_pinn), vmin=0, vmax=vmax_E, cmap='jet', shading='auto')
    axs[1,1].set_title("(d) PINN Error", pad=12)
    axs[1,1].set_xlabel("x")
    axs[1,1].set_ylabel("y")
    axs[1,1].grid(True, linestyle=':', alpha=0.6)
    fig.colorbar(pcm4, ax=axs[1,1], **cbar_kw)

    # 添加总标题和调整布局 | Add overall title and adjust layout
    fig.suptitle("Comparison of Regression and PINN E-Solutions", y=1.04, fontsize=15)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.savefig(os.path.join(data_path, 'fig_E.png'), dpi=300, bbox_inches='tight')


def plot_T(T_reg, T_pinn, T_ref, X, Y, data_path, vmax_T=None):
    """
    绘制关于T的回归解和PINN解的对比图
    Plot comparison of regression and PINN solutions of T
    
    Args:
        T_reg:  [cfg.Nx, cfg.Ny] 回归解 | [cfg.Nx, cfg.Ny] Regression solution
        T_pinn: [cfg.Nx, cfg.Ny] PINN解 | [cfg.Nx, cfg.Ny] PINN solution
        T_ref:  [cfg.Nx, cfg.Ny] 参考解 | [cfg.Nx, cfg.Ny] Reference solution
        X:      [cfg.Nx, cfg.Ny] X坐标网格 | [cfg.Nx, cfg.Ny] X coordinate grid
        Y:      [cfg.Nx, cfg.Ny] Y坐标网格 | [cfg.Nx, cfg.Ny] Y coordinate grid
        data_path: 图像保存路径 | Path to save figure
        vmax_T: 误差图的最大值 | Maximum value for error plots
    """
    # 设置matplotlib绘图参数 | Set matplotlib plotting parameters
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9

    # 创建2x2子图 | Create 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 7), layout='constrained', 
                            sharex=True, sharey=True)
    # 设置参考解的颜色范围 | Set color range based on reference solution
    vmin_ref = T_ref.min()
    vmax_ref = T_ref.max()
    cbar_kw = {'fraction': 0.046, 'pad': 0.04}

    # 子图1: 回归解 | Subplot 1: Regression Solution
    pcm1 = axs[0,0].pcolormesh(X, Y, T_reg, vmin=vmin_ref, vmax=vmax_ref, cmap='jet', shading='auto')
    axs[0,0].set_title("(a) Regression Solution", pad=12)
    axs[0,0].set_xlabel("x")
    axs[0,0].set_ylabel("y")
    axs[0,0].grid(True, linestyle=':', alpha=0.6)
    fig.colorbar(pcm1, ax=axs[0,0], **cbar_kw)

    # 子图2: 回归误差 | Subplot 2: Regression Error
    pcm2 = axs[0,1].pcolormesh(X, Y, np.abs(T_ref - T_reg), vmin=0, vmax=vmax_T, cmap='jet', shading='auto')
    axs[0,1].set_title("(b) Regression Error", pad=12)
    axs[0,1].set_xlabel("x")
    axs[0,1].set_ylabel("y")
    axs[0,1].grid(True, linestyle=':', alpha=0.6)
    fig.colorbar(pcm2, ax=axs[0,1], **cbar_kw)

    # 子图3: PINN解 | Subplot 3: PINN Solution
    pcm3 = axs[1,0].pcolormesh(X, Y, T_pinn, vmin=vmin_ref, vmax=vmax_ref, cmap='jet', shading='auto')
    axs[1,0].set_title("(c) PINN Solution", pad=12)
    axs[1,0].set_xlabel("x")
    axs[1,0].set_ylabel("y")
    axs[1,0].grid(True, linestyle=':', alpha=0.6)
    fig.colorbar(pcm3, ax=axs[1,0], **cbar_kw)

    # 子图4: PINN误差 | Subplot 4: PINN Error
    pcm4 = axs[1,1].pcolormesh(X, Y, np.abs(T_ref - T_pinn), vmin=0, vmax=vmax_T, cmap='jet', shading='auto')
    axs[1,1].set_title("(d) PINN Error", pad=12)
    axs[1,1].set_xlabel("x")
    axs[1,1].set_ylabel("y")
    axs[1,1].grid(True, linestyle=':', alpha=0.6)
    fig.colorbar(pcm4, ax=axs[1,1], **cbar_kw)

    # 添加总标题和调整布局 | Add overall title and adjust layout
    fig.suptitle("Comparison of Regression and PINN T-Solutions", y=1.04, fontsize=15)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.savefig(os.path.join(data_path, 'fig_T.png'), dpi=300, bbox_inches='tight')
