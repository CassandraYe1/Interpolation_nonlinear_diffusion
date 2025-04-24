import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import random


def set_seed(seed=42):
    """
    Set all relevant random number seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def relative_l2(Eref, Epred):
    """
    Set relative_l2 error.

    Args:
        Eref:  [cfg.Nx, cfg.Ny] Reference solution
        Epred: [cfg.Nx, cfg.Ny] Predictive solution

    Returns:
        [cfg.Nx, cfg.Ny] Relative_l2 error between predictive solution and reference solution
    """
    return ((Eref - Epred)**2).mean() / (Eref**2).mean()


def pde_res(E, T, D, K, E_, T_, sigma, X, dt):
    """
    Set the PDE residual.

    Args:
        E, T   : E^n, T^n         [cfg.Nx, cfg.Ny] current solution
        D_, K_ : D^n, K^n         [cfg.Nx, cfg.Ny] current coefficient
        E_, T_ : E^{n-1}, T^{n-1} [cfg.Nx, cfg.Ny] previous coefficient
        sigma  : sigma^n * ((T^n)**4 - (E^n)) [cfg.Nx, cfg.Ny]
        X      : [cfg.Nx, cfg.Ny] Grid points on X-axis
        dt     : [cfg.Nx, cfg.Ny] Time step

    Returns:
        [PDE residual E, PDE residual T]: [2, cfg.Nx, cfg.Ny] According to target nonlinear radiation diffusion
    """
    ones = torch.ones_like(E)
    Egrad = torch.autograd.grad(E, X, grad_outputs=ones, create_graph=True)[0]
    Ex = Egrad[:,[0]]
    Ey = Egrad[:,[1]]
    Exx = torch.autograd.grad(Ex, X, grad_outputs=ones, create_graph=True)[0][:,[0]]
    Eyy = torch.autograd.grad(Ey, X, grad_outputs=ones, create_graph=True)[0][:,[1]]

    ones = torch.ones_like(T)
    Tgrad = torch.autograd.grad(T, X, grad_outputs=ones, create_graph=True)[0]
    Tx = Tgrad[:,[0]]
    Ty = Tgrad[:,[1]]
    Txx = torch.autograd.grad(Tx, X, grad_outputs=ones, create_graph=True)[0][:,[0]]
    Tyy = torch.autograd.grad(Ty, X, grad_outputs=ones, create_graph=True)[0][:,[1]]

    res_E = (((Exx + Eyy) * D * dt + E_ - E)**2) + sigma * dt 
    res_T = (((Txx + Tyy) * K * dt + T_ - T)**2) - sigma * dt 

    return [torch.abs(res_E.mean()), torch.abs(res_T.mean())]