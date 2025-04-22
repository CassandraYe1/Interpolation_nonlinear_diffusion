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


def pde_res(E, D, E_, X, dt):
    """
    Set the PDE residual.

    Args:
        E  : E^n     [cfg.Nx, cfg.Ny] current solution
        D_ : D^n     [cfg.Nx, cfg.Ny] current coefficient
        E_ : E^{n-1} [cfg.Nx, cfg.Ny] previous coefficient
        X  : [cfg.Nx, cfg.Ny] Grid points on X-axis
        dt : Time step

    Returns:
        [cfg.Nx, cfg.Ny] PDE residual according to target nonlinear radiation diffusion
    """
    ones = torch.ones_like(E)
    Egrad = torch.autograd.grad(E, X, grad_outputs=ones, create_graph=True)[0]
    Ex = Egrad[:,[0]]
    Ey = Egrad[:,[1]]
    Exx = torch.autograd.grad(Ex, X, grad_outputs=ones, create_graph=True)[0][:,[0]]
    Eyy = torch.autograd.grad(Ey, X, grad_outputs=ones, create_graph=True)[0][:,[1]]

    res = (((Exx + Eyy) * D * dt + E_ - E)**2)

    return res.mean()
