import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import config as cfg


def load_data():
    """
    Load the fine-grid reference solution.

    Returns:
        D_ref  : D^n     [cfg.Nx, cfg.Ny] current coefficient
        E_prev : E^{n-1} [cfg.Nx, cfg.Ny] previous coefficient
        E_ref  : E^n     [cfg.Nx, cfg.Ny] current solution
        X      : [cfg.Nx, cfg.Ny] Grid points on X-axis
        Y      : [cfg.Nx, cfg.Ny] Grid points on Y-axis
    """
    sol = np.load('./' + cfg.model_name + '/data/sol.npy')
    kappa = np.load('./' + cfg.model_name + '/data/kappa.npy')
    X = np.load('./' + cfg.model_name + '/data/X.npy')
    Y = np.load('./' + cfg.model_name + '/data/Y.npy')

    D_ref = torch.tensor(kappa[-1]).cuda()
    E_prev = torch.tensor(sol[-2]).cuda()
    E_ref = torch.tensor(sol[-1]).cuda()
    X = torch.tensor(X).requires_grad_().cuda()
    Y = torch.tensor(Y).requires_grad_().cuda()

    return D_ref, E_prev, E_ref, X, Y


def z_const(X, Y):
    """
    Generate the continuous ionization function that converts to Boolean values with 2 as the threshold.

    Args:
        X: [cfg.Nx, cfg.Ny] Grid points on X-axis
        Y: [cfg.Nx, cfg.Ny] Grid points on Y-axis

    Returns:
        Z: [cfg.Nx, cfg.Ny] Boolean values of the ionization function type "zconst"
    """
    Z = torch.ones(cfg.Nx, cfg.Ny)
    Z = (Z>2).cuda()

    return Z


def z_line(X, Y):
    """
    Generate the intermittent ionization function that converts to Boolean values with 2 as the threshold.

    Args:
        X: [cfg.Nx, cfg.Ny] Grid points on X-axis
        Y: [cfg.Nx, cfg.Ny] Grid points on Y-axis

    Returns:
        Z: [cfg.Nx, cfg.Ny] Boolean values of the ionization function type "zline"
    """
    Z = torch.zeros(cfg.Nx, cfg.Ny)
    for i in range(cfg.Nx):
        for j in range(cfg.Nx):
            Z[i,j] = (X[i,j]<1./2.)*10.0 + (X[i,j]>=1./2.)*1.0
    Z = (Z>2).cuda()

    return Z


def z_square(X, Y):
    """
    Generate the two-squares ionization function that converts to Boolean values with 2 as the threshold.

    Args:
        X: [cfg.Nx, cfg.Ny] Grid points on X-axis
        Y: [cfg.Nx, cfg.Ny] Grid points on Y-axis

    Returns:
        Z: [cfg.Nx, cfg.Ny] Boolean values of the ionization function type "zsquare"
    """
    ax, ay, bx, by = 3., 9., 9., 3.
    Z = torch.zeros(cfg.Nx, cfg.Ny)
    for i in range(cfg.Nx):
        for j in range(cfg.Nx):
            Z[i,j] = (X[i,j]<(ax+4.)/16.)*(X[i,j]>ax/16.0)*(Y[i,j]<(ay+4.)/16.)*(Y[i,j]>ay/16.0)*9.0 + \
                     (X[i,j]<(bx+4.)/16.)*(X[i,j]>bx/16.0)*(Y[i,j]<(by+4.)/16.)*(Y[i,j]>by/16.0)*9.0 + 1.0
    Z = (Z>2).cuda()

    return Z
