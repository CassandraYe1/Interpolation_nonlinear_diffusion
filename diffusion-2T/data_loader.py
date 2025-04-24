import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import os
import config as cfg


def load_data():
    """
    Load the fine-grid reference solution.

    Returns:
        D_ref, K_ref   : D^n, K^n         [cfg.Nx, cfg.Ny] current coefficient
        E_prev, T_prev : E^{n-1}, T^{n-1} [cfg.Nx, cfg.Ny] previous coefficient
        E_ref, T_ref   : E^n, T^n         [cfg.Nx, cfg.Ny] current solution
        sigma_ref      : sigma^n * ((T^n)**4 - (E^n)) [cfg.Nx, cfg.Ny]
        X              : [cfg.Nx, cfg.Ny] Grid points on X-axis
        Y              : [cfg.Nx, cfg.Ny] Grid points on Y-axis
    """
    # The file "data_loader.py" must be located in the project root directory.
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, cfg.model_name, 'data')

    sol_E = np.load(os.path.join(data_path, 'sol-E.npy'))
    sol_T = np.load(os.path.join(data_path, 'sol-T.npy'))
    kappa_E = np.load(os.path.join(data_path, 'kappa-E.npy'))
    kappa_T = np.load(os.path.join(data_path, 'kappa-T.npy'))
    sigma = np.load(os.path.join(data_path, 'sigma.npy'))
    X = np.load(os.path.join(data_path, 'X.npy'))
    Y = np.load(os.path.join(data_path, 'Y.npy'))

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
