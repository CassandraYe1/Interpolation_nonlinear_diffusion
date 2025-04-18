import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import config as cfg


def load_data():
    """
    Load the known fine grid data

    D_ref, K_ref   : D^n, K^n         current coefficient
    E_prev, T_prev : E^{n-1}, T^{n-1} previous coefficient
    E_ref, T_ref   : E^n, T^n         current solution
    sigma_ref      : sigma^n * ((T^n)**4 - (E^n))
    X              : grid points on X-axis
    Y              : grid points on Y-axis
    """
    sol_E = np.load('./' + cfg.model_name + '/data/sol-E.npy')
    sol_T = np.load('./' + cfg.model_name + '/data/sol-T.npy')
    kappa_E = np.load('./' + cfg.model_name + '/data/kappa-E.npy')
    kappa_T = np.load('./' + cfg.model_name + '/data/kappa-T.npy')
    sigma = np.load('./' + cfg.model_name + '/data/sigma.npy')
    X = np.load('./' + cfg.model_name + '/data/X.npy')
    Y = np.load('./' + cfg.model_name + '/data/Y.npy')

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
    Set continuous z-bool-function
    """
    Z = torch.ones(cfg.Nx, cfg.Ny)
    Z = (Z>2).cuda()

    return Z


def z_line(X, Y):
    """
    Set intermittent z-bool-function
    """
    Z = torch.zeros(cfg.Nx, cfg.Ny)
    for i in range(cfg.Nx):
        for j in range(cfg.Nx):
            Z[i,j] = (X[i,j]<1./2.)*10.0 + (X[i,j]>=1./2.)*1.0
    Z = (Z>2).cuda()

    return Z


def z_square(X, Y):
    """
    Set z-bool-function with two squares
    """
    ax, ay, bx, by = 3., 9., 9., 3.
    Z = torch.zeros(cfg.Nx, cfg.Ny)
    for i in range(cfg.Nx):
        for j in range(cfg.Nx):
            Z[i,j] = (X[i,j]<(ax+4.)/16.)*(X[i,j]>ax/16.0)*(Y[i,j]<(ay+4.)/16.)*(Y[i,j]>ay/16.0)*9.0 + \
                     (X[i,j]<(bx+4.)/16.)*(X[i,j]>bx/16.0)*(Y[i,j]<(by+4.)/16.)*(Y[i,j]>by/16.0)*9.0 + 1.0
    Z = (Z>2).cuda()

    return Z
