import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from data_loader import *


## Phase 1: Declare all configuration parameters.
# model_name = "ionization function type - initial condition type"
model_name = "zsquare-gauss"
# device_name = "cpu" or "cuda"
device_name = "cuda"
# Type of ionization function
zconst = False
zline = False
zsquare = False

Nx = 257 # Number of grid point on x-axis
Ny = 257 # Number of grid point on y-axis
n = 4 # Downsampling factor

# Model trained by coarse-grid reference data
Nfit_reg = 300 # Number of training iterations
lr_E_reg = 1e-2 # Learning rate for LBFGS optimizer of E
lr_T_reg = 1e-2 # Learning rate for LBFGS optimizer of T
epoch_reg = 50 # Epoch
# Model trained by coarse-grid reference data and PDE residual
Nfit_pde = 200 # Number of training iterations
lr_E_pde = 1e-1 # Learning rate for LBFGS optimizer of E
lr_T_pde = 1e-1 # Learning rate for LBFGS optimizer of T
epoch_pde = 10 # Epoch

## Phase 2: Define variables (to be initialized in init_config())
D_ref = None
K_ref = None
E_prev = None
E_ref = None
T_prev = None
T_ref = None
sigma_ref = None
X = None
Y = None
Z = None
inp_fine = None
Z_fine = None

X_coarse = None
Y_coarse = None
Z_coarse = None
inp_coarse = None
Z_coarse = None
D_coarse = None
K_coarse = None
E_coarse_prev = None
E_coarse_ref = None
T_coarse_prev = None
T_coarse_ref = None
sigma_coarse_ref = None

Xd = None
Yd = None
Zd = None
inp_d = None
Ed = None
Ed_ = None
Td = None
Td_ = None
Dd = None
Kd = None
sigma_d = None

Xl = None
Yl = None
Zl = None
inp_l = None
El = None
Tl = None
Dl = None
Kl = None
sigma_l = None

Xr = None
Yr = None
Zr = None
inp_r = None
Er = None
Tr = None
Dr = None
Kr = None
sigma_r = None

Xb = None
Yb = None
Zb = None
inp_b = None
Eb = None
Tb = None
Db = None
Kb = None
sigma_b = None

Xt = None
Yt = None
Zt = None
inp_t = None
Et = None
Tt = None
Dt = None
Kt = None
sigma_t = None

# Initialize status flags
_internal_initialized = False


def init_config():
    """
    Explicitly initialize configuration-dependent data components.
    Must be called after all configuration parameters are updated.
    """
    global D_ref, K_ref, E_prev, E_ref, T_prev, T_ref, sigma_ref, X, Y, Z, inp_fine, Z_fine, \
           X_coarse, Y_coarse, Z_coarse, inp_coarse, Z_coarse, D_coarse, K_coarse, E_coarse_prev, E_coarse_ref, T_coarse_prev, T_coarse_ref, sigma_coarse_ref, \
           Xd, Yd, Zd, inp_d, Ed, Ed_, Td, Td_, Dd, Kd, sigma_d, \
           Xl, Yl, Zl, inp_l, El, Tl, Dl, Kl, sigma_l, \
           Xr, Yr, Zr, inp_r, Er, Tr, Dr, Kr, sigma_r, \
           Xb, Yb, Zb, inp_b, Eb, Tb, Db, Kb, sigma_b, \
           Xt, Yt, Zt, inp_t, Et, Tt, Dt, Kt, sigma_t, _internal_initialized
    
    if _internal_initialized:
        return
    
    # Validate the type of ionization function.
    if not (zconst or zline or zsquare):
        raise ValueError("An ionization function type must be specified (--zconst/--zline/--zsquare).")
    
    # Load the known fine-grid data
    D_ref, K_ref, E_prev, E_ref, T_prev, T_ref, sigma_ref, X, Y = load_data()
    if zconst:
        Z = z_const(X, Y)
    elif zline:
        Z = z_line(X, Y)
    elif zsquare:
        Z = z_square(X, Y)

    inp_fine = torch.concat(
        [X.reshape(-1,1), Y.reshape(-1,1)], 
        axis=1).requires_grad_().cuda()
    Z_fine = Z.reshape(-1,1)

    # coarse grid data
    X_coarse = X[::n,::n]
    Y_coarse = Y[::n,::n]
    Z_coarse = Z[::n,::n].reshape(-1,1)
    inp_coarse = torch.concat(
        [X_coarse.reshape(-1,1), Y_coarse.reshape(-1,1)], 
        axis=1).requires_grad_().cuda()
    D_coarse = D_ref[::n,::n]
    K_coarse = K_ref[::n,::n]
    E_coarse_prev = E_prev[::n,::n]
    E_coarse_ref = E_ref[::n,::n]
    T_coarse_prev = T_prev[::n,::n]
    T_coarse_ref = T_ref[::n,::n]
    sigma_coarse_ref = sigma_ref[::n,::n]

    # internal data
    Xd = X[1:-1,1:-1]
    Yd = Y[1:-1,1:-1]
    Zd = Z[1:-1,1:-1].reshape(-1,1)
    inp_d = torch.concat(
        [Xd.reshape(-1,1), Yd.reshape(-1,1)], 
        axis=1).requires_grad_().cuda()
    Ed = E_ref[1:-1,1:-1]
    Ed_ = E_prev[1:-1,1:-1]
    Td = T_ref[1:-1,1:-1]
    Td_ = T_prev[1:-1,1:-1]
    Dd = D_ref[1:-1,1:-1]
    Kd = K_ref[1:-1,1:-1]
    sigma_d = sigma_ref[1:-1,1:-1]

    # left boundary data
    Xl = X[:,0]
    Yl = Y[:,0]
    Zl = Z[:,0].reshape(-1,1)
    inp_l = torch.concat(
        [Xl.reshape(-1,1), Yl.reshape(-1,1)], 
        axis=1).requires_grad_().cuda()
    El = E_ref[:,[0]]
    Tl = T_ref[:,[0]]
    Dl = D_ref[:,[0]]
    Kl = K_ref[:,[0]]
    sigma_l = sigma_ref[:,[0]]

    # right boundary data
    Xr = X[:,-1]
    Yr = Y[:,-1]
    Zr = Z[:,-1].reshape(-1,1)
    inp_r = torch.concat(
        [Xr.reshape(-1,1), Yr.reshape(-1,1)], 
        axis=1).requires_grad_().cuda()
    Er = E_ref[:,[-1]]
    Tr = T_ref[:,[-1]]
    Dr = D_ref[:,[-1]]
    Kr = K_ref[:,[-1]]
    sigma_r = sigma_ref[:,[-1]]

    # bottom boundary data
    Xb = X[0]
    Yb = Y[0]
    Zb = Z[0].reshape(-1,1)
    inp_b = torch.concat(
        [Xb.reshape(-1,1), Yb.reshape(-1,1)], 
        axis=1).requires_grad_().cuda()
    Eb = E_ref[0].reshape(-1,1)
    Tb = T_ref[0].reshape(-1,1)
    Db = D_ref[0].reshape(-1,1)
    Kb = K_ref[0].reshape(-1,1)
    sigma_b = sigma_ref[0].reshape(-1,1)

    # top boundary data
    Xt = X[-1]
    Yt = Y[-1]
    Zt = Z[-1].reshape(-1,1)
    inp_t = torch.concat(
        [Xt.reshape(-1,1), Yt.reshape(-1,1)], 
        axis=1).requires_grad_().cuda()
    Et = E_ref[-1].reshape(-1,1)
    Tt = T_ref[-1].reshape(-1,1)
    Dt = D_ref[-1].reshape(-1,1)
    Kt = K_ref[-1].reshape(-1,1)
    sigma_t = sigma_ref[-1].reshape(-1,1)

    _internal_initialized = True
    print("Configuration initialization completed.")
