import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from data_loader import load_data, z_const, z_line, z_square


# model_name = "z function" - "initial condition"
model_name = "zsquare-gauss"
device_name = "cuda"

# case of z function
zconst = False
zline = False
zsquare = True

Nx = 257 # number of grid point on x-axis
Ny = 257 # number of grid point on y-axis


# load the known fine grid data
D_ref, E_prev, E_ref, X, Y = load_data()
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
X_coarse = X[::4,::4]
Y_coarse = Y[::4,::4]
Z_coarse = Z[::4,::4]
inp_coarse = torch.concat(
    [X_coarse.reshape(-1,1), Y_coarse.reshape(-1,1)], 
    axis=1).requires_grad_().cuda()
Z_coarse = Z_coarse.reshape(-1,1)
D_coarse = D_ref[::4,::4]
E_coarse_prev = E_prev[::4,::4]
E_coarse_ref = E_ref[::4,::4]

# internal data
Xd = X[1:-1,1:-1]
Yd = Y[1:-1,1:-1]
Zd = Z[1:-1,1:-1].reshape(-1,1)
inp_d = torch.concat(
    [Xd.reshape(-1,1), Yd.reshape(-1,1)], 
    axis=1).requires_grad_().cuda()
Ed = E_ref[1:-1,1:-1]
Ed_ = E_prev[1:-1,1:-1]
Dd = D_ref[1:-1,1:-1]

# left boundary data
Xl = X[:,0]
Yl = Y[:,0]
Zl = Z[:,0].reshape(-1,1)
inp_l = torch.concat(
    [Xl.reshape(-1,1), Yl.reshape(-1,1)], 
    axis=1).requires_grad_().cuda()
El = E_ref[:,[0]]
Dl = D_ref[:,[0]]

# right boundary data
Xr = X[:,-1]
Yr = Y[:,-1]
Zr = Z[:,-1].reshape(-1,1)
inp_r = torch.concat(
    [Xr.reshape(-1,1), Yr.reshape(-1,1)], 
    axis=1).requires_grad_().cuda()
Er = E_ref[:,[-1]]
Dr = D_ref[:,[-1]]

# bottom boundary data
Xb = X[0]
Yb = Y[0]
Zb = Z[0].reshape(-1,1)
inp_b = torch.concat(
    [Xb.reshape(-1,1), Yb.reshape(-1,1)], 
    axis=1).requires_grad_().cuda()
Eb = E_ref[0].reshape(-1,1)
Db = D_ref[0].reshape(-1,1)

# top boundary data
Xt = X[-1]
Yt = Y[-1]
Zt = Z[-1].reshape(-1,1)
inp_t = torch.concat(
    [Xt.reshape(-1,1), Yt.reshape(-1,1)], 
    axis=1).requires_grad_().cuda()
Et = E_ref[-1].reshape(-1,1)
Dt = D_ref[-1].reshape(-1,1)