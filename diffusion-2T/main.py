import numpy as np
import torch 
torch.set_default_dtype(torch.float64)
import argparse
import os
import copy
import time
import matplotlib.pyplot as plt
import matplotlib as mpl


def parse_args():
    """
    Parameter parsing and configuration update.
    """
    parser = argparse.ArgumentParser(description="Train the model with flexible parameters")

    parser.add_argument('--model_name', type=str, default='zconst-const', help='Model name (e.g., zconst-const)')
    parser.add_argument('--device_name', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for training (cuda/cpu)')

    parser.add_argument('--zconst', action='store_true', help='Use z_const function')
    parser.add_argument('--zline', action='store_true', help='Use z_line function')
    parser.add_argument('--zsquare', action='store_true', help='Use z_square function')

    parser.add_argument('--Nx', type=int, default=257, help='Number of grid points on x-axis')
    parser.add_argument('--Ny', type=int, default=257, help='Number of grid points on y-axis')
    parser.add_argument('--n', type=int, default=4, help='Downsampling factor')

    parser.add_argument('--Nfit_reg', type=int, default=300, help='Number of training iterations for regularization phase')
    parser.add_argument('--lr_E_reg', type=float, default=1e-2, help='Learning rate for LBFGS optimizer of E in regularization phase')
    parser.add_argument('--lr_T_reg', type=float, default=1e-2, help='Learning rate for LBFGS optimizer of T in regularization phase')
    parser.add_argument('--epoch_reg', type=int, default=50, help='Epochs for regularization training')

    parser.add_argument('--Nfit_pde', type=int, default=200, help='Number of training iterations for PDE phase')
    parser.add_argument('--lr_E_pde', type=float, default=1e-1, help='Learning rate for LBFGS optimizer of E in PDE phase')
    parser.add_argument('--lr_T_pde', type=float, default=1e-1, help='Learning rate for LBFGS optimizer of T in PDE phase')
    parser.add_argument('--epoch_pde', type=int, default=10, help='Epochs for PDE training')

    parser.add_argument('--vmax_E', type=float, default=0.25, help='Maximum value of error colorbar E')
    parser.add_argument('--vmax_T', type=float, default=0.25, help='Maximum value of error colorbar T')
    args = parser.parse_args()

    # Verify that the three parameters are mutually exclusive.
    z_flags = [args.zconst, args.zline, args.zsquare]
    if sum(z_flags) != 1:
        raise ValueError("Exactly one of --zconst/--zline/--zsquare must be specified")
    
    return args

# Update global parameters in config.py
args = parse_args()
import config as cfg
cfg.model_name = args.model_name
cfg.device_name = args.device_name
cfg.zconst = args.zconst
cfg.zline = args.zline
cfg.zsquare = args.zsquare
cfg.Nx = args.Nx
cfg.Ny = args.Ny
cfg.n = args.n
cfg.Nfit_reg = args.Nfit_reg
cfg.lr_E_reg = args.lr_E_reg
cfg.lr_T_reg = args.lr_T_reg
cfg.epoch_reg = args.epoch_reg
cfg.Nfit_pde = args.Nfit_pde
cfg.lr_E_pde = args.lr_E_pde
cfg.lr_T_pde = args.lr_T_pde
cfg.epoch_pde = args.epoch_pde
cfg.vmax_E = args.vmax_E
cfg.vmax_T = args.vmax_T
cfg.init_config()


# Import other modules after ensuring config updates.
from model import DeepNN
from utils import set_seed, relative_l2
from train_reg import train_model_reg
from train_pde import train_model_pde

# The file "main.py" must be located in the project root directory.
project_root = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_root, cfg.model_name, 'results')
if not os.path.exists(data_path):
    os.makedirs(data_path)

# First-stage training: Use only coarse-grid reference data.
set_seed(0)
print('Train by coarse-grid data:')
model_E = DeepNN().to(cfg.device_name)
model_T = DeepNN().to(cfg.device_name)
start_time = time.time()
[model_E, model_T] = train_model_reg(model_E, model_T, Nfit=cfg.Nfit_reg, lr_E=cfg.lr_E_reg, lr_T=cfg.lr_T_reg, epo=cfg.epoch_reg)
end_time = time.time()
training_time_reg = end_time - start_time
print(f"Regression training time: {training_time_reg:.6e} seconds")

E_reg = model_E(cfg.inp_fine, cfg.Z_fine).detach().cpu().reshape(cfg.Nx, cfg.Ny)
T_reg = model_T(cfg.inp_fine, cfg.Z_fine).detach().cpu().reshape(cfg.Nx, cfg.Ny)
np.save(os.path.join(data_path, 'sol_reg_E'), E_reg)
np.save(os.path.join(data_path, 'sol_reg_T'), T_reg)
torch.save(model_E.state_dict(), os.path.join(data_path, 'model_reg_E.pt'))
torch.save(model_T.state_dict(), os.path.join(data_path, 'model_reg_T.pt'))

# Second-stage training: Uses both coarse-grid reference data and PDE physical constraints.
set_seed(50)
print('Train by both coarse-grid data and PDE residual:')
model_E_cur = DeepNN().cuda()
model_E_cur.load_state_dict(copy.deepcopy(model_E.state_dict()))
model_T_cur = DeepNN().cuda()
model_T_cur.load_state_dict(copy.deepcopy(model_T.state_dict()))
start_time = time.time()
[model_E_cur, model_T_cur] = train_model_pde(model_E_cur, model_T_cur, Nfit=cfg.Nfit_pde, lr_E=cfg.lr_E_pde, lr_T=cfg.lr_T_pde, epo=cfg.epoch_pde)
end_time = time.time()
training_time_pinn = end_time - start_time
print(f"PINN training time: {training_time_pinn:.6e} seconds")

E_pinn = model_E_cur(cfg.inp_fine, cfg.Z_fine).detach().cpu().reshape(cfg.Nx, cfg.Ny)
T_pinn = model_T_cur(cfg.inp_fine, cfg.Z_fine).detach().cpu().reshape(cfg.Nx, cfg.Ny)
np.save(os.path.join(data_path, 'sol_pinn_E'), E_pinn)
np.save(os.path.join(data_path, 'sol_pinn_T'), T_pinn)
torch.save(model_E_cur.state_dict(), os.path.join(data_path, 'model_pinn_E.pt'))
torch.save(model_T_cur.state_dict(), os.path.join(data_path, 'model_pinn_T.pt'))


# L2 error
X = cfg.X.detach().cpu()
Y = cfg.Y.detach().cpu()
E_ref = cfg.E_ref.cpu()
T_ref = cfg.T_ref.cpu()
print('E: Regression Solution rl2: {:.4e}'.format(relative_l2(E_ref, E_reg)))
print('E: PINN Solution rl2: {:.4e}'.format(relative_l2(E_ref, E_pinn)))
print('T: Regression Solution rl2: {:.4e}'.format(relative_l2(T_ref, T_reg)))
print('T: PINN Solution rl2: {:.4e}'.format(relative_l2(T_ref, T_pinn)))


# Plot the comparison of regression and PINN E-solutions.
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9

fig, axs = plt.subplots(2, 2, figsize=(8, 7), layout='constrained', 
                        sharex=True, sharey=True)

vmin = E_ref.min()
vmax = E_ref.max()
cbar_kw = {'fraction': 0.046, 'pad': 0.04}

# Plot 1: Regression Solution
pcm1 = axs[0,0].pcolormesh(X, Y, E_reg, vmin=vmin, vmax=vmax, cmap='jet', shading='auto')
axs[0,0].set_title("(a) Regression Solution", pad=12)
axs[0,0].set_xlabel("x")
axs[0,0].set_ylabel("y")
axs[0,0].grid(True, linestyle=':', alpha=0.6)
fig.colorbar(pcm1, ax=axs[0,0], **cbar_kw)

# Plot 2: Regression Error
pcm2 = axs[0,1].pcolormesh(X, Y, np.abs(E_ref - E_reg), vmin=0, vmax=cfg.vmax_E, cmap='jet', shading='auto')
axs[0,1].set_title("(b) Regression Error", pad=12)
axs[0,1].set_xlabel("x")
axs[0,1].set_ylabel("y")
axs[0,1].grid(True, linestyle=':', alpha=0.6)
fig.colorbar(pcm2, ax=axs[0,1], **cbar_kw)

# Plot 3: PINN Solution
pcm3 = axs[1,0].pcolormesh(X, Y, E_pinn, vmin=vmin, vmax=vmax, cmap='jet', shading='auto')
axs[1,0].set_title("(c) PINN Solution", pad=12)
axs[1,0].set_xlabel("x")
axs[1,0].set_ylabel("y")
axs[1,0].grid(True, linestyle=':', alpha=0.6)
fig.colorbar(pcm3, ax=axs[1,0], **cbar_kw)

# Plot 4: PINN Error
pcm4 = axs[1,1].pcolormesh(X, Y, np.abs(E_ref - E_pinn), vmin=0, vmax=cfg.vmax_E, cmap='jet', shading='auto')
axs[1,1].set_title("(d) PINN Error", pad=12)
axs[1,1].set_xlabel("x")
axs[1,1].set_ylabel("y")
axs[1,1].grid(True, linestyle=':', alpha=0.6)
fig.colorbar(pcm4, ax=axs[1,1], **cbar_kw)

# Add overall title and adjust layout
fig.suptitle("Comparison of Regression and PINN E-Solutions", y=1.04, fontsize=15)
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.savefig(os.path.join(data_path, 'fig_E.png'), dpi=300, bbox_inches='tight')


# Plot the comparison of regression and PINN T-solutions.
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9

fig, axs = plt.subplots(2, 2, figsize=(8, 7), layout='constrained', 
                        sharex=True, sharey=True)

vmin = T_ref.min()
vmax = T_ref.max()
cbar_kw = {'fraction': 0.046, 'pad': 0.04}

# Plot 1: Regression Solution
pcm1 = axs[0,0].pcolormesh(X, Y, T_reg, vmin=vmin, vmax=vmax, cmap='jet', shading='auto')
axs[0,0].set_title("(a) Regression Solution", pad=12)
axs[0,0].set_xlabel("x")
axs[0,0].set_ylabel("y")
axs[0,0].grid(True, linestyle=':', alpha=0.6)
fig.colorbar(pcm1, ax=axs[0,0], **cbar_kw)

# Plot 2: Regression Error
pcm2 = axs[0,1].pcolormesh(X, Y, np.abs(T_ref - T_reg), vmin=0, vmax=cfg.vmax_T, cmap='jet', shading='auto')
axs[0,1].set_title("(b) Regression Error", pad=12)
axs[0,1].set_xlabel("x")
axs[0,1].set_ylabel("y")
axs[0,1].grid(True, linestyle=':', alpha=0.6)
fig.colorbar(pcm2, ax=axs[0,1], **cbar_kw)

# Plot 3: PINN Solution
pcm3 = axs[1,0].pcolormesh(X, Y, T_pinn, vmin=vmin, vmax=vmax, cmap='jet', shading='auto')
axs[1,0].set_title("(c) PINN Solution", pad=12)
axs[1,0].set_xlabel("x")
axs[1,0].set_ylabel("y")
axs[1,0].grid(True, linestyle=':', alpha=0.6)
fig.colorbar(pcm3, ax=axs[1,0], **cbar_kw)

# Plot 4: PINN Error
pcm4 = axs[1,1].pcolormesh(X, Y, np.abs(T_ref - T_pinn), vmin=0, vmax=cfg.vmax_T, cmap='jet', shading='auto')
axs[1,1].set_title("(d) PINN Error", pad=12)
axs[1,1].set_xlabel("x")
axs[1,1].set_ylabel("y")
axs[1,1].grid(True, linestyle=':', alpha=0.6)
fig.colorbar(pcm4, ax=axs[1,1], **cbar_kw)

# Add overall title and adjust layout
fig.suptitle("Comparison of Regression and PINN T-Solutions", y=1.04, fontsize=15)
plt.subplots_adjust(wspace=0.3, hspace=0.3)

plt.savefig(os.path.join(data_path, 'fig_T.png'), dpi=300, bbox_inches='tight')
