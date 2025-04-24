import numpy as np
import torch 
torch.set_default_dtype(torch.float64)
import argparse
import numpy as np
import torch 
torch.set_default_dtype(torch.float64)
import os
import copy

# Parameter parsing and configuration update.
def parse_args():
    parser = argparse.ArgumentParser(description="Train the model with flexible parameters")

    parser.add_argument('--model_name', type=str, default='zsquare-gauss', help='Model name (e.g., zsquare-gauss)')
    parser.add_argument('--device_name', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for training (cuda/cpu)')

    parser.add_argument('--zconst', action='store_true', help='Use z_const function')
    parser.add_argument('--zline', action='store_true', help='Use z_line function')
    parser.add_argument('--zsquare', action='store_true', help='Use z_square function')

    parser.add_argument('--Nx', type=int, default=257, help='Number of grid points on x-axis')
    parser.add_argument('--Ny', type=int, default=257, help='Number of grid points on y-axis')

    parser.add_argument('--Nfit_reg', type=int, default=300, help='Number of training iterations for regularization phase')
    parser.add_argument('--lr_reg', type=float, default=1e-2, help='Learning rate for LBFGS optimizer in regularization phase')
    parser.add_argument('--epoch_reg', type=int, default=50, help='Epochs for regularization training')

    parser.add_argument('--Nfit_pde', type=int, default=200, help='Number of training iterations for PDE phase')
    parser.add_argument('--lr_pde', type=float, default=1e-1, help='Learning rate for LBFGS optimizer in PDE phase')
    parser.add_argument('--epoch_pde', type=int, default=10, help='Epochs for PDE training')
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
cfg.Nfit_reg = args.Nfit_reg
cfg.lr_reg = args.lr_reg
cfg.epoch_reg = args.epoch_reg
cfg.Nfit_pde = args.Nfit_pde
cfg.lr_pde = args.lr_pde
cfg.epoch_pde = args.epoch_pde

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
model = DeepNN().to(cfg.device_name)
model = train_model_reg(model, Nfit=cfg.Nfit_reg, lr=cfg.lr_reg, epo=cfg.epoch_reg)
E_reg = model(cfg.inp_fine, cfg.Z_fine).detach().cpu().reshape(cfg.Nx, cfg.Ny)
np.save(os.path.join(data_path, 'sol_reg'), E_reg)

# Second-stage training: Uses both coarse-grid reference data and PDE physical constraints.
set_seed(50)
print('Train by both coarse-grid data and PDE residual:')
model_cur = DeepNN().to(cfg.device_name)
model_cur.load_state_dict(copy.deepcopy(model.state_dict()))
model_cur = train_model_pde(model_cur, Nfit=cfg.Nfit_pde, lr=cfg.lr_pde, epo=cfg.epoch_pde)
E_pinn = model_cur(cfg.inp_fine, cfg.Z_fine).detach().cpu().reshape(cfg.Nx, cfg.Ny)
np.save(os.path.join(data_path, 'sol_pinn'), E_pinn)

X = cfg.X.detach().cpu()
Y = cfg.Y.detach().cpu()
E_ref = cfg.E_ref.cpu()
print('Regression Solution rl2: {:.4e}'.format(relative_l2(E_ref, E_reg)))
print('PINN Solution rl2: {:.4e}'.format(relative_l2(E_ref, E_pinn)))