import numpy as np
import torch 
torch.set_default_dtype(torch.float64)
import copy

import config as cfg
from model import DeepNN
from utils import set_seed
from train_reg import train_model_reg
from train_pde import train_model_pde


# First-stage training: Use only coarse-grid reference data.
# # # # # # # # # # # # # # #
#  model_name    Nfit   lr  #
# zconst-const   300   1e-2 #
# zconst-gauss   300   1e-2 #
#  zline-const   150   1e-2 #
#  zline-gauss   150   1e-2 #
# zsquare-const  150   1e-2 #
# zsquare-gauss  400   1e-1 #
# # # # # # # # # # # # # # #
set_seed(0)
print('Train by coarse-grid data:')
model = DeepNN().to(cfg.device_name)
model = train_model_reg(model, Nfit=300, lr=1e-2)
E_reg = model(cfg.inp_fine, cfg.Z_fine).detach().cpu().reshape(cfg.Nx, cfg.Ny)
np.save('./' + cfg.model_name + '/result/sol_reg', E_reg)

# Second-stage training: Uses both coarse-grid reference data and PDE physical constraints.
# # # # # # # # # # # # # # #
#  model_name    Nfit   lr  #
# zconst-const   200    1   #
# zconst-gauss   200    1   #
#  zline-const   200    1   #
#  zline-gauss   200    1   #
# zsquare-const  300   1e-1 #
# zsquare-gauss  350    1   #
# # # # # # # # # # # # # # #
set_seed(50)
print('Train by both coarse-grid data and PDE residual:')
model_cur = DeepNN().to(cfg.device_name)
model_cur.load_state_dict(copy.deepcopy(model.state_dict()))
model_cur = train_model_pde(model_cur, Nfit=200, lr=1)
E_pinn = model_cur(cfg.inp_fine, cfg.Z_fine).detach().cpu().reshape(cfg.Nx, cfg.Ny)
np.save('./' + cfg.model_name + '/result/sol_pinn', E_pinn)
