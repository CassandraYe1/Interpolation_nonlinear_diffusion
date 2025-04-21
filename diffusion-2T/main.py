import numpy as np
import matplotlib.pyplot as plt
import torch 
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import copy
import random

import config as cfg
from model import DeepNN
from utils import relative_l2, set_seed
from train_reg import train_model_reg
from train_pde import train_model_pde


# First-stage training: Use only coarse-grid reference data.
# # # # # # # # # # # # # # # # # #
#  model_name    Nfit  lr_E  lr_T #
# zconst-const   300   1e-3  1e-3 #
# zconst-gauss   300   1e-3  1e-3 #
#  zline-const   300   1e-2  1e-2 #
#  zline-gauss   200   1e-3  1e-4 #
# zsquare-const  300   1e-3  1e-3 #
# zsquare-gauss  700   1e-3  1e-3 #
# # # # # # # # # # # # # # # # # #
set_seed(0)
print('Train by coarse-grid data:')
model_E = DeepNN().cuda()
model_T = DeepNN().cuda()
[model_E, model_T] = train_model_reg(model_E, model_T, Nfit=300, lr_E=1e-3, lr_T=1e-3)
E_reg = model_E(cfg.inp_fine, cfg.Z_fine).detach().cpu().reshape(cfg.Nx, cfg.Ny)
T_reg = model_T(cfg.inp_fine, cfg.Z_fine).detach().cpu().reshape(cfg.Nx, cfg.Ny)
np.save('./' + cfg.model_name + '/result/sol_reg_E', E_reg)
np.save('./' + cfg.model_name + '/result/sol_reg_T', T_reg)


# Second-stage training: Uses both coarse-grid reference data and PDE physical constraints.
# # # # # # # # # # # # # # # # # #
#  model_name    Nfit  lr_E  lr_T #
# zconst-const   200   1e-2  1e-2 #
# zconst-gauss   200   1e-1  1e-1 #
#  zline-const   200   1e-1  1e-1 #
#  zline-gauss   200   1e-1  1e-1 #
# zsquare-const  200   1e-1  1e-1 #
# zsquare-gauss  100   1e-1  1e-1 #
# # # # # # # # # # # # # # # # # #
set_seed(50)
print('Train by both coarse-grid data and PDE residual:')
model_E_cur = DeepNN().cuda()
model_E_cur.load_state_dict(copy.deepcopy(model_E.state_dict()))
model_T_cur = DeepNN().cuda()
model_T_cur.load_state_dict(copy.deepcopy(model_T.state_dict()))
[model_E_cur, model_T_cur] = train_model_pde(model_E_cur, model_T_cur, Nfit=200, lr_E=1e-2, lr_T=1e-2)
E_pinn = model_E_cur(cfg.inp_fine, cfg.Z_fine).detach().cpu().reshape(cfg.Nx, cfg.Ny)
T_pinn = model_T_cur(cfg.inp_fine, cfg.Z_fine).detach().cpu().reshape(cfg.Nx, cfg.Ny)
np.save('./' + cfg.model_name + '/result/sol_pinn_E', E_pinn)
np.save('./' + cfg.model_name + '/result/sol_pinn_T', T_pinn)
