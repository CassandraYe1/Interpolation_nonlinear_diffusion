import torch
from utils import relative_l2
import config as cfg


def train_model_reg(model_E, model_T, Nfit=cfg.Nfit_reg, lr_E=cfg.lr_E_reg, lr_T=cfg.lr_T_reg, epo=cfg.epoch_reg):
    """
    Train the model using only coarse-grid reference data.

    Args:
        model_E: The neural network model of E to train
        model_T: The neural network model of T to train
        Nfit   : Number of training iterations
        lr     : Learning rate for LBFGS optimizer
        epo    : Number of training epoch

    Returns:
        [model_E, model_T]: The trained model of E and T
    """
    
    opt_lbfgs_E = torch.optim.LBFGS(model_E.parameters(), lr=lr_E)
    opt_lbfgs_T = torch.optim.LBFGS(model_T.parameters(), lr=lr_T)

    for i in range(Nfit):
        model_E.train()
        model_T.train()

        def closure_E():
            opt_lbfgs_E.zero_grad()
            E_pred = model_E(cfg.inp_coarse, cfg.Z_coarse)
            loss_E = relative_l2(cfg.E_coarse_ref.reshape(-1,1), E_pred)
            loss_E.backward()
            return loss_E

        def closure_T():
            opt_lbfgs_T.zero_grad()
            T_pred = model_T(cfg.inp_coarse, cfg.Z_coarse)
            loss_T = relative_l2(cfg.T_coarse_ref.reshape(-1,1), T_pred)
            loss_T.backward()
            return loss_T

        loss_E = closure_E()
        loss_T = closure_T()
        opt_lbfgs_E.step(closure_E)
        opt_lbfgs_T.step(closure_T)

        if i % epo == 0:
            model_E.eval()
            Epred = model_E(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(257,257)
            ref_rl2_E = relative_l2(cfg.E_ref.cpu().numpy().reshape(-1), Epred.reshape(-1))
            print("E: lbfgs : {:d} - ref_rl2 {:.4e} ".format(i, ref_rl2_E))

            model_T.eval()
            Tpred = model_T(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(257,257)
            ref_rl2_T = relative_l2(cfg.T_ref.cpu().numpy().reshape(-1), Tpred.reshape(-1))
            print("T: lbfgs : {:d} - ref_rl2 {:.4e} ".format(i, ref_rl2_T))

    model_E.eval()
    Epred = model_E(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(257,257)
    ref_rl2_E = relative_l2(cfg.E_ref.cpu().numpy().reshape(-1), Epred.reshape(-1))
    print("E: lbfgs : {:d} - ref_rl2 {:.4e} ".format(i, ref_rl2_E))

    model_T.eval()
    Tpred = model_T(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(257,257)
    ref_rl2_T = relative_l2(cfg.T_ref.cpu().numpy().reshape(-1), Tpred.reshape(-1))
    print("T: lbfgs : {:d} - ref_rl2 {:.4e} ".format(i, ref_rl2_T))

    return [model_E, model_T]