import torch
from utils import relative_l2, pde_res
import config as cfg


def train_model_pde(model_cur, Nfit=cfg.Nfit_pde, lr=cfg.lr_pde, epo=cfg.epoch_pde):
    """
    Train model using both coarse-grid reference data and PDE residuals.
    
    Args:
        model_cur: Current model instance to train
        Nfit     : Number of training iterations
        lr       : Learning rate for LBFGS optimizer
        epo      : Number of training epoch
        
    Returns:
        model_cur: The trained model
    """

    opt_lbfgs = torch.optim.LBFGS(model_cur.parameters(), lr=lr)

    for i in range(Nfit):
        model_cur.train()

        def closure():
            opt_lbfgs.zero_grad()
            E_pred = model_cur(cfg.inp_coarse, cfg.Z_coarse)
            ref_loss = relative_l2(cfg.E_coarse_ref.reshape(-1,1), E_pred)
            pde_loss = pde_res(E_pred, cfg.D_coarse.reshape(-1,1), cfg.E_coarse_prev.reshape(-1,1), cfg.inp_coarse, 0.001)

            loss = pde_loss*10 + ref_loss
            loss.backward()

            return loss

        loss = closure()
        opt_lbfgs.step(closure)
            
        if i % epo == 0:
            model_cur.eval()

            Ed_pred = model_cur(cfg.inp_d, cfg.Zd)
            El_pred = model_cur(cfg.inp_l, cfg.Zl)
            Er_pred = model_cur(cfg.inp_r, cfg.Zr)
            Et_pred = model_cur(cfg.inp_t, cfg.Zt)
            Eb_pred = model_cur(cfg.inp_b, cfg.Zb)

            pde_loss = pde_res(Ed_pred, cfg.Dd.reshape(-1,1), cfg.Ed_.reshape(-1,1), cfg.inp_d, 0.001)
            lbc_loss = relative_l2(cfg.El, El_pred)
            rbc_loss = (Er_pred ** 2).mean()
            tbc_loss = relative_l2(cfg.Et, Et_pred)
            bbc_loss = relative_l2(cfg.Eb, Eb_pred)

            Epred = model_cur(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(257,257)
            ref_rl2 = relative_l2(cfg.E_ref.cpu().numpy().reshape(-1), Epred.reshape(-1))
            print("lbfgs : {:d} - ref_rl2 {:.4e} - pde {:.4e} - lbc {:.4e} - rbc {:.4e} - tbc {:.4e} - bbc {:.4e}".format(
                i, ref_rl2, pde_loss, lbc_loss, rbc_loss, tbc_loss, bbc_loss))
            
    return model_cur
