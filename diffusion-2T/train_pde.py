import torch
from utils import relative_l2, pde_res
import config as cfg


def train_model_pde(model_E_cur, model_T_cur, Nfit=cfg.Nfit_pde, lr_E=cfg.lr_E_pde, lr_T=cfg.lr_T_pde, epo=cfg.epoch_pde):
    """
    Train model using both coarse-grid reference data and PDE residuals.
    
    Args:
        model_E_cur: Current model instance of E to train
        model_T_cur: Current model instance of T to train
        Nfit       : Number of training iterations
        lr_E       : Learning rate for LBFGS optimizer of E
        lr_T       : Learning rate for LBFGS optimizer of T
        epo        : Number of training epoch
        
    Returns:
        [model_E_cur, model_T_cur]: The trained model of E and T
    """
    
    opt_lbfgs_E = torch.optim.LBFGS(model_E_cur.parameters(), lr=lr_E)
    opt_lbfgs_T = torch.optim.LBFGS(model_T_cur.parameters(), lr=lr_T)

    for i in range(Nfit):
        model_E_cur.train()
        model_T_cur.train()

        def closure():
            opt_lbfgs_E.zero_grad()
            opt_lbfgs_T.zero_grad()
            E_pred = model_E_cur(cfg.inp_coarse, cfg.Z_coarse)
            T_pred = model_T_cur(cfg.inp_coarse, cfg.Z_coarse)

            ref_loss_E = relative_l2(cfg.E_coarse_ref.reshape(-1,1), E_pred)
            ref_loss_T = relative_l2(cfg.T_coarse_ref.reshape(-1,1), T_pred)
            pde_loss = pde_res(E_pred, T_pred, cfg.D_coarse.reshape(-1,1), cfg.K_coarse.reshape(-1,1), cfg.E_coarse_prev.reshape(-1,1), cfg.T_coarse_prev.reshape(-1,1), cfg.sigma_coarse_ref.reshape(-1,1), cfg.inp_coarse, 0.001)
            loss_E = pde_loss[0]*10 + ref_loss_E
            loss_T = pde_loss[1]*10 - ref_loss_T

            loss = loss_E + loss_T
            loss.backward()
            return loss

        loss = closure()
        opt_lbfgs_E.step(closure)
        opt_lbfgs_T.step(closure)
            
        if i % epo == 0:
            model_E_cur.eval()
            Ed_pred = model_E_cur(cfg.inp_d, cfg.Zd)
            El_pred = model_E_cur(cfg.inp_l, cfg.Zl)
            Er_pred = model_E_cur(cfg.inp_r, cfg.Zr)
            Et_pred = model_E_cur(cfg.inp_t, cfg.Zt)
            Eb_pred = model_E_cur(cfg.inp_b, cfg.Zb)

            model_T_cur.eval()
            Td_pred = model_T_cur(cfg.inp_d, cfg.Zd)
            Tl_pred = model_T_cur(cfg.inp_l, cfg.Zl)
            Tr_pred = model_T_cur(cfg.inp_r, cfg.Zr)
            Tt_pred = model_T_cur(cfg.inp_t, cfg.Zt)
            Tb_pred = model_T_cur(cfg.inp_b, cfg.Zb)

            pde_loss = pde_res(Ed_pred, Td_pred, cfg.Dd.reshape(-1,1), cfg.Kd.reshape(-1,1), cfg.Ed_.reshape(-1,1), cfg.Td_.reshape(-1,1), cfg.sigma_d.reshape(-1,1), cfg.inp_d, 0.001)
            lbc_loss_E = relative_l2(cfg.El, El_pred)
            rbc_loss_E = (Er_pred ** 2).mean()
            tbc_loss_E = relative_l2(cfg.Et, Et_pred)
            bbc_loss_E = relative_l2(cfg.Eb, Eb_pred)
            ref_loss_E = relative_l2(cfg.E_ref[1:-1,1:-1].reshape(-1,1), Ed_pred)
            lbc_loss_T = relative_l2(cfg.Tl, Tl_pred)
            rbc_loss_T = (Tr_pred ** 2).mean()
            tbc_loss_T = relative_l2(cfg.Tt, Tt_pred)
            bbc_loss_T = relative_l2(cfg.Tb, Tb_pred)
            ref_loss_T = relative_l2(cfg.T_ref[1:-1,1:-1].reshape(-1,1), Td_pred)

            Epred = model_E_cur(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(257,257)
            ref_rl2_E = relative_l2(cfg.E_ref.cpu().numpy().reshape(-1), Epred.reshape(-1))
            print("E: lbfgs : {:d} - ref_rl2 {:.4e} - pde {:.4e} - lbc {:.4e} - rbc {:.4e} - tbc {:.4e} - bbc {:.4e}".format(
                i, ref_rl2_E, pde_loss[0], lbc_loss_E, rbc_loss_E, tbc_loss_E, bbc_loss_E))

            Tpred = model_T_cur(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(257,257)
            ref_rl2_T = relative_l2(cfg.T_ref.cpu().numpy().reshape(-1), Tpred.reshape(-1))
            print("T: lbfgs : {:d} - ref_rl2 {:.4e} - pde {:.4e} - lbc {:.4e} - rbc {:.4e} - tbc {:.4e} - bbc {:.4e}".format(
                i, ref_rl2_T, pde_loss[1], lbc_loss_T, rbc_loss_T, tbc_loss_T, bbc_loss_T))
            
    return [model_E_cur, model_T_cur]
