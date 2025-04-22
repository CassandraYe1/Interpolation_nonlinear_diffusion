import torch
from utils import relative_l2
import config as cfg


def train_model_reg(model, Nfit=cfg.Nfit_reg, lr=cfg.lr_reg, epo=cfg.epoch_reg):
    """
    Train the model using only coarse-grid reference data.

    Args:
        model: The neural network model to train
        Nfit : Number of training iterations
        lr   : Learning rate for LBFGS optimizer
        epo  : Number of training epoch

    Returns:
        model: The trained model
    """

    opt_lbfgs = torch.optim.LBFGS(model.parameters(), lr=lr)

    for i in range(Nfit):
        model.train()

        def closure():
            opt_lbfgs.zero_grad()
            E_pred = model(cfg.inp_coarse, cfg.Z_coarse)
            loss = relative_l2(cfg.E_coarse_ref.reshape(-1,1), E_pred)
            loss.backward()
            return loss

        loss = closure()
        opt_lbfgs.step(closure)

        if i % epo == 0:
            model.eval()
            Epred = model(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(257,257)
            ref_rl2 = relative_l2(cfg.E_ref.cpu().numpy().reshape(-1), Epred.reshape(-1))
            print("lbfgs : {:d} - ref_rl2 {:.4e} ".format(i, ref_rl2))

    model.eval()
    Epred = model(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(257,257)
    ref_rl2 = relative_l2(cfg.E_ref.cpu().numpy().reshape(-1), Epred.reshape(-1))
    print("lbfgs : {:d} - ref_rl2 {:.4e} ".format(i, ref_rl2))

    return model
