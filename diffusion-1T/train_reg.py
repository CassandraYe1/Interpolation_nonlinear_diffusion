import torch
import torch.optim as optim
from utils import relative_l2
import config as cfg


def train_model_reg(model, Nfit=300, lr=1e-2):
    """
    Model trained only by coarse grid known data

    Nfit : training steps
    lr   : learning rate
    """

    opt_lbfgs = torch.optim.LBFGS(model.parameters(), lr=lr)
    # sch_adam = torch.optim.lr_scheduler.StepLR(opt_adam, 5000, gamma=0.9)

    for i in range(Nfit):
        model.train()

        def closure():
            opt_lbfgs.zero_grad()
            E_pred = model(cfg.inp_coarse, cfg.Z_coarse)
            loss = relative_l2(cfg.E_coarse_ref.reshape(-1,1), E_pred)
            loss.backward()
            return loss

        # E_prev_ = model(inp_full)
        # loss = relative_l2(E_prev.reshape(-1,1), E_prev_)
        # loss.backward()
        # opt_adam.step()
        # sch_adam.step()

        loss = closure()
        opt_lbfgs.step(closure)

        if i % 50 == 0:
            model.eval()
            Epred = model(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(257,257)
            ref_rl2 = relative_l2(cfg.E_ref.cpu().numpy().reshape(-1), Epred.reshape(-1))
            print("lbfgs : {:d} - ref_rl2 {:.4e} ".format(i, ref_rl2))

    model.eval()
    Epred = model(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(257,257)
    ref_rl2 = relative_l2(cfg.E_ref.cpu().numpy().reshape(-1), Epred.reshape(-1))
    print("lbfgs : {:d} - ref_rl2 {:.4e} ".format(i, ref_rl2))

    return model