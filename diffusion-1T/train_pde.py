import numpy as np
import torch
from utils import relative_l2, pde_res
from config import Config


def train_model_pde(model_cur, cfg: Config, Nfit=None, lr=None, epo=None):
    """
    使用粗网格参考数据和PDE残差联合训练模型
    Train model using both coarse-grid reference data and PDE residuals.
    
    Args:
        model_cur: 当前要训练的模型实例
                   Current model instance to train
        cfg  : 配置类实例，包含所有参数
               Config class instance containing all parameters
        Nfit : 训练迭代次数
               Number of training iterations
        lr   : LBFGS优化器的学习率
               Learning rate for LBFGS optimizer
        epo  : 训练周期数(打印间隔)
               Number of training epoch (print interval)
        
    Returns:
        model_cur: 训练完成的模型
                   The trained model
    """
    m = 10
    rl2_loss = []
    opt_lbfgs = torch.optim.LBFGS(model_cur.parameters(), lr=lr, line_search_fn='strong_wolfe')
    
    for i in range(Nfit):
        model_cur.train()

        def closure():
            """
            LBFGS需要闭包函数计算损失
            LBFGS requires closure function for loss calculation
            """
            opt_lbfgs.zero_grad()
            E_pred = model_cur(cfg.inp_fine, cfg.Z_fine_bool).reshape(cfg.Nx,cfg.Ny)

            # 计算参考解损失(粗网格) | Calculate reference solution loss (coarse-grid)
            ref_loss = relative_l2(cfg.E_coarse_ref, E_pred[::cfg.n, ::cfg.n])
            # 计算PDE残差损失 | Calculate PDE residual loss
            #pde_loss = pde_res(E_pred[::cfg.n, ::cfg.n], cfg.D_coarse, cfg.E_coarse_prev, cfg.inp_coarse, 0.001)
            Exx = (E_pred[cfg.n:-cfg.n:cfg.n, (cfg.n+1)::cfg.n] + E_pred[cfg.n:-cfg.n:cfg.n, (cfg.n-1):-(cfg.n+1):cfg.n] - 2 * E_pred[cfg.n:-cfg.n:cfg.n, cfg.n:-cfg.n:cfg.n]) / (1 / 256)**2
            Eyy = (E_pred[(cfg.n+1)::cfg.n, cfg.n:-cfg.n:cfg.n] + E_pred[(cfg.n-1):-(cfg.n+1):cfg.n, cfg.n:-cfg.n:cfg.n] - 2 * E_pred[cfg.n:-cfg.n:cfg.n, cfg.n:-cfg.n:cfg.n]) / (1 / 256)**2
            res = (((Exx + Eyy) * cfg.D_coarse[1:-1, 1:-1] * 0.001 + cfg.E_coarse_prev[1:-1, 1:-1] - E_pred[cfg.n:-cfg.n:cfg.n, cfg.n:-cfg.n:cfg.n])**2)
            pde_loss = res.mean()

            loss = pde_loss + ref_loss*m
            loss.backward()

            return loss

        # 执行优化步骤 | Perform optimization step
        loss = closure()
        opt_lbfgs.step(closure)
            
        model_cur.eval()

        # 在各边界和内部点进行预测 | Predict on boundaries and internal points
        Ed_pred = model_cur(cfg.inp_fine, cfg.Z_fine_bool).reshape(cfg.Nx,cfg.Ny)
        El_pred = model_cur(cfg.inp_l, cfg.Zl_bool)
        Er_pred = model_cur(cfg.inp_r, cfg.Zr_bool)
        Et_pred = model_cur(cfg.inp_t, cfg.Zt_bool)
        Eb_pred = model_cur(cfg.inp_b, cfg.Zb_bool)

        # 计算各项损失 | Calculate various losses
        #pde_loss = pde_res(Ed_pred, cfg.Dd, cfg.Ed_, cfg.inp_d, 0.001)
        Exx = (Ed_pred[1:-1, 2:] + Ed_pred[1:-1, :-2] - 2 * Ed_pred[1:-1, 1:-1]) / (1 / 256)**2
        Eyy = (Ed_pred[2:, 1:-1] + Ed_pred[:-2, 1:-1] - 2 * Ed_pred[1:-1, 1:-1]) / (1 / 256)**2
        res = (((Exx + Eyy) * cfg.Dd * 0.001 + cfg.Ed_ - Ed_pred[1:-1, 1:-1])**2)
        pde_loss = res.mean()

        lbc_loss = relative_l2(cfg.El, El_pred)
        rbc_loss = relative_l2(cfg.Er, Er_pred)
        tbc_loss = relative_l2(cfg.Et, Et_pred)
        bbc_loss = relative_l2(cfg.Eb, Eb_pred)

        # 精细网格评估 | Fine grid evaluation
        Epred = model_cur(cfg.inp_fine, cfg.Z_fine_bool).detach().reshape(cfg.Nx,cfg.Ny)
        ref_rl2 = relative_l2(cfg.E_ref.reshape(-1), Epred.reshape(-1))
        rl2_loss.append(ref_rl2)

        if pde_loss > ref_rl2:
            log = int(torch.log10(pde_loss / ref_rl2))
            m = (int((pde_loss / ref_rl2) / (10**log))) * (10**(log+1))
        else:
            log = int(torch.log10(ref_rl2 / pde_loss))
            m = (int((ref_rl2 / pde_loss) / (10**log))) * (10**(log-1))
            m = 1 / m

        # 定期验证 | Periodic validation
        if i % epo == 0:
            print("lbfgs : {:d} - ref_rl2 {:.4e} - pde {:.4e} - lbc {:.4e} - rbc {:.4e} - tbc {:.4e} - bbc {:.4e}".format(
                i, ref_rl2, pde_loss, lbc_loss, rbc_loss, tbc_loss, bbc_loss))
            print(f"Updated weights: m = {m}")
            
    return model_cur, rl2_loss
