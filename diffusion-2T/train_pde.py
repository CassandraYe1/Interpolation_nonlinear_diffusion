import numpy as np
import torch
from utils import relative_l2, pde_res
from config import Config


def train_model_pde(model_E_cur, model_T_cur, cfg: Config, Nfit=None, lr_E=None, lr_T=None, epo=None):
    """
    使用粗网格参考数据和PDE残差联合训练模型
    Train model using both coarse-grid reference data and PDE residuals.
    
    Args:
        model_E_cur: 当前要训练的关于E的模型实例
                     Current model of E instance to train
        model_T_cur: 当前要训练的关于T的模型实例
                     Current model of T instance to train
        cfg        : 配置类实例，包含所有参数
                     Config class instance containing all parameters
        Nfit       : 训练迭代次数
                     Number of training iterations
        lr_E       : 关于E的LBFGS优化器的学习率
                     Learning rate for LBFGS optimizer of E
        lr_T       : 关于T的LBFGS优化器的学习率
                     Learning rate for LBFGS optimizer of T
        epo        : 训练周期数(打印间隔)
                     Number of training epoch (print interval)
        
    Returns:
        [model_E_cur, model_T_cur]: 训练完成的模型
                                    The trained model
    """
    m_E = 10
    m_T = 10
    rl2_loss_E = []
    rl2_loss_T = []
    opt_lbfgs_E = torch.optim.LBFGS(model_E_cur.parameters(), lr=lr_E, line_search_fn='strong_wolfe')
    opt_lbfgs_T = torch.optim.LBFGS(model_T_cur.parameters(), lr=lr_T, line_search_fn='strong_wolfe')

    for i in range(Nfit):
        model_E_cur.train()
        model_T_cur.train()

        def closure_E():
            """
            LBFGS需要闭包函数计算损失
            LBFGS requires closure function for loss calculation
            """
            opt_lbfgs_E.zero_grad()
            E_pred = model_E_cur(cfg.inp_fine, cfg.Z_fine_bool).reshape(cfg.Nx,cfg.Ny)

            # 计算参考解损失(粗网格) | Calculate reference solution loss (coarse-grid)
            ref_loss_E = relative_l2(cfg.E_coarse_ref, E_pred[::cfg.n, ::cfg.n])
            # 计算PDE残差损失 | Calculate PDE residual loss
            Exx = (E_pred[2:-2:2, 3::2] + E_pred[2:-2:2, 1:-3:2] - 2 * E_pred[2:-2:2, 2:-2:2]) / (1 / 256)**2
            Eyy = (E_pred[3::2, 2:-2:2] + E_pred[1:-3:2, 2:-2:2] - 2 * E_pred[2:-2:2, 2:-2:2]) / (1 / 256)**2
            res_E = (((Exx + Eyy) * cfg.D_coarse[1:-1, 1:-1] * 0.001 + cfg.E_coarse_prev[1:-1, 1:-1] - E_pred[2:-2:2, 2:-2:2] + cfg.sigma_coarse_ref[1:-1, 1:-1] * 0.001)**2)
            pde_loss_E = res_E.mean()

            if pde_loss_E > ref_loss_E:
                loss_E = pde_loss_E + ref_loss_E*m_E
            else:
                loss_E = pde_loss_E*m_E + ref_loss_E
            loss_E.backward()
            return loss_E

        def closure_T():
            """
            LBFGS需要闭包函数计算损失
            LBFGS requires closure function for loss calculation
            """
            opt_lbfgs_T.zero_grad()
            T_pred = model_T_cur(cfg.inp_fine, cfg.Z_fine_bool).reshape(cfg.Nx,cfg.Ny)

            # 计算参考解损失(粗网格) | Calculate reference solution loss (coarse-grid)
            ref_loss_T = relative_l2(cfg.T_coarse_ref, T_pred[::cfg.n, ::cfg.n])
            # 计算PDE残差损失 | Calculate PDE residual loss
            Txx = (T_pred[2:-2:2, 3::2] + T_pred[2:-2:2, 1:-3:2] - 2 * T_pred[2:-2:2, 2:-2:2]) / (1 / 256)**2
            Tyy = (T_pred[3::2, 2:-2:2] + T_pred[1:-3:2, 2:-2:2] - 2 * T_pred[2:-2:2, 2:-2:2]) / (1 / 256)**2
            res_T = (((Txx + Tyy) * cfg.K_coarse[1:-1, 1:-1] * 0.001 + cfg.T_coarse_prev[1:-1, 1:-1] - T_pred[2:-2:2, 2:-2:2] - cfg.sigma_coarse_ref[1:-1, 1:-1] * 0.001)**2)
            pde_loss_T = res_T.mean()

            if pde_loss_T > ref_loss_T:
                loss_T = pde_loss_T + ref_loss_T*m_T
            else:
                loss_T = pde_loss_T*m_T + ref_loss_T
            loss_T.backward()
            return loss_T

        # 执行优化步骤 | Perform optimization step
        loss_E = closure_E()
        loss_T = closure_T()
        opt_lbfgs_E.step(closure_E)
        opt_lbfgs_T.step(closure_T)

        model_E_cur.eval()
        model_T_cur.eval()

        # 在各边界和内部点进行预测 | Predict on boundaries and internal points
        Ed_pred = model_E_cur(cfg.inp_fine, cfg.Z_fine_bool).reshape(cfg.Nx,cfg.Ny)
        El_pred = model_E_cur(cfg.inp_l, cfg.Zl_bool)
        Er_pred = model_E_cur(cfg.inp_r, cfg.Zr_bool)
        Et_pred = model_E_cur(cfg.inp_t, cfg.Zt_bool)
        Eb_pred = model_E_cur(cfg.inp_b, cfg.Zb_bool)

        Td_pred = model_T_cur(cfg.inp_fine, cfg.Z_fine_bool).reshape(cfg.Nx,cfg.Ny)
        Tl_pred = model_T_cur(cfg.inp_l, cfg.Zl_bool)
        Tr_pred = model_T_cur(cfg.inp_r, cfg.Zr_bool)
        Tt_pred = model_T_cur(cfg.inp_t, cfg.Zt_bool)
        Tb_pred = model_T_cur(cfg.inp_b, cfg.Zb_bool)

        # 计算各项损失 | Calculate various losses
        Exx = (Ed_pred[1:-1, 2:] + Ed_pred[1:-1, :-2] - 2 * Ed_pred[1:-1, 1:-1]) / (1 / 256)**2
        Eyy = (Ed_pred[2:, 1:-1] + Ed_pred[:-2, 1:-1] - 2 * Ed_pred[1:-1, 1:-1]) / (1 / 256)**2
        res_E = (((Exx + Eyy) * cfg.Dd * 0.001 + cfg.Ed_ - Ed_pred[1:-1, 1:-1] + cfg.sigma_d * 0.001)**2)
        pde_loss_E = res_E.mean()
        Txx = (Td_pred[1:-1, 2:] + Td_pred[1:-1, :-2] - 2 * Td_pred[1:-1, 1:-1]) / (1 / 256)**2
        Tyy = (Td_pred[2:, 1:-1] + Td_pred[:-2, 1:-1] - 2 * Td_pred[1:-1, 1:-1]) / (1 / 256)**2
        res_T = (((Txx + Tyy) * cfg.Kd * 0.001 + cfg.Td_ - Td_pred[1:-1, 1:-1] - cfg.sigma_d * 0.001)**2)
        pde_loss_T = res_T.mean()

        lbc_loss_E = relative_l2(cfg.El, El_pred)
        rbc_loss_E = relative_l2(cfg.Er, Er_pred)
        tbc_loss_E = relative_l2(cfg.Et, Et_pred)
        bbc_loss_E = relative_l2(cfg.Eb, Eb_pred)

        lbc_loss_T = relative_l2(cfg.Tl, Tl_pred)
        rbc_loss_T = relative_l2(cfg.Tr, Tr_pred)
        tbc_loss_T = relative_l2(cfg.Tt, Tt_pred)
        bbc_loss_T = relative_l2(cfg.Tb, Tb_pred)

        # 精细网格评估 | Fine grid evaluation
        Epred = model_E_cur(cfg.inp_fine, cfg.Z_fine_bool).detach().reshape(cfg.Nx,cfg.Ny)
        ref_rl2_E = relative_l2(cfg.E_ref.reshape(-1), Epred.reshape(-1))
        Tpred = model_T_cur(cfg.inp_fine, cfg.Z_fine_bool).detach().reshape(cfg.Nx,cfg.Ny)
        ref_rl2_T = relative_l2(cfg.T_ref.reshape(-1), Tpred.reshape(-1))
        rl2_loss_E.append(ref_rl2_E)
        rl2_loss_T.append(ref_rl2_T)

        if pde_loss_E > ref_rl2_E:
            log = int(torch.log10(pde_loss_E / ref_rl2_E))
            m_E = (int((pde_loss_E / ref_rl2_E) / (10**log))) * (10**(log+1))
        else:
            log = int(torch.log10(ref_rl2_E / pde_loss_E))
            m_E = (int((ref_rl2_E / pde_loss_E) / (10**log))) * (10**(log-1))
            #m_E = 1 / m_E
        if pde_loss_T > ref_rl2_T:
            log = int(torch.log10(pde_loss_T / ref_rl2_T))
            m_T = (int((pde_loss_T / ref_rl2_T) / (10**log))) * (10**(log+1))
        else:
            log = int(torch.log10(ref_rl2_T / pde_loss_T))
            m_T = (int((ref_rl2_T / pde_loss_T) / (10**log))) * (10**(log-1))
            #m_T = 1 / m_T
            
        # 定期验证 | Periodic validation
        if i % epo == 0:
            print("E: lbfgs : {:d} - ref_rl2 {:.4e} - pde {:.4e} - lbc {:.4e} - rbc {:.4e} - tbc {:.4e} - bbc {:.4e}".format(
                i, ref_rl2_E, pde_loss_E, lbc_loss_E, rbc_loss_E, tbc_loss_E, bbc_loss_E))
            print("T: lbfgs : {:d} - ref_rl2 {:.4e} - pde {:.4e} - lbc {:.4e} - rbc {:.4e} - tbc {:.4e} - bbc {:.4e}".format(
                i, ref_rl2_T, pde_loss_T, lbc_loss_T, rbc_loss_T, tbc_loss_T, bbc_loss_T))
            print(f"Updated weights: m_E = {m_E}, m_T = {m_T}")
 
    return [model_E_cur, model_T_cur, rl2_loss_E, rl2_loss_T]
