import numpy as np
import torch
from utils import relative_l2
from config import Config


def train_model_reg(model_E, model_T, cfg: Config, Nfit=None, lr_E=None, lr_T=None, epo=None):
    """
    使用粗网格参考数据训练模型
    Train the model using only coarse-grid reference data.

    Args:
        model_E: 要训练的关于E的神经网络模型
                 The neural network model of E to train
        model_T: 要训练的关于T的神经网络模型
                 The neural network model of T to train
        cfg    : 配置类实例，包含所有参数
                 Config class instance containing all parameters
        Nfit   : 训练迭代次数
                 Number of training iterations
        lr_E   : 关于E的LBFGS优化器的学习率
                 Learning rate for LBFGS optimizer of E
        lr_T   : 关于T的LBFGS优化器的学习率
                 Learning rate for LBFGS optimizer of T
        epo    : 训练周期数(打印间隔)
                 Number of training epoch (print interval)

    Returns:
        [model_E, model_T]: 训练好的模型
                            The trained model
    """
    rl2_loss_E = []
    rl2_loss_T = []
    opt_adam_E = torch.optim.Adam(model_E.parameters(), lr=lr_E*0.1)
    opt_adam_T = torch.optim.Adam(model_T.parameters(), lr=lr_T*0.1)
    opt_lbfgs_E = torch.optim.LBFGS(model_E.parameters(), lr=lr_E, line_search_fn='strong_wolfe')
    opt_lbfgs_T = torch.optim.LBFGS(model_T.parameters(), lr=lr_T, line_search_fn='strong_wolfe')

    for warmup in range(Nfit):
        model_E.train()
        model_T.train()

        def closure_E():
            """
            关于E的LBFGS需要闭包函数计算损失
            LBFGS of E requires closure function for loss calculation
            """
            opt_lbfgs_E.zero_grad()
            E_pred = model_E(cfg.inp_coarse, cfg.Z_coarse_bool)

            # 计算相对L2损失 | Calculate relative L2 loss
            loss_E = relative_l2(cfg.E_coarse_ref.reshape(-1,1), E_pred)

            loss_E.backward()
            return loss_E

        def closure_T():
            """
            关于T的LBFGS需要闭包函数计算损失
            LBFGS of T requires closure function for loss calculation
            """
            opt_lbfgs_T.zero_grad()
            T_pred = model_T(cfg.inp_coarse, cfg.Z_coarse_bool)

            # 计算相对L2损失 | Calculate relative L2 loss
            loss_T = relative_l2(cfg.T_coarse_ref.reshape(-1,1), T_pred)

            loss_T.backward()
            return loss_T

        # 执行优化步骤 | Perform optimization step
        loss_E = closure_E()
        loss_T = closure_T()
        opt_lbfgs_E.step(closure_E)
        opt_lbfgs_T.step(closure_T)

        if warmup % 10 == 0:
            model_E.eval()
            model_T.eval()

            # 在各边界和内部点进行预测 | Predict on boundaries and internal points
            El_pred = model_E(cfg.inp_l, cfg.Zl_bool)
            Er_pred = model_E(cfg.inp_r, cfg.Zr_bool)
            Et_pred = model_E(cfg.inp_t, cfg.Zt_bool)
            Eb_pred = model_E(cfg.inp_b, cfg.Zb_bool)

            Tl_pred = model_T(cfg.inp_l, cfg.Zl_bool)
            Tr_pred = model_T(cfg.inp_r, cfg.Zr_bool)
            Tt_pred = model_T(cfg.inp_t, cfg.Zt_bool)
            Tb_pred = model_T(cfg.inp_b, cfg.Zb_bool)

            # 计算各项损失 | Calculate various losses
            lbc_loss_E = relative_l2(cfg.El, El_pred)
            rbc_loss_E = relative_l2(cfg.Er, Er_pred)
            tbc_loss_E = relative_l2(cfg.Et, Et_pred)
            bbc_loss_E = relative_l2(cfg.Eb, Eb_pred)

            lbc_loss_T = relative_l2(cfg.Tl, Tl_pred)
            rbc_loss_T = relative_l2(cfg.Tr, Tr_pred)
            tbc_loss_T = relative_l2(cfg.Tt, Tt_pred)
            bbc_loss_T = relative_l2(cfg.Tb, Tb_pred)

            # 精细网格评估 | Fine grid evaluation
            Epred = model_E(cfg.inp_fine, cfg.Z_fine_bool).detach().reshape(cfg.Nx,cfg.Ny)
            ref_rl2_E = relative_l2(cfg.E_ref.reshape(-1), Epred.reshape(-1))
            print("E: adam : {:d} - ref_rl2 {:.4e} - lbc {:.4e} - rbc {:.4e} - tbc {:.4e} - bbc {:.4e}".format(
                warmup, ref_rl2_E, lbc_loss_E, rbc_loss_E, tbc_loss_E, bbc_loss_E))

            Tpred = model_T(cfg.inp_fine, cfg.Z_fine_bool).detach().reshape(cfg.Nx,cfg.Ny)
            ref_rl2_T = relative_l2(cfg.T_ref.reshape(-1), Tpred.reshape(-1))
            print("T: adam : {:d} - ref_rl2 {:.4e} - lbc {:.4e} - rbc {:.4e} - tbc {:.4e} - bbc {:.4e}".format(
                warmup, ref_rl2_T, lbc_loss_T, rbc_loss_T, tbc_loss_T, bbc_loss_T))

    for i in range(Nfit):
        model_E.train()
        model_T.train()

        def closure_E():
            """
            关于E的LBFGS需要闭包函数计算损失
            LBFGS of E requires closure function for loss calculation
            """
            opt_lbfgs_E.zero_grad()
            E_pred = model_E(cfg.inp_coarse, cfg.Z_coarse_bool)

            # 计算相对L2损失 | Calculate relative L2 loss
            loss_E = relative_l2(cfg.E_coarse_ref.reshape(-1,1), E_pred)

            loss_E.backward()
            return loss_E

        def closure_T():
            """
            关于T的LBFGS需要闭包函数计算损失
            LBFGS of T requires closure function for loss calculation
            """
            opt_lbfgs_T.zero_grad()
            T_pred = model_T(cfg.inp_coarse, cfg.Z_coarse_bool)

            # 计算相对L2损失 | Calculate relative L2 loss
            loss_T = relative_l2(cfg.T_coarse_ref.reshape(-1,1), T_pred)

            loss_T.backward()
            return loss_T

        # 执行优化步骤 | Perform optimization step
        loss_E = closure_E()
        loss_T = closure_T()
        opt_lbfgs_E.step(closure_E)
        opt_lbfgs_T.step(closure_T)

        model_E.eval()
        model_T.eval()

        # 在各边界和内部点进行预测 | Predict on boundaries and internal points
        El_pred = model_E(cfg.inp_l, cfg.Zl_bool)
        Er_pred = model_E(cfg.inp_r, cfg.Zr_bool)
        Et_pred = model_E(cfg.inp_t, cfg.Zt_bool)
        Eb_pred = model_E(cfg.inp_b, cfg.Zb_bool)

        Tl_pred = model_T(cfg.inp_l, cfg.Zl_bool)
        Tr_pred = model_T(cfg.inp_r, cfg.Zr_bool)
        Tt_pred = model_T(cfg.inp_t, cfg.Zt_bool)
        Tb_pred = model_T(cfg.inp_b, cfg.Zb_bool)

        # 计算各项损失 | Calculate various losses
        lbc_loss_E = relative_l2(cfg.El, El_pred)
        rbc_loss_E = relative_l2(cfg.Er, Er_pred)
        tbc_loss_E = relative_l2(cfg.Et, Et_pred)
        bbc_loss_E = relative_l2(cfg.Eb, Eb_pred)

        lbc_loss_T = relative_l2(cfg.Tl, Tl_pred)
        rbc_loss_T = relative_l2(cfg.Tr, Tr_pred)
        tbc_loss_T = relative_l2(cfg.Tt, Tt_pred)
        bbc_loss_T = relative_l2(cfg.Tb, Tb_pred)

        # 精细网格评估 | Fine grid evaluation
        Epred = model_E(cfg.inp_fine, cfg.Z_fine_bool).detach().reshape(cfg.Nx,cfg.Ny)
        ref_rl2_E = relative_l2(cfg.E_ref.reshape(-1), Epred.reshape(-1))
        Tpred = model_T(cfg.inp_fine, cfg.Z_fine_bool).detach().reshape(cfg.Nx,cfg.Ny)
        ref_rl2_T = relative_l2(cfg.T_ref.reshape(-1), Tpred.reshape(-1))
        rl2_loss_E.append(ref_rl2_E)
        rl2_loss_T.append(ref_rl2_T)

        # 定期验证并打印进度 | Periodically validate and print progress
        if i % epo == 0:
            print("E: lbfgs : {:d} - ref_rl2 {:.4e} - lbc {:.4e} - rbc {:.4e} - tbc {:.4e} - bbc {:.4e}".format(
                  i, ref_rl2_E, lbc_loss_E, rbc_loss_E, tbc_loss_E, bbc_loss_E))
            print("T: lbfgs : {:d} - ref_rl2 {:.4e} - lbc {:.4e} - rbc {:.4e} - tbc {:.4e} - bbc {:.4e}".format(
                  i, ref_rl2_T, lbc_loss_T, rbc_loss_T, tbc_loss_T, bbc_loss_T))

    # 最终评估 | Final evaluation
    model_E.eval()
    model_T.eval()

    # 在各边界和内部点进行预测 | Predict on boundaries and internal points
    El_pred = model_E(cfg.inp_l, cfg.Zl_bool)
    Er_pred = model_E(cfg.inp_r, cfg.Zr_bool)
    Et_pred = model_E(cfg.inp_t, cfg.Zt_bool)
    Eb_pred = model_E(cfg.inp_b, cfg.Zb_bool)

    Tl_pred = model_T(cfg.inp_l, cfg.Zl_bool)
    Tr_pred = model_T(cfg.inp_r, cfg.Zr_bool)
    Tt_pred = model_T(cfg.inp_t, cfg.Zt_bool)
    Tb_pred = model_T(cfg.inp_b, cfg.Zb_bool)

    # 计算各项损失 | Calculate various losses
    lbc_loss_E = relative_l2(cfg.El, El_pred)
    rbc_loss_E = relative_l2(cfg.Er, Er_pred)
    tbc_loss_E = relative_l2(cfg.Et, Et_pred)
    bbc_loss_E = relative_l2(cfg.Eb, Eb_pred)

    lbc_loss_T = relative_l2(cfg.Tl, Tl_pred)
    rbc_loss_T = relative_l2(cfg.Tr, Tr_pred)
    tbc_loss_T = relative_l2(cfg.Tt, Tt_pred)
    bbc_loss_T = relative_l2(cfg.Tb, Tb_pred)

    # 精细网格评估 | Fine grid evaluation
    Epred = model_E(cfg.inp_fine, cfg.Z_fine_bool).detach().reshape(cfg.Nx,cfg.Ny)
    ref_rl2_E = relative_l2(cfg.E_ref.reshape(-1), Epred.reshape(-1))
    print("E: lbfgs : {:d} - ref_rl2 {:.4e} - lbc {:.4e} - rbc {:.4e} - tbc {:.4e} - bbc {:.4e}".format(
            i, ref_rl2_E, lbc_loss_E, rbc_loss_E, tbc_loss_E, bbc_loss_E))

    Tpred = model_T(cfg.inp_fine, cfg.Z_fine_bool).detach().reshape(cfg.Nx,cfg.Ny)
    ref_rl2_T = relative_l2(cfg.T_ref.reshape(-1), Tpred.reshape(-1))
    print("T: lbfgs : {:d} - ref_rl2 {:.4e} - lbc {:.4e} - rbc {:.4e} - tbc {:.4e} - bbc {:.4e}".format(
            i, ref_rl2_T, lbc_loss_T, rbc_loss_T, tbc_loss_T, bbc_loss_T))

    return [model_E, model_T, rl2_loss_E, rl2_loss_T]