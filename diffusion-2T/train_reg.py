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
    opt_lbfgs_E = torch.optim.LBFGS(model_E.parameters(), lr=lr_E)
    opt_lbfgs_T = torch.optim.LBFGS(model_T.parameters(), lr=lr_T)

    for i in range(Nfit):
        model_E.train()
        model_T.train()

        def closure_E():
            """
            关于E的LBFGS需要闭包函数计算损失
            LBFGS of E requires closure function for loss calculation
            """
            opt_lbfgs_E.zero_grad()
            E_pred = model_E(cfg.inp_coarse, cfg.Z_coarse)

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
            T_pred = model_T(cfg.inp_coarse, cfg.Z_coarse)

            # 计算相对L2损失 | Calculate relative L2 loss
            loss_T = relative_l2(cfg.T_coarse_ref.reshape(-1,1), T_pred)

            loss_T.backward()
            return loss_T

        # 执行优化步骤 | Perform optimization step
        loss_E = closure_E()
        loss_T = closure_T()
        opt_lbfgs_E.step(closure_E)
        opt_lbfgs_T.step(closure_T)

        # 定期验证并打印进度 | Periodically validate and print progress
        if i % epo == 0:
            model_E.eval()
            Epred = model_E(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(257,257)
            ref_rl2_E = relative_l2(cfg.E_ref.cpu().numpy().reshape(-1), Epred.reshape(-1))
            print("E: lbfgs : {:d} - ref_rl2 {:.4e} ".format(i, ref_rl2_E))

            model_T.eval()
            Tpred = model_T(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(257,257)
            ref_rl2_T = relative_l2(cfg.T_ref.cpu().numpy().reshape(-1), Tpred.reshape(-1))
            print("T: lbfgs : {:d} - ref_rl2 {:.4e} ".format(i, ref_rl2_T))

    # 最终评估 | Final evaluation
    model_E.eval()
    Epred = model_E(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(257,257)
    ref_rl2_E = relative_l2(cfg.E_ref.cpu().numpy().reshape(-1), Epred.reshape(-1))
    print("E: lbfgs : {:d} - ref_rl2 {:.4e} ".format(i, ref_rl2_E))

    model_T.eval()
    Tpred = model_T(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(257,257)
    ref_rl2_T = relative_l2(cfg.T_ref.cpu().numpy().reshape(-1), Tpred.reshape(-1))
    print("T: lbfgs : {:d} - ref_rl2 {:.4e} ".format(i, ref_rl2_T))

    return [model_E, model_T]