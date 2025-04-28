import torch
from utils import relative_l2
from config import Config


def train_model_reg(model, cfg: Config, Nfit=None, lr=None, epo=None):
    """
    使用粗网格参考数据训练模型
    Train the model using only coarse-grid reference data.

    Args:
        model: 要训练的神经网络模型
               The neural network model to train
        cfg  : 配置类实例，包含所有参数
               Config class instance containing all parameters
        Nfit : 训练迭代次数
               Number of training iterations
        lr   : LBFGS优化器的学习率
               Learning rate for LBFGS optimizer
        epo  : 训练周期数(打印间隔)
               Number of training epoch (print interval)

    Returns:
        model: 训练好的模型
               The trained model
    """

    opt_lbfgs = torch.optim.LBFGS(model.parameters(), lr=lr)

    for i in range(Nfit):
        model.train()

        def closure():
            """
            LBFGS需要闭包函数计算损失
            LBFGS requires closure function for loss calculation
            """
            opt_lbfgs.zero_grad()
            E_pred = model(cfg.inp_coarse, cfg.Z_coarse)

            # 计算相对L2损失 | Calculate relative L2 loss
            loss = relative_l2(cfg.E_coarse_ref.reshape(-1,1), E_pred)
            
            loss.backward()
            return loss

        # 执行优化步骤 | Perform optimization step
        loss = closure()
        opt_lbfgs.step(closure)

        # 定期验证并打印进度 | Periodically validate and print progress
        if i % epo == 0:
            model.eval()
            Epred = model(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(cfg.Nx, cfg.Ny)
            ref_rl2 = relative_l2(cfg.E_ref.cpu().numpy().reshape(-1), Epred.reshape(-1))
            print("lbfgs : {:d} - ref_rl2 {:.4e} ".format(i, ref_rl2))

    # 最终评估 | Final evaluation
    model.eval()
    Epred = model(cfg.inp_fine, cfg.Z_fine).cpu().detach().numpy().reshape(cfg.Nx, cfg.Ny)
    ref_rl2 = relative_l2(cfg.E_ref.cpu().numpy().reshape(-1), Epred.reshape(-1))
    print("lbfgs : {:d} - ref_rl2 {:.4e} ".format(i, ref_rl2))

    return model
