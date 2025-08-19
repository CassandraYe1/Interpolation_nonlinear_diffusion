import numpy as np
import torch
from utils import *
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
    rl2_loss = []
    opt_adam = torch.optim.Adam(model.parameters(), lr=lr*0.1)
    opt_lbfgs = torch.optim.LBFGS(model.parameters(), lr=lr, line_search_fn='strong_wolfe')
    
    for warmup in range(Nfit):
       opt_adam.zero_grad()
       E_pred = model(cfg.inp_coarse, cfg.Z_coarse_bool)

       # 计算相对L2损失 | Calculate relative L2 loss
       loss = relative_l2(cfg.E_coarse_ref.reshape(-1,1), E_pred)

       loss.backward()
       opt_adam.step()

       if warmup % 10 == 0:
            model.eval()

            # 在各边界和内部点进行预测 | Predict on boundaries and internal points
            El_pred = model(cfg.inp_l, cfg.Zl_bool)
            Er_pred = model(cfg.inp_r, cfg.Zr_bool)
            Et_pred = model(cfg.inp_t, cfg.Zt_bool)
            Eb_pred = model(cfg.inp_b, cfg.Zb_bool)

            # 计算各项损失 | Calculate various losses
            lbc_loss = relative_l2(cfg.El, El_pred)
            rbc_loss = relative_l2(cfg.Er, Er_pred)
            tbc_loss = relative_l2(cfg.Et, Et_pred)
            bbc_loss = relative_l2(cfg.Eb, Eb_pred)

            # 精细网格评估 | Fine grid evaluation
            Epred = model(cfg.inp_fine, cfg.Z_fine_bool).detach().reshape(cfg.Nx,cfg.Ny)
            ref_rl2 = relative_l2(cfg.E_ref.reshape(-1), Epred.reshape(-1))
            print("adam : {:d} - ref_rl2 {:.4e} - lbc {:.4e} - rbc {:.4e} - tbc {:.4e} - bbc {:.4e}".format(
                  warmup, ref_rl2, lbc_loss, rbc_loss, tbc_loss, bbc_loss))
    
    for i in range(Nfit):
       model.train()

       def closure():
            """
            LBFGS需要闭包函数计算损失
            LBFGS requires closure function for loss calculation
            """
            opt_lbfgs.zero_grad()
            E_pred = model(cfg.inp_coarse, cfg.Z_coarse_bool)

            # 计算相对L2损失 | Calculate relative L2 loss
            loss = relative_l2(cfg.E_coarse_ref.reshape(-1,1), E_pred)

            loss.backward()
            return loss

       # 执行优化步骤 | Perform optimization step
       loss = closure()
       opt_lbfgs.step(closure)

       model.eval()

       # 在各边界和内部点进行预测 | Predict on boundaries and internal points
       El_pred = model(cfg.inp_l, cfg.Zl_bool)
       Er_pred = model(cfg.inp_r, cfg.Zr_bool)
       Et_pred = model(cfg.inp_t, cfg.Zt_bool)
       Eb_pred = model(cfg.inp_b, cfg.Zb_bool)

       # 计算各项损失 | Calculate various losses
       lbc_loss = relative_l2(cfg.El, El_pred)
       rbc_loss = relative_l2(cfg.Er, Er_pred)
       tbc_loss = relative_l2(cfg.Et, Et_pred)
       bbc_loss = relative_l2(cfg.Eb, Eb_pred)

       # 精细网格评估 | Fine grid evaluation
       Epred = model(cfg.inp_fine, cfg.Z_fine_bool).detach().reshape(cfg.Nx,cfg.Ny)
       ref_rl2 = relative_l2(cfg.E_ref.reshape(-1), Epred.reshape(-1))
       rl2_loss.append(ref_rl2)

       # 定期验证并打印进度 | Periodically validate and print progress
       if i % epo == 0:
            print("lbfgs : {:d} - ref_rl2 {:.4e} - lbc {:.4e} - rbc {:.4e} - tbc {:.4e} - bbc {:.4e}".format(
                  i, ref_rl2, lbc_loss, rbc_loss, tbc_loss, bbc_loss))
    
    # 最终评估 | Final evaluation
    model.eval()

    # 在各边界和内部点进行预测 | Predict on boundaries and internal points
    El_pred = model(cfg.inp_l, cfg.Zl_bool)
    Er_pred = model(cfg.inp_r, cfg.Zr_bool)
    Et_pred = model(cfg.inp_t, cfg.Zt_bool)
    Eb_pred = model(cfg.inp_b, cfg.Zb_bool)

    # 计算各项损失 | Calculate various losses
    lbc_loss = relative_l2(cfg.El, El_pred)
    rbc_loss = relative_l2(cfg.Er, Er_pred)
    tbc_loss = relative_l2(cfg.Et, Et_pred)
    bbc_loss = relative_l2(cfg.Eb, Eb_pred)

    # 精细网格评估 | Fine grid evaluation
    Epred = model(cfg.inp_fine, cfg.Z_fine_bool).detach().reshape(cfg.Nx,cfg.Ny)
    ref_rl2 = relative_l2(cfg.E_ref.reshape(-1), Epred.reshape(-1))
    print("lbfgs : {:d} - ref_rl2 {:.4e} - lbc {:.4e} - rbc {:.4e} - tbc {:.4e} - bbc {:.4e}".format(
          i, ref_rl2, lbc_loss, rbc_loss, tbc_loss, bbc_loss))
    
    return model, rl2_loss
