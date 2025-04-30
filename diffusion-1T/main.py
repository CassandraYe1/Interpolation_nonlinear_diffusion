import numpy as np
import torch 
torch.set_default_dtype(torch.float64)
import argparse
import os
import copy
import time
from model import DeepNN
from config import Config
from utils import set_seed, relative_l2, plot
from train_reg import train_model_reg
from train_pde import train_model_pde


def parse_args():
    """
    参数解析和配置更新
    Parameter parsing and configuration update
    
    Returns:
        argparse.Namespace: 包含所有命令行参数的命名空间对象
                            Namespace object containing all command line arguments
    """
    parser = argparse.ArgumentParser(description="Train the model with flexible parameters")

    # 模型设置参数 | Model settings parameters
    parser.add_argument('--model_name', type=str, default='zconst-const', help='Model name (e.g., zconst-const)')
    parser.add_argument('--device_name', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for training (cuda/cpu)')
    parser.add_argument('--ionization_type', type=str, default='zconst', help='Type of ionization function')
    # 网格设置参数 | Grid settings parameters
    parser.add_argument('--Nx', type=int, default=257, help='Number of grid points on x-axis')
    parser.add_argument('--Ny', type=int, default=257, help='Number of grid points on y-axis')
    parser.add_argument('--n', type=int, default=4, help='Downsampling factor')
    # 网络设置参数 | Network settings parameters
    parser.add_argument('--width', type=int, default=512, help='Number of units in each hidden layer')
    # 回归训练参数 | Regression training parameters
    parser.add_argument('--Nfit_reg', type=int, default=300, help='Number of training iterations for regularization phase')
    parser.add_argument('--lr_reg', type=float, default=1e-2, help='Learning rate for LBFGS optimizer in regularization phase')
    parser.add_argument('--epoch_reg', type=int, default=50, help='Epochs for regularization training')
    # PDE训练参数 | PDE training parameters
    parser.add_argument('--Nfit_pde', type=int, default=200, help='Number of training iterations for PDE phase')
    parser.add_argument('--lr_pde', type=float, default=1e-1, help='Learning rate for LBFGS optimizer in PDE phase')
    parser.add_argument('--epoch_pde', type=int, default=10, help='Epochs for PDE training')
    # 可视化参数 | Visualization parameters
    parser.add_argument('--vmax', type=float, default=0.25, help='Maximum value of error colorbar')
    
    return parser.parse_args()

def override_config_with_args(config: Config, args: argparse.Namespace) -> None:
    """
    用命令行参数覆盖配置类的属性值
    Override config class attributes with command line arguments
    
    Args:
        config: 需要被覆盖的配置类实例
                Config class instance to be overridden
        args: 从argparse解析得到的命名空间对象
              Namespace object parsed from argparse
        
    Raises:
        AttributeError: 当尝试覆盖config中不存在的属性时抛出
                      When attempting to override non-existent attributes in config
        ValueError: 当参数值不合法时抛出
                   When parameter values are invalid
    """
    # 将命名空间转为字典 | Convert namespace to dictionary
    args_dict = vars(args)
    
    valid_args = {}
    for arg_name, arg_value in args_dict.items():
        # 跳过None值和config路径参数 | Skip None values and config path parameters
        if arg_value is not None and arg_name != 'config':
            valid_args[arg_name] = arg_value
    
    # 逐个覆盖配置项 | Override config items one by one
    for param_name, param_value in valid_args.items():
        if not hasattr(config, param_name):
            raise AttributeError(
                f"Config class does not have attribute '{param_name}'. "
                f"Please ensure the Config class defines this attribute or check parameter spelling."
            )
        
        try:
            setattr(config, param_name, param_value)
        except Exception as e:
            raise ValueError(
                f"Cannot set parameter {param_name} to {param_value}. "
                f"Type or value may be incompatible: {str(e)}"
            )
        
        # 可选：打印调试信息
    print("Configuration initialization completed.")


if __name__ == "__main__":
    # === 配置初始化阶段 === | === Configuration Initialization Phase ===
    # 解析命令行参数 | Parse command line arguments
    args = parse_args()
    # 创建并更新配置对象 | Create and update config object
    cfg = Config()
    override_config_with_args(cfg, args)
    cfg.init_config()

    # === 输出目录准备 === | === Output Directory Preparation ===
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, 'results', cfg.model_name)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # === 第一阶段训练：仅使用粗网格参考数据 === 
    # === Phase 1 Training: Coarse-grid Reference Data Only ===
    set_seed(0)
    print('Train by coarse-grid data:')
    model = DeepNN(cfg=cfg).to(cfg.device_name)

    # 训练并计时 | Train and time
    start_time = time.time()
    model = train_model_reg(model, cfg=cfg, Nfit=cfg.Nfit_reg, lr=cfg.lr_reg, epo=cfg.epoch_reg)
    end_time = time.time()
    training_time_reg = end_time - start_time
    print(f"Regression training time: {training_time_reg:.6e} seconds")

    # 保存第一阶段结果 | Save phase 1 results
    E_reg = model(cfg.inp_fine, cfg.Z_fine).detach().cpu().reshape(cfg.Nx, cfg.Ny)
    np.save(os.path.join(data_path, 'sol_reg'), E_reg)
    torch.save(model.state_dict(), os.path.join(data_path, 'model_reg.pt'))

    # === 第二阶段训练：结合粗网格数据和PDE约束 ===
    # === Phase 2 Training: Coarse-grid Data + PDE Constraints ===
    set_seed(50)
    print('Train by both coarse-grid data and PDE residual:')
    model_cur = DeepNN(cfg=cfg).to(cfg.device_name)
    model_cur.load_state_dict(copy.deepcopy(model.state_dict()))

    # 训练并计时 | Train and time
    start_time = time.time()
    model_cur = train_model_pde(model_cur, cfg=cfg, Nfit=cfg.Nfit_pde, lr=cfg.lr_pde, epo=cfg.epoch_pde)
    end_time = time.time()
    training_time_pinn = end_time - start_time
    print(f"PINN training time: {training_time_pinn:.6e} seconds")

    # 保存第二阶段结果 | Save phase 2 results
    E_pinn = model_cur(cfg.inp_fine, cfg.Z_fine).detach().cpu().reshape(cfg.Nx, cfg.Ny)
    np.save(os.path.join(data_path, 'sol_pinn'), E_pinn)
    torch.save(model_cur.state_dict(), os.path.join(data_path, 'model_pinn.pt'))

    # === 结果评估 === | === Result Evaluation ===
    X = cfg.X.detach().cpu()
    Y = cfg.Y.detach().cpu()
    E_ref = cfg.E_ref.cpu()
    print('Regression Solution rl2: {:.4e}'.format(relative_l2(E_ref, E_reg)))
    print('PINN Solution rl2: {:.4e}'.format(relative_l2(E_ref, E_pinn)))

    # === 可视化 === | === Visualization ===
    plot(E_reg, E_pinn, E_ref, X, Y, data_path, vmax=cfg.vmax)
