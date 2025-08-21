import numpy as np
import torch 
torch.set_default_dtype(torch.float64)
import argparse
import os
import copy
import time
from model import DeepNN
from config import Config
from utils import *
from train_reg import *
from train_pde import *


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
    parser.add_argument('--model_name', type=str, default='zconst_const', help='Model name (e.g., zconst_const)')
    parser.add_argument('--device_name', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for training (cuda/cpu)')
    parser.add_argument('--ionization_type', type=str, default='zconst', help='Type of ionization function')
    # 网格设置参数 | Grid settings parameters
    parser.add_argument('--Nx', type=int, default=257, help='Number of grid points on x-axis')
    parser.add_argument('--Ny', type=int, default=257, help='Number of grid points on y-axis')
    parser.add_argument('--n', type=int, default=2, help='Downsampling factor')
    parser.add_argument('--t', type=float, default=1, help='Target time step')
    # 网络设置参数 | Network settings parameters
    parser.add_argument('--depth', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--width', type=int, default=512, help='Number of units in each hidden layer')
    # 回归训练参数 | Regression training parameters
    parser.add_argument('--Nfit_reg', type=int, default=500, help='Number of training iterations for regularization phase')
    parser.add_argument('--lr_E_reg', type=float, default=1e-1, help='Learning rate for LBFGS optimizer in regularization phase')
    parser.add_argument('--epoch_reg', type=int, default=50, help='Epochs for regularization training')
    # PDE训练参数 | PDE training parameters
    parser.add_argument('--Nfit_pde', type=int, default=500, help='Number of training iterations for PDE phase')
    parser.add_argument('--lr_E_pde', type=float, default=1, help='Learning rate for LBFGS optimizer in PDE phase')
    parser.add_argument('--epoch_pde', type=int, default=10, help='Epochs for PDE training')
    
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
        
    print("Configuration initialization completed.")


if __name__ == "__main__":
    # === 配置初始化阶段 === | === Configuration Initialization Phase ===
    # 设置一个固定的种子以实现可重复性 | Set a fixed seed for reproducibility
    set_seed(0)
    np.random.seed(0)
    # 解析命令行参数 | Parse command line arguments
    args = parse_args()
    regions = ["top_left", "top_right", "bottom_left", "bottom_right"] # "top_left", "top_right", "bottom_left", "bottom_right"
    for region in regions:
        print(f"\n=== Processing {region} region ===")
        # 创建并更新配置对象 | Create and update config object
        cfg = Config(region)
        override_config_with_args(cfg, args)
        cfg.init_config()

        # === 输出目录准备 === | === Output Directory Preparation ===
        project_root = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(project_root, 'results', cfg.model_name, f't={cfg.t}')
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        # === 第一阶段训练：仅使用粗网格参考数据 === 
        # === Phase 1 Training: Coarse-grid Reference Data Only ===
        print('Train by coarse-grid data:')
        model = DeepNN(cfg=cfg).to(cfg.device_name)

        # 训练并计时 | Train and time
        start_time = time.time()
        [model, rl2_loss_reg] = train_model_reg(model, cfg=cfg, Nfit=cfg.Nfit_reg, lr=cfg.lr_E_reg, epo=cfg.epoch_reg)
        end_time = time.time()
        training_time_reg = end_time - start_time
        print(f"Regression training time: {training_time_reg:.6e} seconds")

        # 保存第一阶段结果 | Save phase 1 results
        E_reg = model(cfg.inp_fine, cfg.Z_fine_bool).detach().cpu().reshape(cfg.Nx,cfg.Ny)
        np.save(os.path.join(data_path, 'sol_reg_'+region), E_reg)
        rl2_loss_reg = [tensor.cpu() for tensor in rl2_loss_reg]
        np.save(os.path.join(data_path, 'loss_reg_'+region), rl2_loss_reg)
        torch.save(model.state_dict(), os.path.join(data_path, 'model_reg_'+region+'.pt'))

        # === 第二阶段训练：结合粗网格数据和PDE约束 ===
        # === Phase 2 Training: Coarse-grid Data + PDE Constraints ===
        print('Train by both coarse-grid data and PDE residual:')
        model_cur = DeepNN(cfg=cfg).to(cfg.device_name)
        model_cur.load_state_dict(copy.deepcopy(model.state_dict()))

        # 训练并计时 | Train and time
        start_time = time.time()
        [model_cur, rl2_loss_pinn] = train_model_pde(model_cur, cfg=cfg, Nfit=cfg.Nfit_pde, lr=cfg.lr_E_pde, epo=cfg.epoch_pde)
        end_time = time.time()
        training_time_pinn = end_time - start_time
        print(f"PINN training time: {training_time_pinn:.6e} seconds")

        # 保存第二阶段结果 | Save phase 2 results
        E_pinn = model_cur(cfg.inp_fine, cfg.Z_fine_bool).detach().cpu().reshape(cfg.Nx,cfg.Ny)
        np.save(os.path.join(data_path, 'sol_pinn_'+region), E_pinn)
        rl2_loss_pinn = [tensor.cpu() for tensor in rl2_loss_pinn]
        np.save(os.path.join(data_path, 'loss_pinn_'+region), rl2_loss_pinn)
        torch.save(model_cur.state_dict(), os.path.join(data_path, 'model_pinn_'+region+'.pt'))
        
    # === 结果评估 === | === Result Evaluation ===
    data_path_test = os.path.join(project_root, 'data', cfg.model_name)
    sol = np.load(os.path.join(data_path_test, 'sol.npy'))
    X = np.load(os.path.join(data_path_test, 'X.npy'))
    Y = np.load(os.path.join(data_path_test, 'Y.npy'))
    E_ref = torch.tensor(sol[int(1000*cfg.t-1)])
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    E_reg_final = torch.zeros((257,257))
    E_pinn_final = torch.zeros((257,257))
    count = torch.zeros((257,257))
    for region in regions:
        x_start, x_end, y_start, y_end = cfg.region_boundaries[region]
        
        E_reg = np.load(os.path.join(data_path, 'sol_reg_' + region + '.npy'))
        E_pinn = np.load(os.path.join(data_path, 'sol_pinn_' + region + '.npy'))
        
        E_reg_final[x_start:x_end, y_start:y_end] += torch.tensor(E_reg)
        E_pinn_final[x_start:x_end, y_start:y_end] += torch.tensor(E_pinn)
        count[x_start:x_end, y_start:y_end] += 1 
    E_reg_final = torch.where(count != 0, E_reg_final / count, torch.zeros_like(E_reg_final))
    E_pinn_final = torch.where(count != 0, E_pinn_final / count, torch.zeros_like(E_pinn_final))
    np.save(os.path.join(data_path, 'sol_reg'), E_reg_final)
    np.save(os.path.join(data_path, 'sol_pinn'), E_pinn_final)

    print('Regression Solution rl2: {:.4e}'.format(relative_l2(E_ref, E_reg_final)))
    print('PINN Solution rl2: {:.4e}'.format(relative_l2(E_ref, E_pinn_final)))
