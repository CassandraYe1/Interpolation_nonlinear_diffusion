import numpy as np
import torch 
torch.set_default_dtype(torch.float64)
import argparse
import os
import copy
import time
from model import DeepNN
from config import Config
from utils import set_seed, relative_l2, plot_E, plot_T
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
    parser.add_argument('--n', type=int, default=2, help='Downsampling factor')
    parser.add_argument('--t', type=float, default=1, help='Target time step')
    # 网络设置参数 | Network settings parameters
    parser.add_argument('--depth', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--width', type=int, default=512, help='Number of units in each hidden layer')
    # 回归训练参数 | Regression training parameters
    parser.add_argument('--Nfit_reg', type=int, default=300, help='Number of training iterations for regularization phase')
    parser.add_argument('--lr_E_reg', type=float, default=1e-2, help='Learning rate for LBFGS optimizer of E in regularization phase')
    parser.add_argument('--lr_T_reg', type=float, default=1e-2, help='Learning rate for LBFGS optimizer of T in regularization phase')
    parser.add_argument('--epoch_reg', type=int, default=50, help='Epochs for regularization training')
    # PDE训练参数 | PDE training parameters
    parser.add_argument('--Nfit_pde', type=int, default=200, help='Number of training iterations for PDE phase')
    parser.add_argument('--lr_E_pde', type=float, default=1e-1, help='Learning rate for LBFGS optimizer of E in PDE phase')
    parser.add_argument('--lr_T_pde', type=float, default=1e-1, help='Learning rate for LBFGS optimizer of T in PDE phase')
    parser.add_argument('--epoch_pde', type=int, default=10, help='Epochs for PDE training')
    # 可视化参数 | Visualization parameters
    parser.add_argument('--vmax_E', type=float, default=0.25, help='Maximum value of error colorbar E')
    parser.add_argument('--vmax_T', type=float, default=0.25, help='Maximum value of error colorbar T')
    
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
        model_E = DeepNN(cfg=cfg).to(cfg.device_name)
        model_T = DeepNN(cfg=cfg).to(cfg.device_name)

        # 训练并计时 | Train and time
        start_time = time.time()
        [model_E, model_T, rl2_loss_reg_E, rl2_loss_reg_T] = train_model_reg(model_E, model_T, cfg=cfg, Nfit=cfg.Nfit_reg, lr_E=cfg.lr_E_reg, lr_T=cfg.lr_T_reg, epo=cfg.epoch_reg)
        #model_E.load_state_dict(torch.load(os.path.join(data_path, 'model_reg_E_'+region+'.pt')))
        #model_T.load_state_dict(torch.load(os.path.join(data_path, 'model_reg_T_'+region+'.pt')))
        end_time = time.time()
        training_time_reg = end_time - start_time
        print(f"Regression training time: {training_time_reg:.6e} seconds")

        # 保存第一阶段结果 | Save phase 1 results
        E_reg = model_E(cfg.inp_fine, cfg.Z_fine_bool).detach().cpu().reshape(cfg.Nx,cfg.Ny)
        T_reg = model_T(cfg.inp_fine, cfg.Z_fine_bool).detach().cpu().reshape(cfg.Nx,cfg.Ny)
        np.save(os.path.join(data_path, 'sol_reg_E_'+region), E_reg)
        np.save(os.path.join(data_path, 'sol_reg_T_'+region), T_reg)
        rl2_loss_reg_E = [tensor.cpu() for tensor in rl2_loss_reg_E]
        rl2_loss_reg_T = [tensor.cpu() for tensor in rl2_loss_reg_T]
        np.save(os.path.join(data_path, 'loss_reg_E_'+region), rl2_loss_reg_E)
        np.save(os.path.join(data_path, 'loss_reg_T_'+region), rl2_loss_reg_T)
        torch.save(model_E.state_dict(), os.path.join(data_path, 'model_reg_E_'+region+'.pt'))
        torch.save(model_T.state_dict(), os.path.join(data_path, 'model_reg_T_'+region+'.pt'))
        
        # === 第二阶段训练：结合粗网格数据和PDE约束 ===
        # === Phase 2 Training: Coarse-grid Data + PDE Constraints ===
        print('Train by both coarse-grid data and PDE residual:')
        model_E_cur = DeepNN(cfg=cfg).to(cfg.device_name)
        model_E_cur.load_state_dict(copy.deepcopy(model_E.state_dict()))
        model_T_cur = DeepNN(cfg=cfg).to(cfg.device_name)
        model_T_cur.load_state_dict(copy.deepcopy(model_T.state_dict()))

        # 训练并计时 | Train and time
        start_time = time.time()
        [model_E_cur, model_T_cur, rl2_loss_pinn_E, rl2_loss_pinn_T] = train_model_pde(model_E_cur, model_T_cur, cfg=cfg, Nfit=cfg.Nfit_pde, lr_E=cfg.lr_E_pde, lr_T=cfg.lr_T_pde, epo=cfg.epoch_pde)
        end_time = time.time()
        training_time_pinn = end_time - start_time
        print(f"PINN training time: {training_time_pinn:.6e} seconds")

        # 保存第二阶段结果 | Save phase 2 results
        E_pinn = model_E_cur(cfg.inp_fine, cfg.Z_fine_bool).detach().cpu().reshape(cfg.Nx,cfg.Ny)
        T_pinn = model_T_cur(cfg.inp_fine, cfg.Z_fine_bool).detach().cpu().reshape(cfg.Nx,cfg.Ny)
        np.save(os.path.join(data_path, 'sol_pinn_E_'+region), E_pinn)
        np.save(os.path.join(data_path, 'sol_pinn_T_'+region), T_pinn)
        rl2_loss_pinn_E = [tensor.cpu() for tensor in rl2_loss_pinn_E]
        rl2_loss_pinn_T = [tensor.cpu() for tensor in rl2_loss_pinn_T]
        np.save(os.path.join(data_path, 'loss_pinn_E_'+region), rl2_loss_pinn_E)
        np.save(os.path.join(data_path, 'loss_pinn_T_'+region), rl2_loss_pinn_T)
        torch.save(model_E_cur.state_dict(), os.path.join(data_path, 'model_pinn_E_'+region+'.pt'))
        torch.save(model_T_cur.state_dict(), os.path.join(data_path, 'model_pinn_T_'+region+'.pt'))
        
    # === 结果评估 === | === Result Evaluation ===
    data_path_test = os.path.join(project_root, 'data', cfg.model_name)
    sol_E = np.load(os.path.join(data_path_test, 'sol_E.npy'))
    sol_T = np.load(os.path.join(data_path_test, 'sol_T.npy'))
    X = np.load(os.path.join(data_path_test, 'X.npy'))
    Y = np.load(os.path.join(data_path_test, 'Y.npy'))
    E_ref = torch.tensor(sol_E[int(1000*cfg.t-1)])
    T_ref = torch.tensor(sol_T[int(1000*cfg.t-1)])
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    E_reg_final = torch.zeros((257,257))
    E_pinn_final = torch.zeros((257,257))
    T_reg_final = torch.zeros((257,257))
    T_pinn_final = torch.zeros((257,257))
    count = torch.zeros((257,257))
    for region in regions:
        x_start, x_end, y_start, y_end = cfg.region_boundaries[region]
        
        E_reg = np.load(os.path.join(data_path, 'sol_reg_E_' + region + '.npy'))
        E_pinn = np.load(os.path.join(data_path, 'sol_pinn_E_' + region + '.npy'))
        T_reg = np.load(os.path.join(data_path, 'sol_reg_T_' + region + '.npy'))
        T_pinn = np.load(os.path.join(data_path, 'sol_pinn_T_' + region + '.npy'))
        
        E_reg_final[x_start:x_end, y_start:y_end] += torch.tensor(E_reg)
        E_pinn_final[x_start:x_end, y_start:y_end] += torch.tensor(E_pinn)
        T_reg_final[x_start:x_end, y_start:y_end] += torch.tensor(T_reg)
        T_pinn_final[x_start:x_end, y_start:y_end] += torch.tensor(T_pinn)
        count[x_start:x_end, y_start:y_end] += 1 
    E_reg_final = torch.where(count != 0, E_reg_final / count, torch.zeros_like(E_reg_final))
    E_pinn_final = torch.where(count != 0, E_pinn_final / count, torch.zeros_like(E_pinn_final))
    T_reg_final = torch.where(count != 0, T_reg_final / count, torch.zeros_like(T_reg_final))
    T_pinn_final = torch.where(count != 0, T_pinn_final / count, torch.zeros_like(T_pinn_final))
    np.save(os.path.join(data_path, 'sol_reg_E'), E_reg_final)
    np.save(os.path.join(data_path, 'sol_reg_T'), T_reg_final)
    np.save(os.path.join(data_path, 'sol_pinn_E'), E_pinn_final)
    np.save(os.path.join(data_path, 'sol_pinn_T'), T_pinn_final)

    print('E: Regression Solution rl2: {:.4e}'.format(relative_l2(E_ref, E_reg_final)))
    print('E: PINN Solution rl2: {:.4e}'.format(relative_l2(E_ref, E_pinn_final)))
    print('T: Regression Solution rl2: {:.4e}'.format(relative_l2(T_ref, T_reg_final)))
    print('T: PINN Solution rl2: {:.4e}'.format(relative_l2(T_ref, T_pinn_final)))

    # === 可视化 === | === Visualization ===
    plot_E(E_reg_final, E_pinn_final, E_ref, X, Y, data_path, vmax=cfg.vmax_E, t=cfg.t)
    plot_T(T_reg_final, T_pinn_final, T_ref, X, Y, data_path, vmax=cfg.vmax_T, t=cfg.t)
