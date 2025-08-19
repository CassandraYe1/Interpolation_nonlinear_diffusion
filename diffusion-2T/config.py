import yaml
import torch
torch.set_default_dtype(torch.float64)
from data_loader import *


class Config:
    def __init__(self, region="top_left"):
        """
        默认配置初始化
        Default configuration initialization
        """
        self.region = region
        # 获取项目根目录和配置文件路径 | Get project root and config file path
        project_root = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(project_root, 'config.yaml')
        self.config_path = config_path

        # 初始化各配置组字典 | Initialize configuration group dictionaries
        self.model_settings = {}
        self.grid_settings = {}
        self.network_settings = {}
        self.training_settings = {}
        self.visualization = {}

        # 内部初始化标志 | Internal initialization flag
        self._internal_initialized = False
        self._init_data_vars()
        self.load_from_yaml()

    def _init_data_vars(self):
        """
        初始化所有数据相关变量为None
        Initialize all data-related variables to None
        """
        # 精细网格 | Fine grid
        self.D_ref = None     # 当前时间步扩散系数张量 | Diffusion coefficient tensor from current time step
        self.K_ref = None     # 当前时间步热导率张量 | Thermal conductivity tensor from current time step
        self.E_prev = None    # 前一时间步电场分布张量 | Electric field distribution tensor from previous time step
        self.E_ref = None     # 当前时间步电场分布张量 | Electric field distribution tensor from current time step
        self.T_prev = None    # 前一时间步温度分布张量 | Temperature distribution tensor from previous time step
        self.T_ref = None     # 当前时间步温度分布张量 | Temperature distribution tensor from current time step
        self.sigma_ref = None # 当前时间步电导率张量 | Electrical conductivity tensor from current time step
        self.X = None         # X坐标矩阵 | X-coordinate matrix
        self.Y = None         # Y坐标矩阵 | Y-coordinate matrix
        self.Z = None        # 电离函数矩阵 | Ionization function matrix
        self.Z_bool = None   # 电离函数布尔值矩阵 | Ionization function bool matrix
        self.inp_fine = None  # 输入张量 | Input tensor
        self.Z_fine = None    # 展平后的电离函数布尔值张量 | Flattened ionization function bool tensor

        # 粗网格 | Coarse grid
        self.X_coarse = None         # X坐标矩阵 | X-coordinate matrix
        self.Y_coarse = None         # Y坐标矩阵 | Y-coordinate matrix
        self.Z_coarse = None      # 电离函数布尔值矩阵 | Ionization function bool matrix
        self.Z_coarse_bool = None      # 电离函数布尔值矩阵 | Ionization function bool matrix
        self.inp_coarse = None       # 输入张量 | Input tensor
        self.D_coarse = None         # 当前时间步扩散系数张量 | Diffusion coefficient tensor from current time step
        self.K_coarse = None         # 当前时间步热导率张量 | Thermal conductivity tensor from current time step
        self.E_coarse_prev = None    # 前一时间步电场分布张量 | Electric field distribution tensor from previous time step
        self.E_coarse_ref = None     # 当前时间步电场分布张量 | Electric field distribution tensor from current time step
        self.T_coarse_prev = None    # 前一时间步温度分布张量 | Temperature distribution tensor from previous time step
        self.T_coarse_ref = None     # 当前时间步温度分布张量 | Temperature distribution tensor from current time step
        self.sigma_coarse_ref = None # 当前时间步电导率张量 | Electrical conductivity tensor from current time step

        # 内部点 | Internal points (excluding boundaries)
        self.Xd = None      # X坐标矩阵 | X-coordinate matrix
        self.Yd = None      # Y坐标矩阵 | Y-coordinate matrix
        self.Zd = None    # 电离函数布尔值矩阵 | Ionization function bool matrix
        self.Zd_bool = None    # 电离函数布尔值矩阵 | Ionization function bool matrix
        self.inp_d = None   # 输入张量 | Input tensor
        self.Ed = None      # 当前时间步电场分布张量 | Electric field distribution tensor from current time step
        self.Ed_ = None     # 前一时间步电场分布张量 | Electric field distribution tensor from previous time step
        self.Td = None      # 当前时间步温度分布张量 | Temperature distribution tensor from current time step
        self.Td_ = None     # 前一时间步温度分布张量 | Temperature distribution tensor from previous time step
        self.Dd = None      # 当前时间步扩散系数张量 | Diffusion coefficient tensor from current time step
        self.Kd = None      # 当前时间步热导率张量 | Thermal conductivity tensor from current time step
        self.sigma_d = None # 当前时间步电导率张量 | Electrical conductivity tensor from current time step

        # 左边界 | Left boundary
        self.Xl = None      # X坐标矩阵 | X-coordinate matrix
        self.Yl = None      # Y坐标矩阵 | Y-coordinate matrix
        self.Zl = None    # 电离函数布尔值矩阵 | Ionization function bool matrix
        self.Zl_bool = None    # 电离函数布尔值矩阵 | Ionization function bool matrix
        self.inp_l = None   # 输入张量 | Input tensor
        self.El = None      # 当前时间步电场分布张量 | Electric field distribution tensor from current time step
        self.Tl = None      # 当前时间步温度分布张量 | Temperature distribution tensor from current time step
        self.Dl = None      # 当前时间步扩散系数张量 | Diffusion coefficient tensor from current time step
        self.Kl = None      # 当前时间步热导率张量 | Thermal conductivity tensor from current time step
        self.sigma_l = None # 当前时间步电导率张量 | Electrical conductivity tensor from current time step

        # 右边界 | Right boundary
        self.Xr = None      # X坐标矩阵 | X-coordinate matrix
        self.Yr = None      # Y坐标矩阵 | Y-coordinate matrix
        self.Zr = None    # 电离函数布尔值矩阵 | Ionization function bool matrix
        self.Zr_bool = None    # 电离函数布尔值矩阵 | Ionization function bool matrix
        self.inp_r = None   # 输入张量 | Input tensor
        self.Er = None      # 当前时间步电场分布张量 | Electric field distribution tensor from current time step
        self.Tr = None      # 当前时间步温度分布张量 | Temperature distribution tensor from current time step
        self.Dr = None      # 当前时间步扩散系数张量 | Diffusion coefficient tensor from current time step
        self.Kr = None      # 当前时间步热导率张量 | Thermal conductivity tensor from current time step
        self.sigma_r = None # 当前时间步电导率张量 | Electrical conductivity tensor from current time step

        # 下边界 | Bottom boundary
        self.Xb = None      # X坐标矩阵 | X-coordinate matrix
        self.Yb = None      # Y坐标矩阵 | Y-coordinate matrix
        self.Zb = None    # 电离函数布尔值矩阵 | Ionization function bool matrix
        self.Zb_bool = None    # 电离函数布尔值矩阵 | Ionization function bool matrix
        self.inp_b = None   # 输入张量 | Input tensor
        self.Eb = None      # 当前时间步电场分布张量 | Electric field distribution tensor from current time step
        self.Tb = None      # 当前时间步温度分布张量 | Temperature distribution tensor from current time step
        self.Db = None      # 当前时间步扩散系数张量 | Diffusion coefficient tensor from current time step
        self.Kb = None      # 当前时间步热导率张量 | Thermal conductivity tensor from current time step
        self.sigma_b = None # 当前时间步电导率张量 | Electrical conductivity tensor from current time step

        # 上边界 | Top boundary
        self.Xt = None      # X坐标矩阵 | X-coordinate matrix
        self.Yt = None      # Y坐标矩阵 | Y-coordinate matrix
        self.Zt = None    # 电离函数布尔值矩阵 | Ionization function bool matrix
        self.Zt_bool = None    # 电离函数布尔值矩阵 | Ionization function bool matrix
        self.inp_t = None   # 输入张量 | Input tensor
        self.Et = None      # 当前时间步电场分布张量 | Electric field distribution tensor from current time step
        self.Tt = None      # 当前时间步温度分布张量 | Temperature distribution tensor from current time step
        self.Dt = None      # 当前时间步扩散系数张量 | Diffusion coefficient tensor from current time step
        self.Kt = None      # 当前时间步热导率张量 | Thermal conductivity tensor from current time step
        self.sigma_t = None # 当前时间步电导率张量 | Electrical conductivity tensor from current time step

    def load_from_yaml(self):
        """
        从YAML配置文件加载各项设置
        Load settings from YAML configuration file
        """
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            # 获取各配置组 | Get each configuration group
            self.model_settings = config_data.get('model_settings', {})
            self.grid_settings = config_data.get('grid_settings', {})
            self.network_settings = config_data.get('network_settings', {})
            self.training_settings = config_data.get('training_settings', {})
            self.visualization = config_data.get('visualization', {})

        # 将配置映射到类属性 | Map configurations to class attributes
        self.model_name = self.model_settings.get('model_name', 'zconst-const')
        self.device_name = self.model_settings.get('device_name', 'cuda')
        self.ionization_type = self.model_settings.get('ionization_type', 'zconst')
        self.Nx = self.grid_settings.get('Nx', 257)
        self.Ny = self.grid_settings.get('Ny', 257)
        self.n = self.grid_settings.get('n', 2)
        self.t = self.grid_settings.get('t', 1)
        self.depth = self.network_settings.get('depth', 2)
        self.width = self.network_settings.get('width', 512)
        self.Nfit_reg = self.training_settings['regression'].get('Nfit', 500)
        self.lr_E_reg = self.training_settings['regression'].get('lr_E', 1e-1)
        self.lr_T_reg = self.training_settings['regression'].get('lr_T', 1e-1)
        self.epoch_reg = self.training_settings['regression'].get('epochs', 50)
        self.Nfit_pde = self.training_settings['pde'].get('Nfit', 500)
        self.lr_E_pde = self.training_settings['pde'].get('lr_E', 1)
        self.lr_T_pde = self.training_settings['pde'].get('lr_T', 1)
        self.epoch_pde = self.training_settings['pde'].get('epochs', 10)

    def init_config(self):
        """
        根据配置初始化数据组件
        Initialize data components based on configuration
        """
        # 防止重复初始化 | Prevent re-initialization
        if self._internal_initialized:
            return

        # 验证电离函数类型 | Validate ionization type
        if self.ionization_type not in ['zconst', 'zline', 'zsquare']:
            raise ValueError(f"Invalid ionization_type: {self.ionization_type}")

        self.region_boundaries = {
            "top_left": (0, self.Nx, 0, self.Ny),
            "top_right": (0, self.Nx, 257-self.Ny, 257),
            "bottom_left": (257-self.Nx, 257, 0, self.Ny),
            "bottom_right": (257-self.Nx, 257, 257-self.Ny, 257)
        }
        
        # 加载数据并计算电离函数 | Load data and compute ionization function
        self.D_ref, self.K_ref, self.E_prev, self.E_ref, self.T_prev, self.T_ref, self.sigma_ref, self.X, self.Y = load_data(self, self.region)
        if self.ionization_type == 'zconst':
            self.Z, self.Z_bool = z_const(self.X, self.Y, self)
        elif self.ionization_type == 'zline':
            self.Z, self.Z_bool = z_line(self.X, self.Y, self)
        elif self.ionization_type == 'zsquare':
            self.Z, self.Z_bool = z_square(self.X, self.Y, self)

        # 构建精细网格输入张量 | Construct fine grid input tensor
        self.inp_fine = torch.concat(
            [self.X.reshape(-1,1), self.Y.reshape(-1,1)], 
            axis=1).requires_grad_().cuda()
        self.Z_fine = self.Z
        self.Z_fine_bool = self.Z_bool.reshape(-1,1)

        # 粗网格数据准备 | Prepare coarse grid data
        self.X_coarse = self.X[::self.n,::self.n]
        self.Y_coarse = self.Y[::self.n,::self.n]
        self.Z_coarse = self.Z[::self.n,::self.n]
        self.Z_coarse_bool = self.Z_bool[::self.n,::self.n].reshape(-1,1)
        self.inp_coarse = torch.concat(
            [self.X_coarse.reshape(-1,1), self.Y_coarse.reshape(-1,1)], 
            axis=1).requires_grad_().cuda()
        self.D_coarse = self.D_ref[::self.n,::self.n]
        self.K_coarse = self.K_ref[::self.n,::self.n]
        self.E_coarse_prev = self.E_prev[::self.n,::self.n]
        self.E_coarse_ref = self.E_ref[::self.n,::self.n]
        self.T_coarse_prev = self.T_prev[::self.n,::self.n]
        self.T_coarse_ref = self.T_ref[::self.n,::self.n]
        self.sigma_coarse_ref = self.sigma_ref[::self.n,::self.n]

        # 内部点数据准备 | Prepare internal points data (excluding boundaries)
        self.Xd = self.X[1:-1,1:-1]
        self.Yd = self.Y[1:-1,1:-1]
        self.Zd = self.Z[1:-1,1:-1]
        self.Zd_bool = self.Z_bool[1:-1,1:-1].reshape(-1,1)
        self.inp_d = torch.concat(
            [self.Xd.reshape(-1,1), self.Yd.reshape(-1,1)], 
            axis=1).requires_grad_().cuda()
        self.Ed = self.E_ref[1:-1,1:-1]
        self.Ed_ = self.E_prev[1:-1,1:-1]
        self.Td = self.T_ref[1:-1,1:-1]
        self.Td_ = self.T_prev[1:-1,1:-1]
        self.Dd = self.D_ref[1:-1,1:-1]
        self.Kd = self.K_ref[1:-1,1:-1]
        self.sigma_d = self.sigma_ref[1:-1,1:-1]

        # 左边界数据准备 | Prepare left boundary data
        self.Xl = self.X[:,0]
        self.Yl = self.Y[:,0]
        self.Zl = self.Z[:,0]
        self.Zl_bool = self.Z_bool[:,0].reshape(-1,1)
        self.inp_l = torch.concat(
            [self.Xl.reshape(-1,1), self.Yl.reshape(-1,1)], 
            axis=1).requires_grad_().cuda()
        self.El = self.E_ref[:,[0]]
        self.Tl = self.T_ref[:,[0]]
        self.Dl = self.D_ref[:,[0]]
        self.Kl = self.K_ref[:,[0]]
        self.sigma_l = self.sigma_ref[:,[0]]

        # 右边界数据准备 | Prepare right boundary data
        self.Xr = self.X[:,-1]
        self.Yr = self.Y[:,-1]
        self.Zr = self.Z[:,-1]
        self.Zr_bool = self.Z_bool[:,-1].reshape(-1,1)
        self.inp_r = torch.concat(
            [self.Xr.reshape(-1,1), self.Yr.reshape(-1,1)], 
            axis=1).requires_grad_().cuda()
        self.Er = self.E_ref[:,[-1]]
        self.Tr = self.T_ref[:,[-1]]
        self.Dr = self.D_ref[:,[-1]]
        self.Kr = self.K_ref[:,[-1]]
        self.sigma_r = self.sigma_ref[:,[-1]]

        # 下边界数据准备 | Prepare bottom boundary data
        self.Xb = self.X[0]
        self.Yb = self.Y[0]
        self.Zb = self.Z[0]
        self.Zb_bool = self.Z_bool[0].reshape(-1,1)
        self.inp_b = torch.concat(
            [self.Xb.reshape(-1,1), self.Yb.reshape(-1,1)], 
            axis=1).requires_grad_().cuda()
        self.Eb = self.E_ref[0].reshape(-1,1)
        self.Tb = self.T_ref[0].reshape(-1,1)
        self.Db = self.D_ref[0].reshape(-1,1)
        self.Kb = self.K_ref[0].reshape(-1,1)
        self.sigma_b = self.sigma_ref[0].reshape(-1,1)

        # 上边界数据准备 | Prepare top boundary data
        self.Xt = self.X[-1]
        self.Yt = self.Y[-1]
        self.Zt = self.Z[-1]
        self.Zt_bool = self.Z_bool[-1].reshape(-1,1)
        self.inp_t = torch.concat(
            [self.Xt.reshape(-1,1), self.Yt.reshape(-1,1)], 
            axis=1).requires_grad_().cuda()
        self.Et = self.E_ref[-1].reshape(-1,1)
        self.Tt = self.T_ref[-1].reshape(-1,1)
        self.Dt = self.D_ref[-1].reshape(-1,1)
        self.Kt = self.K_ref[-1].reshape(-1,1)
        self.sigma_t = self.sigma_ref[-1].reshape(-1,1)

        # 标记初始化完成 | Mark initialization as complete
        self._internal_initialized = True
