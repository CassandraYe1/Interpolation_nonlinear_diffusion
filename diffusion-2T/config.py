import yaml
import torch
torch.set_default_dtype(torch.float64)
from data_loader import *


class Config:
    def __init__(self):
        """
        默认配置初始化
        Default configuration initialization
        """
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
        D_ref = None
        K_ref = None
        E_prev = None
        E_ref = None
        T_prev = None
        T_ref = None
        sigma_ref = None
        X = None
        Y = None
        Z = None
        inp_fine = None
        Z_fine = None

        X_coarse = None
        Y_coarse = None
        Z_coarse = None
        inp_coarse = None
        Z_coarse = None
        D_coarse = None
        K_coarse = None
        E_coarse_prev = None
        E_coarse_ref = None
        T_coarse_prev = None
        T_coarse_ref = None
        sigma_coarse_ref = None

        Xd = None
        Yd = None
        Zd = None
        inp_d = None
        Ed = None
        Ed_ = None
        Td = None
        Td_ = None
        Dd = None
        Kd = None
        sigma_d = None

        Xl = None
        Yl = None
        Zl = None
        inp_l = None
        El = None
        Tl = None
        Dl = None
        Kl = None
        sigma_l = None

        Xr = None
        Yr = None
        Zr = None
        inp_r = None
        Er = None
        Tr = None
        Dr = None
        Kr = None
        sigma_r = None

        Xb = None
        Yb = None
        Zb = None
        inp_b = None
        Eb = None
        Tb = None
        Db = None
        Kb = None
        sigma_b = None

        Xt = None
        Yt = None
        Zt = None
        inp_t = None
        Et = None
        Tt = None
        Dt = None
        Kt = None
        sigma_t = None

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
        self.n = self.grid_settings.get('n', 4)
        self.depth = self.network_settings.get('depth', 2)
        self.width = self.network_settings.get('width', 512)
        self.Nfit_reg = self.training_settings['regression'].get('Nfit', 300)
        self.lr_E_reg = self.training_settings['regression'].get('lr_E', 1e-2)
        self.lr_T_reg = self.training_settings['regression'].get('lr_T', 1e-2)
        self.epoch_reg = self.training_settings['regression'].get('epochs', 50)
        self.Nfit_pde = self.training_settings['pde'].get('Nfit', 200)
        self.lr_E_pde = self.training_settings['pde'].get('lr_E', 1e-1)
        self.lr_T_pde = self.training_settings['pde'].get('lr_T', 1e-1)
        self.epoch_pde = self.training_settings['pde'].get('epochs', 10)
        self.vmax_E = self.visualization.get('vmax_E', 0.25)
        self.vmax_T = self.visualization.get('vmax_T', 0.25)

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
        
        # 加载数据并计算电离函数 | Load data and compute ionization function
        self.D_ref, self.K_ref, self.E_prev, self.E_ref, self.T_prev, self.T_ref, self.sigma_ref, self.X, self.Y = load_data(self)
        if self.ionization_type == 'zconst':
            self.Z = z_const(self.X, self.Y, self)
        elif self.ionization_type == 'zline':
            self.Z = z_line(self.X, self.Y, self)
        elif self.ionization_type == 'zsquare':
            self.Z = z_square(self.X, self.Y, self)

        # 构建精细网格输入张量 | Construct fine grid input tensor
        self.inp_fine = torch.concat(
            [self.X.reshape(-1,1), self.Y.reshape(-1,1)], 
            axis=1).requires_grad_().cuda()
        self.Z_fine = self.Z.reshape(-1,1)

        # 粗网格数据准备 | Prepare coarse grid data
        self.X_coarse = self.X[::self.n,::self.n]
        self.Y_coarse = self.Y[::self.n,::self.n]
        self.Z_coarse = self.Z[::self.n,::self.n].reshape(-1,1)
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
        self.Zd = self.Z[1:-1,1:-1].reshape(-1,1)
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
        self.Zl = self.Z[:,0].reshape(-1,1)
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
        self.Zr = self.Z[:,-1].reshape(-1,1)
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
        self.Zb = self.Z[0].reshape(-1,1)
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
        self.Zt = self.Z[-1].reshape(-1,1)
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
