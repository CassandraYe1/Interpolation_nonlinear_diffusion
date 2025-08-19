# 非线性辐射扩散问题的神经网络超分辨率算法研究

## 超分辨率方法：

超分辨率技术在科学计算领域展现出突破传统数值方法效率瓶颈的革命性潜力。特别是在涉及多尺度、强非线性特征的物理场重构任务中，传统基于插值或数据驱动的超分辨率方法往往因缺乏物理规律约束，导致重构结果存在非物理解或守恒性破坏等根本缺陷。这一局限性在辐射输运、湍流模拟等对物理一致性有严格要求的领域尤为突出，严重制约了超分辨率技术在实际工程上的应用。

针对这些问题，本项目提出了一种超分辨率神经网络框架，用神经网络直接学习从低分辨率网格到高分辨率网格的映射关系。该网络构建了"粗网格输入→网络预测→物理校正"的架构，在传统数据驱动损失的基础上，引入基于方程的物理约束，确保预测解严格满足物理规律，在保证非线性辐射扩散问题求解精度的同时提升求解效率。

## 非线性辐射扩散问题：

非线性辐射扩散问题是一类典型的多尺度强耦合输运方程，其核心在于描述辐射能量与物质能量通过光子输运产生的非线性能量交换过程。该过程的控制方程可表述为：

### 单温问题：

$$
\begin{aligned}
   & \frac{\partial E}{\partial t}-\nabla\cdot(D_L\nabla E) = 0, \quad(x,y,t)\in\Omega\times[0,1] \\
   & 0.5E+D_L\nabla E\cdot n = \beta(x,y,t), \quad(x,y,t)\in\lbrace x=0\rbrace\times[0,1] \\
   & 0.5E+D_L\nabla E\cdot n = 0, \quad(x,y,t)\in\partial\Omega\setminus\lbrace x=0\rbrace\times[0,1] \\
   & E|_{t=0} = g(x,y,0)
\end{aligned}
$$

其中 $\Omega = [0,1]\times[0,1]$ ；辐射扩散系数 $D_L$ 选用限流形式，即 $D_L = \frac{1}{3\sigma_{\alpha}+\frac{|\nabla E|}{E}}, \sigma_{\alpha} = \frac{z^3}{E^{3/4}}$ 。

方程初值条件设置为常数初值，即 $g(x,y,0) = 0.01$ 。

边值条件在左边界处设置为线性边值，即 $\beta(x,y,t) = \max$ { $20t,10$ }。

### 双温问题：

$$
\begin{aligned}
   & \frac{\partial E}{\partial t} - \nabla \cdot (D_L \nabla E) = \sigma_{\alpha}(T^4 - E), \quad(x,y,t)\in\Omega\times[0,1] \\
   & \frac{\partial T}{\partial t} - \nabla \cdot (K_L \nabla T) = \sigma_{\alpha}(E - T^4), \quad(x,y,t)\in\Omega\times[0,1] \\
   & 0.5E + D_L \nabla E \cdot n = \beta(x,y,t), \quad (x,y,t) \in \lbrace x=0 \rbrace \times [0,1] \\
   & 0.5E + D_L \nabla E \cdot n = 0, \quad (x,y,t) \in \partial\Omega \setminus \lbrace x=0 \rbrace \times [0,1] \\
   & K_L \nabla T \cdot n = 0, \quad (x,y,t) \in \partial\Omega \times [0,1] \\
   & E\vert_{t=0} = g(x,y,0) \\
   & T^4\vert_{t=0} = g(x,y,0)
\end{aligned}
$$

其中 $\Omega = [0,1]\times[0,1]$ ；辐射扩散系数 $D_L, K_L$ 同样选用限流形式，即 $D_L = \frac{1}{3\sigma_{\alpha}+\frac{|\nabla E|}{E}}, \sigma_{\alpha} = \frac{z^3}{E^{3/4}}, K_L = \frac{T^4}{T^{3/2}z+T^{5/2}|\nabla T|}$ 。

方程初值条件设置为高斯初值，即 $g(x,y,0) = 0.01+100e^{-(x^2+y^2)/0.01}$ 。

边值条件在左边界处设置为零边值，即 $\beta(x,y,t) = 0$ 。

## 神经网络超分辨率算法设计：

结合低分辨率数值解 $E_{\text{coarse}}$ 以及方程本身，本项目设计了针对单一方程的新型神经网络解法。

以单温问题为例，用向后差分处理方程中的时间偏导项，从而将目标问题在时间上进行离散。同时，参考Picard迭代的格式，提出了一种基于粗网格参考解的显式扩散系数更新方法，对非线性项进行线性化处理，得到如下方程：

$$
\begin{equation}
   E^n-E^{n-1}-D^n\nabla\cdot(\nabla E^n)\Delta t = 0
\end{equation}
$$

用同样的方法处理双温问题，得到如下方程：

$$
\begin{aligned}
   & E^n-E^{n-1}-D^n\nabla\cdot(\nabla E^n)\Delta t-\sigma_{\alpha}(T^4-E)\Delta t = 0 \\
   & T^n-T^{n-1}-K^n\nabla\cdot(\nabla T^n)\Delta t-\sigma_{\alpha}(E-T^4)\Delta t = 0
\end{aligned}
$$

该算法的神经网络优化可以分为两个阶段。第一阶段为回归训练，该阶段通过优化基于低分辨率数据构建的损失函数 $L_{\text{reg}}$ ，得到粗略的解函数，确保网络在已知粗网格数据对应网格点处的预测足够准确；第二阶段为PDE训练，该阶段在 $L_{\text{reg}}$ 的基础上引入基于目标方程式构建的包含物理约束的损失函数 $L_{\text{pde}}$ ，从而构建复合损失函数 $L_{\text{reg+pde}}$ ，进一步提升精度，且确保预测结果满足物理规律。单温问题的损失函数如下：

$$
\begin{aligned}
   & L_{\text{reg+pde}} = L_{\text{reg}}+wL_{\text{pde}} \\
   & L_{\text{reg}} = \frac{\Vert E^n_{\text{coarse}}-E^n \Vert_2}{\Vert E^n_{\text{coarse}} \Vert_2} \\
   & L_{\text{pde}} = \Vert E^n-D^n_{\text{coarse}}\nabla\cdot(\nabla E^n)\Delta t-E^{n-1}_{\text{coarse}} \Vert_2^2
\end{aligned}
$$

双温问题的损失函数如下：

$$
\begin{aligned}
   & L_{\text{reg+pde}} = L_{\text{reg}} + wL_{\text{pde}} \\
   & L_{\text{reg}} = \frac{\Vert E^n_{\text{coarse}}-E^n \Vert_2}{\Vert E^n_{\text{coarse}} \Vert_2} + \frac{\Vert T^n_{\text{coarse}}-T^n \Vert_2}{\Vert T^n_{\text{coarse}} \Vert_2} 
\end{aligned}
\begin{aligned}
   L_{\text{pde}} &= \Vert E^n - D^n_{\text{coarse}} \nabla \cdot (\nabla E^n) \Delta t - \sigma_{\alpha} (T^4 - E) \Delta t - E^{n-1}_{\text{coarse}} \Vert_2^2 \\
   & + \Vert T^n - K^n_{\text{coarse}} \nabla \cdot (\nabla T^n) \Delta t - \sigma_{\alpha} (E - T^4) \Delta t - T^{n-1}_{\text{coarse}} \Vert_2^2
\end{aligned}
$$

其中 $E^n_{\text{coarse}},T^n_{\text{coarse}}$ 表示当前时间步的已知低分辨率解， $E^{n-1}_{\text{coarse}},T^{n-1}_{\text{coarse}}$ 表示前一时间步的已知低分辨率解， $D^n_{\text{coarse}}$ 表示当前时间步的已知辐射扩散系数， $K^n_{\text{coarse}}$ 表示当前时间步的已知材料导热系数， $w$ 是动态权重系数。使用有限差分法对物理信息损失项 $L_{\text{pde}}$ 进行离散线性化处理。

考虑到材料函数材料交界面处的强梯度效应对神经网络收敛性产生的负面影响，我们在基线模型（即上述算法）的基础上设计了掩码模型。首先根据材料分布属性将计算域分解为互补的几何子区域，随后训练神经网络，以各个子区域的局部解为输出。最终，通过预设的空间掩码函数将各子区域输出整合为全局连续解场。

又考虑到不同介质间交界面以及介质自身几何形状的不规则特征，我们在掩码模型的基础上设计了分块掩码模型。该模型基于空间区域分解，将空间域 $\Omega$ 划分四个部分重叠的子区域 $\Omega_i,i=1,2,3,4$ 。在每个子域 $\Omega_i$ 上独立训练网络分支，分别学习得到局部输出 $E_i$ ，将所有子域预测结果 $E_i$ 进行对应拼接获得最终的预测结果。

针对分块掩码模型中各子域 $\Omega_i$ 内数据驱动损失 $L_{\text{reg}}$ 与物理信息损失 $L_{\text{pde}}$ 的差异，我们设计了一种动态权重调整机制。令权重 $w$ 按以下规则自适应调整

$$
\begin{equation}
   w = \alpha\frac{L_{\text{reg}}}{10L_{\text{pde}}}
\end{equation}
$$

其中 $\alpha$ 表示与 $\lg\frac{L_{\text{pde}}}{L_{\text{reg}}}$ 有关的平滑系数，以捕捉两个损失项之间的数量级差异。

## 代码介绍

本代码实现了一个结合数据驱动与物理约束的双阶段神经网络训练框架，用于求解非线性辐射扩散方程的高分辨率数值解。代码采用模块化设计，支持灵活的参数配置与跨平台（CPU/GPU）训练。

取“Nx”×“Ny”的细网格点（默认设置为257×257），设置时间步长为0.001，皮卡迭代的收敛极限为0.001，将有限元法求出的结果作为参考解。已知的粗网格解 $E_{\text{coarse}}$ 由参考解经过“n”倍下采样得到（默认设置为4），分辨率为65×65。

我们的神经网络采用LBFGS优化器。

### 项目结构：

从[https://pan.baidu.com/s/1yS07Ebv-AE2ta2DWwui3GQ?pwd=9j42](https://pan.baidu.com/s/1yS07Ebv-AE2ta2DWwui3GQ?pwd=9j42)访问数据集。根据下述项目结构，将提取的`/diffusion-1T/data`和`/diffusion-2T/data`分别放在根目录中。
   
```
Interpolation_nonlinear_diffusion/
├── diffusion-1T/
│   ├── data/
│   ├── results/
│   ├── config.yaml
│   ├── config.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train_reg.py
│   ├── train_pde.py
│   ├── main.py
│   └── plot.ipynb
├── diffusion-2T/
│   ├── data/
│   ├── results/
│   ├── config.yaml
│   ├── config.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train_reg.py
│   ├── train_pde.py
│   ├── main.py
│   └── plot.ipynb
└── README.md
```

### 参数设置：

#### 全局参数：

##### 模型参数：

|参数      |说明      |默认值      |
|:--------:|:--------:|:--------:|
|model_name    |目标模型（"材料函数类型-初边值函数类型"）    |zconst-const  |
|device_name   |计算设备（"cuda"或"cpu"）    |"cuda"          |
|ionization_type   |材料函数类型（"zconst"或"zline"或"zsquare"）    |zconst          |

##### 网格参数：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|Nx   |x轴细网格点数    |257    |
|Ny   |y轴细网格点数    |257    |
|n    |下采样倍数     |4      |

##### 网络参数：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|depth   |隐藏层层数    |2    |
|width   |隐藏层单元数    |512    |

#### 单温问题算法参数：

##### 第一阶段训练参数：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|Nfit_reg   |训练步数    |300    |
|lr_reg   |LBFGS优化器学习率    |1e-2    |
|epoch_reg    |训练轮次     |50      |

##### 第二阶段训练参数：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|Nfit_pde   |训练步数    |200    |
|lr_pde   |LBFGS优化器学习率    |1e-1    |
|epoch_pde    |训练轮次     |10      |

##### 可视化参数：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|vmax   |误差图像色带的最大值    |0.25    |

#### 双温问题算法参数：

##### 第一阶段训练参数：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|Nfit_reg   |训练步数    |300    |
|lr_E_reg   |关于E的LBFGS优化器学习率    |1e-2    |
|lr_T_reg   |关于T的LBFGS优化器学习率    |1e-2    |
|epoch_reg    |训练轮次     |50      |

##### 第二阶段训练参数：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|Nfit_pde   |训练步数    |200    |
|lr_E_pde   |关于E的LBFGS优化器学习率    |1e-1    |
|lr_T_pde   |关于T的LBFGS优化器学习率    |1e-1    |
|epoch_pde    |训练轮次     |10      |

##### 可视化参数：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|vmax_E   |E的误差图像色带的最大值    |0.25    |
|vmax_T   |T的误差图像色带的最大值    |0.25    |

### 输出结果：

#### 单温问题：

训练完成后会在`./diffusion-1T/results/<model_name>/`目录下生成：

(1) `model_reg.pt`: 第一阶段训练模型

(2) `model_pinn.pt`: 第二阶段训练模型

(3) `sol_reg.npy`: 第一阶段预测结果

(4) `sol_pinn.npy`: 第二阶段预测结果

(5) `fig.png`: 第一、第二阶段预测结果误差图像

#### 双温问题：

训练完成后会在`./diffusion-2T/results/<model_name>/`目录下生成：

(1) `model_reg_E.pt`: 第一阶段关于E的训练模型

(2) `model_reg_T.pt`: 第一阶段关于T的训练模型

(3) `model_pinn_E.pt`: 第二阶段关于E的训练模型

(4) `model_pinn_T.pt`: 第二阶段关于T的训练模型

(5) `sol_reg_E.npy`: 第一阶段对E的预测结果

(6) `sol_reg_T.npy`: 第一阶段对T的预测结果

(7) `sol_pinn_E.npy`: 第二阶段对E的预测结果

(8) `sol_pinn_T.npy`: 第二阶段对T的预测结果

(9) `fig_E.png`: 关于E的第一、第二阶段预测结果误差图像

(10) `fig_T.png`: 关于T的第一、第二阶段预测结果误差图像

## 数值实验：

### 单温问题：

#### (1) zconst-const

材料函数为常数（zconst）： $z=1$

初边值条件为常数初值+线性边值（const）： $\beta (x,y,t) = \max$ { $20t, 10$ }, $g(x,y,t) = 0.01$

设置第一次回归训练时的训练步数为Nfit_reg=100，学习率为lr_reg=1e-3；第二次PDE训练时的训练步数为Nfit_pde=200，学习率为lr_pde=1。可视化参数设置为vmax=0.1。

命令行参数如下：

```bash
python ./diffusion-1T/main.py --model_name "zconst-const" --ionization_type "zconst" --Nfit_reg 100 --lr_reg 1e-3 --Nfit_pde 200 --lr_pde 1 --vmax 0.1
```

两次训练结果与参考解之间的NMSE误差以及绝对误差可视化如下：

|训练      |l2相对误差 |
|:--------:|:--------:|
|第一次训练   |1.3803e-4|
|第二次训练   |7.9253e-7|

<img src="./diffusion-1T/results/zconst-const/fig.png" alt="1T-zconst-const" width="400" />

#### (2) zconst-gauss

材料函数为常数（zconst）： $z=1$

初边值条件为高斯初值+零边值（gauss）： $\beta (x,y,t) = 0, g(x,y,t) = 0.01+100e^{-(x^2+y^2)/0.01}$

设置第一次回归训练时的训练步数为Nfit_reg=200，学习率为lr_reg=1e-3；第二次PDE训练时的训练步数为Nfit_pde=200，学习率为lr_pde=1。可视化参数设置为vmax=0.02。

命令行参数如下：

```bash
python ./diffusion-1T/main.py --model_name "zconst-gauss" --ionization_type "zconst" --Nfit_reg 200 --lr_reg 1e-3 --Nfit_pde 200 --lr_pde 1 --vmax 0.02
```

两次训练结果与参考解之间的NMSE误差以及绝对误差可视化如下：

|训练      |l2相对误差 |
|:--------:|:--------:|
|第一次训练   |3.1160e-3|
|第二次训练   |1.0960e-5|

<img src="./diffusion-1T/results/zconst-gauss/fig.png" alt="1T-zconst-gauss" width="400" />

#### (3) zline-const

材料函数为间断线性（zline）：当 $x\leq0.5$ 时， $z=1$ ；当 $x>0.5$ 时， $z=10$

初边值条件为常数初值+线性边值（const）： $\beta (x,y,t) = \max$ { $20t, 10$ }, $g(x,y,t) = 0.01$

设置第一次回归训练时的训练步数为Nfit_reg=150，学习率为lr_reg=1e-2；第二次PDE训练时的训练步数为Nfit_pde=200，学习率为lr_pde=1。可视化参数设置为vmax=0.25。

命令行参数如下：

```bash
python ./diffusion-1T/main.py --model_name "zline-const" --ionization_type "zline" --Nfit_reg 150 --lr_reg 1e-2 --Nfit_pde 200 --lr_pde 1 --vmax 0.25
```

两次训练结果与参考解之间的NMSE误差以及绝对误差可视化如下：

|训练      |l2相对误差 |
|:--------:|:--------:|
|第一次训练   |8.6974e-5|
|第二次训练   |2.6432e-5|

<img src="./diffusion-1T/results/zline-const/fig.png" alt="1T-zline-const" width="400" />

#### (4) zline-gauss

材料函数为间断线性（zline）：当 $x\leq0.5$ 时， $z=1$ ；当 $x>0.5$ 时， $z=10$

初边值条件为高斯初值+零边值（gauss）： $\beta (x,y,t) = 0, g(x,y,t) = 0.01+100e^{-(x^2+y^2)/0.01}$

设置第一次回归训练时的训练步数为Nfit_reg=200，学习率为lr_reg=1e-2；第二次PDE训练时的训练步数为Nfit_pde=100，学习率为lr_pde=1。可视化参数设置为vmax=0.072。

命令行参数如下：

```bash
python ./diffusion-1T/main.py --model_name "zline-gauss" --ionization_type "zline" --Nfit_reg 200 --lr_reg 1e-2 --Nfit_pde 100 --lr_pde 1 --vmax 0.072
```

两次训练结果与参考解之间的NMSE误差以及绝对误差可视化如下：

|训练      |l2相对误差 |
|:--------:|:--------:|
|第一次训练   |1.2046e-3|
|第二次训练   |8.1885e-4|

<img src="./diffusion-1T/results/zline-gauss/fig.png" alt="1T-zline-gauss" width="400" />

#### (5) zsquare-const

材料函数为双方形（zsquare）：当 $\frac{3}{16}<x<\frac{7}{16}, \frac{9}{16}<y<\frac{13}{16}$ 或 $\frac{9}{16}<x<\frac{13}{16}, \frac{3}{16}<y<\frac{7}{16}$ 时， $z=10$ ；其他时候 $z=1$

初边值条件为常数初值+线性边值（const）： $\beta (x,y,t) = \max$ { $20t, 10$ }, $g(x,y,t) = 0.01$

设置第一次回归训练时的训练步数为Nfit_reg=150，学习率为lr_reg=1e-2；第二次PDE训练时的训练步数为Nfit_pde=300，学习率为lr_pde=1e-1。可视化参数设置为vmax=1.0。

命令行参数如下：

```bash
python ./diffusion-1T/main.py --model_name "zsquare-const" --ionization_type "zsquare" --Nfit_reg 150 --lr_reg 1e-2 --Nfit_pde 300 --lr_pde 1e-1 --vmax 1.0
```

两次训练结果与参考解之间的NMSE误差以及绝对误差可视化如下：

|训练      |l2相对误差 |
|:--------:|:--------:|
|第一次训练   |7.7425e-4|
|第二次训练   |3.1866e-4|

<img src="./diffusion-1T/results/zsquare-const/fig.png" alt="1T-zsquare-const" width="400" />

#### (6) zsquare-gauss

材料函数为双方形（zsquare）：当 $\frac{3}{16}<x<\frac{7}{16}, \frac{9}{16}<y<\frac{13}{16}$ 或 $\frac{9}{16}<x<\frac{13}{16}, \frac{3}{16}<y<\frac{7}{16}$ 时， $z=10$ ；其他时候 $z=1$

初边值条件为高斯初值+零边值（gauss）： $\beta (x,y,t) = 0, g(x,y,t) = 0.01+100e^{-(x^2+y^2)/0.01}$

设置第一次回归训练时的训练步数为Nfit_reg=400，学习率为lr_reg=1e-1；第二次PDE训练时的训练步数为Nfit_pde=350，学习率为lr_pde=1。可视化参数设置为vmax=0.16。

命令行参数如下：

```bash
python ./diffusion-1T/main.py --model_name "zsquare-gauss" --ionization_type "zsquare" --Nfit_reg 400 --lr_reg 1e-1 --Nfit_pde 350 --lr_pde 1 --vmax 0.16
```

两次训练结果与参考解之间的NMSE误差以及绝对误差可视化如下：

|训练      |l2相对误差 |
|:--------:|:--------:|
|第一次训练   |4.7518e-3|
|第二次训练   |3.9955e-3|

<img src="./diffusion-1T/results/zsquare-gauss/fig.png" alt="1T-zsquare-gauss" width="400" />

### 双温问题：

#### (1) zconst-const

材料函数为常数（zconst）： $z=1$

初边值条件为常数初值+线性边值（const）： $\beta (x,y,t) = \max$ { $20t, 10$ }, $g(x,y,t) = 0.01$

设置第一次回归训练时的训练步数为Nfit_reg=300，关于E的学习率为lr_E_reg=1e-3，关于T的学习率为lr_T_reg=1e-3；第二次PDE训练时的训练步数为Nfit_pde=200，关于E的学习率为lr_E_pde=1e-2，关于T的学习率为lr_T_pde=1e-2。可视化参数设置为vmax_E=0.28，vmax_T=0.02。

命令行参数如下：

```bash
python ./diffusion-2T/main.py --model_name "zconst-const" --ionization_type "zconst" --Nfit_reg 300 --lr_E_reg 1e-3 --lr_T_reg 1e-3 --Nfit_pde 200 --lr_E_pde 1e-2 --lr_T_pde 1e-2 --vmax_E 0.28 --vmax_T 0.02
```

关于变量E和T的两次训练结果与参考解之间的NMSE误差以及绝对误差可视化如下：

|训练      |E的l2相对误差 |T的l2相对误差 |
|:--------:|:-----------:|:-----------:|
|第一次训练 |1.0433e-4    |3.8226e-5    |
|第二次训练 |4.1970e-5    |4.8864e-6    |

<img src="./diffusion-2T/results/zconst-const/fig_E.png" alt="2T-zconst-const-E" width="400" /> <img src="./diffusion-2T/results/zconst-const/fig_T.png" alt="2T-zconst-const-T" width="400" />

#### (2) zconst-gauss

材料函数为常数（zconst）： $z=1$

初边值条件为高斯初值+零边值（gauss）： $\beta (x,y,t) = 0, g(x,y,t) = 0.01+100e^{-(x^2+y^2)/0.01}$

设置第一次回归训练时的训练步数为Nfit_reg=300，关于E的学习率为lr_E_reg=1e-3，关于T的学习率为lr_T_reg=1e-3；第二次PDE训练时的训练步数为Nfit_pde=200，关于E的学习率为lr_E_pde=1e-1，关于T的学习率为lr_T_pde=1e-1。可视化参数设置为vmax_E=0.004，vmax_T=0.015。

命令行参数如下：

```bash
python ./diffusion-2T/main.py --model_name "zconst-gauss" --ionization_type "zconst" --Nfit_reg 300 --lr_E_reg 1e-3 --lr_T_reg 1e-3 --Nfit_pde 200 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 0.004 --vmax_T 0.015
```

两次训练结果与参考解之间的NMSE误差以及绝对误差可视化如下：

|训练      |E的l2相对误差 |T的l2相对误差 |
|:--------:|:-----------:|:-----------:|
|第一次训练 |1.2102e-4    |5.3335e-5    |
|第二次训练 |7.9144e-6    |2.6790e-6    |

<img src="./diffusion-2T/results/zconst-gauss/fig_E.png" alt="2T-zconst-gauss-E" width="400" /> <img src="./diffusion-2T/results/zconst-gauss/fig_T.png" alt="2T-zconst-gauss-T" width="400" />

#### (3) zline-const

材料函数为间断线性（zline）：当 $x\leq0.5$ 时， $z=1$ ；当 $x>0.5$ 时， $z=10$

初边值条件为常数初值+线性边值（const）： $\beta (x,y,t) = \max$ { $20t, 10$ }, $g(x,y,t) = 0.01$

设置第一次回归训练时的训练步数为Nfit_reg=300，关于E的学习率为lr_E_reg=1e-2，关于T的学习率为lr_T_reg=1e-2；第二次PDE训练时的训练步数为Nfit_pde=200，关于E的学习率为lr_E_pde=1e-1，关于T的学习率为lr_T_pde=1e-1。可视化参数设置为vmax_E=1.7，vmax_T=0.03。

命令行参数如下：

```bash
python ./diffusion-2T/main.py --model_name "zline-const" --ionization_type "zline" --Nfit_reg 300 --lr_E_reg 1e-2 --lr_T_reg 1e-2 --Nfit_pde 200 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 1.7 --vmax_T 0.03
```

关于变量E和T的两次训练结果与参考解之间的NMSE误差以及绝对误差可视化如下：

|训练      |E的l2相对误差 |T的l2相对误差 |
|:--------:|:-----------:|:-----------:|
|第一次训练 |3.3480e-4    |9.3573e-6    |
|第二次训练 |9.2222e-4    |4.3018e-6    |

<img src="./diffusion-2T/results/zline-const/fig_E.png" alt="2T-zline-const-E" width="400" /> <img src="./diffusion-2T/results/zline-const/fig_T.png" alt="2T-zline-const-T" width="400" />

#### (4) zline-gauss

材料函数为间断线性（zline）：当 $x\leq0.5$ 时， $z=1$ ；当 $x>0.5$ 时， $z=10$

初边值条件为高斯初值+零边值（gauss）： $\beta (x,y,t) = 0, g(x,y,t) = 0.01+100e^{-(x^2+y^2)/0.01}$

设置第一次回归训练时的训练步数为Nfit_reg=200，关于E的学习率为lr_E_reg=1e-3，关于T的学习率为lr_T_reg=1e-4；第二次PDE训练时的训练步数为Nfit_pde=200，关于E的学习率为lr_E_pde=1e-1，关于T的学习率为lr_T_pde=1e-1。可视化参数设置为vmax_E=0.012，vmax_T=0.15。

命令行参数如下：

```bash
python ./diffusion-2T/main.py --model_name "zline-gauss" --ionization_type "zline" --Nfit_reg 200 --lr_E_reg 1e-3 --lr_T_reg 1e-4 --Nfit_pde 200 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 0.012 --vmax_T 0.15
```

两次训练结果与参考解之间的NMSE误差以及绝对误差可视化如下：

|训练      |E的l2相对误差 |T的l2相对误差 |
|:--------:|:-----------:|:-----------:|
|第一次训练 |3.2893e-4    |2.0064e-2    |
|第二次训练 |9.5702e-6    |4.9225e-4    |

<img src="./diffusion-2T/results/zline-gauss/fig_E.png" alt="2T-zline-gauss-E" width="400" /> <img src="./diffusion-2T/results/zline-gauss/fig_T.png" alt="2T-zline-gauss-T" width="400" />

#### (5) zsquare-const

材料函数为双方形（zsquare）：当 $\frac{3}{16}<x<\frac{7}{16}, \frac{9}{16}<y<\frac{13}{16}$ 或 $\frac{9}{16}<x<\frac{13}{16}, \frac{3}{16}<y<\frac{7}{16}$ 时， $z=10$ ；其他时候 $z=1$

初边值条件为常数初值+线性边值（const）： $\beta (x,y,t) = \max$ { $20t, 10$ }, $g(x,y,t) = 0.01$

设置第一次回归训练时的训练步数为Nfit_reg=300，关于E的学习率为lr_E_reg=1e-3，关于T的学习率为lr_T_reg=1e-3；第二次PDE训练时的训练步数为Nfit_pde=200，关于E的学习率为lr_E_pde=1e-1，关于T的学习率为lr_T_pde=1e-1。可视化参数设置为vmax_E=0.6，vmax_T=0.15。

命令行参数如下：

```bash
python ./diffusion-2T/main.py --model_name "zsquare-const" --ionization_type "zsquare" --Nfit_reg 300 --lr_E_reg 1e-3 --lr_T_reg 1e-3 --Nfit_pde 200 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 0.6 --vmax_T 0.15
```

关于变量E和T的两次训练结果与参考解之间的NMSE误差以及绝对误差可视化如下：

|训练      |E的l2相对误差 |T的l2相对误差 |
|:--------:|:-----------:|:-----------:|
|第一次训练 |1.6882e-3    |9.2393e-4    |
|第二次训练 |3.9618e-5    |5.0227e-6    |

<img src="./diffusion-2T/results/zsquare-const/fig_E.png" alt="2T-zsquare-const-E" width="400" /> <img src="./diffusion-2T/results/zsquare-const/fig_T.png" alt="2T-zsquare-const-T" width="400" />

#### (6) zsquare-gauss

材料函数为双方形（zsquare）：当 $\frac{3}{16}<x<\frac{7}{16}, \frac{9}{16}<y<\frac{13}{16}$ 或 $\frac{9}{16}<x<\frac{13}{16}, \frac{3}{16}<y<\frac{7}{16}$ 时， $z=10$ ；其他时候 $z=1$

初边值条件为高斯初值+零边值（gauss）： $\beta (x,y,t) = 0, g(x,y,t) = 0.01+100e^{-(x^2+y^2)/0.01}$

设置第一次回归训练时的训练步数为Nfit_reg=700，关于E的学习率为lr_E_reg=1e-3，关于T的学习率为lr_T_reg=1e-3；第二次PDE训练时的训练步数为Nfit_pde=100，关于E的学习率为lr_E_pde=1e-1，关于T的学习率为lr_T_pde=1e-1。可视化参数设置为vmax_E=0.006，vmax_T=0.11。

命令行参数如下：

```bash
python ./diffusion-2T/main.py --model_name "zsquare-gauss" --ionization_type "zsquare" --Nfit_reg 700 --lr_E_reg 1e-3 --lr_T_reg 1e-3 --Nfit_pde 100 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 0.006 --vmax_T 0.11
```

两次训练结果与参考解之间的NMSE误差以及绝对误差可视化如下：

|训练      |E的l2相对误差 |T的l2相对误差 |
|:--------:|:-----------:|:-----------:|
|第一次训练 |3.5218e-4    |1.4630e-3    |
|第二次训练 |2.2470e-5    |6.6278e-4    |

<img src="./diffusion-2T/results/zsquare-gauss/fig_E.png" alt="2T-zsquare-gauss-E" width="400" /> <img src="./diffusion-2T/results/zsquare-gauss/fig_T.png" alt="2T-zsquare-gauss-T" width="400" />

-----------------------------------------------------------------------------------------------------------------------------

# Research on Neural Network Super-Resolution Algorithms for Nonlinear Radiation Diffusion Problems

## Super-Resolution Method:

Super-resolution technology demonstrates revolutionary potential in scientific computing to overcome the efficiency limitations of traditional numerical methods. Particularly in physical field reconstruction tasks involving multi-scale and strongly nonlinear features, conventional interpolation-based or data-driven super-resolution approaches often suffer from fundamental flaws such as unphysical solutions or conservation law violations due to the lack of physics-based constraints. This limitation becomes especially pronounced in fields demanding strict physical consistency—such as radiation transport and turbulence simulation—severely restricting the practical engineering applications of super-resolution techniques.

To address these challenges, this project proposes a neural network-based super-resolution framework that directly learns the mapping from low-resolution to high-resolution computational grids. The architecture follows a "coarse-grid input → network prediction → physics correction" pipeline, incorporating governing equation-derived physical constraints alongside traditional data-driven loss functions. This ensures strict adherence to physical laws while maintaining solution accuracy for nonlinear radiation diffusion problems and significantly improving computational efficiency.

## Nonlinear Radiation Diffusion Problem:

The nonlinear radiation diffusion problem represents a classic example of multiscale strongly coupled transport equations. At its core, it describes the nonlinear energy exchange process between radiation energy and material energy mediated by photon transport. The governing equations for this process can be expressed as follows.

### Single-Temperature Problem:

$$
\begin{aligned}
   & \frac{\partial E}{\partial t}-\nabla\cdot(D_L\nabla E) = 0, \quad(x,y,t)\in\Omega\times[0,1] \\
   & 0.5E+D_L\nabla E\cdot n = \beta(x,y,t), \quad(x,y,t)\in\lbrace x=0\rbrace\times[0,1] \\
   & 0.5E+D_L\nabla E\cdot n = 0, \quad(x,y,t)\in\partial\Omega\setminus\lbrace x=0\rbrace\times[0,1] \\
   & E|_{t=0} = g(x,y,0)
\end{aligned}
$$

where $\Omega = [0,1]\times[0,1]$ , while the radiation diffusion coefficient $D_L$ adopts the flux-limited form, expressed as $D_L = \frac{1}{3\sigma_{\alpha}+\frac{|\nabla E|}{E}}, \sigma_{\alpha} = \frac{z^3}{E^{3/4}}$ .

### Two-Temperature Problem:

$$
\begin{aligned}
   & \frac{\partial E}{\partial t} - \nabla \cdot (D_L \nabla E) = \sigma_{\alpha}(T^4 - E), \quad(x,y,t)\in\Omega\times[0,1] \\
   & \frac{\partial T}{\partial t} - \nabla \cdot (K_L \nabla T) = \sigma_{\alpha}(E - T^4), \quad(x,y,t)\in\Omega\times[0,1] \\
   & 0.5E + D_L \nabla E \cdot n = \beta(x,y,t), \quad (x,y,t) \in \lbrace x=0 \rbrace \times [0,1] \\
   & 0.5E + D_L \nabla E \cdot n = 0, \quad (x,y,t) \in \partial\Omega \setminus \lbrace x=0 \rbrace \times [0,1] \\
   & K_L \nabla T \cdot n = 0, \quad (x,y,t) \in \partial\Omega \times [0,1] \\
   & E\vert_{t=0} = T^4\vert_{t=0} = g(x,y,0)
\end{aligned}
$$

where $\Omega = [0,1]\times[0,1]$ , while the radiation diffusion coefficient $D_L, K_L$ also adopts the flux-limited form, expressed as $D_L = \frac{1}{3\sigma_{\alpha}+\frac{|\nabla E|}{E}}, \sigma_{\alpha} = \frac{z^3}{E^{3/4}}, K_L = \frac{T^4}{T^{3/2}z+T^{5/2}|\nabla T|}$ .

For the single-temperature and two-temperature problems mentioned above, the material function $z$ can be classified into three cases: "zconst" (constant), "zline" (intermittent linear) and "zsquare" (two-squares). Initial and boundary conditions $\beta(x,y,t), g(x,y,t)$ can also be classified into two cases: "const" (constant initial+linear boundary) and "gauss" (gauss initial+zero boundary). The specific formula for each case will be given later.

## Design of Neural Network Super-Resolution Algorithm:

We propose a novel neural network-based solver for a single equation by combining low-resolution numerical solutions $E_{\text{coarse}}$ with the governing equation itself.

We construct a fully connected neural network that takes the spatial coordinates of target points $(x,y)$ as input data. The network features a uniform-width architecture, where both the number of hidden layers "depth" and neurons per hidden layer "width" are configurable (default configuration: 2 hidden layers with 512 neurons each), employing ReLU as the activation function. The output layer is configured as a two-channel structure, where a Boolean mask is applied based on the magnitude of the material function $z$ at different target points to select the corresponding output channel for each target point.

To ensure solving efficiency, we first utilize the low-resolution reference solution to construct a data-driven loss function $L_{\text{reg}}$ for training. Then, incorporating the equations of the target radiation diffusion problem, we design a physics-constrained loss function $L_{\text{reg+pde}}$ to further improve the solving accuracy of the neural network model.

Taking the single-temperature problem as an example, we discretize the temporal derivative in the equation using backward differencing and employ a neural network to compute the results at each time step. The loss function $L_{\text{reg+pde}}$ of the network incorporates both data-driven loss $L_{\text{reg}}$ and physics-informed loss $L_{\text{pde}}$ as constraints. The specific formulation is as follows:

$$
\begin{aligned}
   & L_{\text{reg+pde}} = L_{\text{reg}}+10L_{\text{pde}} \\
   & L_{\text{reg}} = \Vert E^n-E^n_{\text{coarse}} \Vert \\
   & L_{\text{pde}} = \Vert E^n-D^n_{\text{coarse}}\nabla\cdot(\nabla E^n)\Delta t-E^{n-1}_{\text{coarse}} \Vert
\end{aligned}
$$

Using the same methodology, we also derive the specific formulation of the loss function for the two-temperature problem as follows:

$$
\begin{aligned}
   L_{\text{reg+pde}} &= L_{\text{reg}} + 10L_{\text{pde}} \\
   L_{\text{reg}} &= \Vert E^n - E^n_{\text{coarse}} \Vert + \Vert T^n - T^n_{\text{coarse}} \Vert \\
   L_{\text{pde}} &= \Vert E^n - D^n_{\text{coarse}} \nabla \cdot (\nabla E^n) \Delta t - \sigma_{\alpha} (T^4 - E) \Delta t - E^{n-1}_{\text{coarse}} \Vert \\
   & + \Vert T^n - K^n_{\text{coarse}} \nabla \cdot (\nabla T^n) \Delta t - \sigma_{\alpha} (E - T^4) \Delta t - T^{n-1}_{\text{coarse}} \Vert
\end{aligned}
$$

## Code Introduction:

This code implements a dual-phase neural network training framework that integrates data-driven learning with physics-based constraints for obtaining high-resolution numerical solutions to nonlinear radiation diffusion equations. The modular-designed code supports flexible parameter configuration and cross-platform (CPU/GPU) training.

The reference solution is obtained through finite element method with the following specifications: (1) a fine grid of "Nx" × "Ny" points (default: 257×257); (2) time step size of 0.001; (3) Picard iteration convergence limit of 0.001. The known coarse-grid solution $E_{\text{coarse}}$ is derived by "n"-times downsampling (default: n=4) of the reference solution, yielding a resolution of 65×65.

Our neural network employs the LBFGS optimizer for training.

### Project Structure：
   
```
Interpolation_nonlinear_diffusion/
├── diffusion-1T/
│   ├── data/
│   ├── results/
│   ├── config.yaml
│   ├── config.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train_reg.py
│   ├── train_pde.py
│   ├── main.py
│   └── plot.ipynb
├── diffusion-2T/
│   ├── data/
│   ├── results/
│   ├── config.yaml
│   ├── config.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── model.py
│   ├── train_reg.py
│   ├── train_pde.py
│   ├── main.py
│   └── plot.ipynb
└── README.md
```

### Parameter Specification:

#### Global parameters:

##### Model parameters:

|Parameter      |Description      |Default      |
|:--------:|:--------:|:--------:|
|model_name    |target model ("material function type-initial&boundary condition type")   |zconst-const  |
|device_name   |computation device ("cuda" or "cpu")    |cuda          |
|ionization_type   |material function type ("zconst", "zline" or "zsquare")    |zconst          |

##### Grid parameters:

|Parameter      |Description      |Default      |
|:--------:|:--------:|:---------:|
|Nx   |fine-grid points on x-axis    |257    |
|Ny   |fine-grid points on x-axis    |257    |
|n    |downsampling factor      |4      |

##### Model parameters:

|Parameter      |Description      |Default      |
|:--------:|:--------:|:---------:|
|depth   |Number of hidden layers    |2    |
|width   |Number of units in each hidden layer    |512    |

#### Parameters of single-temperature problem:

##### Training parameters (Phase 1):

|Parameter      |Description      |Default      |
|:--------:|:--------:|:---------:|
|Nfit_reg   |training iterations    |300    |
|lr_reg   |LBFGS optimizer learning rate    |1e-2    |
|epoch_reg    |training epochs     |50      |

##### Training parameters (Phase 2):

|Parameter      |Description      |Default      |
|:--------:|:--------:|:---------:|
|Nfit_pde   |training iterations    |200    |
|lr_pde   |LBFGS optimizer learning rate    |1e-1    |
|epoch_pde    |training epochs     |10      |

##### Visualization parameters:

|Parameter      |Description      |Default      |
|:--------:|:--------:|:---------:|
|vmax   |Maximum value of error colorbar    |0.25    |

#### Parameters of two-temperature problem:

##### Training parameters (Phase 1):

|Parameter      |Description      |Default      |
|:--------:|:--------:|:---------:|
|Nfit_reg   |training iterations    |300    |
|lr_E_reg   |LBFGS optimizer learning rate of E   |1e-2    |
|lr_T_reg   |LBFGS optimizer learning rate of T    |1e-2    |
|epoch_reg    |training epochs     |50      |

##### Training parameters (Phase 2):

|Parameter      |Description      |Default      |
|:--------:|:--------:|:---------:|
|Nfit_pde   |training iterations    |200    |
|lr_E_pde   |LBFGS optimizer learning rate of E    |1e-1    |
|lr_T_pde   |LBFGS optimizer learning rate of T    |1e-1    |
|epoch_pde    |training epochs     |10      |

##### Visualization parameters:

|Parameter      |Description      |Default      |
|:--------:|:--------:|:---------:|
|vmax_E   |Maximum value of error colorbar E   |0.25    |
|vmax_T   |Maximum value of error colorbar T   |0.25    |

### Output:

#### Single-temperature problem:

Results will be saved in `./diffusion-1T/results/<model_name>/`:

(1) `model_reg.pt`: Phase 1 regression trained model

(2) `model_pinn.pt`: Phase 2 physics-informed model

(3) `sol_reg.npy`: Phase 1 predictions

(4) `sol_pinn.npy`: Phase 2 predictions

(5) `fig.png`: figure about the comparison of regression and PINN solutions

#### Two-temperature problem:

Results will be saved in `./diffusion-2T/results/<model_name>/`:

(1) `model_reg_E.pt`: Phase 1 regression trained model of E

(2) `model_reg_T.pt`: Phase 1 regression trained model of T

(3) `model_pinn_E.pt`: Phase 2 physics-informed model of E

(4) `model_pinn_T.pt`: Phase 2 physics-informed model of T

(5) `sol_reg_E.npy`: Phase 1 E-predictions

(6) `sol_reg_T.npy`: Phase 1 T-predictions

(7) `sol_pinn_E.npy`: Phase 2 E-predictions

(8) `sol_pinn_T.npy`: Phase 2 T-predictions

(9) `fig_E.png`: figure about the comparison of regression and PINN E-solutions

(10) `fig_T.png`: figure about the comparison of regression and PINN T-solutions

## Numerical Experiments:

### Single-Temperature Problem:

#### (1) zconst-const

material function uses "zconst" type: Always $z=1$

Initial and boundary condition uses "const" type: $\beta (x,y,t) = \max$ { $20t, 10$ }, $g(x,y,t) = 0.01$

Set the number of training steps for the first regression training to Nfit_reg=100, with a learning rate of lr_reg=1e-3. For the second PDE training, set Nfit_pde=200 training steps, with a learning rate of lr_pde=1. The visualization parameter is set to vmax=0.1.

The command-line parameters are as follows:

```bash
python ./diffusion-1T/main.py --model_name "zconst-const" --zconst --Nfit_reg 100 --lr_reg 1e-3 --Nfit_pde 200 --lr_pde 1 --vmax 0.1
```

NMSE between the two training results and the reference solution, along with the absolute error visualization, are shown below:

|Training      |L2 relative error |
|:--------:|:--------:|
|Phase 1   |1.3803e-4|
|Phase 2   |7.9253e-7|

<img src="./diffusion-1T/results/zconst-const/fig.png" alt="1T-zconst-const" width="400" />

#### (2) zconst-gauss

material function uses "zconst" type: Always $z=1$

Initial and boundary condition uses "gauss" type: $\beta (x,y,t) = 0, g(x,y,t) = 0.01+100e^{-(x^2+y^2)/0.01}$

Set the number of training steps for the first regression training to Nfit_reg=200, with a learning rate of lr_reg=1e-3. For the second PDE training, set Nfit_pde=200 training steps, with a learning rate of lr_pde=1. The visualization parameter is set to vmax=0.02.

The command-line parameters are as follows:

```bash
python ./diffusion-1T/main.py --model_name "zconst-gauss" --ionization_type "zconst" --Nfit_reg 200 --lr_reg 1e-3 --Nfit_pde 200 --lr_pde 1 --vmax 0.02
```

NMSE between the two training results and the reference solution, along with the absolute error visualization, are shown below:

|Training      |L2 relative error |
|:--------:|:--------:|
|Phase 1   |3.1160e-3|
|Phase 2   |1.0960e-5|

<img src="./diffusion-1T/results/zconst-gauss/fig.png" alt="1T-zconst-gauss" width="400" />

#### (3) zline-const

material function uses "zline" type: When $x\leq0.5$, $z=1$; when $x>0.5$, $z=10$

Initial and boundary condition uses "const" type: $\beta (x,y,t) = \max$ { $20t, 10$ }, $g(x,y,t) = 0.01$

Set the number of training steps for the first regression training to Nfit_reg=150, with a learning rate of lr_reg=1e-2. For the second PDE training, set Nfit_pde=200 training steps, with a learning rate of lr_pde=1. The visualization parameter is set to vmax=0.25.

The command-line parameters are as follows:

```bash
python ./diffusion-1T/main.py --model_name "zline-const" --ionization_type "zline" --Nfit_reg 150 --lr_reg 1e-2 --Nfit_pde 200 --lr_pde 1 --vmax 0.25
```

NMSE between the two training results and the reference solution, along with the absolute error visualization, are shown below:

|Training      |L2 relative error |
|:--------:|:--------:|
|Phase 1   |8.6974e-5|
|Phase 2   |2.6432e-5|

<img src="./diffusion-1T/results/zline-const/fig.png" alt="1T-zline-const" width="400" />

#### (4) zline-gauss

material function uses "zline" type: When $x\leq0.5$, $z=1$; when $x>0.5$, $z=10$

Initial and boundary condition uses "gauss" type: $\beta (x,y,t) = 0, g(x,y,t) = 0.01+100e^{-(x^2+y^2)/0.01}$

Set the number of training steps for the first regression training to Nfit_reg=200, with a learning rate of lr_reg=1e-2. For the second PDE training, set Nfit_pde=100 training steps, with a learning rate of lr_pde=1. The visualization parameter is set to vmax=0.072.

The command-line parameters are as follows:

```bash
python ./diffusion-1T/main.py --model_name "zline-gauss" --ionization_type "zline" --Nfit_reg 200 --lr_reg 1e-2 --Nfit_pde 100 --lr_pde 1 --vmax 0.072
```

NMSE between the two training results and the reference solution, along with the absolute error visualization, are shown below:

|Training      |L2 relative error |
|:--------:|:--------:|
|Phase 1   |1.2046e-3|
|Phase 2   |8.1885e-4|

<img src="./diffusion-1T/results/zline-gauss/fig.png" alt="1T-zline-gauss" width="400" />

#### (5) zsquare-const

material function uses "zsquare" type: When $\frac{3}{16}<x<\frac{7}{16}, \frac{9}{16}<y<\frac{13}{16}$ or $\frac{9}{16}<x<\frac{13}{16}, \frac{3}{16}<y<\frac{7}{16}$, $z=10$; otherwise $z=1$

Initial and boundary condition uses "const" type: $\beta (x,y,t) = \max$ { $20t, 10$ }, $g(x,y,t) = 0.01$

Set the number of training steps for the first regression training to Nfit_reg=150, with a learning rate of lr_reg=1e-2. For the second PDE training, set Nfit_pde=300 training steps, with a learning rate of lr_pde=1e-1. The visualization parameter is set to vmax=1.0.

The command-line parameters are as follows:

```bash
python ./diffusion-1T/main.py --model_name "zsquare-const" --ionization_type "zsquare" --Nfit_reg 150 --lr_reg 1e-2 --Nfit_pde 300 --lr_pde 1e-1 --vmax 1.0
```

NMSE between the two training results and the reference solution, along with the absolute error visualization, are shown below:

|Training      |L2 relative error |
|:--------:|:--------:|
|Phase 1   |7.7425e-4|
|Phase 2   |3.1866e-4|

<img src="./diffusion-1T/results/zsquare-const/fig.png" alt="1T-zsquare-const" width="400" />

#### (6) zsquare-gauss

material function uses "zsquare" type: When $\frac{3}{16}<x<\frac{7}{16}, \frac{9}{16}<y<\frac{13}{16}$ or $\frac{9}{16}<x<\frac{13}{16}, \frac{3}{16}<y<\frac{7}{16}$, $z=10$; otherwise $z=1$

Initial and boundary condition uses "gauss" type: $\beta (x,y,t) = 0, g(x,y,t) = 0.01+100e^{-(x^2+y^2)/0.01}$

Set the number of training steps for the first regression training to Nfit_reg=400, with a learning rate of lr_reg=1e-1. For the second PDE training, set Nfit_pde=350 training steps, with a learning rate of lr_pde=1. The visualization parameter is set to vmax=0.16.

The command-line parameters are as follows:

```bash
python ./diffusion-1T/main.py --model_name "zsquare-gauss" --ionization_type "zsquare" --Nfit_reg 400 --lr_reg 1e-1 --Nfit_pde 350 --lr_pde 1 --vmax 0.16
```

NMSE between the two training results and the reference solution, along with the absolute error visualization, are shown below:

|Training      |L2 relative error |
|:--------:|:--------:|
|Phase 1   |4.7518e-3|
|Phase 2   |3.9955e-3|

<img src="./diffusion-1T/results/zsquare-gauss/fig.png" alt="1T-zsquare-gauss" width="400" />

### Two-Temperature Problem:

#### (1) zconst-const

material function uses "zconst" type: Always $z=1$

Initial and boundary condition uses "const" type: $\beta (x,y,t) = \max$ { $20t, 10$ }, $g(x,y,t) = 0.01$

Set the number of training steps for the first regression training to Nfit_reg=300, with a learning rate of E lr_E_reg=1e-3 and a learning rate of T lr_T_reg=1e-3. For the second PDE training, set Nfit_pde=200 training steps, with a learning rate of E lr_E_pde=1e-2 and a learning rate of T lr_T_pde=1e-2. The visualization parameter is set to vmax_E=0.28 and vmax_T=0.02.

The command-line parameters are as follows:

```bash
python ./diffusion-2T/main.py --model_name "zconst-const" --ionization_type "zconst" --Nfit_reg 300 --lr_E_reg 1e-3 --lr_T_reg 1e-3 --Nfit_pde 200 --lr_E_pde 1e-2 --lr_T_pde 1e-2 --vmax_E 0.28 --vmax_T 0.02
```

NMSE between the two training results and the reference solution, along with the absolute error visualization, are shown below:

|Training      |L2 relative error of E |L2 relative error of T |
|:--------:|:-----------:|:-----------:|
|Phase 1 |1.0433e-4    |3.8226e-5    |
|Phase 2 |4.1970e-5    |4.8864e-6    |

<img src="./diffusion-2T/results/zconst-const/fig_E.png" alt="2T-zconst-const-E" width="400" /> <img src="./diffusion-2T/results/zconst-const/fig_T.png" alt="2T-zconst-const-T" width="400" />

#### (2) zconst-gauss

material function uses "zconst" type: Always $z=1$

Initial and boundary condition uses "gauss" type: $\beta (x,y,t) = 0, g(x,y,t) = 0.01+100e^{-(x^2+y^2)/0.01}$

Set the number of training steps for the first regression training to Nfit_reg=300, with a learning rate of E lr_E_reg=1e-3 and a learning rate of T lr_T_reg=1e-3. For the second PDE training, set Nfit_pde=200 training steps, with a learning rate of E lr_E_pde=1e-1 and a learning rate of T lr_T_pde=1e-1. The visualization parameter is set to vmax_E=0.004 and vmax_T=0.015.

The command-line parameters are as follows:

```bash
python ./diffusion-2T/main.py --model_name "zconst-gauss" --ionization_type "zconst" --Nfit_reg 300 --lr_E_reg 1e-3 --lr_T_reg 1e-3 --Nfit_pde 200 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 0.004 --vmax_T 0.015
```

NMSE between the two training results and the reference solution, along with the absolute error visualization, are shown below:

|Training      |L2 relative error of E |L2 relative error of T |
|:--------:|:-----------:|:-----------:|
|Phase 1 |1.2102e-4    |5.3335e-5    |
|Phase 2 |7.9144e-6    |2.6790e-6    |

<img src="./diffusion-2T/results/zconst-gauss/fig_E.png" alt="2T-zconst-gauss-E" width="400" /> <img src="./diffusion-2T/results/zconst-gauss/fig_T.png" alt="2T-zconst-gauss-T" width="400" />

#### (3) zline-const

material function uses "zline" type: When $x\leq0.5$, $z=1$; when $x>0.5$, $z=10$

Initial and boundary condition uses "const" type: $\beta (x,y,t) = \max$ { $20t, 10$ }, $g(x,y,t) = 0.01$

Set the number of training steps for the first regression training to Nfit_reg=300, with a learning rate of E lr_E_reg=1e-2 and a learning rate of T lr_T_reg=1e-2. For the second PDE training, set Nfit_pde=200 training steps, with a learning rate of E lr_E_pde=1e-1 and a learning rate of T lr_T_pde=1e-1. The visualization parameter is set to vmax_E=1.7 and vmax_T=0.03.

The command-line parameters are as follows:

```bash
python ./diffusion-2T/main.py --model_name "zline-const" --ionization_type "zline" --Nfit_reg 300 --lr_E_reg 1e-2 --lr_T_reg 1e-2 --Nfit_pde 200 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 1.7 --vmax_T 0.03
```

NMSE between the two training results and the reference solution, along with the absolute error visualization, are shown below:

|Training      |L2 relative error of E |L2 relative error of T |
|:--------:|:-----------:|:-----------:|
|Phase 1 |3.3480e-4    |9.3573e-6    |
|Phase 2 |9.2222e-4    |4.3018e-6    |

<img src="./diffusion-2T/results/zline-const/fig_E.png" alt="2T-zline-const-E" width="400" /> <img src="./diffusion-2T/results/zline-const/fig_T.png" alt="2T-zline-const-T" width="400" />

#### (4) zline-gauss

material function uses "zline" type: When $x\leq0.5$, $z=1$; when $x>0.5$, $z=10$

Initial and boundary condition uses "gauss" type: $\beta (x,y,t) = 0, g(x,y,t) = 0.01+100e^{-(x^2+y^2)/0.01}$

Set the number of training steps for the first regression training to Nfit_reg=200, with a learning rate of E lr_E_reg=1e-3 and a learning rate of T lr_T_reg=1e-4. For the second PDE training, set Nfit_pde=200 training steps, with a learning rate of E lr_E_pde=1e-1 and a learning rate of T lr_T_pde=1e-1. The visualization parameter is set to vmax_E=0.012 and vmax_T=0.15.

The command-line parameters are as follows:

```bash
python ./diffusion-2T/main.py --model_name "zline-gauss" --ionization_type "zline" --Nfit_reg 200 --lr_E_reg 1e-3 --lr_T_reg 1e-4 --Nfit_pde 200 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 0.012 --vmax_T 0.15
```

NMSE between the two training results and the reference solution, along with the absolute error visualization, are shown below:

|Training      |L2 relative error of E |L2 relative error of T |
|:--------:|:-----------:|:-----------:|
|Phase 1 |3.2893e-4    |2.0064e-2    |
|Phase 2 |9.5702e-6    |4.9225e-4    |

<img src="./diffusion-2T/results/zline-gauss/fig_E.png" alt="2T-zline-gauss-E" width="400" /> <img src="./diffusion-2T/results/zline-gauss/fig_T.png" alt="2T-zline-gauss-T" width="400" />

#### (5) zsquare-const

material function uses "zsquare" type: When $\frac{3}{16}<x<\frac{7}{16}, \frac{9}{16}<y<\frac{13}{16}$ or $\frac{9}{16}<x<\frac{13}{16}, \frac{3}{16}<y<\frac{7}{16}$, $z=10$; otherwise $z=1$

Initial and boundary condition uses "const" type: $\beta (x,y,t) = \max$ { $20t, 10$ }, $g(x,y,t) = 0.01$

Set the number of training steps for the first regression training to Nfit_reg=300, with a learning rate of E lr_E_reg=1e-3 and a learning rate of T lr_T_reg=1e-3. For the second PDE training, set Nfit_pde=200 training steps, with a learning rate of E lr_E_pde=1e-1 and a learning rate of T lr_T_pde=1e-1. The visualization parameter is set to vmax_E=0.6 and vmax_T=0.15.

The command-line parameters are as follows:

```bash
python ./diffusion-2T/main.py --model_name "zsquare-const" --ionization_type "zsquare" --Nfit_reg 300 --lr_E_reg 1e-3 --lr_T_reg 1e-3 --Nfit_pde 200 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 0.6 --vmax_T 0.15
```

NMSE between the two training results and the reference solution, along with the absolute error visualization, are shown below:

|Training      |L2 relative error of E |L2 relative error of T |
|:--------:|:-----------:|:-----------:|
|Phase 1 |1.6882e-3    |9.2393e-4    |
|Phase 2 |3.9618e-5    |5.0227e-6    |

<img src="./diffusion-2T/results/zsquare-const/fig_E.png" alt="2T-zsquare-const-E" width="400" /> <img src="./diffusion-2T/results/zsquare-const/fig_T.png" alt="2T-zsquare-const-T" width="400" />

#### (6) zsquare-gauss

material function uses "zsquare" type: When $\frac{3}{16}<x<\frac{7}{16}, \frac{9}{16}<y<\frac{13}{16}$ or $\frac{9}{16}<x<\frac{13}{16}, \frac{3}{16}<y<\frac{7}{16}$, $z=10$; otherwise $z=1$

Initial and boundary condition uses "gauss" type: $\beta (x,y,t) = 0, g(x,y,t) = 0.01+100e^{-(x^2+y^2)/0.01}$

Set the number of training steps for the first regression training to Nfit_reg=700, with a learning rate of E lr_E_reg=1e-3 and a learning rate of T lr_T_reg=1e-3. For the second PDE training, set Nfit_pde=100 training steps, with a learning rate of E lr_E_pde=1e-1 and a learning rate of T lr_T_pde=1e-1. The visualization parameter is set to vmax_E=0.006 and vmax_T=0.11.

The command-line parameters are as follows:

```bash
python ./diffusion-2T/main.py --model_name "zsquare-gauss" --ionization_type "zsquare" --Nfit_reg 700 --lr_E_reg 1e-3 --lr_T_reg 1e-3 --Nfit_pde 100 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 0.006 --vmax_T 0.11
```

NMSE between the two training results and the reference solution, along with the absolute error visualization, are shown below:

|Training      |L2 relative error of E |L2 relative error of T |
|:--------:|:-----------:|:-----------:|
|Phase 1 |3.5218e-4    |1.4630e-3    |
|Phase 2 |2.2470e-5    |6.6278e-4    |

<img src="./diffusion-2T/results/zsquare-gauss/fig_E.png" alt="2T-zsquare-gauss-E" width="400" /> <img src="./diffusion-2T/results/zsquare-gauss/fig_T.png" alt="2T-zsquare-gauss-T" width="400" />
