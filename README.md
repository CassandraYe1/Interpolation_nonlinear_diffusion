# 非线性辐射扩散问题的神经网络超分辨率算法研究

## 背景介绍：

超分辨率技术在科学计算领域展现出突破传统数值方法效率瓶颈的革命性潜力。特别是在涉及多尺度、强非线性特征的物理场重构任务中，传统基于插值或数据驱动的超分辨率方法往往因缺乏物理规律约束，导致重构结果存在非物理解或守恒性破坏等根本缺陷。这一局限性在辐射输运、湍流模拟等对物理一致性有严格要求的领域尤为突出，严重制约了超分辨率技术在实际工程上的应用。

针对这些问题，本项目提出了一种超分辨率神经网络框架，用神经网络直接学习从低分辨率网格到高分辨率网格的映射关系。该网络构建了"粗网格输入→网络预测→物理校正"的架构，在传统数据驱动损失的基础上，引入基于方程的物理约束，确保预测解严格满足物理规律，在保证非线性辐射扩散问题求解精度的同时提升求解效率。

## 非线性辐射扩散问题描述：

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

### 典型测试问题：

在辐射扩散方程中，材料函数 $z(x,y)$ 是描述计算域内介质吸收辐射能量并将其转化为内能效率的关键物理量。我们构造了以下三种材料函数 $z(x,y)$ 的参数化模型：

常数材料函数： $z=1$

间断线性材料函数：当 $x\leq0.5$ 时， $z=1$ ；当 $x>0.5$ 时， $z=10$

双方形材料函数：当 $\frac{3}{16}<x<\frac{7}{16}, \frac{9}{16}<y<\frac{13}{16}$ 或 $\frac{9}{16}<x<\frac{13}{16}, \frac{3}{16}<y<\frac{7}{16}$ 时， $z=10$ ；其他时候 $z=1$

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
   & L_{\text{reg}} = \frac{\Vert E^n_{\text{coarse}}-E^n \Vert_2}{\Vert E^n_{\text{coarse}} \Vert_2} + \frac{\Vert T^n_{\text{coarse}}-T^n \Vert_2}{\Vert T^n_{\text{coarse}} \Vert_2} \\
   & L_{\text{pde}} = \Vert E^n - D^n_{\text{coarse}} \nabla \cdot (\nabla E^n) \Delta t - \sigma_{\alpha} (T^4 - E) \Delta t - E^{n-1}_{\text{coarse}} \Vert_2^2 + 
\end{aligned}
$$

$$
\begin{aligned}
   \Vert T^n - K^n_{\text{coarse}} \nabla \cdot (\nabla T^n) \Delta t - \sigma_{\alpha} (E - T^4) \Delta t - T^{n-1}_{\text{coarse}} \Vert_2^2
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

### 项目结构：
   
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
│   └── main.py
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
│   └── main.py
├── requirements.txt
└── README.md
```

### 参数设置：

##### 模型参数：

|参数      |说明      |默认值      |
|:--------:|:--------:|:--------:|
|model_name    |目标模型（"材料函数类型_初值函数类型"）    |zconst_const  |
|device_name   |计算设备（"cuda"或"cpu"）    |"cuda"          |
|ionization_type   |材料函数类型（"zconst"或"zline"或"zsquare"）    |zconst          |

##### 网格参数：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|Nx   |x轴细网格点数    |257    |
|Ny   |y轴细网格点数    |257    |
|n    |下采样倍数     |2      |
|t    |目标时刻       |1      |

##### 网络参数：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|depth   |隐藏层层数    |2    |
|width   |隐藏层单元数    |512    |

#### 训练参数：

##### 回归训练参数：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|Nfit_reg   |训练步数    |500    |
|lr_E_reg   |关于E的LBFGS优化器学习率    |1e-1    |
|lr_T_reg   |关于T的LBFGS优化器学习率    |1e-1    |
|epoch_reg    |训练轮次     |50      |

##### PDE训练参数：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|Nfit_pde   |训练步数    |500    |
|lr_E_pde   |关于E的LBFGS优化器学习率    |1    |
|lr_T_pde   |关于T的LBFGS优化器学习率    |1    |
|epoch_pde    |训练轮次     |10      |

## 数值实验：

### 数据集：

参考解由FreeFem++的有限元求解器生成，具体设置如下：对于单温及双温问题，在上取个节点的四边形网格，时间离散采用隐式向后欧拉。时间步长设置为，时间域从计算至，共1000步。每一时间步都使用Picard迭代求解非线性系统，更新解直至两次迭代间的残差降低至0.001或迭代次数达到100次。低分辨率数据由参考解经过下采样得到。

从[https://pan.baidu.com/s/1yS07Ebv-AE2ta2DWwui3GQ?pwd=9j42](https://pan.baidu.com/s/1yS07Ebv-AE2ta2DWwui3GQ?pwd=9j42)访问数据集。根据上述项目结构，将提取的`/diffusion-1T/data`和`/diffusion-2T/data`分别放在根目录中。

### 单温问题：

#### (1) zconst_const

常数材料函数+常数初值+线性边值的情况下，命令行参数如下：

```bash
python ./diffusion-1T/main.py --model_name "zconst_const" --ionization_type "zconst" --Nx 129 --Ny 129 --Nfit_reg 300 --Nfit_pde 200
```

两次训练结果与参考解之间的相对 $L_2$ 误差如下：

|训练      |相对 $L_2$ 误差 |
|:--------:|:--------:|
|第一次训练   | $1.30\times10^{-4}$ |
|第二次训练   | $1.02\times10^{-4}$ |

#### (2) zline_const

间断线性材料函数+常数初值+线性边值的情况下，命令行参数如下：

```bash
python ./diffusion-1T/main.py --model_name "zline_const" --ionization_type "zline" --Nx 129 --Ny 129
```

两次训练结果与参考解之间的相对 $L_2$ 误差如下：

|训练      |相对 $L_2$ 误差 |
|:--------:|:--------:|
|第一次训练   | $1.83\times10^{-3}$ |
|第二次训练   | $1.81\times10^{-3}$ |

#### (3) zsquare_const

双方形材料函数+常数初值+线性边值的情况下，命令行参数如下：

```bash
python ./diffusion-1T/main.py --model_name "zsquare_const" --ionization_type "zsquare" --Nx 145 --Ny 145
```

两次训练结果与参考解之间的相对 $L_2$ 误差如下：

|训练      |相对 $L_2$ 误差 |
|:--------:|:--------:|
|第一次训练   | $4.12\times10^{-3}$ |
|第二次训练   | $1.58\times10^{-3}$ |

### 双温问题：

#### (1) zconst_gauss

常数材料函数+高斯初值+零边值的情况下，命令行参数如下：

```bash
python ./diffusion-2T/main.py --model_name "zconst_gauss" --ionization_type "zconst" --Nx 129 --Ny 129 --Nfit_reg 300 --Nfit_pde 200
```

两次训练结果与参考解之间的相对 $L_2$ 误差以及绝对误差可视化如下：

|训练      |E的相对 $L_2$ 误差 |T的相对 $L_2$ 误差 |
|:--------:|:-----------:|:-----------:|
|第一次训练 | $7.80\times10^{-4}$ | $1.65\times10^{-4}$ |
|第二次训练 | $7.42\times10^{-4}$ | $1.62\times10^{-4}$ |

#### (2) zline_gauss

间断线性材料函数+高斯初值+零边值的情况下，命令行参数如下：

```bash
python ./diffusion-2T/main.py --model_name "zline_gauss" --ionization_type "zline" --Nx 129 --Ny 129
```

两次训练结果与参考解之间的相对 $L_2$ 误差以及绝对误差可视化如下：

|训练      |E的相对 $L_2$ 误差 |T的相对 $L_2$ 误差 |
|:--------:|:-----------:|:-----------:|
|第一次训练 | $1.51\times10^{-3}$ | $2.86\times10^{-4}$ |
|第二次训练 | $1.47\times10^{-3}$ | $2.86\times10^{-4}$ |

#### (3) zsquare_gauss

双方形材料函数+高斯初值+零边值的情况下，命令行参数如下：

```bash
python ./diffusion-2T/main.py --model_name "zsquare_gauss" --ionization_type "zsquare" --Nx 145 --Ny 145
```

两次训练结果与参考解之间的相对 $L_2$ 误差以及绝对误差可视化如下：

|训练      |E的相对 $L_2$ 误差 |T的相对 $L_2$ 误差 |
|:--------:|:-----------:|:-----------:|
|第一次训练 | $1.12\times10^{-3}$ | $6.07\times10^{-4}$ |
|第二次训练 | $1.16\times10^{-3}$ | $5.27\times10^{-4}$ |
