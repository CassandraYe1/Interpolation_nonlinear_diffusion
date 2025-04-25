项目标题：

非线性辐射扩散问题的神经网络算法研究

项目描述：

给定 $\Omega = [0,1]\times[0,1]$ ，单温非线性辐射扩散问题的具体模型如下：

$$
\begin{aligned}
   & \frac{\partial E}{\partial t}-\nabla\cdot(D_L\nabla E) = 0 \\
   & 0.5E+D_L\nabla E\cdot n = \beta(x,y,t), \quad(x,y,t)\in\lbrace x=0\rbrace\times[0,1] \\
   & 0.5E+D_L\nabla E\cdot n = 0, \quad(x,y,t)\in\partial\Omega\setminus\lbrace x=0\rbrace\times[0,1] \\
   & E|_{t=0} = g(x,y,0)
\end{aligned}
$$

其中辐射扩散系数 $D_L$ 选用限流形式，即 $D_L = \frac{1}{3\sigma_{\alpha}+\frac{|\nabla E|}{E}}, \sigma_{\alpha} = \frac{z^3}{E^{3/4}}$ 。

双温非线性辐射扩散问题的具体模型如下：

$$
\begin{aligned}
   & \frac{\partial E}{\partial t} - \nabla \cdot (D_L \nabla E) = \sigma_{\alpha}(T^4 - E) \\
   & \frac{\partial T}{\partial t} - \nabla \cdot (K_L \nabla T) = \sigma_{\alpha}(E - T^4) \\
   & 0.5E + D_L \nabla E \cdot n = \beta(x,y,t), \quad (x,y,t) \in \lbrace x=0 \rbrace \times [0,1] \\
   & 0.5E + D_L \nabla E \cdot n = 0, \quad (x,y,t) \in \partial\Omega \setminus \lbrace x=0 \rbrace \times [0,1] \\
   & K_L \nabla T \cdot n = 0, \quad (x,y,t) \in \partial\Omega \times [0,1] \\
   & E\vert_{t=0} = T^4\vert_{t=0} = g(x,y,0)
\end{aligned}
$$

其中 $\Omega = [0,1]\times[0,1]$ 辐射扩散系数 $D_L, K_L$ 同样选用限流形式，即 $D_L = \frac{1}{3\sigma_{\alpha}+\frac{|\nabla E|}{E}}, \sigma_{\alpha} = \frac{z^3}{E^{3/4}}, K_L = \frac{T^4}{T^{3/2}z+T^{5/2}|\nabla T|}$ 。

对于上述单温、双温问题，电离度函数 $z$ 可以分为以下三种情况：

   (1)zconst(连续)： $z=1$

   (2)zline(左右突变)：当 $x\leq0.5$ 时， $z=1$ ；当 $x\textgreater0.5$ 时， $z=10$

   (3)zsquare(两块突变)：当 $\frac{3}{16}\textless x \textless\frac{7}{16}, \frac{9}{16}\textless y \textless\frac{13}{16}$ 或 $\frac{9}{16}\textless x \textless\frac{13}{16}, \frac{3}{16}\textless y \textless\frac{7}{16}$ 时， $z=10$ ；其他时候 $z=1$

初边值条件 $\beta(x,y,t), g(x,y,t)$ 可以分为以下两种情况：

   (1)const(常数初值+线性边值)： $\beta(x,y,t) = \max\{20t, 10\}, g(x,y,t) = 0.01$

   (2)gauss(高斯初值+零边值)： $\beta(x,y,t) = 0, g(x,y,t) = 0.01+100e^{-(x^2+y^2)/0.01}$

考虑到经典数值方法(如有限体积、有限元)存在如下问题：
   
   (1)大变形网格上插值计算量大且精度难以保证；

   (2)非线性方程难以保持正性；

   (3)格式对网格变形和系数分布有限制，推广至时空高阶比较困难。
   
因此，本项目希望借助神经网络，发展融合方程及数据驱动的高精度神经网络求解算法，在提升单温、双温非线性辐射扩散问题求解精度的同时提升求解效率。

功能特性：

结合低分辨率数值解 $E_{coarse}$ 以及方程本身，我们设计了针对单一方程的新型神经网络解法。

以单温问题为例，我们用向后差分处理方程中的时间偏导，用神经网络分别计算每个时间层的结果，并在网络损失函数 $L_{reg+pde}$ 的设计部分考虑添加数据驱动损失 $L_{reg}$和物理信息损失 $L_{pde}$ 的约束，具体公式如下：

$$
\begin{aligned}
   & L_{reg+pde} = L_{reg}+10L_{pde} \\
   & L_{reg} = \Vert E^n-E^n_{coarse} \Vert \\
   & L_{pde} = \Vert E^n-D^n_{coarse}\nabla\cdot(\nabla E^n)\Delta t-E^{n-1}_{coarse} \Vert
\end{aligned}
$$

用同样的方法，我们也给出了双温问题的损失函数具体公式：

$$
\begin{aligned}
   L_{reg+pde} &= L_{reg} + 10L_{pde} \\
   L_{reg} &= \Vert E^n - E^n_{coarse} \Vert + \Vert T^n - T^n_{coarse} \Vert \\
   L_{pde} &= \Vert E^n - D^n_{coarse} \nabla \cdot (\nabla E^n) \Delta t - \sigma_{\alpha} (T^4 - E) \Delta t - E^{n-1}_{coarse} \Vert \\
   & + \Vert T^n - K^n_{coarse} \nabla \cdot (\nabla T^n) \Delta t - \sigma_{\alpha} (E - T^4) \Delta t - T^{n-1}_{coarse} \Vert
\end{aligned}
$$

为了获取目标单温和双温问题的数据，取257×257的细网格点，设时间步长为0.001，设皮卡迭代的收敛极限为0.001，将有限元法求出的结果作为参考解，并通过4倍下采样得到65×65的粗网格解 $E_{coarse}$ 。

构建一个全连接神经网络，该网络包含两个隐藏层，每一层包含512个神经元，激活函数选用relu函数。将目标点的空间坐标值作为输入数据，输出层设置为二维的，按电离度函数 $z$ 在不同目标点的大小设置布尔值，从而对各目标点的输出结果进行选择。

为了保证求解效率，我们首先利用低分辨率参考解，构建数据驱动损失函数 $L_{reg}$ 进行训练。然后，结合目标辐射扩散问题的方程，设计包含物理约束的损失函数 $L_{reg+pde}$，进一步提升神经网络模型的求解精度。

使用说明：

1. 参数说明

(1)系统参数：

|参数      |说明      |默认值      |
|:--------:|:--------:|:--------:|
|model_name    |目标模型（"电离度函数类型-初边值函数类型"）    |"zconst-const"  |
|device_name   |计算设备（"cuda"或"cpu"）    |"cuda"          |

(2)必选参数：

参数zconst、zline和zsquare中，有且仅有一个值为True，其余两个值为False

(3)网格配置参数：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|Nx   |x轴网格点数    |257    |
|Ny   |y轴网格点数    |257    |
|n    |下采样倍数     |4      |

(4)第一阶段训练参数：

单温问题：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|Nfit_reg   |训练步数    |300    |
|lr_reg   |LBFGS优化器学习率    |1e-2    |
|epoch_reg    |训练轮次     |50      |

双温问题：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|Nfit_reg   |训练步数    |300    |
|lr_E_reg   |关于E的LBFGS优化器学习率    |1e-2    |
|lr_T_reg   |关于T的LBFGS优化器学习率    |1e-2    |
|epoch_reg    |训练轮次     |50      |

(5)第二阶段训练参数：

单温问题：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|Nfit_pde   |训练步数    |200    |
|lr_pde   |LBFGS优化器学习率    |1e-1    |
|epoch_pde    |训练轮次     |10      |

双温问题：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|Nfit_pde   |训练步数    |200    |
|lr_E_pde   |关于E的LBFGS优化器学习率    |1e-1    |
|lr_T_pde   |关于T的LBFGS优化器学习率    |1e-1    |
|epoch_pde    |训练轮次     |10      |

(6)可视化参数：

单温问题：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|vmax   |误差图像色带的最大值    |0.25    |

双温问题：

|参数      |说明      |默认值      |
|:--------:|:--------:|:---------:|
|vmax_E   |E的误差图像色带的最大值    |0.25    |
|vmax_T   |T的误差图像色带的最大值    |0.25    |

2. 使用场景

这里给出每种情况对应的命令行语句。

```bash
## 单温问题：
# zconst-const
python ./diffusion-1T/main.py --model_name "zconst-const" --zconst --Nfit_reg 100 --lr_reg 1e-3 --Nfit_pde 200 --lr_pde 1 --vmax 0.25
# zconst-gauss
python ./diffusion-1T/main.py --model_name "zconst-gauss" --zconst --Nfit_reg 200 --lr_reg 1e-3 --Nfit_pde 200 --lr_pde 1 --vmax 0.2
# zline-const
python ./diffusion-1T/main.py --model_name "zline-const" --zline --Nfit_reg 150 --lr_reg 1e-2 --Nfit_pde 200 --lr_pde 1 --vmax 0.25
# zline-gauss
python ./diffusion-1T/main.py --model_name "zline-gauss" --zline --Nfit_reg 200 --lr_reg 1e-2 --Nfit_pde 100 --lr_pde 1 --vmax 0.072
# zsquare-const
python ./diffusion-1T/main.py --model_name "zsquare-const" --zsquare --Nfit_reg 150 --lr_reg 1e-2 --Nfit_pde 300 --lr_pde 1e-1 --vmax 1.0
# zsquare-gauss
python ./diffusion-1T/main.py --model_name "zsquare-gauss" --zsquare --Nfit_reg 400 --lr_reg 1e-1 --Nfit_pde 350 --lr_pde 1 --vmax 0.16

## 双温问题：
# zconst-const
python ./diffusion-2T/main.py --model_name "zconst-const" --zconst --Nfit_reg 300 --lr_E_reg 1e-3 --lr_T_reg 1e-3 --Nfit_pde 200 --lr_E_pde 1e-2 --lr_T_pde 1e-2 --vmax_E 0.28 --vmax_T 0.02
# zconst-gauss
python ./diffusion-2T/main.py --model_name "zconst-gauss" --zconst --Nfit_reg 300 --lr_E_reg 1e-3 --lr_T_reg 1e-3 --Nfit_pde 200 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 0.004 --vmax_T 0.015
# zline-const
python ./diffusion-2T/main.py --model_name "zline-const" --zline --Nfit_reg 300 --lr_E_reg 1e-2 --lr_T_reg 1e-2 --Nfit_pde 200 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 1.7 --vmax_T 0.03
# zline-gauss
python ./diffusion-2T/main.py --model_name "zline-gauss" --zline --Nfit_reg 200 --lr_E_reg 1e-3 --lr_T_reg 1e-4 --Nfit_pde 200 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 0.012 --vmax_T 0.15
# zsquare-const
python ./diffusion-2T/main.py --model_name "zsquare-const" --zsquare --Nfit_reg 300 --lr_E_reg 1e-3 --lr_T_reg 1e-3 --Nfit_pde 200 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 0.6 --vmax_T 0.15
# zsquare-gauss
python ./diffusion-2T/main.py --model_name "zsquare-gauss" --zsquare --Nfit_reg 700 --lr_E_reg 1e-3 --lr_T_reg 1e-3 --Nfit_pde 100 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 0.006 --vmax_T 0.11
```

3. 输出说明

对于单温问题，训练完成后会在 ./diffusion-1T/<model_name>/results/ 目录下生成：

(1) model_reg.pt : 第一阶段训练模型

(2) model_pinn.pt : 第二阶段训练模型

(3) sol_reg.npy : 第一阶段预测结果

(4) sol_pinn.npy : 第二阶段预测结果

(5) fig.png : 第一、第二阶段预测结果误差图像

对于单温问题，训练完成后会在 ./diffusion-2T/<model_name>/results/ 目录下生成：

(1) model_reg_E.pt : 第一阶段关于E的训练模型

(2) model_reg_T.pt : 第一阶段关于T的训练模型

(3) model_pinn_E.pt : 第二阶段关于E的训练模型

(4) model_pinn_T.pt : 第二阶段关于T的训练模型

(5) sol_reg_E.npy : 第一阶段对E的预测结果

(6) sol_reg_T.npy : 第一阶段对T的预测结果

(7) sol_pinn_E.npy : 第二阶段对E的预测结果

(8) sol_pinn_T.npy : 第二阶段对T的预测结果

(9) fig_E.png : 关于E的第一、第二阶段预测结果误差图像

(10) fig_T.png : 关于T的第一、第二阶段预测结果误差图像

-----------------------------------------------------------------------------------------------------------------------------

Project Title:

Neural Network Algorithm Research for Nonlinear Radiation Diffusion Problems

Project Description:

With $\Omega = [0,1]\times[0,1]$ , the specific model for the single-temperature nonlinear radiation diffusion problem is as follows:

$$
\begin{aligned}
   & \frac{\partial E}{\partial t}-\nabla\cdot(D_L\nabla E) = 0 \\
   & 0.5E+D_L\nabla E\cdot n = \beta(x,y,t), \quad(x,y,t)\in\lbrace x=0\rbrace\times[0,1] \\
   & 0.5E+D_L\nabla E\cdot n = 0, \quad(x,y,t)\in\partial\Omega\setminus\lbrace x=0\rbrace\times[0,1] \\
   & E|_{t=0} = g(x,y,0)
\end{aligned}
$$

where the radiation diffusion coefficient $D_L$ adopts the flux-limited form, expressed as $D_L = \frac{1}{3\sigma_{\alpha}+\frac{|\nabla E|}{E}}, \sigma_{\alpha} = \frac{z^3}{E^{3/4}}$ .

The specific model for the two-temperature nonlinear radiation diffusion problem is as follows:

$$
\begin{aligned}
   & \frac{\partial E}{\partial t} - \nabla \cdot (D_L \nabla E) = \sigma_{\alpha}(T^4 - E) \\
   & \frac{\partial T}{\partial t} - \nabla \cdot (K_L \nabla T) = \sigma_{\alpha}(E - T^4) \\
   & 0.5E + D_L \nabla E \cdot n = \beta(x,y,t), \quad (x,y,t) \in \lbrace x=0 \rbrace \times [0,1] \\
   & 0.5E + D_L \nabla E \cdot n = 0, \quad (x,y,t) \in \partial\Omega \setminus \lbrace x=0 \rbrace \times [0,1] \\
   & K_L \nabla T \cdot n = 0, \quad (x,y,t) \in \partial\Omega \times [0,1] \\
   & E\vert_{t=0} = T^4\vert_{t=0} = g(x,y,0)
\end{aligned}
$$

where the radiation diffusion coefficient $D_L, K_L$ also adopts the flux-limited form, expressed as $D_L = \frac{1}{3\sigma_{\alpha}+\frac{|\nabla E|}{E}}, \sigma_{\alpha} = \frac{z^3}{E^{3/4}}, K_L = \frac{T^4}{T^{3/2}z+T^{5/2}|\nabla T|}$ .

For the single-temperature and two-temperature problems mentioned above, the ionization function $z$ can be classified into the following three cases:

   (1)zconst: Always $z=1$

   (2)zline: When $x\leq0.5$, $z=1$; when $x\textgreater0.5$, $z=10$

   (3)zsquare: When $\frac{3}{16}\textless x \textless\frac{7}{16}, \frac{9}{16}\textless y \textless\frac{13}{16}$ or $\frac{9}{16}\textless x \textless\frac{13}{16}, \frac{3}{16}\textless y \textless\frac{7}{16}$, $z=10$; otherwise $z=1$

Initial and boundary conditions $\beta(x,y,t), g(x,y,t)$ can be classified into the following two cases:

   (1)const: $\beta(x,y,t) = \max\{20t, 10\}, g(x,y,t) = 0.01$

   (2)gauss: $\beta(x,y,t) = 0, g(x,y,t) = 0.01+100e^{-(x^2+y^2)/0.01}$

Classical numerical methods (e.g., finite volume/element) face three key challenges:

   (1) computational cost and accuracy loss on highly deformed meshes;

   (2) positivity preservation in nonlinear equations;

   (3) mesh/coefficient sensitivity and high-order limitations.

To address these, we propose a hybrid equation-data-driven neural solver to enhance accuracy and efficiency for single-/two-temperature nonlinear radiation diffusion problems.

Key Features:

We propose a novel neural network-based solver for a single equation by combining low-resolution numerical solutions $E_{coarse}$ with the governing equation itself.

Taking the single-temperature problem as an example, we discretize the temporal derivative in the equation using backward differencing and employ a neural network to compute the results at each time step. The loss function $L_{reg+pde}$ of the network incorporates both data-driven loss $L_{reg}$ and physics-informed loss $L_{pde}$ as constraints. The specific formulation is as follows:

$$
\begin{aligned}
   & L_{reg+pde} = L_{reg}+10L_{pde} \\
   & L_{reg} = \Vert E^n-E^n_{coarse} \Vert \\
   & L_{pde} = \Vert E^n-D^n_{coarse}\nabla\cdot(\nabla E^n)\Delta t-E^{n-1}_{coarse} \Vert
\end{aligned}
$$

Using the same methodology, we also derive the specific formulation of the loss function for the two-temperature problem as follows:

$$
\begin{aligned}
   L_{reg+pde} &= L_{reg} + 10L_{pde} \\
   L_{reg} &= \Vert E^n - E^n_{coarse} \Vert + \Vert T^n - T^n_{coarse} \Vert \\
   L_{pde} &= \Vert E^n - D^n_{coarse} \nabla \cdot (\nabla E^n) \Delta t - \sigma_{\alpha} (T^4 - E) \Delta t - E^{n-1}_{coarse} \Vert \\
   &\quad + \Vert T^n - K^n_{coarse} \nabla \cdot (\nabla T^n) \Delta t - \sigma_{\alpha} (E - T^4) \Delta t - T^{n-1}_{coarse} \Vert
\end{aligned}
$$

To generate the required data for both single-temperature and two-temperature target problems, we implemented the following numerical procedure:

   (1) Employed a fine-grid of 256×256 spatial points, with the time step size 0.001 and the tolerance of Picard iteration convergence criterion 0.001, to setup high-resolution simulation;

   (2) Computed high-fidelity reference solutions using the finite element method, which serve as the fine-grid reference solution;

   (3) Generated corresponding coarse-grid data $E_{coarse}$ through 4x downsampling the FEM results as 65×65.

Construct a fully connected neural network that contains two hidden layers. Each layer contains 512 neurons with ReLU as the activation function. Take the spatial coordinate values of the target points as input data and set a two-dimensional output layer. Configure Boolean values in the output layer based on the magnitude of the ionization function $z$ at different target points, so that the model will select the output results for each target point.

To ensure solving efficiency, we first utilize the low-resolution reference solution to construct a data-driven loss function $L_{reg}$ for training. Then, incorporating the equations of the target radiation diffusion problem, we design a physics-constrained loss function $L_{reg+pde}$ to further improve the solving accuracy of the neural network model.

Usage Instructions:

1. Parameter specification

(1) System parameters:

|Parameter      |Description      |Default      |
|:--------:|:--------:|:--------:|
|model_name    |target model ("ionization function type-initial&boundary condition type")   |"zconst-const"  |
|device_name   |computation device ("cuda"或"cpu")    |"cuda"          |

(2) Required parameters:

The parameters "zconst", "zline" and "zsquare" must satisfy the condition that exactly one of them is set to True, while the other two must be False.

(3) Grid configuration:

|Parameter      |Description      |Default      |
|:--------:|:--------:|:---------:|
|Nx   |grid points on x-axis    |257    |
|Ny   |grid points on x-axis    |257    |
|n    |downsampling factor      |4      |

(4) Training parameters (Phase 1):

Single-temperature problem:

|Parameter      |Description      |Default      |
|:--------:|:--------:|:---------:|
|Nfit_reg   |training iterations    |300    |
|lr_reg   |LBFGS optimizer learning rate    |1e-2    |
|epoch_reg    |training epochs     |50      |

Two-temperature problem:

|Parameter      |Description      |Default      |
|:--------:|:--------:|:---------:|
|Nfit_reg   |training iterations    |300    |
|lr_E_reg   |LBFGS optimizer learning rate of E   |1e-2    |
|lr_T_reg   |LBFGS optimizer learning rate of T    |1e-2    |
|epoch_reg    |training epochs     |50      |

(5) Training parameters (Phase 2):

Single-temperature problem:

|Parameter      |Description      |Default      |
|:--------:|:--------:|:---------:|
|Nfit_pde   |training iterations    |200    |
|lr_pde   |LBFGS optimizer learning rate    |1e-1    |
|epoch_pde    |training epochs     |10      |

Two-temperature problem:

|Parameter      |Description      |Default      |
|:--------:|:--------:|:---------:|
|Nfit_pde   |training iterations    |200    |
|lr_E_pde   |LBFGS optimizer learning rate of E    |1e-1    |
|lr_T_pde   |LBFGS optimizer learning rate of T    |1e-1    |
|epoch_pde    |training epochs     |10      |

(6) Visualization parameters:

Single-temperature problem:

|Parameter      |Description      |Default      |
|:--------:|:--------:|:---------:|
|vmax   |Maximum value of error colorbar    |0.25    |

Two-temperature problem:

|Parameter      |Description      |Default      |
|:--------:|:--------:|:---------:|
|vmax_E   |Maximum value of error colorbar E   |0.25    |
|vmax_T   |Maximum value of error colorbar T   |0.25    |

2. Use cases

Here are the corresponding command-line statements for each case:

```bash
## Single-temperature problem:
# zconst-const
python ./diffusion-1T/main.py --model_name "zconst-const" --zconst --Nfit_reg 100 --lr_reg 1e-3 --Nfit_pde 200 --lr_pde 1 --vmax 0.25
# zconst-gauss
python ./diffusion-1T/main.py --model_name "zconst-gauss" --zconst --Nfit_reg 200 --lr_reg 1e-3 --Nfit_pde 200 --lr_pde 1 --vmax 0.2
# zline-const
python ./diffusion-1T/main.py --model_name "zline-const" --zline --Nfit_reg 150 --lr_reg 1e-2 --Nfit_pde 200 --lr_pde 1 --vmax 0.25
# zline-gauss
python ./diffusion-1T/main.py --model_name "zline-gauss" --zline --Nfit_reg 200 --lr_reg 1e-2 --Nfit_pde 100 --lr_pde 1 --vmax 0.072
# zsquare-const
python ./diffusion-1T/main.py --model_name "zsquare-const" --zsquare --Nfit_reg 150 --lr_reg 1e-2 --Nfit_pde 300 --lr_pde 1e-1 --vmax 1.0
# zsquare-gauss
python ./diffusion-1T/main.py --model_name "zsquare-gauss" --zsquare --Nfit_reg 400 --lr_reg 1e-1 --Nfit_pde 350 --lr_pde 1 --vmax 0.16

## Two-temperature problem:
# zconst-const
python ./diffusion-2T/main.py --model_name "zconst-const" --zconst --Nfit_reg 300 --lr_E_reg 1e-3 --lr_T_reg 1e-3 --Nfit_pde 200 --lr_E_pde 1e-2 --lr_T_pde 1e-2 --vmax_E 0.28 --vmax_T 0.02
# zconst-gauss
python ./diffusion-2T/main.py --model_name "zconst-gauss" --zconst --Nfit_reg 300 --lr_E_reg 1e-3 --lr_T_reg 1e-3 --Nfit_pde 200 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 0.004 --vmax_T 0.015
# zline-const
python ./diffusion-2T/main.py --model_name "zline-const" --zline --Nfit_reg 300 --lr_E_reg 1e-2 --lr_T_reg 1e-2 --Nfit_pde 200 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 1.7 --vmax_T 0.03
# zline-gauss
python ./diffusion-2T/main.py --model_name "zline-gauss" --zline --Nfit_reg 200 --lr_E_reg 1e-3 --lr_T_reg 1e-4 --Nfit_pde 200 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 0.012 --vmax_T 0.15
# zsquare-const
python ./diffusion-2T/main.py --model_name "zsquare-const" --zsquare --Nfit_reg 300 --lr_E_reg 1e-3 --lr_T_reg 1e-3 --Nfit_pde 200 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 0.6 --vmax_T 0.15
# zsquare-gauss
python ./diffusion-2T/main.py --model_name "zsquare-gauss" --zsquare --Nfit_reg 700 --lr_E_reg 1e-3 --lr_T_reg 1e-3 --Nfit_pde 100 --lr_E_pde 1e-1 --lr_T_pde 1e-1 --vmax_E 0.006 --vmax_T 0.11
```

3. Output specification

For single-temperature problem, results will be saved in "./diffusion-1T/<model_name>/results/":

(1) model_reg.pt : Phase 1 regression trained model

(2) model_pinn.pt : Phase 2 physics-informed model

(3) sol_reg.npy : Phase 1 predictions

(4) sol_pinn.npy : Phase 2 predictions

(5) fig.png : figure about the comparison of regression and PINN solutions

For two-temperature problem, results will be saved in "./diffusion-2T/<model_name>/results/":

(1) model_reg_E.pt : Phase 1 regression trained model of E

(2) model_reg_T.pt : Phase 1 regression trained model of T

(3) model_pinn_E.pt : Phase 2 physics-informed model of E

(4) model_pinn_T.pt : Phase 2 physics-informed model of T

(5) sol_reg_E.npy : Phase 1 E-predictions

(6) sol_reg_T.npy : Phase 1 T-predictions

(7) sol_pinn_E.npy : Phase 2 E-predictions

(8) sol_pinn_T.npy : Phase 2 T-predictions

(9) fig_E.png : figure about the comparison of regression and PINN E-solutions

(10) fig_T.png : figure about the comparison of regression and PINN T-solutions
