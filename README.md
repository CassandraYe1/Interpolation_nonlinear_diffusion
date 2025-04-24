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
   &\quad + \Vert T^n - K^n_{coarse} \nabla \cdot (\nabla T^n) \Delta t - \sigma_{\alpha} (E - T^4) \Delta t - T^{n-1}_{coarse} \Vert
\end{aligned}
$$

为了获取目标单温和双温问题的数据，取256×256的细网格点，设时间步长为0.001，设皮卡迭代的收敛极限为0.001，将有限元法求出的结果作为参考解，并通过4倍下采样得到65×65的粗网格解 $E_{coarse}$ 。

构建一个全连接神经网络，该网络包含两个隐藏层，每一层包含512个神经元，激活函数选用relu函数。将目标点的空间坐标值作为输入数据，输出层设置为二维的，按电离度函数 $z$ 在不同目标点的大小设置布尔值，从而对各目标点的输出结果进行选择。

为了保证求解效率，我们首先利用低分辨率参考解，构建数据驱动损失函数 $L_{reg}$ 进行训练。然后，结合目标辐射扩散问题的方程，设计包含物理约束的损失函数 $L_{reg+pde}$，进一步提升神经网络模型的求解精度。

使用说明：

更改

输出结果为(1)低分辨率数据驱动损失函数训练的结果sol_reg.npy；(2)PDE方程物理约束驱动损失函数训练的结果sol_pinn.npy。

运行可视化脚本interp_plot.ipynb。

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

For the single-temperature and two-temperature problems mentioned above, the ionization degree function $z$ can be classified into the following three cases:

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

1. Configure config.py:

   (1) Model naming convention: Model name format is like "ionization function type - initial condition type". There are three types of ionization function (zconst, zline and zsquare) and two types of initial condition (const and gauss) in total.

   (2) Compute device: Set to either "cpu" or "cuda".

   (3) Path configuration: Set "True" for the corresponding ionization function (zconst/zline/zsquare) being used, and "False" for the other two.

2. Adjust training iterations (Nfit) and learning rates (lr, lr_E and lr_T) in the neural network according to the comments in main.py.

3. Run the main program "python main.py",

   with output results: (1) sol_reg.npy - Results from coarse-grid data-driven training; (2) sol_pinn.npy - Results from PDE physics-constrained training.

4. Execute the visualization script "interp_plot.ipynb".
