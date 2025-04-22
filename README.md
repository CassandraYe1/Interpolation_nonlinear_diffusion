项目标题：

非线性辐射扩散问题的神经网络算法研究

项目描述：

单温非线性辐射扩散问题的具体模型如下：

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
   
   (1)大变形网格上节点量插值计算量大且精度\保正难以保证；

   (2)保正格式为非线性，是否存在更好地到局部数值流的映射关系；

   (3)格式对网格变形和系数分布有限制，且推广至时空高阶比较困难，
   
因此，本项目希望借助神经网络，发展融合方程及数据驱动的高精度神经网络求解算法，在提升单温、双温非线性辐射扩散问题求解精度的同时提升求解效率。

功能特性：

结合低分辨率数值解 $E_{coarse}$ 以及方程本身，我们设计了针对单一方程的新型神经网络解法。

以单温问题为例，我们用向后差分处理方程中的时间偏导，用神经网络分别计算每个时间层的结果，并在网络损失函数 $L_{data+pinn}$ 的设计部分考虑添加数据驱动损失 $L_{ref}$和物理信息损失 $L_{pde}$ 的约束，具体公式如下：

$$
\begin{aligned}
   & L_{data+pinn} = \omega_{ref}L_{ref}+\omega_{pde}L_{pde} \\
   & L_{ref} = \Vert E^n-E^n_{coarse} \Vert \\
   & L_{pde} = \Vert E^n-D^n_{coarse}\nabla\cdot(\nabla E^n)\Delta t-E^{n-1}_{coarse} \Vert
\end{aligned}
$$

用同样的方法，我们也给出了双温问题的损失函数具体公式：

$$
\begin{aligned}
   L_{data+pinn} &= \omega_{ref} L_{ref} + \omega_{pde} L_{pde} \\
   L_{ref} &= \Vert E^n - E^n_{coarse} \Vert + \Vert T^n - T^n_{coarse} \Vert \\
   L_{pde} &= \Vert E^n - D^n_{coarse} \nabla \cdot (\nabla E^n) \Delta t - \sigma_{\alpha} (T^4 - E) \Delta t - E^{n-1}_{coarse} \Vert \\
   &\quad + \Vert T^n - K^n_{coarse} \nabla \cdot (\nabla T^n) \Delta t - \sigma_{\alpha} (E - T^4) \Delta t - T^{n-1}_{coarse} \Vert
\end{aligned}
$$

具体来说，对于单温和双温问题，我们分别取128×128和256×256的细网格点，设时间步长为0.001，皮卡迭代的收敛极限为0.001，将有限元法求出的结果作为参考解，并利用下采样得到65×65的粗网格解，也就是E_coarse。我们选择构建一个全连接神经网络，包含两个隐藏层，激活函数选用relu函数，有效确保了结果的正性。将目标点的空间坐标值作为输入数据进行训练，这使得结果跳出了传统数值解法只能求解网格点值的局限，保证了解在空间上的连续性。另外，考虑到由于电离度函数可能发生由1到10的突变，从而导致结果在不同空间发生剧烈变化，我们设置了一个二维的输出层，并按电离度函数在不同目标点的大小设置了布尔值，从而对各目标点的输出结果进行选择。
为了保证求解效率，我们先在仅有低分辨率参考解约束损失函数的情况下进行训练，学习率设为0.01，训练300步。然后再向损失函数中加入目标方程的物理约束，继续训练，此时学习率设为0.1，训练200步。如果还想再次提升网络学习效率，也可以考虑将初边值条件强制嵌入神经网络，进一步缩小解的函数空间。

首先，利用低分辨率数值解，构建数据驱动损失函数，提升神经网络模型的训练效率；然后，结合目标辐射扩散问题的方程，设计包含物理约束的损失函数，进一步提升神经网络模型的求解精度。

使用说明：

1. 修改config.py配置：
   
   (1)定义模型名称：模型名称的构成为"电离度函数类别-初值条件类别"，其中电离度函数分为zconst(连续)、zline(左右突变)和zsquare(两块突变)三类，初值条件分为const(常数)和gauss(高斯函数)两类；

   (2)设置计算设备：cpu或cuda；

   (3)定义路径：该模型对应哪一种电离度函数，就将哪一种电离度函数(zconst/zline/zsquare)设置为True，其他两种设置为False。

2. 根据main.py中的注释，修改神经网络的训练步数(Nfit)和学习率(lr, lr_E and lr_T)。

3. 运行主程序python main.py，

   输出结果为(1)低分辨率数据驱动损失函数训练的结果sol_reg.npy；(2)PDE方程物理约束驱动损失函数训练的结果sol_pinn.npy。

5. 运行可视化脚本interp_plot.ipynb。

-----------------------------------------------------------------------------------------------------------------------------

Project Title:

Neural Network Algorithm Research for Nonlinear Radiation Diffusion Problems

Project Description:

This project develops high-precision neural network that integrate equation-driven and data-driven approaches for both single-temperature and two-temperature nonlinear radiation diffusion problems.

Key Features:

Firstly, utilize coarse-grid reference solution to construct data-driven loss function, which can enhance the training efficiency of our neural network model.

Then, design physics-informed loss function, incorporating target nonlinear radiation diffusion equations, in order to improve solution accuracy.

Usage Instructions:

1. Configure config.py:

   (1) Model naming convention: Model name format is like "ionization function type - initial condition type". There are three types of ionization function (zconst, zline and zsquare) and two types of initial condition (const and gauss) in total.

   (2) Compute device: Set to either "cpu" or "cuda".

   (3) Path configuration: Set "True" for the corresponding ionization function (zconst/zline/zsquare) being used, and "False" for the other two.

2. Adjust training iterations (Nfit) and learning rates (lr, lr_E and lr_T) in the neural network according to the comments in main.py.

3. Run the main program "python main.py",

   with output results: (1) sol_reg.npy - Results from coarse-grid data-driven training; (2) sol_pinn.npy - Results from PDE physics-constrained training.

4. Execute the visualization script "interp_plot.ipynb".
