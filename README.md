\textbf{项目标题：}

非线性辐射扩散问题的神经网络算法研究

项目描述：

针对单温、双温非线性辐射扩散问题，发展融合方程及数据驱动的高精度神经网络求解算法。

功能特性：

首先，利用低分辨率数值解，构建数据驱动损失函数，提升神经网络模型的训练效率；然后，结合目标辐射扩散问题的方程，设计包含物理约束的损失函数，进一步提升神经网络模型的求解精度。

使用说明：

1. 修改config.py配置：
   
   (1)定义模型名称：模型名称的构成为"电离度函数类别-初值条件类别"，其中电离度函数分为zconst(连续)、zline(左右突变)和zsquare(两块突变)三类，初值条件分为const(常数)和gauss(高斯函数)两类；

   (2)设置计算设备：cpu或cuda；

   (3)定义路径：该模型对应哪一种电离度函数，就将哪一种电离度函数(zconst/zline/zsquare)设置为True，其他两种设置为False。

2. 运行可视化脚本interp_plot.ipynb

3. 输出结果：

   (1)低分辨率数据驱动损失函数训练的结果sol_reg.npy；

   (2)PDE方程物理约束驱动损失函数训练的结果sol_pinn.npy；

   (3)上述两种结果与参考解之间的可视化图像对比。

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

2. Run Visualization Script "interp_plot.ipynb".

3. Output Results:

   (1) sol_reg.npy - Results from coarse-grid data-driven training.

   (2) sol_pinn.npy - Results from PDE physics-constrained training.

   (3) Comparative visualizations between both results and reference solutions.
