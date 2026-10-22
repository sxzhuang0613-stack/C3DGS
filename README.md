DynamicGaussianRepresentation

该仓库包含心肌参考相位高斯初始化和文章所述 GaussianEvolution 两阶段流程。

训练

首先运行参考相位高斯初始化：

CUDA_VISIBLE_DEVICES=0 python GetInitialGaussian.py

随后拟合每个序列从参考相位到目标相位的连续速度场与二维低频强度残差：

CUDA_VISIBLE_DEVICES=0 python motionx.py --config config.yaml

Stage 2 默认从已有 motionx_ode_latest.pth 继续训练。若开启运动导出，每个 case 的
stage2_experiments/<case>/motion_sequence.npz 包含：

reference_xyz：参考相位高斯中心 [N, 3]；

xyz：所有相位的连续高斯轨迹 [T, N, 3]；

displacement：由连续速度场积分得到的参考相位到各相位位移 [T, N, 3]；

jacobian_det：各高斯位置处的局部变形行列式 [T, N]。

同目录下的 dense_flow.npz 可直接用于掩膜传播与应变计算：

flow_pixel：参考相位到各相位的二维像素位移 [T, S, 2, H, W]；

through_plane_normalized：归一化坐标中的穿层位移 [T, S, H, W]。

GaussianEvolution 实现

Gaussians/GaussianEvolutionX.py 使用原始 (x,y,z,t) 坐标和 MLP 预测
瞬时速度 v(x,t)，以 RK4 从参考相位积分至目标相位，同时积分
dF/dt=(grad_x v)F。动态协方差按照
Sigma(t)=F(t) Sigma(0) F(t)^T 更新。二维观测分支只接收截断 Fourier 编码，
输出与结构渲染结果相加的低频强度残差，并在参考相位强制为零。

运动维度由 config.yaml 控制：

motion_dimension: "2D" 严格令切片法向位移为零；

motion_dimension: "2D" 还会预测正的法向尺度，使准不可压缩约束可以通过
transverse support expansion 补偿平面内收缩；

motion_dimension: "3D" 同时预测穿层速度，并使用
through_plane_max_displacement * tanh(v_z) 限制其归一化速度；当归一化时间属于
[0,1] 时，相应总穿层位移也不超过该幅度。

训练目标与文章保持一致：

L = L_R + 0.1 L_V + 0.01 L_S

其中 L_R 为图像 L1 损失，L_V=(det(F)-1)^2，L_S=||grad v||_2^2。

定制渲染器

GaussianEvolution 直接向渲染器传递预计算的动态协方差。因此合并本仓库中的
CUDA 修改后，需要在包含完整 third_party/glm 的渲染库目录重新编译：

cd diff-gaussian-rasterization
pip install -e .

当前仓库仅保存了作者修改过的渲染器文件；编译时仍需使用原工程中未上传的依赖文件。

最小测试

pytest -q tests/test_gaussian_evolution.py


nohup env CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python -u Motion.py > stage2.log 2>&1 < /dev/null &

# Stage 2 控制点高斯演化

文件放置位置：

```text
DynamicGaussianRepresentation/
├── motion.py
└── Gaussians/
    └── GaussianEvolution.py
```

## 三种训练路径

```bash
# 原始全高斯 Neural ODE 消融
python motion.py --config config.yaml --control-deformation-mode none

# 自适应控制点 + 核传播（推荐默认方案）
python Motion.py --config config.yaml --control-deformation-mode kernel

# 自适应控制点 + 局部 MLS/仿射变形图
python motion.py --config config.yaml --control-deformation-mode mls
```

可从命令行覆盖控制点覆盖半径和邻居数：

```bash
python motion.py --config config.yaml \
  --control-deformation-mode mls \
  --control-coverage-radius 0.12 \
  --control-neighbors 8
```

## 关键定义

- `flow_jacobian` / `flow_jacobian_det`：真实材料点运动 Jacobian，用于拓扑与折叠判断。
- `support_transform` / `support_volume_ratio`：加入法向尺度后的高斯支撑变换，用于协方差 push-forward 和准不可压缩。
- 控制点只参数化运动；全部 Stage-1 高斯仍然得到中心、协方差和切片占据并参与渲染。
- 控制点由覆盖半径停止的 FPS 按病例自适应选择，阶段二训练期间固定不变。
- `lambda_control_anchor` 约束传播后的控制高斯与其 ODE 控制轨迹一致。

三种传播方式使用不同的 checkpoint 文件名：

```text
net_latest_none.pth
net_latest_kernel.pth
net_latest_mls.pth
```

避免在切换传播方式时误加载语义不同的旧检查点。
