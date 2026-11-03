# 2026-11-03：加性 Gaussian 场与解析厚度二维切片渲染

- 目标：将 v2 归一化薄切片混合改为有符号加性场、解析 Gaussian PSF 和等效二维 rasterization；保留 tile 架构。
- 修改：渲染库 Python/native 参数、Forward/Backward、几何/图像缓存及公共数学辅助；新增 `reference.py`、`tests/test_slice_response.py` 和短基准脚本。训练/测试配置及 `render_options()` 仅接入 PSF，移除 background_weight。README 同步，v2 指导文档标为历史。
- 数学：S=Sigma+jitter*I，E=S+vv^T，幅值 D^-1/2（D=1+v^T S^-1 v）。预处理缓存二维 m、H、log 幅值；像素仅二维核加性累积。反向先累积二维参数梯度，再完整回传 mu/S，包括幅值的 0.5*(S^-1-E^-1) 项，复用尺度/四元数链。
- 坐标：标量 PSF 使用模型空间单位法向；显式向量及 DICOM `psf_sigma_mm` 支持一般仿射，物理积分位移按 M 变换，与 M^-T 的几何法向区分。间距、层厚、sigma 不自动互推；零 PSF 共用 point 路径。
- 支撑：对解析积分后的响应裁剪，Q=(u-m)^T H(u-m)+q+logD；保留 C1 窗及其完整导数。有限支撑与无限域解析积分的误差分开验证。
- 接口：公开 Tensor 形状、两种协方差输入、三个返回值和一阶 autograd 保持；background_weight 仅作弃用 Python 兼容字段。native API_VERSION=3，需重新构建。
- 数值问题：当前 PyTorch 2.4.1/CUDA 环境的 50000×4 float64 行归约/广播曾产生错误范数，独立于扩展复现。四元数单位化改为数学等价的四个一维 double 分量运算；保留有限值/非零检查，不放宽阈值。增加 50000 个大幅值四元数测试。
- 未修改：GaussianModel、Clone/Split/Prune/Densification、Stage2 运动模型、训练入口与损失逻辑；未读患者像素、未运行训练、未安装依赖。

验证：

- 在现有 cardiac-3dgs 环境（PyTorch 2.4.1+cu121）用 CUDA 12.1、现有 gcc12-tools 构建新旧二进制成功。构建使用 `/tmp/c3dgs_cuda121` 临时头文件/库链接，产物位于 `/tmp/c3dgs_slice_v3_build` 与 `/tmp/c3dgs_slice_v2_build`，没有安装或覆盖环境。
- 新增 16 项测试全部通过（无跳过）：解析积分与高精度数值积分、3D/2D 输出及自动求导、幅值梯度、位置/尺度/旋转/intensity 有限差分、两种协方差路径、负强度/零点云/无候选、非等比/shear 物理转换、非默认 stream、几何快照、训练 wrapper 和两份配置接线。CUDA 梯度与 PyTorch 参考按 float32 容差比较；原子累加不承诺逐位确定。
- 随机 double 3D/2D 首次数值核对最大误差 1.11e-16。相关 Python 语法解析、git diff --check 通过。
- 额外执行现有 `test_regressions.py`：8 通过、5 因系统 Python 无 PyTorch 等依赖跳过、1 失败。失败为 HEAD 原有 GaussianModel_noop.training_setup 读取缺失配置 position_lr_delay_mult，已核对 HEAD 同样存在；未扩大范围修复。

性能（A100-SXM4-40GB，50000 点，128/256 正方图像，同 seed=42 的 packed covariance 输入；预热3次、测量10次，中位 GPU 计时，峰值为相对输入常驻内存的 allocated 增量）：

| 分辨率 | 模型 | 前向 ms | 反向 ms | 峰值增量 MiB |
|---|---|---:|---:|---:|
| 128 | v2 normalized point | 0.905 | 1.047 | 7.950 |
| 128 | v3 additive point | 1.037 | 1.261 | 7.987 |
| 128 | v3 gaussian，sigma=0.05 模型单位 | 1.184 | 1.516 | 8.506 |
| 256 | v2 normalized point | 0.885 | 1.538 | 10.260 |
| 256 | v3 additive point | 1.017 | 1.626 | 9.547 |
| 256 | v3 gaussian，sigma=0.05 模型单位 | 1.116 | 1.928 | 10.537 |

- point 有效候选均为5248个 Gaussian；厚度响应增加至10159个，不能把 gaussian 的额外开销完全归因于二维核。新版本当前端到端未加速，预处理/反向链和主机检查开销仍存在；没有宣称与原版等速或更快。
- 另做50000点尺度/旋转路径的 gaussian 短基准，输出与梯度均有限。未进行完整训练/收敛或真实病例性能评估；多切片批量、shared memory 和连续场导数查询留作可选扩展。
