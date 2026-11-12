# Stage1 接入归一化切片与批量高斯操作适配

## 范围与实现

- 仅适配 `GetInitialGaussian.py` 调用链；未修改任何渲染库、`Motion.py`、`evaluate.py` 或 `config_test.yaml`，未安装依赖或扩展。
- `Renderer.py` 正常导入已安装的 `diff_gaussian_rasterization_slice`；灰度适配为三通道，仍返回原单通道字典。位置采用 `A=diag(1/bounds)`，完整协方差采用 `AΣAᵀ`，统一传入预计算协方差；参数、PLY 与结构阈值仍使用原病例模型坐标。支持连续索引、三个轴及单层零平面。
- `Gaussians/utils.py` 提供与原 wxyz 约定一致的可微单位化、旋转与批量协方差构造；结构模型移除对旧渲染包的导入。
- `structural_step()` 保留梯度阈值、面内尺度、增长预算和一次性状态提交。Clone 仅扰动新点，父点及其 Adam 动量保留；Clone/Split 均继承局部强度，删除减半与积分补偿。仅清理无效点，禁止低强度剪枝。
- 初始化保留采样、超参数和 CG；去掉旧 PSF/jitter 的法向尺度扣除；单位响应检查像素覆盖，单位伴随检查高斯实际贡献；背景仅在无覆盖区域扣除。
- 初始化的固定协方差只构造一次；每个训练步的真实切片和插层共享一次可微协方差，优化或结构更新后重建；全病例快照也复用一次构造。不加入额外训练渲染、逐点验收或残差统计。
- 配置使用 `render_model=normalized_slice_v1`、`structural_algorithm=batch_normalized_v1`；删除不支持的旧渲染项，默认 Stage1 输出改为独立 `acdc_experiment_normalized`，其余训练策略、学习率和损失权重保留。
- 几何记录新增渲染模型和 `intensity_semantics=local_value`。完整恢复与 PLY 热启动要求语义一致，不自动把旧加性幅度解释为局部强度。
- 更新 Stage1 测试、README 和阈值文档；测试专用临时扩展加载在 `tests/stage1_test_support.py`，生产代码没有源码路径回退。

## 实际验证

- 环境：`/home/exouser/miniconda3/envs/cardiac-3dgs/bin/python`，A100。该环境当前未安装新包；GPU 检查显式加载上一任务保留的 `/tmp/slice-normalized-build-wkqa_33r/_C_G3D.so` 与配套 Python 包装，未重建或改写库。
- 最终定向集合 36 项：33 项 CPU 检查及 2 项直接 GPU 检查通过，异步主入口插层集成 1 项出现 CUDA 非法访问。集合覆盖坐标/梯度、共享协方差梯度、灰度与背景、CG 伴随、Clone/Split、暗点保护、容量与 Adam、协议拒绝和配置。日志 `/tmp/stage1-final-tests.log`。
- 短 GPU 渲染→更新→分裂→更新通过；初始化颜色伴随与几何 backward 通过。
- 主入口合成 NPZ 的初始化、两步真实/插层训练、结构更新、续训和最终 PLY 记录修复，在 `CUDA_LAUNCH_BLOCKING=1` 下通过；另一个只在诊断进程设置 `torch.backends.cudnn.enabled=False` 的运行也通过。项目默认后端未改变。日志 `/tmp/stage1-interpolation-sync.log`、`/tmp/stage1-main-no-cudnn.log`。
- 对默认路径运行 memcheck：初始化算子用例通过，主入口用例失败，共报告 31 个错误，不能宣称 memcheck 通过。日志 `/tmp/stage1-normalized-memcheck.log`。
- 独立进程未导入或调用任何高斯扩展，仅运行原有 PyTorch `_sample_positions()` 链，也在 PyTorch TensorIterator 加法核报告非法访问，memcheck 共 3 个错误。日志 `/tmp/stage1-pytorch-sampling-memcheck.log`，脚本 `/tmp/probe_stage1_sampling.py`。这证明异常可在渲染器外复现，但未确定全部异常的根因。
- AST 静态检查与 `git diff --check` 通过。修改前快照 `/tmp/stage1-normalized-before-yxcuen09`；对两个库目录共 1604 个文件的 SHA256 核对无差异，三个范围外入口/配置逐字节不变。

## 速度与限制

- 同设备、4096 点批量结构微基准，旧实现约 4.37 ms，新实现约 3.77 ms；使用中位数，只说明该规模未观察到结构管理变慢。
- 同一新库、4096 点、64×64 的前向微基准：已准备输入的原生调用约 0.24 ms，完整包装约 1.46 ms，传入共享模型协方差的包装约 0.55 ms。共享构造仍需每个优化步执行一次；这些数字不能作为完整训练速度或成像模型优劣结论。
- 50000 点旧结构基线测速在原有 PyTorch 路径发生 CUDA 非法访问，未完成该规模对比。测速脚本 `/tmp/benchmark_stage1_normalized.py`，日志 `/tmp/stage1-normalized-benchmark-cached.log`。
- 默认 GPU 端到端路径仍不稳定；没有关闭 cuDNN、修改环境、调整训练损失或改库来掩盖问题。没有真实病例、完整训练或默认规模稳定性证明。
- 单高斯主导/同色区域的位置梯度可能很弱；结构操作不保证归一化图像不变。Stage2 与独立评估入口本轮未适配或验证；历史全仓库测试不作为本次通过项。
