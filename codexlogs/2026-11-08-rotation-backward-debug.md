# Stage1 四元数反向与工作区渲染库排查

## 问题与证据

- 用户在数百次迭代后于 `loss.backward()` 遇到 `native API expects normalized wxyz rotations`。
- 排查工作区 `diff-gaussian-slice-rendering` 的 Python 包装、原生前后向参数检查、缓存布局、tile 分桶与协方差梯度。尚未捕获用户原始抛错现场，不能断言所有 CUDA 异常来自同一原因。
- 当前环境为 `cardiac-3dgs`，PyTorch 2.4.1+cu121，A100，驱动 595.99.02。未安装或升级依赖。
- 独立进程仅导入 PyTorch：CPU 生成固定随机 `[50000,4]` double 数据并复制到 CUDA；5 次逐行 norm 各有 7–11 行与 CPU 不符，最大误差约 0.9354；四个一维分量显式平方求和与 CPU 一致。另一次 float32 `F.normalize` 检查有 2 行真实范数平方超限，最大偏差约 0.3784。
- 纯 PyTorch 的 `F.normalize + Adam + backward` 也曾出现非法访问，重跑可通过。该证据说明仍存在不依赖本渲染库的 CUDA 异常；不能据此确定是 PyTorch、驱动还是硬件原因。

## 修改与边界

- `Renderer.py` 优先导入工作区源码，拒绝进程中已先加载的其他位置同名包；传原始四元数给库入口，保留唯一一次可微单位化。
- 库 `__init__.py` 的单位化使用四个一维 double 分量，避免二维行归约及广播反向路径；支持工作区构建目录，检查新二进制标记。
- `GaussianModel_structural.py` 的属性与无效点检查复用该路径；`local_response.py` 和 `initialization.py` 与正式渲染的单次激活保持一致。
- `rasterize_points.cu` 的前后向原生范数检查改为逐分量 double 求和，保留 `1e-4` 阈值；错误报告增加最大范数平方偏差。未放行非法四元数，未在 CUDA 内重复归一化。
- `ext.cpp` 增加 `ROTATION_CHECK_VERSION=1`，防止 Python 修订与旧扩展混用；README 补充工作区构建与重启说明。
- 未修改训练入口、损失、训练超参数、Dataloader、结构操作策略或 Stage2 运动数学；未添加强制同步、截断梯度、禁用 cuDNN 等运行补丁。

## 实际验证

- 使用已有 CUDA 12.1 工具链和 gcc12 编译器执行 `setup.py build_ext --inplace` 成功；确认 Python 与 CUDA 扩展均来自工作区。
- 新增 `tests/test_rotation_backward.py` 3 项全部通过：5 万点激活与 CPU 独立前向/非零梯度一致；双视图 600 次 Adam 更新；原生前后向均拒绝非单位输入。
- 额外合成检查：5 万点、10 张 128×128 切片、20 步初始化系数拟合、每步两视图，L1/SSIM/各向异性正则与 Adam 连续 600 步通过；未执行结构操作或真实病例训练。另一次启动在初始化、尚未调用 rasterizer 时出现非法访问，说明运行稳定性尚未确认。
- 排除主入口组合测试后运行 73 项：72 项通过，仅历史 `test_clone_and_split_obey_capacity` 失败（8 != 6），未修改旧 Clone/Split。
- 全量 74 项未通过：主入口“训练→结构操作→恢复→评估”组合仍可发生异步非法访问，导致后续 CUDA 测试连带失败。专项通过不等于完整训练稳定。
- 修订过程中用 NVIDIA Compute Sanitizer memcheck 检查选定的库 kernel（预处理、前后向、协方差梯度、分桶及 CUB）：初始化伴随/反向和主入口组合两项通过，约 415 秒，`ERROR SUMMARY: 0 errors`。该运行开始于主渲染入口最后一次消除重复单位化之前；检查器可能改变执行时序，且未覆盖所有 PyTorch/cuDNN kernel，不能据此排除普通运行的问题。工具仅解压到 `/tmp`，没有安装到环境。
- 7 个相关 Python 文件语法检查及 `git diff --check` 通过；运行时确认源码包路径、原生扩展路径均为工作区，版本标记为 `3 / 1`。

## 限制

尚未复现原始病例的同一报错，也未完成真实病例数百迭代后的稳定性验证。已确认并修订四元数链路，但组合 CUDA 非法访问仍需定位，不应把本记录作为完整训练已修复的结论。
