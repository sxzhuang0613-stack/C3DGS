# Stage1 统一高斯结构操作与主入口数据接口

## 目标与范围

新增独立结构模型，统一 Birth / Split / Prune / Merge / Swap；取消 Stage1 入口的直接 Clone。
主入口对接现有 dataloader，默认 ACDC、完整病例 `dim='3D'`、`cmr=[S,T,H,W]`。
不改变正常训练的采样、损失、反向传播、Adam 更新顺序和结构操作调度。

## 修改与决策

- `Gaussians/GaussianModel_structural.py`：继承旧模型的属性、PLY 和优化器接口；新增 `GaussianStructuralOptimizer`，共享候选筛选、局部响应、SVD/加权 Lasso 强度重拟合和统一目标验收。
- Birth 使用真实切片残差块峰，切片仿射定位，邻域几何初始化；Split 使用残差方向、对称位移与正定协方差缩减；Merge 对同号且观测响应相关的近邻做有符号质量矩匹配；异号抵消候选联合 Prune；Swap 联合评价删除和新增。
- 初始矩匹配包含 Renderer 的 covariance jitter；只保证初始体积积分矩，不声称切片响应完全守恒。最终按真实观测 L1/SSIM、原各向异性惩罚及容量/幅值正则验收。
- `diff-gaussian-slice-rendering/.../local_response.py` 与 `__init__.py`：新增稀疏局部二维响应，抽出同一四元数单位化函数；沿用现有 PSF、float32 几何输入舍入、jitter、cutoff 和 transition。未修改 CUDA。完整支撑或邻居支撑并集超预算时跳过候选，不截短响应。
- `CostFunctions.py`：仅抽出未归约 SSIM map，共享原数学表达式。局部评价读取完整窗口 halo，保持图像边界零填充与全图归一化。
- 接受候选后即时更新虚拟预测，周期末一次 gather/concat 提交；未变基元保留 Adam 状态，新行清零动量，旧基元只重拟合强度时只清零其强度动量，参数组 step 保留。正式全量渲染不一致、目标不降或渲染异常时回滚参数、状态与统计引用。
- `GetInitialGaussian.py`：只替换模型导入和原增殖调用。新增模型兼容入口已有初始化关键字及单层 ROI 坐标。
- `config_utils.py`、训练/测试配置：显式完整病例接口、ACDC 默认与数据集别名；移除失效的 `cmr_layout` 和旧增殖阈值，集中新增结构参数；补齐继承调度器所需参数，不改变默认学习率曲线。README 同步说明。
- 原 `Dataloader.py`、`GaussianModel_noop.py`、`GaussianEvolution.py` 的 SHA256 与任务开始时一致；Stage2 文件及 Renderer.py 未修改。

## 实际验证

使用现有 `cardiac-3dgs` 环境、工作区 Python 渲染包和已有 API_VERSION=3 二进制：

```bash
PYTHONDONTWRITEBYTECODE=1 C3DGS_EXTENSION_DIR=/tmp/c3dgs_slice_v3_build \
PYTHONPATH=diff-gaussian-slice-rendering \
/home/exouser/miniconda3/envs/cardiac-3dgs/bin/python -m unittest discover -s tests -v
```

- 共运行 59 项测试，58 项通过；本次新增 21 项全部通过，无跳过。
- 新增覆盖实际 dataloader 合成 NPZ 拼接、主入口参考相位提取、Stage2 序列接口；局部 L1/SSIM 边界增量；五类操作、容量限制、正负强度、质量矩匹配；秩亏 SVD、加权 Lasso；Adam 状态迁移、校验失败及异常回滚、CPU/GPU 随机数流保护。
- GPU 实测覆盖任意斜切片、旋转各向异性、point/Gaussian PSF 的局部响应与正式 CUDA 一致性；带正则及邻居重拟合的结构周期；提交后前向、反向和 Adam 更新；现有 CUDA 梯度/厚度渲染测试全部通过。
- 唯一失败为旧 `test_regressions.TensorTests.test_clone_and_split_obey_capacity`：未修改的 `GaussianModel_noop` Clone 后点数 8，测试期望上限 6。本次保留旧模型，未扩大范围修复其 Clone。
- 受保护文件哈希校验、修改文件语法检查及 `git diff --check` 通过。

## 限制

未运行完整病例训练、大规模性能/显存测试，未安装依赖、提交或推送。CUDA 算子未变，本次复用已有二进制，没有重新编译。
环境中的已安装 Python 包尚未同步新增 `local_response.py`；正式运行前应按 README 同步当前本地渲染包。
预算限制会跳过宽支撑候选，候选筛选与小规模局部拟合不保证全局最优。
默认结构窗口与插层阶段不重叠；配置重叠时使用固定节点近似随机插层期望，验收只对该固定观测库成立。
有限支撑与正式 Renderer 一致，仍区别于无限域 Gaussian 精确积分；极小收益受 float32 CUDA 累加误差影响，最终一致性与目标下降校验可回滚，长期训练表现仍待实际病例实验。
