# C3DGS 心脏 MRI 高斯重建

项目按病例独立优化。Stage1 拟合参考相位的无 opacity、单通道强度三维高斯；Stage2 读取参考高斯，拟合完整序列的演化网络。测试入口仅评估 Stage1，不更新模型。

## 环境

需要 CUDA 版 PyTorch、兼容的 CUDA Toolkit/nvcc、NumPy、PyYAML、plyfile、torchvision、matplotlib、tqdm 和 simple-knn。`temp.py` 额外需要 nibabel；独立使用 `tools.py` 的可视化/指标功能需要 OpenCV、python-pptx、scikit-learn、Pillow。本仓库不自动安装或修改环境。

先编译当前渲染扩展（不依赖旧 third_party/glm）：

```bash
cd diff-gaussian-slice-rendering
python -m pip install --no-build-isolation .
```

需要当前源码的 API_VERSION=3 二进制；详细安装说明见该目录 README。
本次结构优化还需要渲染库新增的 Python 文件 `local_response.py`，因此已有 API_VERSION=3
安装也需要同步 Python 包源码；CUDA 算子本次没有改动。本次开发未安装或覆盖环境中的包。

## 配置和运行

`config_train.yaml` 包含两个阶段的训练参数和显式预处理参数；`config_test.yaml` 只包含 Stage1 测试所需参数。旧 `config.yaml` 已移除。

首先修改配置中的数据路径、输出路径和 `device`。所有相对外部路径以配置文件所在目录为基准，支持环境变量和 `~`。`device: cuda:0` 表示当前可见设备中的第 0 张卡，代码不会覆盖外部 `CUDA_VISIBLE_DEVICES`。

```bash
python GetInitialGaussian.py --config config_train.yaml
python Motion.py --config config_train.yaml
python evaluate.py --config config_test.yaml
```

保留 Stage2 的 CLI 覆盖，例如 `--control-deformation-mode mls`、`--slice-training-mode random_slice`。

默认数据集为 ACDC，训练读取 `dataset_mode: all`，测试读取 `val`；这是逐病例重建评估，不代表对未见病例的泛化。主入口显式使用读取器的 `dim='3D'` 完整病例接口。没有对应 PLY 的病例会明确报错，不会在测试中重新训练。`all/train/val` 继续按排序后的全部病例及 `train_fraction` 选择集合；新增的 `test` 根据 NPZ 的 `original_split` 选择官方测试病例。这两种划分方式不同，不保证 `train/val` 与官方 `test` 互斥。评估入口始终排除无有效标签的病例，`test` 模式也执行相同过滤。

测试的 `stage1_ply_path` 使用 `{case}` 替换 NPZ 文件主名。输出逐切片 L1、PSNR、SSIM 到 `metrics.csv`，可选保存预测数组和固定显示范围的 PNG。默认指标使用未截断预测；`clamp_predictions` 可显式切换。测试配置的归一化、坐标边界和参考相位应与生成 PLY 时一致。

## 数据契约

当前读取器仅适配 ACDC（`datasets='acdc'`）和 M&Ms（`datasets='mnms'`，也接受 `mm/mms/m&ms`）。配置中的 `dataset_name` 使用 `acdc` 或 `mnms`，`dataset_paths` 已指向 `/home/exouser/Datasets` 下的两个 NPZ 目录，也可在构造读取器时通过 `data_path` 覆盖路径。

NPZ 包含 `cmr: [T,S,H,W]`、`labels: [P,S,H,W]` 和 JSON 字符串 `info`。`dataset(..., dim='3D')` 默认返回完整病例：`cmr: [S,T,128,128]`、`label: [S,P,128,128]`；`dim='2D'` 遍历各病例的所有切片，返回 `cmr: [T,128,128]`、`label: [P,128,128]`。先按病例划分集合，再展开切片。图像按完整病例执行 min–max 归一化，输出为 `[0,1]` 的 float32；标签为 long。仅将 H/W 缩放为 128×128，图像使用双线性插值，标签使用最近邻插值，空标签保留空相位轴。保留全部切片和时间帧，不对 S/T 插值或补齐，因此默认使用 `batch_size=1`。`info.source_spatial_shape` 和 `info.spatial_shape` 记录缩放前后尺寸；原始 JSON 的 affine、spacing 仍描述缩放前的数据。

对外统一类别为 ACDC 定义：`0=背景、1=右室腔、2=左室心肌、3=左室腔`；M&Ms 原始编号 1、3 在读取时交换。`P` 仅包含实际标注相位，ED/ES 对应时间帧位于 `info['time_indices']`，无标签时 `P=0`，不会补成背景。M&Ms 官方标记为 unlabeled 或相位元数据无效的病例也会被测试过滤，即使文件中存在标签数组；全背景标签不视为无标签。其他模式可显式指定 `require_labels=True` 使用同一过滤规则。

输出固定为 `cmr、label、info`。`info` 为字典，包含病例标识、文件名、数据集、原始集合、切片索引（3D 时为 -1）、轴序、标注相位、有效性以及统一类别定义和映射；原始元信息保存在 `info['metadata']` JSON 字符串中，避免其中的 null 和异构字段影响默认 DataLoader 拼接。原始类别定义保留在该 JSON 中。新增数据集时，在读取器的适配位置扩展类别映射、字段和轴序处理，共用后续索引及输出逻辑。

读取器输出轴序固定，训练和评估入口不再依据旧的 `cmr_layout` 配置重复转置。`reference_phase` 默认为 0；Stage2 当前要求参考相位为 0，不自动改选 ED。

几何统一使用 `source.files.cmr_4d.zooms[:3]` 的 XYZ spacing，不比较或使用 affine。
crop 不改变 spacing；从实际 NPZ crop 尺寸 Hc×Wc 缩放到 128×128 后，
`spacing_xyz=[sx*Wc/128,sy*Hc/128,sz]`，`source_spacing_xyz` 保留缩放前值。
两个字段均为 CPU float64 Tensor，2D/3D 接口相同；原 JSON 保留。
spacing 按本数据集的毫米约定解释，未知单位仍保留在原始元信息中，不声称恢复患者绝对位置。

旧 `InitializeGaussians` API 保留 ROI 兼容入口；当前 Stage1 主入口使用参考相位图像生成采样权重，不额外读取分割标签。

新模型在 spacing 定义的局部栈坐标中居中，三个轴共用 `L=max(W*sx, H*sy, max(S-1,1)*sz)/2` 缩放，病例 bounds 自动计算并同时传给初始化、Renderer 和结构优化。
面内使用像素中心 `(u+.5-W/2)*sx`，法向使用 `(k-(S-1)/2)*sz`；单层观测位于零平面。
配置中的 `bounds` 仅用于无几何记录的旧模型兼容。
当前渲染为有符号加性 Gaussian 场，支持 point 和解析 Gaussian PSF 厚度响应；Preprocess 转为等效二维核。背景只加一次，不包含 opacity 合成或灰度归一化。
`response_mode` 默认为 point；当前数据未提供逐病例 PSF，不从 spacing 推断层厚。
gaussian 的 `psf_sigma` 使用模型坐标单位，或显式提供 `psf_std_vector`；毫米转换见渲染库 README。

新初始化保持 50000 点，50% 分层均匀、30% 图像前景、20% 图像边缘；采样分位数图不改变训练目标。
80% 高斯面内 sigma 为 1.25 个输出像素，20% 为 2.5 个像素；point 法向 sigma 为 0.4 倍层间距，旋转从单位四元数开始。
初始轴比限制为 5，扩大短轴以保留覆盖。存在已知 PSF 时，从有效法向方差扣除 PSF 和固定 jitter，并保留最小内禀尺度。
先用正式 Renderer 标定共同强度，再用最多 20 步矩阵自由 CG 拟合有符号系数，不构造全局响应矩阵，不增加正常 Adam 预热阶段。
CG 预算耗尽不等于收敛，初始化报告包含相对残差和拟合前后 MSE。病例级采样使用独立稳定种子，不消耗训练的全局随机数流。

## 保存和恢复

Stage1 每例保存：

- `training_state.pth`：参数、Adam 动量、迭代次数、增殖统计、随机状态和损失历史。
- `point_cloud_iter_*.ply` / `point_cloud_final.ply`：用于模型交换的参考高斯。
- 每个新 PLY 的 `.geometry.json`：spacing、病例 bounds、缩放因子、参考相位、归一化、渲染条件、固定几何正则参考容量、迭代号及 PLY 哈希；必须随 PLY 一起复制。
- `reconstructions/iter_*/`：各次重建快照，避免相互覆盖。

默认从完整训练状态恢复；只在迭代已完成且所需最终产物存在时跳过病例。`warm_start_ply_path` 仅用于模型参数热启动，不表示完整续训。恢复的随机状态要求使用相同数据、设备布局和实验参数；CUDA 原子累加仍可能产生非确定性。
新 checkpoint 保存相同几何记录；恢复、测试和 Stage2 读取时核对坐标/观测条件。
缺少几何记录的旧模型显式使用 legacy_box，不转换其参数坐标。新最终 PLY 记录缺失时，可通过完整 checkpoint 重建产物。
续训允许延长迭代数和改变保存频率，拒绝混用已改变的 Stage1 优化条件；初始化不会在续训时重新执行。
评估新初始化应使用独立输出目录，或明确关闭原实验的恢复/跳过选项。

新 PLY 字段为 `x,y,z,intensity,scale_0..2,rot_0..3`；尺度保存为 log 标准差，旋转保存原始 wxyz 四元数。旧灰度 `color_*` 可按 `legacy_color_policy` 迁移，旧 opacity 不再参与模型。两个 stage 均使用此模型格式。

Stage1 默认真实切片无放回均匀采样，支持单层/双层病例。原损失权重不变，插层起点保持原实际行为的 12500 步。`max_num_pts` 默认 200000，是结构容量上限；不将低强度组织当成低贡献点剪除。

Stage1 使用独立的 `Gaussians/GaussianModel_structural.py`，继承旧模型的属性、PLY、Adam 和学习率接口，原 `GaussianModel_noop.py` 保留。
结构周期沿用 `densify_from_iter / densify_until_iter / densification_interval`，只替换原增殖调用；正常训练采样、损失、反向传播和 Adam 更新次序不变。
统一结构目标为全病例真实切片 L1/SSIM、原各向异性惩罚，加 `structural_count_weight*N` 和按观测响应范数归一化的强度 L1 惩罚；新增两项只用于结构验收。
正常训练和结构验收的几何惩罚均为各点惩罚之和除以固定参考容量 N0；N0 保存并恢复，不随增删变化，防止零贡献 Birth 稀释平均正则。

- Birth 从真实观测残差峰生成候选，在对应切片平面定位，继承邻域尺度和方向。
- Split 沿残差结构方向提出对称子基元，保持初始有符号质量及二阶矩；Merge 对同号且响应高度相关的近邻做矩匹配，异号抵消交给联合 Prune。
- Prune 评价删除及少量邻居重拟合的实际收益；Swap 联合删除和新增，在容量上限处重新分配基元。

五种操作共用局部响应、SVD/坐标下降强度重拟合及完整目标验收，不执行直接 Clone。
局部响应匹配正式渲染的四元数激活、PSF、jitter、cutoff 和 transition；SSIM 使用原窗口的完整影响域、边界零填充和全图归一化。
候选按已接受操作更新后的预测依次评价，最后统一提交参数、统计和 Adam 状态；正式 Renderer 再渲染一次校验，失败则恢复原参数与状态引用。
未变基元保留动量，新基元动量清零，仅重拟合强度的旧基元重置强度动量。新增行沿用参数组的 Adam step。

`structural_*` 参数集中在训练配置中，控制候选数、局部邻居、响应缓存和支撑预算。超过预算会跳过整个候选，避免截断局部响应造成错误收益。
默认结构窗口与插层阶段不重叠；若手动配置为重叠，固定 `structural_interp_nodes` 近似随机插层目标的期望，不改变训练随机数流。
结构验收保证该观测集合上的目标下降；不保证后续随机训练损失单调，也不等于全局最优。有限支撑与原 Renderer 一致，仍是完整 Gaussian 的数值近似。

Stage2 的网络、ODE 和控制点传播数学实现保持原样；仅适配数据、配置、PLY 和共享渲染接口。checkpoint 仍按传播方式和切片训练模式分别保存。
新模型的 Stage2 渲染及稠密查询网格使用保存的病例 bounds；像素位移换算同步更新，旧模型保留旧导出约定。
Stage2 checkpoint/运动结果保存几何记录，避免将旧坐标结果误用为新坐标；不匹配时使用独立实验目录。

## 可选预处理

```bash
python temp.py --config config_train.yaml
```

只有显式运行才访问数据。`preprocess_*` 配置控制输入/输出、时间轴、目标帧数和归一化；默认保持原 `slice_uint8` 协议，设为 `none` 可保留原浮点强度。NPZ 同时保存源 affine、spacing、原始帧数等元数据。默认拒绝覆盖已有病例文件。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试区分不依赖 GPU 的接口/配置回归和需要 PyTorch/CUDA 扩展的数值测试；依赖不具备时后者跳过。语法和接口检查不能替代 CUDA 编译、GPU 数值验证或完整训练。
