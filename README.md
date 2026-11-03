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

## 配置和运行

`config_train.yaml` 包含两个阶段的训练参数和显式预处理参数；`config_test.yaml` 只包含 Stage1 测试所需参数。旧 `config.yaml` 已移除。

首先修改配置中的数据路径、输出路径和 `device`。所有相对外部路径以配置文件所在目录为基准，支持环境变量和 `~`。`device: cuda:0` 表示当前可见设备中的第 0 张卡，代码不会覆盖外部 `CUDA_VISIBLE_DEVICES`。

```bash
python GetInitialGaussian.py --config config_train.yaml
python Motion.py --config config_train.yaml
python evaluate.py --config config_test.yaml
```

保留 Stage2 的 CLI 覆盖，例如 `--control-deformation-mode mls`、`--slice-training-mode random_slice`。

训练与测试默认都读取 `dataset_mode: val`，保持原病例划分；这是逐病例重建评估，不代表对未见病例的泛化。没有对应 PLY 的病例会明确报错，不会在测试中重新训练。`all/train/val` 继续按排序后的全部病例及 `train_fraction` 选择集合；新增的 `test` 根据 NPZ 的 `original_split` 选择官方测试病例。这两种划分方式不同，不保证 `train/val` 与官方 `test` 互斥。评估入口始终排除无有效标签的病例，`test` 模式也执行相同过滤。

测试的 `stage1_ply_path` 使用 `{case}` 替换 NPZ 文件主名。输出逐切片 L1、PSNR、SSIM 到 `metrics.csv`，可选保存预测数组和固定显示范围的 PNG。默认指标使用未截断预测；`clamp_predictions` 可显式切换。测试配置的归一化、坐标边界和参考相位应与生成 PLY 时一致。

## 数据契约

当前读取器仅适配 ACDC（`datasets='acdc'`）和 M&Ms（`datasets='mnms'`，也接受 `mm/mms/m&ms`）。配置中的 `dataset_name` 使用 `acdc` 或 `mnms`，`dataset_paths` 已指向 `/home/exouser/Datasets` 下的两个 NPZ 目录，也可在构造读取器时通过 `data_path` 覆盖路径。

NPZ 包含 `cmr: [T,S,H,W]`、`labels: [P,S,H,W]` 和 JSON 字符串 `info`。`dataset(..., dim='3D')` 默认返回完整病例：`cmr: [S,T,128,128]`、`label: [S,P,128,128]`；`dim='2D'` 遍历各病例的所有切片，返回 `cmr: [T,128,128]`、`label: [P,128,128]`。先按病例划分集合，再展开切片。图像按完整病例执行 min–max 归一化，输出为 `[0,1]` 的 float32；标签为 long。仅将 H/W 缩放为 128×128，图像使用双线性插值，标签使用最近邻插值，空标签保留空相位轴。保留全部切片和时间帧，不对 S/T 插值或补齐，因此默认使用 `batch_size=1`。`info.source_spatial_shape` 和 `info.spatial_shape` 记录缩放前后尺寸；原始 JSON 的 affine、spacing 仍描述缩放前的数据。

对外统一类别为 ACDC 定义：`0=背景、1=右室腔、2=左室心肌、3=左室腔`；M&Ms 原始编号 1、3 在读取时交换。`P` 仅包含实际标注相位，ED/ES 对应时间帧位于 `info['time_indices']`，无标签时 `P=0`，不会补成背景。M&Ms 官方标记为 unlabeled 或相位元数据无效的病例也会被测试过滤，即使文件中存在标签数组；全背景标签不视为无标签。其他模式可显式指定 `require_labels=True` 使用同一过滤规则。

输出固定为 `cmr、label、info`。`info` 为字典，包含病例标识、文件名、数据集、原始集合、切片索引（3D 时为 -1）、轴序、标注相位、有效性以及统一类别定义和映射；原始元信息保存在 `info['metadata']` JSON 字符串中，避免其中的 null 和异构字段影响默认 DataLoader 拼接。原始类别定义保留在该 JSON 中。新增数据集时，在读取器的适配位置扩展类别映射、字段和轴序处理，共用后续索引及输出逻辑。

读取器输出轴序固定，训练和评估入口不再依据旧的 `cmr_layout` 配置重复转置。`reference_phase` 默认为 0；Stage2 当前要求参考相位为 0。

`roi_mask` 可选，三维 mask 默认为 `[S,H,W]`，也可用 `roi_layout` 指定其他轴序；`S/D/Z` 为层，`H/Y` 为行，`W/X` 为列。`[M,3]` 点集直接表示模型空间 XYZ。单层 ROI 的层坐标位于模型中心。

`bounds` 为模型空间 XYZ 半边长，初始化和渲染使用同一配置。当前渲染为有符号加性 Gaussian 场，支持 point 和解析 Gaussian PSF 厚度响应；Preprocess 转为等效二维核。背景只加一次，不包含 opacity 合成或灰度归一化。`response_mode` 默认为 point；gaussian 的 `psf_sigma` 使用模型坐标单位，或显式提供 `psf_std_vector`。不能把切片间距当成 PSF sigma；毫米及非等比坐标转换见渲染库 README。

## 保存和恢复

Stage1 每例保存：

- `training_state.pth`：参数、Adam 动量、迭代次数、增殖统计、随机状态和损失历史。
- `point_cloud_iter_*.ply` / `point_cloud_final.ply`：用于模型交换的参考高斯。
- `reconstructions/iter_*/`：各次重建快照，避免相互覆盖。

默认从完整训练状态恢复；只在迭代已完成且所需最终产物存在时跳过病例。`warm_start_ply_path` 仅用于模型参数热启动，不表示完整续训。恢复的随机状态要求使用相同数据、设备布局和实验参数；CUDA 原子累加仍可能产生非确定性。

新 PLY 字段为 `x,y,z,intensity,scale_0..2,rot_0..3`；尺度保存为 log 标准差，旋转保存原始 wxyz 四元数。旧灰度 `color_*` 可按 `legacy_color_policy` 迁移，旧 opacity 不再参与模型。两个 stage 均使用此模型格式。

Stage1 默认真实切片无放回均匀采样，支持单层/双层病例。原损失权重不变，插层起点保持原实际行为的 12500 步。`max_num_pts` 默认 200000，是新增的增殖容量限制；不将低强度组织当成低贡献点剪除。

Stage2 的网络、ODE 和控制点传播数学实现保持原样；仅适配数据、配置、PLY 和共享渲染接口。checkpoint 仍按传播方式和切片训练模式分别保存。

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
