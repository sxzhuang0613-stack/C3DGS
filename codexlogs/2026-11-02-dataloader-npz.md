# ACDC / M&Ms NPZ 读取接口统一

- 目标：保持 Dataset 主流程，适配两个新 NPZ 数据集，支持 2D+T / 3D+T，统一 ACDC 类别定义，测试排除无有效标签病例。
- `Dataloader.py`：新增 `dim` 和 `require_labels`；读取 `cmr/labels/info`，统一输出 `cmr/label/info`。图像按病例归一化后提取切片；图像为 float32，标签为 long。3D 轴序为 STHW/SPHW，2D 为 THW/PHW。
- M&Ms 使用 `[0,3,2,1]` 映射到 ACDC 类别；ED/ES 等真实标注相位通过 `info.time_indices` 关联图像，不生成未标注帧标签。原始 JSON 元信息作为字符串保留，兼容默认 DataLoader 拼接。
- 保留 train/val 的排序比例划分，过滤在划分后、展开切片前执行；新增 test 根据原始 Testing/testing 集合选择病例。评估入口始终要求有效标签。M&Ms 官方 unlabeled 或相位元数据无效的病例，即使标签数组非空也不进入测试。
- `GetInitialGaussian.py`、`Motion.py`、`evaluate.py`：同步元信息读取和固定 STHW 输出，避免二次转置。两个 YAML 更新数据目录及 mnms 名称；README 更新接口约定。未来数据集在集中适配处扩展，不引入额外框架。
- 验证：cardiac-3dgs 环境中 7 项读取器回归测试通过；两套真实 NPZ 的样本读取、归一化、2D/3D 对应关系、类别映射和默认 batch 拼接通过。扫描全部元信息：ACDC 150 例有效、官方测试 50 例（538 切片）；M&Ms 345 例中 320 例有效，排除 25 例官方 unlabeled / 相位无效病例，官方测试 136 例（1621 切片）。没有遍历加载全部病例像素。
- 静态验证：修改的 Python 文件语法解析通过，git diff --check 通过。系统 Python 的现有回归中 4 项通过，17 项因缺少依赖跳过；cardiac-3dgs 环境缺少 PyYAML，5 项配置测试跳过。未安装依赖，未运行训练、CUDA 数值或完整评估。
- 限制：默认 batch_size=1；2D 访问仍按病例加载并归一化，不新增缓存。train/val 比例划分与官方 test 是不同集合选择方式，不能据此假定互斥。保持原有增强行为。
