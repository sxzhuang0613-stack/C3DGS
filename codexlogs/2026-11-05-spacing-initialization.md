# spacing 坐标与 Stage1 高斯初始化

## 目标与实现

- `Dataloader.py` 统一读取 CMR `zooms[:3]` 的 XYZ spacing，不比较 affine；按实际 crop 尺寸修正 128×128 resize 后面内 spacing，层间距不变。新增 `source_spacing_xyz`、`spacing_xyz`，保留其余数据处理和原始 JSON。
- `case_geometry.py` 生成局部栈坐标、统一缩放 L 和病例 bounds；位置采用像素中心，切片中心沿 spacing 分布。初始化、Renderer、结构优化和测试共享同一几何。单位按数据集毫米约定解释，不恢复患者绝对位置。
- `Gaussians/initialization.py` 实现 50% 均匀、30% 图像前景、20% 边缘的分层采样，使用独立病例种子；80%/20% 面内尺度为 1.25/2.5 输出像素，point 法向 sigma 为 0.4 层间距，单位四元数起步，初始轴比最多 5。采样分位数图不修改训练目标、不新增标签监督。
- 使用正式 Renderer 标定共同强度，再做矩阵自由 CG 求解有符号系数；不构造全局响应矩阵、不新增正常 Adam 预热。求解包含相对 ridge、迭代上限和残差报告，不使用强度/输出硬截断。
- `GaussianModel_structural.py` 接入新初始化，并让配置中的 Adam epsilon/betas 生效，默认实际值保持不变。`GetInitialGaussian.py` 保持真实切片采样、损失组合、反向及更新顺序，仅接入病例几何、初始化与保存恢复。
- checkpoint 保存几何与固定参考容量 N0。新 PLY 旁置 `.geometry.json`，记录几何、渲染条件、参考容量、迭代号和 PLY SHA256。恢复核对几何与 Stage1 优化条件；允许延长训练及改变保存频率。最终几何记录缺失时，可从完整 checkpoint 重建产物。
- 无记录的旧 checkpoint/PLY 显式走 legacy_box，保留旧参数坐标，不自动转换。续训不重新初始化。
- `CostFunctions.py` 和结构验收统一采用 `sum(geometry_penalty)/N0`；N0 不随增删改变，修复零贡献 Birth 稀释平均正则的问题。
- `evaluate.py` 读取并核对几何记录。`Motion.py` 只适配新模型的几何读取、checkpoint/导出记录、稠密查询坐标及像素单位换算；未修改运动网络、ODE 或传播数学。
- 训练配置集中新增初始化参数，移除主入口失效的固定强度、Z 拉伸与 ROI 初始化参数；旧初始化 API 保留。测试配置不包含初始化参数。README 更新使用与兼容约定。

## 验证

使用现有 cardiac-3dgs 环境与当前已安装扩展；其 SHA256 与工作区 build 中扩展一致，开头为 `e83c55fd`。未安装或覆盖环境中的包。

```bash
PYTHONDONTWRITEBYTECODE=1 \
C3DGS_EXTENSION_DIR=/home/exouser/miniconda3/envs/cardiac-3dgs/lib/python3.10/site-packages \
PYTHONPATH=diff-gaussian-slice-rendering \
/home/exouser/miniconda3/envs/cardiac-3dgs/bin/python -m unittest discover -s tests -v
```

- 共 71 项，70 项通过，无跳过；本次新增 12 项全部通过。
- 唯一失败仍为未修改的 `GaussianModel_noop` Clone 容量测试：实际点数 8，测试期望 6。未修改旧模型或将此失败改为跳过。
- 新增覆盖非正方形 crop 的 spacing 与物理视野、像素中心和单层坐标、初始化采样与轴比、病例随机数隔离、零图像、线性算子伴随、PLY 几何/哈希、续训配置检查、零贡献 Birth 拒绝、稠密查询坐标与像素位移。
- CUDA 验证初始化、伴随、反向与 Adam 更新；合成主入口运行 2 步（含结构操作），再恢复 1 步，重建缺失最终几何记录，并加载评估。不是完整病例训练。
- CUDA 合成小场景 48 点、8 次系数求解，MSE 约 0.00894→0.00119；只验证数值路径，不代表病例精度提升。
- 9 个实现文件语法检查、配置字段覆盖及 `git diff --check` 通过；原 `GaussianModel_noop.py`、`GaussianEvolution.py` 和 CUDA 源码无修改。

## 限制与诊断记录

- 未运行真实病例训练、大规模性能/显存测试或完整 Stage2 训练；未读取真实病例像素。本次 CUDA 源码未变，没有重新编译。
- 初期使用历史 `/tmp/c3dgs_slice_v3_build` 扩展时，组合测试出现异步非法访问/错位地址；单测或同步诊断可能通过。换用与当前工作区 build 哈希一致的已安装扩展后，专项与完整检查未复现。未查明历史二进制的底层原因，未添加强制同步、禁用 cuDNN 或数值截断规避问题；历史二进制未修改。
- CG 达到预算不保证收敛，报告相对残差；完整病例的初始化耗时、内存及最终精度待实验确认。
- spacing 不代表采集层厚。默认 point，不从 spacing 猜测 PSF；已有 PSF 配置仍使用模型坐标单位。
- 新初始化实验应使用独立输出目录，或明确关闭旧实验的 resume/skip；新几何 PLY 必须与旁置记录一同复制。
- 工作区原有 build/egg-info 改动保留；未提交或推送。
