# C3DGS 无 opacity 仿射切片渲染库（详细中文注释版）

本版 v2.0.1 保留中文注释和渲染公式，修复 CUB 宏冲突并统一新包名。建议先阅读 `CODE_GUIDE_zh.md`，其中解释代码精简、GLM 替代与性能边界。

本交付基于 `sxzhuang0613-stack/C3DGS` 的 `master` 快照
`94a0efe1c4a1b214be638f011a02bfc04e562e9a`，按两份 v1.1 文档及数值改进实现。
这是完整的扩展源码目录，可独立安装；不是已编译二进制，也不是完整上层训练项目。

**状态：已完成源码修改和静态整理。未进行 CUDA 编译、GPU 运行、数值梯度对照或训练验证。**
示例用于说明新调用方式，未运行。没有修改远端 GitHub 仓库。

## 1. 渲染模型

每个像素的模型空间坐标为 `p = slice_to_world @ [column,row,0,1]`。
第三列仅保存几何单位法向，不表示层厚，也不参与本版本采样。

令 `S = covariance + covariance_jitter * I`、`Q = (p-mean)^T S^-1 (p-mean)`。
基本模型为：

```text
w_i = exp(-Q_i / 2)
image = (sum_i w_i * intensity_i + tau * background) / (sum_i w_i + tau)
weight_sum = sum_i w_i
```

`tau = background_weight > 0`，强度是自由标量，不限制到 [0,1]；因此可表达负 HU。
输入强度单位与背景必须一致；库不做 DICOM rescale、窗宽窗位或 MR 强度标准化。
删除逐 Gaussian opacity、alpha/透射率、遮挡提前退出、SH、投影相机参数和外部 means2D。
归一化核混合仍受 Gaussian 密度及 tau 影响；复制／分裂 Gaussian 不保证场值不变。

### 平滑截断改进

`Qmax = -2 log(kernel_cutoff)`。`Q >= Qmax` 时权重为零。
`cutoff_transition = 0` 精确使用 v1.1 的硬截断模型（边界本身不可导）。
默认 `cutoff_transition = 2.0`，在 `[Qmax-transition,Qmax]` 内令
`t=(Q-(Qmax-transition))/transition`，并使用 `h=(1-t)^2(1+2t)`、`w=exp(-Q/2)*h`。
区间之外 `h=1` 或 `0`。这消除外边界的一阶跳变，但确实改变了阈值附近的核形状。

反向使用完整系数 `beta = g*(intensity-image)/denom * exp(-Q/2)*(h-2 dh/dQ)`，
不是把 taper 当作常量。条件核包围范围仍按未加窗 Gaussian 的 Qmax 保守计算。
平滑模式下不再对加窗后的 w 二次执行 `w > kernel_cutoff`，否则会重新引入跳变。
若需要严格对照文档原公式，显式设置 `cutoff_transition=0`。

### 数值处理

- 不使用 `det+epsilon` 逆矩阵；使用双精度 Cholesky 求有效协方差逆。
- 预处理、前向、反向共享同一有效协方差与逆矩阵缓存。
- 平面距离使用 `(n·(mean-origin))^2/(n^T S n)`，避免大数相减得到 q。
- 包围半径使用条件二维协方差最大特征值，保留 1 像素余量。
- 保留 inclusive scan → Gaussian/tile 复制 → radix sort → tile ranges；排序键为 tile 和 Gaussian ID，无深度遮挡语义。
- 图像分母、预测及核累计用 double，公开参数、图像和梯度为 float32。原子梯度累计仍存在浮点非确定性。
- 默认 `covariance_jitter=1e-8` 是模型坐标单位的平方；改变坐标单位应相应选择该值。
  此各向同性稳定项不保证一般仿射变换下的严格不变性。若上游已稳定且正定，可设置为 0。
- 有效协方差非正定或切片数值退化会报错，不静默丢弃坏 Gaussian。
- 支持 PyTorch 当前 CUDA stream 和 device guard；前向为获得分桶总数存在主机同步。
- 当前实现优先保证计算路径清晰一致，未做性能调优；不能假定速度等同原扩展。

## 2. 安装

依赖 Python >=3.9、带 CUDA 支持的 PyTorch、与其兼容的 CUDA Toolkit/nvcc、C++ 编译器和 setuptools。
原始 GLM/第三方目录已不再被本扩展引用，无需拷贝。使用 CUDA Toolkit 提供的 CUB scan/radix-sort 头文件。
原始许可证保留在 `LICENSE.md`，使用范围仍受其约束。

建议在解压后的干净源码目录重新安装，避免旧 `.pyd/.so` 被导入：

```bash
cd diff-gaussian-slice-rendering
python -m pip install --no-build-isolation .
```

架构由 PyTorch 和 `TORCH_CUDA_ARCH_LIST` 决定，不再硬编码 sm_86。
如覆盖旧目录，应先清理该目录内的旧扩展二进制及 `build/`，然后重新安装。
Python 会检查 native `API_VERSION == 2`，检测到旧版本会明确报错。
`CMakeLists.txt` 仅构建原生 CUDA 静态库；Python 绑定请使用上面的安装方式。

## 3. 公共接口

```python
image, radii, weight_sum = rasterize_gaussians(
    means3D, intensities, raster_settings,
    scales=scales, rotations=rotations,
    # 或只传 cov3Ds_precomp=covariance
)
```

| 输入／输出 | 形状与含义 |
|---|---|
| means3D | CUDA float32 `[N,3]`，模型坐标 |
| intensities | CUDA float32 `[N,1]`，标量强度 |
| scales | CUDA float32 `[N,3]`，正的已激活标准差，不是 log scale |
| rotations | CUDA float32 `[N,4]`，`w,x,y,z`，Python 内可微归一化 |
| cov3Ds_precomp | CUDA float32 `[N,6]`，`xx,xy,xz,yy,yz,zz`，协方差而非精度矩阵 |
| image | float32 `[1,H,W]`，可微 |
| radii | int32 `[N]`，不可微；仅用于候选范围／粗可见性 |
| weight_sum | float32 `[1,H,W]`，不可微诊断量，不包含 tau |

两个 covariance 输入路径必须二选一。支持非连续 Tensor，Python 会作可微 contiguous 转换。
各 Tensor 必须同设备，不自动转 dtype。`N=0` 和零候选输出背景图、零权重。
内部 autograd 六输入对应六项返回，native backward 返回五项 Gaussian 梯度；仅支持一阶梯度。
`weight_sum` 不能直接用于可微 coverage loss；切片矩阵、tau、背景、截断和 jitter 都不学习。

设置项：

```python
GaussianRasterizationSettings(
    image_height=128, image_width=128, slice_to_world=T,
    background=0., background_weight=.01, kernel_cutoff=.001,
    cutoff_transition=2., covariance_jitter=1e-8, debug=False,
)
```

矩阵构造与保存使用 CPU float64；渲染时快照转换为同设备 float32。
CUDA 使用 row-major `[4,4]`，前后向保存并使用同一次快照。
底行为 `[0,0,0,1]`，前两列必须线性独立。第三列由前两列叉积重新归一化。
若大绝对物理坐标与很小尺度导致 float32 精度不足，应先统一平移／缩放模型和切片坐标。

## 4. 三种几何入口

### 无 DICOM 的三轴模式

```python
settings = GaussianRasterizationSettings.from_axis(
    image_height=128, image_width=128,
    axis=2, slice_idx=4, num_slices=9, bounds=(1.,1.,1.),
)
```

bounds 为模型坐标半边长。面内步长为 `2*b/W`、`2*b/H`，首像素在 `-b+b/W`、`-b+b/H`。
切片位置为 `-b_axis+2*b_axis*slice_idx/(num_slices-1)`；单层居中为 0。
axis 0 对应列 y／行 z，axis 1 对应列 x／行 z，axis 2 对应列 x／行 y。
axis 1 的叉积法向为 -y，不影响点采样，不强行改成 +y。

### DICOM／SAX／LAX

```python
settings = GaussianRasterizationSettings.from_dicom(
    ds, model_from_patient=None,  # None 表示直接使用 LPS mm
    # frame_index=0,             # 多帧必须显式指定，零起点
)
```

列方向 = IOP 前三个值 × PixelSpacing[1]；行方向 = IOP 后三个值 × PixelSpacing[0]；
首像素中心 = ImagePositionPatient。支持 classic single-frame 和增强型 per-frame/shared functional groups。
多帧必须提供所选帧的 PlanePositionSequence/ImagePositionPatient，拒绝凭 SliceThickness 推断位置。
缺失／不兼容的几何请用已知 `slice_to_world`，不静默补成正交轴向。
无硬性 pydicom 依赖；传入提供相应属性的 Dataset 即可。

若模型不是 LPS mm，提供显式的患者→模型 `model_from_patient [4,4]`。
Gaussian 均值也要施加相同变换，协方差需用 `A @ covariance @ A.T` 变换。
本库只变换切片几何，不自动重写 Gaussian。LPS→RAS 可使用 diag(-1,-1,1,1)。
切片几何不读取像素强度，不自动处理时间配准或跨序列 MR 强度一致性。

### 直接矩阵与裁剪／缩放

```python
settings = GaussianRasterizationSettings(H, W, T)
# M 把新像素 [column,row,1] 映射回原图像素坐标
new_settings = settings.with_pixel_transform(M, image_height=new_H, image_width=new_W)
```

例如从原图 `(left,top)` 开始裁剪，`M=[[1,0,left],[0,1,top],[0,0,1]]`。
按半像素约定 resize：`sx=old_W/new_W, sy=old_H/new_H`，
`M=[[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]]`。
这对应常见 `align_corners=False` 的采样坐标；其他插值坐标约定应使用其实际 M。
不能仅改 H/W 后复用旧矩阵。DICOM 构造器也可传 pixel_to_source 与新 H/W。

## 5. 迁移范围与文件

修改文档列出的全部 12 个核心文件，另修改 `auxiliary.h`、`setup.py`、`CMakeLists.txt`：
前者集中前后向共用的核／矩阵运算并去掉遗留 GLM/投影辅助依赖，后两者配套构建与架构配置。
新增本说明、`MANIFEST.in` 和 `examples/render_slice.py`。

本 v2.0.1 为独立的完整修复版源码目录，不需要应用旧 `changes.patch`。
修复细节与安装命令见 `BUILD_FIX_zh.md`。

上层调用需删除旧 `opacities/means2D/SH/camera` 参数，RGB 强度改为明确的 `[N,1]`。
Stage 1 继续 scales+rotations；Stage 2 使用预计算 covariance。
训练代码、PLY、optimizer opacity group、densify/prune 及运动模型属于上层项目，本包未修改。
不要把此源码包理解为上层训练项目已完成适配。

本版实现无限薄平面上的标量场采样；不是 CT 投影积分，也未加入 MR slice profile／有限层厚积分。

## 6. 内部接口差异

为统一参数校验和双向配置，native forward/backward 共享前缀：
`(means,intensities,scales,rotations,covariance,slice_to_world,H,W,background,tau,cutoff,transition,jitter,debug)`。
与文档建议的 native 参数排列不同，但 Python/C++/CUDA 在此版本同步，公共 Python 接口保持文档约定。
forward 返回 `(instances,image,radii,geomBuffer,binningBuffer,imageBuffer,weight_sum)`。
backward 在共享前缀后接 `(grad_image,instances,geomBuffer,binningBuffer,imageBuffer)`，
返回 `(grad_means,grad_intensities,grad_scales,grad_rotations,grad_covariance)`。
不要从上层直接调用私有 `_C_G3D`，尤其不要混用旧二进制和新 Python 文件。

缓存不再采用旧版本字节布局：保存 double 精度 precision、denom/prediction，64 位扫描计数与矩形范围。
分桶总数超过 INT_MAX-255 会报错，避免整数溢出。扫描／排序的临时空间保留在前向缓存中。
所有 Python 输出和 native 签名已成套改动；旧缓存不可跨版本复用。
