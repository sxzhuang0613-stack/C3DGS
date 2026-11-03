# 加性 Gaussian 场与解析厚度切片渲染（v3.0.0）

当前版本用有符号加性 Gaussian 场代替 v2 的归一化灰度混合，并在 Preprocess 中将解析 Gaussian slice response 转成等效二维 footprint。保留 tile 分桶、CUB scan/radix sort、当前 CUDA stream 和两种协方差参数化。

## 模型

```text
S_i = covariance_i + covariance_jitter * I
E_i = S_i + v v^T
D_i = 1 + v^T S_i^-1 v
response_i(p) = exp[-(p-mu_i)^T E_i^-1 (p-mu_i)/2] / sqrt(D_i)
image(p) = background + sum_i intensity_i * response_i(p)
```

`intensity` 是自由有符号贡献，不使用 opacity、SH、alpha compositing、Sigmoid、归一化分母或渲染输出截断。背景只加一次。零 Gaussian 或无候选时输出背景。

`v` 是固定的一倍标准差位移，不是可学习 Gaussian 属性：

- `response_mode='point'`：`v=0`。
- `response_mode='gaussian', psf_sigma=sigma`：`v=sigma*n`，`n` 为当前模型空间切片单位法向。
- `response_mode='gaussian', psf_std_vector=[vx,vy,vz]`：直接使用模型空间位移向量，适合非等比物理坐标变换。两种 PSF 表达互斥。

Gaussian profile 是沿法向归一化的无限域 Gaussian 权重，不是有限区间均匀盒积分。零 PSF 与加性点采样共用同一路径；对非零 PSF，幅值修正 `1/sqrt(D)` 不可省略。固定 jitter 在基础核和幅值中使用同一个 S；一般仿射不变性验证应设 `covariance_jitter=0`，或一致地变换稳定协方差。

## 二维计算和反向

切片映射 `p=o+J*u`，`u=[column,row]`。Preprocess 计算：

```text
P = E^-1
H = J^T P J
g = J^T P (mu-o)
m = H^-1 g
q = [n_geo^T(mu-o)]^2 / (n_geo^T E n_geo)
log_amplitude = -(q + log(D))/2
response(u) = exp(log_amplitude - (u-m)^T H (u-m)/2)
```

使用 double Cholesky 和 `log1p`，不使用 `det+epsilon` 或通过增大 jitter 掩盖退化。法向商避免补平方时的大数相减。像素核只读取二维中心、二维精度和对数幅值。

反向先在像素阶段累积二维中心、对称精度、对数幅值和 intensity 梯度，再逐 Gaussian 回传三维位置和协方差。无支撑窗时，令 `w=E^-1(p-mu)`：

```text
d log(response)/d mu = w
d log(response)/d S = (S^-1 - E^-1 + w w^T)/2
```

其中 `S^-1-E^-1` 来自幅值修正。之后复用尺度/四元数链式求导；预计算协方差直接返回 packed 梯度。Python 在 custom Function 外可微归一化 wxyz，CUDA 不重复归一化。当前仅支持一阶梯度。

## 有限支撑与 tile culling

完整解析积分是数学精确的；实际 rasterizer 对积分后的二维响应施加有限支撑近似，而不是先截断三维 Gaussian 再积分。

```text
Q = (u-m)^T H (u-m) + q + log(D)
Qmax = -2*log(kernel_cutoff)
```

`kernel_cutoff` 作用于不含有符号 intensity 的响应。`Q>=Qmax` 时裁剪；footprint 半径由 `Qmax-q-log(D)` 和 `H^-1` 的最大特征值决定。中心不在切片上的 Gaussian 仍可能产生厚度响应，不能按原始薄切片距离提前剔除。

`cutoff_transition=0` 使用硬支撑，边界本身不可导。默认 `2` 在最后一段 Q 区间施加 C1 三次窗；前后向均包含窗函数导数。这会改变尾部核形状，不再是严格完整 Gaussian。窗口内、窗口外误差单独验证。整图裁剪误差受 `sum(abs(intensity))` 影响，不能把单核阈值当成整图固定误差。

Tile 是保守候选，`radii` 不是精确像素贡献。排序仍按 tile/ID，不做遮挡或正负贡献早停。

## Python 接口

```python
settings = GaussianRasterizationSettings.from_axis(
    image_height=128, image_width=128, axis=2,
    slice_idx=4.25, num_slices=9, bounds=(1.,1.,1.),
    response_mode='gaussian', psf_sigma=.05,  # 示例值，单位是模型坐标
)
image, radii, weight_sum = rasterize_gaussians(
    means3D, intensities, settings, scales=scales, rotations=rotations,
)
# 或只传 cov3Ds_precomp=covariance
```

| 输入/输出 | 契约 |
|---|---|
| means3D | CUDA float32 `[N,3]` |
| intensities | CUDA float32 `[N,1]`，自由有符号贡献 |
| scales / rotations | CUDA float32 `[N,3]` 正标准差 / `[N,4]` wxyz |
| cov3Ds_precomp | CUDA float32 `[N,6]`，xx,xy,xz,yy,yz,zz；与尺度/旋转互斥 |
| image | float32 `[1,H,W]`，可微，无输出截断 |
| radii | int32 `[N]`，不可微 |
| weight_sum | float32 `[1,H,W]`，不可微的响应和，不含背景或 intensity |

`GaussianRasterizer` 保留同一薄包装。公开输出和模型属性不变。旧 Python `background_weight` 参数只用于构造兼容，会发出弃用提示且完全不参与计算；配置已移除此参数。旧归一化模型的 intensity 数值不具有新的加性含义，加载同形状 PLY 不保证重建结果相同。

## 物理坐标

`slice_to_world` 保持 row-major 4×4 约定：前两列是像素列/行三维位移，第四列是首像素中心，第三列保存由叉积计算的几何单位法向。支持任意斜切片、旋转各向异性核和显式裁剪/resize 映射。

Slice spacing 定位中心平面；slice thickness 是采集层厚；PSF sigma 是 profile 标准差，三者不自动互相替代。不从 DICOM `SliceThickness` 推断 sigma。若另行明确 profile 的 FWHM，调用方可显式换算 `sigma=FWHM/(2*sqrt(2*log(2)))`。

```python
settings = GaussianRasterizationSettings.from_dicom(
    ds, response_mode='gaussian', psf_sigma_mm=2.0,
    model_from_patient=patient_to_model,
)
```

该入口先在患者 LPS mm 坐标中定义 `v_mm=sigma_mm*n_mm`，再用仿射线性部分 M 转换 `v_model=M*v_mm`。几何法向按 `M^-T*n_mm` 变化，非等比缩放/shear 下不能拿它替代物理 PSF 位移。调用方须同步变换 Gaussian 均值和协方差。`psf_sigma_mm` 不能与模型空间 PSF 参数混用。

`with_pixel_transform()` 接受新像素到旧像素的 3×3 仿射，保留 PSF；resize 不改变采集 profile。几何与 PSF 为固定输入，前向保存独立快照，不支持学习它们。

归一化训练入口使用 `from_axis`，无法仅凭病例尺寸/bounds 恢复毫米 PSF。训练/测试配置新增 `response_mode`、`psf_sigma`、`psf_std_vector`，默认 point；需要已知的模型空间 PSF 才开启 gaussian。

## 构建和验证

依赖现有 CUDA PyTorch、兼容 CUDA Toolkit/nvcc、C++17 编译器、setuptools；使用 Toolkit CUB，不依赖 GLM。未增加运行时依赖。

```bash
cd diff-gaussian-slice-rendering
python -m pip install --no-build-isolation .
```

native `API_VERSION=3`；Python 拒绝旧二进制。`CMakeLists.txt` 仅构建原生静态库。`BUILD_FIX_zh.md` 和 `CODE_GUIDE_zh.md` 为 v2 历史说明，当前模型以本文件为准。

从项目根目录运行：

```bash
python -m unittest discover -s tests -p test_slice_response.py -v
```

独立 `build_ext --build-lib <dir>` 后，可设置 `C3DGS_EXTENSION_DIR=<dir>` 运行测试，无需安装。`reference.py` 提供小规模 float64 PyTorch 参考：完整三维响应、二维 footprint 和可选裁剪。测试覆盖高精度数值积分、解析等价性、负强度、物理仿射、两条协方差梯度、裁剪窗口和训练 wrapper。

`examples/benchmark_slice.py` 进行短时间前向/反向及峰值显存测试，参数由 CLI 指定。不同模型的数值输出不应比较为相同；性能比较必须说明点数、分辨率、PSF 和支撑范围。
