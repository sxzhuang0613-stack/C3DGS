# 详细注释版：阅读顺序、代码精简与依赖说明

原中文注释版在上一份 `C3DGS_OpacityFree_Slice_v2.zip` 的完整源码上增加中文注释。
16 个代码／构建文件新增 238 行解释性注释，渲染公式、API、默认参数和渲染逻辑保持不变。v2.0.1 另修复构建常量和包命名，详见 BUILD_FIX_zh.md。
Python AST 与原交付一致；C++/CUDA 去掉独立行注释后与原交付一致。
这仅验证本次注释编辑没有改变可执行内容，不等于证明上一版算法或 CUDA 实现正确。
尚未进行 CUDA 编译、GPU 运行、数值梯度与性能验证。

## 1. 建议阅读顺序

| 顺序 | 文件 | 重点 |
|---|---|---|
| 1 | diff_gaussian_slice_rendering/__init__.py | 三种切片入口、坐标约定、六输入 autograd、两种参数路径 |
| 2 | rasterize_points.h / rasterize_points.cu | Tensor 检查、七项前向结果、五项反向梯度、设备与 stream |
| 3 | cuda_rasterizer/rasterizer.h / rasterizer_impl.h | 参数结构、三类缓存和内存布局 |
| 4 | cuda_rasterizer/rasterizer_impl.cu | scan、复制、radix sort、tile ranges 和内核调度 |
| 5 | cuda_rasterizer/auxiliary.h | 四元数矩阵、Cholesky、统一像素坐标、核与 taper 导数 |
| 6 | cuda_rasterizer/forward.cu | 条件二维核、保守包围半径、三维采样与归一化混合 |
| 7 | cuda_rasterizer/backward.cu | 商法则、均值/协方差梯度、packed 对称项、尺度/旋转链 |
| 8 | setup.py / CMakeLists.txt / examples/render_slice.py | 编译入口与两种调用示例 |

关键数据流：Python 几何构造 → native Tensor 桥接 → 每 Gaussian 条件核预处理 →
Gaussian/tile 分桶排序 → 每像素归一化强度 → 保存缓存 → 像素损失反向 → Gaussian 参数梯度。

## 2. 为什么代码明显减少

以下统计仅比较固定 master 快照与上一份未加中文注释交付中相同的 14 个 `.cu/.h/.cpp/.py` 文件。
包含空行与注释，不包含 third_party，不包含 CMake、README 和新增示例：原版 2945 行，上一版 892 行。
物理行数不是功能完整度、算法正确性或性能指标；上一版也使用了更紧凑的书写格式。

| 文件 | 原快照行数 | 未加本次中文注释的 v2 行数 |
|---|---:|---:|
| forward.cu | 719 | 96 |
| backward.cu | 569 | 66 |
| rasterizer_impl.cu | 471 | 84 |
| rasterize_points.cu | 252 | 93 |
| Python __init__.py | 213 | 274 |

减少有四类来源，不能全部解释为“删除了无用代码”。

1. **任务范围改变。** 删除公开的 opacity、相机投影、SH、多通道 RGB、深度等接口和对应缓存／梯度管理。
   原始项目中有些图形学函数可能属于保留代码或特定分支，并非每次原渲染都会执行；这里是源码层面的删减。
   新库提供固定切片标量场采样，不再承担通用相机图像渲染。
2. **数学与接口合并。** 三个 axis 分支统一为矩阵，前后向共用核函数，参数集合使用结构体。
   单通道归一化混合的商法则也比保留通用 alpha 合成及其梯度接口更简洁。
3. **矩阵实现替换。** 少量固定维度运算直接用标量公式／局部数组表达，因而不再需要 GLM 的完整头文件目录。
   这是明确实现了所需运算，不是遗漏了旋转或协方差计算。
4. **执行实现也被简化。** 上一版删除了原像素核的 shared-memory 分批搬运、协作同步等实现，
   改成每像素线程遍历所属 tile 的候选并直接读取全局内存。CUB 分桶与排序仍然保留。
   因此这是包含重写的完整源码实现，不是严格最小修改版，也不保留原版全部性能优化。

第 4 点需要特别区分：没有遮挡的数学模型仍可以使用 shared memory 加速，
删除 shared-memory 搬运并不是删除 opacity 在数学上必然要求的步骤。
上一版采用该简化是实现选择；未来可以在不改变模型的情况下恢复协作加载。
双精度内部计算、逐次有限值检查、获取分桶总数时的主机同步也可能影响速度。
没有基准测试前，不能承诺与原版等速，更不能据代码少就声称更快。

## 3. 为什么不需要旧 third_party

你提到的目录应是 `third_party`。原 setup.py 显式将 `third_party/glm/` 加入 include 搜索路径，
原 CUDA 使用 `glm::vec3/vec4/mat2/mat3`、矩阵乘法和 transpose 等操作。
GLM 是通用数学头文件库，并不是 Gaussian rasterization 的 CUDA 主算法本体。

本版源码没有 `#include <glm/...>` 或 `glm::...` 调用，setup.py 和 CMakeLists.txt 也没有 GLM 路径。
相关运算现位于以下文件：

| 原用途 | 当前实现 |
|---|---|
| 四元数转旋转矩阵 | auxiliary.h / rotation：wxyz 的显式九元素公式 |
| R diag(s²) R^T | forward.cu / preprocessCUDA：固定 3×3 标量循环 |
| 对称 3×3 逆矩阵 | auxiliary.h / inverseSPD：double Cholesky 与三列求解 |
| 二维条件核求逆 | forward.cu：二维 Cholesky/Schur 补计算 V |
| 像素到模型坐标 | auxiliary.h / pixelPoint：读取 4×4 仿射的前两列和第四列 |
| 协方差到尺度/旋转的梯度 | backward.cu / covarianceBackward：显式矩阵链与四元数 Jacobian |

因此本包无需复制旧仓库的 GLM 子目录。这个结论只针对当前源码的依赖，不意味着
可以从其他仍使用 GLM 的项目里删除 third_party，也不意味着本包是无依赖软件。

仍需的外部组件：

- PyTorch：Tensor、自动微分衔接、扩展构建和 CUDA stream/device 管理。
- CUDA Toolkit / Runtime / nvcc：GPU 内核编译和执行。
- CUB：inclusive scan 与 radix sort；通过 CUDA 工具链中的头文件提供，本包按需包含 `cub/device/device_scan.cuh` 与 `cub/device/device_radix_sort.cuh`。
- Python、setuptools 和兼容的 C++ 编译器。

原目录中的预编译二进制不是新接口可直接复用的运行文件；本版必须重新编译。
压缩包大小还会受第三方文件、二进制、构建产物和压缩率影响，不能拿 ZIP 大小推断主算法是否完整。
本次没有对原仓库全目录大小作定量核算，以上行数比较严格限定在相同核心源文件。

## 4. 完整性与边界

这是完整的修改后渲染扩展源码，可以在具备兼容编译环境时尝试独立构建；不需要与旧库拼接。
这里的完整表示依赖于本项目自身的源码与构建入口齐全，不表示已通过编译或 GPU 验证。
它仍不包含上层 Gaussian 初始化、PLY、优化器、densification、运动网络或完整训练流程。
数学范围为无限薄切片平面上的标量场采样，不包含 CT 射线积分或 MR 有限层厚 profile。

v2.0.1 直接提供完整修复源码，不需要应用旧版本的 changes.patch。
