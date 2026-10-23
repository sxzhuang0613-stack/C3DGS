# v2.0.1 安装错误修复说明

依据上传的安装日志修复两处问题；保留中文注释及当前渲染模型。

## 1. 直接导致编译失败的原因

原 `config.h` 定义了 `#define NUM_CHANNELS 1`。`rasterizer_impl.cu` 在包含该定义后又包含
`<cub/cub.cuh>`。预处理器将 CUB histogram 模板中的同名参数替换为数字 1，
于是合法的 `template<int NUM_CHANNELS,...>` 变成非法的 `template<int 1,...>`。
日志中的 11 条 histogram 报错是同一类宏冲突造成的，不代表本算法使用了 histogram。

本版修复：

- `config.h` 的三个通用宏替换为 `CudaRasterizer` 命名空间中的 C++17 inline constexpr 常量。
- 使用 `kNumChannels`、`kTileWidth`、`kTileHeight`，同步更新前向、反向和调度中的 tile 尺寸引用。
- `rasterizer_impl.cu` 只包含 CUB 的 device_scan 和 device_radix_sort 头文件。
  按需包含用于减少依赖范围；真正消除文本替换冲突的是移除通用宏。

不能通过修改 CUDA 安装目录中的 CUB 文件解决此问题，也无需为这条错误重装 CUDA。

## 2. 新名称必须区分发行包名与 Python 包名

日志显示 Python 包和扩展使用了 `diff-gaussian-slice-rendering`。
连字符可以用于发行包名及项目目录，但 Python 普通 import 使用合法标识符，应改成下划线。
本版已统一：

| 位置 | 名称 |
|---|---|
| 外层项目目录 | diff-gaussian-slice-rendering |
| setup.py / name | diff-gaussian-slice-rendering |
| 内层 Python 包目录与 packages | diff_gaussian_slice_rendering |
| CUDAExtension.name | diff_gaussian_slice_rendering._C_G3D |

公开 Python 类/函数继续保留 GaussianRasterizationSettings、GaussianRasterizer、rasterize_gaussians。
此次未同步重命名公共 API，避免将构建修复和上层接口迁移混在一起。
可以与旧 `diff_gaussian_rasterization_slice` 共存，无需卸载旧库。

## 3. 安装

建议将新 ZIP 解压到全新的父目录，避免旧的连字符包目录与 build 缓存混入。
例如先把 ZIP 放到当前目录，执行：

```bash
conda activate 3dgs
unzip C3DGS_Slice_Rendering_v2_0_1_BuildFix.zip -d "$HOME/c3dgs-buildfix-v201"
cd "$HOME/c3dgs-buildfix-v201/diff-gaussian-slice-rendering"
python -m pip install --no-build-isolation -v .
```

本版增加 pyproject.toml，声明 setuptools 构建后端；仍复用当前环境已安装的 PyTorch。
没有强制更改 CUDA 架构，没有自动升级/降级 PyTorch、CUDA 或编译器。
若需要限制编译并行度，可在同一终端设置 `export MAX_JOBS=2` 后再安装。

安装后同时检查 Python 包及实际二进制扩展：

```bash
python - <<'PY'
import torch
import diff_gaussian_slice_rendering as package
from diff_gaussian_slice_rendering import _C_G3D
print('Python package:', package.__file__)
print('Native extension:', _C_G3D.__file__)
print('API_VERSION:', _C_G3D.API_VERSION)
assert _C_G3D.API_VERSION == 2
PY
```

只 import Python 包不足以证明扩展可用，因为本库允许扩展未加载时先构造 CPU 切片几何。

## 4. 本次验证的范围

- 用本机 C++17 编译器构造了与 CUB 同名模板参数的最小复现：旧 config.h 编译失败；新 config.h 通过。
- 对全部 Python 源码进行了语法检查，并检查构建包名、目录与扩展前缀一致。
- 当前执行环境没有 nvcc，因此没有完成实际 CUDA 扩展编译或 GPU 运行验证。
- 日志中的 legacy setup.py、CUDA 12.1 编译器版本范围、未设置 TORCH_CUDA_ARCH_LIST 是警告，
  本次日志中真正使构建中断的是上述 CUB 模板编译错误。

这是针对已观察到错误的完整源码修复版；不应将最小 C++ 验证理解为整个 CUDA 库已通过测试。
