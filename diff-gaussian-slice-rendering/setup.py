from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# 此文件构建 PyTorch C++/CUDA 扩展；需要当前环境已有兼容的 PyTorch 和 CUDA Toolkit。
# 源码不再 include glm，因此没有 third_party/glm 的 include 路径。
# 这不表示没有外部依赖：Torch、CUDA Runtime、CUB 和编译工具链仍然必需。
root = Path(__file__).resolve().parent
setup(
    name="diff-gaussian-slice-rendering",
    version="2.0.1",
    description="Opacity-free differentiable scalar Gaussian slice rasterization",
    packages=["diff_gaussian_slice_rendering"],
    python_requires=">=3.9",
    # 五个翻译单元分别是调度、前向、反向、Tensor 桥接与 PyBind 注册。
    # 头文件由这些源文件 include，不需单独放进 sources 列表。
    ext_modules=[CUDAExtension(
        name="diff_gaussian_slice_rendering._C_G3D",
        sources=["cuda_rasterizer/rasterizer_impl.cu", "cuda_rasterizer/forward.cu",
                 "cuda_rasterizer/backward.cu", "rasterize_points.cu", "ext.cpp"],
        include_dirs=[str(root)],
        # Architecture is selected by PyTorch/TORCH_CUDA_ARCH_LIST, not hardcoded.
        # No --use_fast_math: the same weight/cutoff must be used in both passes.
        extra_compile_args={"cxx": ["-O3"] if __import__("os").name != "nt" else ["/O2"],
                            "nvcc": ["-O3", "-lineinfo"]},
    )],
    cmdclass={"build_ext": BuildExtension},
)
