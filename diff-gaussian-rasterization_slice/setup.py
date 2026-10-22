from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

# this_dir = os.path.dirname(os.path.abspath(__file__))

# setup(
#     name="diff_gaussian_rasterization",
#     packages=['diff_gaussian_rasterization'],
#     ext_modules=[
#         CUDAExtension(
#             name="diff_gaussian_rasterization._C_G3D",
#             sources=[
#                 "cuda_rasterizer/rasterizer_impl.cu",
#                 "cuda_rasterizer/forward.cu",
#                 "cuda_rasterizer/backward.cu",
#                 "rasterize_points.cu",
#                 "ext.cpp"
#             ],
#             extra_compile_args={
#                 "nvcc": [
#                     "-I" + os.path.join(this_dir, "third_party/glm/"),
#                     "-gencode=arch=compute_86,code=sm_86",   # ← 新增
#                 ]
#             }
#         )
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )


this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    # 1. 【name】：这是 pip 管理工具识别的包名。
    # 当你运行 `pip list` 或 `pip uninstall` 时，显示的就是这个名字。
    # 这里改名是为了防止 pip 覆盖你原版的库。
    name="diff_gaussian_rasterization_slice",
    
    # 2. 【packages】：这告诉 setuptools 哪些本地文件夹应该作为 Python 包被打包安装。
    # 这个名字必须和你源码里包含 __init__.py 的那个“核心文件夹”的名字一模一样！
    # 稍后你需要把你的文件夹名字也改成这个。
    packages=['diff_gaussian_rasterization_slice'],
    
    ext_modules=[
        CUDAExtension(
            # 3. 【CUDAExtension 的 name】：这是 C++/CUDA 编译出来的底层二进制模块的导入路径。
            # 它的前半部分（点号之前）必须和上面的 packages 名字（即 Python 文件夹名）完全一致，
            # 这样编译出来的 _C_G3D.so 动态链接库才会正确地放入你的自定义包目录下。
            name="diff_gaussian_rasterization_slice._C_G3D",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp"
            ],
            extra_compile_args={
                "nvcc": [
                    "-I" + os.path.join(this_dir, "third_party/glm/"),
                    "-gencode=arch=compute_86,code=sm_86", 
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)