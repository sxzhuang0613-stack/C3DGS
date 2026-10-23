cmake_minimum_required(VERSION 3.18)
project(OpacityFreeSliceRasterizer LANGUAGES CXX CUDA)
# This target builds the native CUDA library only. Use setup.py for PyTorch.
find_package(CUDAToolkit REQUIRED)
# 这个 CMake 目标只编译 CUDA 静态库，不生成 Python 的 _C_G3D 扩展。
# 要在 PyTorch 中调用，请使用 setup.py 对应的 pip install 命令。
add_library(CudaRasterizer STATIC cuda_rasterizer/forward.cu
    cuda_rasterizer/backward.cu cuda_rasterizer/rasterizer_impl.cu)
target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(CudaRasterizer PUBLIC CUDA::cudart)
set_target_properties(CudaRasterizer PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17
    POSITION_INDEPENDENT_CODE ON)
# Select architectures via -DCMAKE_CUDA_ARCHITECTURES=... at configure time.
