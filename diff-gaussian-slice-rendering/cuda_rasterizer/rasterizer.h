/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <cuda_runtime.h>
#include <functional>
#include <cstddef>
namespace CudaRasterizer {
// All matrices are row-major; covariance packing is xx,xy,xz,yy,yz,zz.
// 跨层参数结构：汇总固定观测条件和 Gaussian 数据指针。
// count=N，width/height 为像素数；全部输入指针指向当前 CUDA 设备的 contiguous float32。
// background/tau/qmax/transition/jitter 用 double 传入，避免多处常量不一致。
// covariance 非空表示预计算路径；否则 scales 与 rotations 必须有效。
struct Parameters {
    int count, width, height;
    const float *means, *intensities, *scales, *rotations, *covariance, *slice_to_world;
    double background, tau, qmax, transition, jitter;
};
// 梯度数组由 C++ bridge 零初始化，CUDA 累加后按约定顺序返回 Python。
// 协方差路径不使用 scales/rotations 梯度；两者在 Python 被转换为 None。
struct Gradients { float *means, *intensities, *covariance, *scales, *rotations; };
using Allocator = std::function<char*(size_t)>;
// 库的原生调度层，不接触 Python autograd，也不使用 DICOM/axis 分支。
// 几何入口差异全部已在 Python 中转换成同一种 slice_to_world。
class Rasterizer {
public:
    static int forward(Allocator geometry, Allocator binning, Allocator image,
        const Parameters&, float* output, float* weight_sum, int* radii, cudaStream_t);
    static void backward(const Parameters&, int instances, char* geometry,
        char* binning, char* image, const float* grad_image, Gradients, cudaStream_t);
};
}
