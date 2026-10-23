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

#include <torch/extension.h>
#include "rasterize_points.h"
// 将两个 C++ 函数绑定成 Python 可调用的扩展入口。
// API_VERSION 用来拒绝旧版 .so/.pyd，避免新 Python 包装错误加载旧签名的二进制。
// 这里不负责自动微分；自动微分协议由 Python 的 torch.autograd.Function 实现。
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.attr("API_VERSION")=2;
    m.def("rasterize_gaussians",&RasterizeGaussiansCUDA,"Opacity-free scalar slice forward (CUDA)");
    m.def("rasterize_gaussians_backward",&RasterizeGaussiansBackwardCUDA,"Opacity-free scalar slice backward (CUDA)");
}
