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
#include "rasterizer_impl.h"
// 像素阶段求二维 footprint 梯度；预处理阶段完整回传三维属性。
// 没有 opacity、SH、深度或切片矩阵梯度输出。
namespace BACKWARD {
void render(const CudaRasterizer::Parameters&, CudaRasterizer::GeometryState,
    CudaRasterizer::BinningState, CudaRasterizer::ImageState, const float*,
    CudaRasterizer::Gradients, dim3, cudaStream_t);
void preprocess(const CudaRasterizer::Parameters&, CudaRasterizer::GeometryState,
    CudaRasterizer::Gradients, cudaStream_t);
}
