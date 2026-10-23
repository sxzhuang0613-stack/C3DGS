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
// 前向接口：preprocess 每 Gaussian 构造候选范围；render 每像素形成归一化强度。
// Parameters 与 State 结构避免原版大量同类型裸指针参数的顺序错配。
namespace FORWARD {
void preprocess(const CudaRasterizer::Parameters&, CudaRasterizer::GeometryState, int*, dim3, cudaStream_t);
void render(const CudaRasterizer::Parameters&, CudaRasterizer::GeometryState,
    CudaRasterizer::BinningState, CudaRasterizer::ImageState, float*, float*, dim3, cudaStream_t);
}
