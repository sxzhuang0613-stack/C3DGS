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
#include <torch/extension.h>
#include <tuple>
// native forward 七项顺序：实例数、图像、半径、几何缓存、分桶缓存、图像缓存、权重和。
// native backward 五项顺序：均值、强度、尺度、旋转、协方差梯度。
// Tensor 在 C++ 类型层面无法区分语义，Python 解包顺序必须与实现对应。
using ForwardResult=std::tuple<int,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>;
using BackwardResult=std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>;
// 前向与反向共享参数前缀，便于使用同次前向保存的配置快照。
// 这些是私有 native 接口，用户应通过 Python rasterize_gaussians 调用。
ForwardResult RasterizeGaussiansCUDA(
    const torch::Tensor& means,const torch::Tensor& intensities,
    const torch::Tensor& scales,const torch::Tensor& rotations,
    const torch::Tensor& covariance,const torch::Tensor& slice_to_world,
    int height,int width,double background,double tau,double cutoff,
    double transition,double jitter,bool debug);
BackwardResult RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& means,const torch::Tensor& intensities,
    const torch::Tensor& scales,const torch::Tensor& rotations,
    const torch::Tensor& covariance,const torch::Tensor& slice_to_world,
    int height,int width,double background,double tau,double cutoff,
    double transition,double jitter,bool debug,
    const torch::Tensor& incoming,int instances,const torch::Tensor& geometry,
    const torch::Tensor& binning,const torch::Tensor& image);
