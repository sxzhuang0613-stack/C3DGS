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

// 使用带命名空间的 C++ 常量，不使用通用名称的预处理宏。
// 旧的 #define NUM_CHANNELS 1 会把 CUB 中的 template<int NUM_CHANNELS>
// 替换成非法的 template<int 1>，导致 histogram 头文件编译报错。
// C++ 常量不会进行这种跨头文件的文本替换。
namespace CudaRasterizer {
inline constexpr int kNumChannels = 1;
// tile 大小同时决定候选分桶和逐像素 kernel 的 block 大小。
inline constexpr int kTileWidth = 16;
inline constexpr int kTileHeight = 16;
}
