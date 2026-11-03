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
#include "rasterizer.h"
#include <cstdint>
namespace CudaRasterizer {
// 每 Gaussian 缓存像素中心、二维精度 xx/xy/yy 与对数幅值；不保存三维像素运算。
struct GeometryState {
    double2* means2D;
    double *precision, *log_amplitude;
    uint4* rectangles;
    uint64_t *tiles_touched, *point_offsets;
    int* invalid;
    size_t scan_size;
    char* scan_space;
    static GeometryState fromChunk(char*&, size_t);
};
// 按 Gaussian/tile 实例数分配，每个实例是一个 Gaussian 与一个 tile 的配对。
struct BinningState {
    uint64_t *keys_unsorted, *keys;
    uint32_t *list_unsorted, *list;
    size_t sort_size;
    char* sort_space;
    static BinningState fromChunk(char*&, size_t);
};
// 仅保存 tile 候选范围；加性模型不需要图像分母或预测缓存。
struct ImageState {
    uint2* ranges;
    static ImageState fromChunk(char*&, size_t);
};
// 通过 uintptr_t 完成对齐，计算方式与数据类型字节大小显式对应。
// 即使不同数组类型混用，也不会依赖编译器对 struct 成员的隐式布局。
template<class T> inline void obtain(char*& p,T*& out,size_t n) {
    auto a=(reinterpret_cast<uintptr_t>(p)+127)&~uintptr_t(127);
    out=reinterpret_cast<T*>(a); p=reinterpret_cast<char*>(a+n*sizeof(T));
}
// 在无真实存储的地址计数模式下执行同一布局计算，返回所需字节数并加对齐余量。
// 这里只计算位置；fromChunk 的 CUB 调用处于临时空间大小查询模式。
template<class T> inline size_t required(size_t n) {
    char* p=nullptr; T::fromChunk(p,n); return reinterpret_cast<uintptr_t>(p)+128;
}
}
