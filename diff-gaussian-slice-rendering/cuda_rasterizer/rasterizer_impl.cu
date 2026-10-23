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

#include "rasterizer_impl.h"
#include "forward.h"
#include "backward.h"
#include "auxiliary.h"
// 仅包含使用的 CUB 算法；消除宏污染是 config.h 中 scoped constexpr 的职责。
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <climits>
using namespace CudaRasterizer;
// 将一块连续 uint8 CUDA Tensor 解释为各内部数组；每段起点按 128 字节对齐。
// required<T>() 与实际分配都调用同一布局函数，避免前后向缓存解释不同。
// 这些 Tensor 由 PyTorch 管理生命周期，不另行 cudaMalloc/cudaFree。
GeometryState GeometryState::fromChunk(char*& p,size_t n) {
    GeometryState g{};
    obtain(p,g.means2D,n); obtain(p,g.precision,6*n); obtain(p,g.rectangles,n);
    obtain(p,g.tiles_touched,n); obtain(p,g.point_offsets,n); obtain(p,g.invalid,1);
    // 首次以临时存储指针 nullptr 查询 CUB scan 需要的字节数。
    // 后续从同一 chunk 中分配该空间；计数/前缀和用 uint64_t 防止累加先溢出。
    if(n) cudaCheck(cub::DeviceScan::InclusiveSum(nullptr,g.scan_size,g.tiles_touched,g.point_offsets,(int)n));
    obtain(p,g.scan_space,g.scan_size); return g;
}
// 分桶缓存包括排序前/后的 key 与 Gaussian ID，以及 CUB radix sort 临时存储。
// 这里保留排序流程，并没有把全部 Gaussian 暴力广播给全部像素。
BinningState BinningState::fromChunk(char*& p,size_t n) {
    BinningState b{};
    obtain(p,b.keys_unsorted,n); obtain(p,b.keys,n); obtain(p,b.list_unsorted,n); obtain(p,b.list,n);
    if(n) cudaCheck(cub::DeviceRadixSort::SortPairs(nullptr,b.sort_size,b.keys_unsorted,b.keys,b.list_unsorted,b.list,(int)n));
    obtain(p,b.sort_space,b.sort_size); return b;
}
// ranges 按像素数预留，实际只使用 tile 数那一段，属于保守分配。
// denom/prediction 各有 H*W 个 double，既不复用旧 alpha 缓存，也不复用输出 float 图像。
ImageState ImageState::fromChunk(char*& p,size_t n) {
    ImageState i{}; obtain(p,i.ranges,n); obtain(p,i.denom,n); obtain(p,i.prediction,n); return i;
}
// 每个 Gaussian 生成它覆盖的全部 tile 实例。
// 包含 M 个 tile 的 Gaussian 会在列表中出现 M 次，但在某个 tile 内只出现一次。
// offsets[i-1] 是该 Gaussian 的输出起点，来自 inclusive scan。
__global__ void duplicateWithKeys(int n,const uint4* rectangles,const uint64_t* offsets,uint64_t* keys,uint32_t* list,unsigned tiles_x) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    uint64_t out=i?offsets[i-1]:0; uint4 r=rectangles[i];
    for(unsigned y=r.y;y<r.w;++y) for(unsigned x=r.x;x<r.z;++x) {
        // Stable Gaussian ID replaces camera-depth tie breaker. No occlusion.
        // 64 位 key：高 32 位 tile ID，低 32 位 Gaussian ID。
        // 高位分组形成每个 tile 的连续候选区间；低位只是确定排序次序，不再代表深度。
        // 数学模型没有遮挡，因此这个顺序不会改变理想实数运算下的图像。
        keys[out]=(uint64_t(y*tiles_x+x)<<32)|uint32_t(i); list[out]=i; ++out;
    }
}
// 扫描已排序 keys，在 tile ID 发生变化处写入半开区间 [start,end)。
// 主机先将 ranges 清零，所以没有任何候选的 tile 保持 [0,0)。
__global__ void identifyTileRanges(int n,const uint64_t* keys,uint2* ranges) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    unsigned tile=keys[i]>>32;
    if(!i) ranges[tile].x=0;
    else { unsigned prev=keys[i-1]>>32; if(prev!=tile) { ranges[prev].y=i; ranges[tile].x=i; } }
    if(i==n-1) ranges[tile].y=n;
}
// 完整前向调度顺序：分配缓存 -> 几何预处理 -> scan -> 获取实例总数 ->
// 分配排序空间 -> 复制候选 -> radix sort -> 标记 ranges -> 逐像素渲染。
// 调用方提供三个 allocator，保证缓存最终是可由 autograd 保存的 Torch Tensor。
int Rasterizer::forward(Allocator geometry,Allocator binning,Allocator image,const Parameters& p,float* out,float* weights,int* radii,cudaStream_t s) {
    char* gptr=geometry(required<GeometryState>(p.count)); GeometryState g=GeometryState::fromChunk(gptr,p.count);
    char* iptr=image(required<ImageState>(size_t(p.width)*p.height)); ImageState im=ImageState::fromChunk(iptr,size_t(p.width)*p.height);
    dim3 grid((p.width+kTileWidth-1)/kTileWidth,(p.height+kTileHeight-1)/kTileHeight);
    cudaCheck(cudaMemsetAsync(g.invalid,0,sizeof(int),s));
    cudaCheck(cudaMemsetAsync(im.ranges,0,size_t(grid.x)*grid.y*sizeof(uint2),s));
    // 零 Gaussian 时跳过 scan 和最后一个元素读取，防止 P-1 访问。
    // 无论是否有候选，最终都会运行像素核以写出背景和合法图像缓存。
    uint64_t total=0; int invalid=0;
    if(p.count) {
        FORWARD::preprocess(p,g,radii,grid,s); cudaCheck(cudaGetLastError());
        cudaCheck(cub::DeviceScan::InclusiveSum(g.scan_space,g.scan_size,g.tiles_touched,g.point_offsets,p.count,s));
        cudaCheck(cudaMemcpyAsync(&total,g.point_offsets+p.count-1,sizeof(total),cudaMemcpyDeviceToHost,s));
        cudaCheck(cudaMemcpyAsync(&invalid,g.invalid,sizeof(invalid),cudaMemcpyDeviceToHost,s));
        // 这里必须获得主机侧 total 才能决定排序缓存大小，因此当前实现需要同步。
        // 这也使 invalid 检查及时生效；它属于性能开销，不能称为完全异步实现。
        cudaCheck(cudaStreamSynchronize(s)); // host needs instance count to allocate
    }
    if(invalid) throw std::runtime_error("Invalid effective covariance or numerically degenerate slice geometry; check units, conditioning and covariance_jitter.");
    // 虽然扫描计数为 64 位，后面的实例索引/launch 仍使用 int。
    // 保留线程块余量后检查上限，拒绝过大的任务，避免后续整数转换溢出。
    if(total>INT_MAX-255) throw std::runtime_error("Gaussian/tile instance count exceeds INT_MAX-255; reduce image or support size.");
    char* bptr=binning(required<BinningState>(total)); BinningState b=BinningState::fromChunk(bptr,total);
    if(total) {
        duplicateWithKeys<<<(p.count+255)/256,256,0,s>>>(p.count,g.rectangles,g.point_offsets,b.keys_unsorted,b.list_unsorted,grid.x);
        cudaCheck(cudaGetLastError());
        cudaCheck(cub::DeviceRadixSort::SortPairs(b.sort_space,b.sort_size,b.keys_unsorted,b.keys,b.list_unsorted,b.list,(int)total,0,64,s));
        identifyTileRanges<<<(total+255)/256,256,0,s>>>((int)total,b.keys,im.ranges);
        cudaCheck(cudaGetLastError());
    }
    FORWARD::render(p,g,b,im,out,weights,grid,s); cudaCheck(cudaGetLastError()); return (int)total;
}
// 反向重新解释同次 forward 保存的三块缓存，不重新分桶或重新计算预测。
// 第一步累积 means/intensity/covariance 梯度；第二步仅在尺度+旋转路径传播矩阵链。
// 当无实例时梯度应全为零，调用方已经分配了零初始化梯度，因此可直接返回。
void Rasterizer::backward(const Parameters& p,int instances,char* geometry,char* binning,char* image,const float* incoming,Gradients grads,cudaStream_t s) {
    if(!p.count || !instances) return;
    GeometryState g=GeometryState::fromChunk(geometry,p.count);
    BinningState b=BinningState::fromChunk(binning,instances);
    ImageState im=ImageState::fromChunk(image,size_t(p.width)*p.height);
    dim3 grid((p.width+kTileWidth-1)/kTileWidth,(p.height+kTileHeight-1)/kTileHeight);
    BACKWARD::render(p,g,b,im,incoming,grads,grid,s); cudaCheck(cudaGetLastError());
    BACKWARD::preprocess(p,grads,s); cudaCheck(cudaGetLastError());
}
