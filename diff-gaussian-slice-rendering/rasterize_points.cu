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

#include "rasterize_points.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include "cuda_rasterizer/auxiliary.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <climits>
#include <cmath>
#include <limits>
using torch::Tensor;
using namespace CudaRasterizer;
// Python 与 CUDA 之间的最后一道 Tensor 契约检查：CUDA、float32、连续、同设备、有限值。
// 有限值检查会触发设备计算及主机读回；当前实现选择显式检查，未优化这部分开销。
static void tensorCheck(const Tensor& t,const Tensor& ref,const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type()==torch::kFloat32 && t.is_contiguous(),name," must be contiguous CUDA float32");
    TORCH_CHECK(t.device()==ref.device(),name," must be on the means3D device");
    TORCH_CHECK(torch::isfinite(t).all().item<bool>(),name," must contain finite values");
}
// 组装并校验原生参数。forward 和 backward 共用该函数，确保两次解释一致。
// 支持 N=0；使用 shape 而不是 numel 判定预计算分支，以区分 [0,6] 与空占位 [0]。
static Parameters parameters(const Tensor& m,const Tensor& c,const Tensor& scales,const Tensor& rot,const Tensor& cov,const Tensor& t,
    int H,int W,double bg,double tau,double cutoff,double transition,double jitter) {
    tensorCheck(m,m,"means3D"); tensorCheck(c,m,"intensities"); tensorCheck(scales,m,"scales");
    tensorCheck(rot,m,"rotations"); tensorCheck(cov,m,"cov3Ds_precomp"); tensorCheck(t,m,"slice_to_world");
    TORCH_CHECK(m.dim()==2 && m.size(1)==3 && m.size(0)<=INT_MAX/9,"means3D must have shape [N,3] with N <= INT_MAX/9");
    int n=(int)m.size(0);
    TORCH_CHECK(c.dim()==2 && c.size(0)==n && c.size(1)==1,"intensities must have shape [N,1]");
    TORCH_CHECK(H>0 && W>0 && H<=INT_MAX-16 && W<=INT_MAX-16 && int64_t(H)*W<=INT_MAX,"invalid image dimensions");
    TORCH_CHECK(std::isfinite(bg) && std::abs(bg)<=std::numeric_limits<float>::max() && std::isfinite(tau) && tau>0 && std::isfinite(cutoff) && cutoff>0 && cutoff<1,"invalid background, tau or kernel_cutoff");
    double qmax=-2*std::log(cutoff);
    TORCH_CHECK(std::isfinite(transition) && transition>=0 && transition<qmax,"cutoff_transition must be in [0,-2 log(kernel_cutoff))");
    TORCH_CHECK(std::isfinite(jitter) && jitter>=0,"covariance_jitter must be finite and nonnegative");
    // 判断 [N,6] 协方差路径；未使用的参数使用合法的同设备空 Tensor。
    // Python 已保证互斥，这里再次检查，以防直接调用 private native 接口绕过包装。
    bool precomputed=(cov.dim()==2);
    if(precomputed) {
        TORCH_CHECK(cov.size(0)==n && cov.size(1)==6 && scales.numel()==0 && rot.numel()==0,"covariance path requires [N,6] and empty scales/rotations");
    } else {
        TORCH_CHECK(cov.numel()==0 && scales.dim()==2 && scales.size(0)==n && scales.size(1)==3 && rot.dim()==2 && rot.size(0)==n && rot.size(1)==4,"supply scales [N,3] and rotations [N,4], or covariance [N,6]");
        TORCH_CHECK((scales>0).all().item<bool>(),"scales must be positive activated standard deviations");
        auto norms=(rot*rot).sum(1);
        TORCH_CHECK((torch::abs(norms-1)<1e-4).all().item<bool>(),"native API expects normalized wxyz rotations");
    }
    TORCH_CHECK(t.dim()==2 && t.size(0)==4 && t.size(1)==4,"slice_to_world must have shape [4,4]");
    // 将仅 4×4 的矩阵读到主机检查仿射底行和面内独立性。
    // CUDA 内只需要前两列和第四列。此处主机读回同样意味着同步开销。
    auto cpu=t.cpu(); const float* T=cpu.data_ptr<float>();
    TORCH_CHECK(T[12]==0 && T[13]==0 && T[14]==0 && T[15]==1,"slice_to_world must be affine, bottom row [0,0,0,1]");
    double a[3]={T[0],T[4],T[8]},b[3]={T[1],T[5],T[9]},aa=0,bb=0;
    for(int k=0;k<3;++k) { aa+=a[k]*a[k]; bb+=b[k]*b[k]; }
    double cross[3]={a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
    double nn=0; for(double v:cross) nn+=v*v;
    TORCH_CHECK(aa>0 && bb>0 && nn>1e-12*aa*bb,"slice pixel axes are degenerate or nearly collinear");
    return {n,W,H,m.data_ptr<float>(),c.data_ptr<float>(),precomputed?nullptr:scales.data_ptr<float>(),precomputed?nullptr:rot.data_ptr<float>(),precomputed?cov.data_ptr<float>():nullptr,t.data_ptr<float>(),bg,tau,qmax,transition,jitter};
}
// 把 resize_ 封装成 Rasterizer 所需的字节分配回调。
// 对应 Tensor 存储由 PyTorch 管理，返回的数据指针只在本次调用及缓存存活期内使用。
static Allocator allocator(Tensor& t) {
    return [&t](size_t n) { t.resize_({(int64_t)n}); return reinterpret_cast<char*>(t.data_ptr<uint8_t>()); };
}
// PyBind 调用的前向入口：设置设备 guard，分配 image/radii/weight_sum 及三块缓存。
// 返回七项，其中缓存不直接暴露给公开 API，而是交给 autograd ctx 保存。
// 最后一项是 weight_sum；必须与 Python 解包顺序严格相同。
ForwardResult RasterizeGaussiansCUDA(const Tensor& m,const Tensor& c,const Tensor& scales,const Tensor& rot,const Tensor& cov,const Tensor& t,
    int H,int W,double bg,double tau,double cutoff,double transition,double jitter,bool debug) {
    TORCH_CHECK(m.is_cuda(),"means3D must be CUDA"); c10::cuda::CUDAGuard guard(m.device());
    auto p=parameters(m,c,scales,rot,cov,t,H,W,bg,tau,cutoff,transition,jitter);
    auto out=torch::empty({1,H,W},m.options()),weights=torch::empty_like(out);
    auto radii=torch::zeros({p.count},m.options().dtype(torch::kInt32));
    auto options=m.options().dtype(torch::kUInt8);
    auto geom=torch::empty({0},options),bin=torch::empty({0},options),im=torch::empty({0},options);
    cudaStream_t stream=c10::cuda::getCurrentCUDAStream(m.get_device()).stream();
    // 沿用调用方当前 stream，防止非默认 stream 上的数据尚未准备好就被其他 stream 读取。
    // 设备 guard 与 stream 管理是执行正确性的一部分，不能因代码精简而省略。
    int n=Rasterizer::forward(allocator(geom),allocator(bin),allocator(im),p,out.data_ptr<float>(),weights.data_ptr<float>(),radii.data_ptr<int>(),stream);
    if(debug) cudaCheck(cudaStreamSynchronize(stream));
    return {n,out,radii,geom,bin,im,weights};
}
// PyBind 调用的反向入口：检查 incoming 与缓存形状/设备，零初始化五类梯度。
// CUDA 的 atomicAdd 依赖零初始化；empty 分配后直接累加会产生未定义结果。
BackwardResult RasterizeGaussiansBackwardCUDA(const Tensor& m,const Tensor& c,const Tensor& scales,const Tensor& rot,const Tensor& cov,const Tensor& t,
    int H,int W,double bg,double tau,double cutoff,double transition,double jitter,bool debug,
    const Tensor& incoming,int instances,const Tensor& geometry,const Tensor& binning,const Tensor& image) {
    TORCH_CHECK(m.is_cuda(),"means3D must be CUDA"); c10::cuda::CUDAGuard guard(m.device());
    auto p=parameters(m,c,scales,rot,cov,t,H,W,bg,tau,cutoff,transition,jitter);
    tensorCheck(incoming,m,"grad_image");
    TORCH_CHECK(incoming.dim()==3 && incoming.size(0)==1 && incoming.size(1)==H && incoming.size(2)==W,"grad_image shape mismatch");
    TORCH_CHECK(instances>=0,"invalid instance count");
    // 核对缓存类型与最小字节数，防止将旧版本的 cache 误解释为新布局。
    // 这不是对任意外部伪造 buffer 的完整内容验证；private native API 应只由本包装调用。
    const Tensor* buffers[3]={&geometry,&binning,&image};
    size_t sizes[3]={required<GeometryState>(p.count),required<BinningState>(instances),required<ImageState>(size_t(H)*W)};
    for(int i=0;i<3;++i) TORCH_CHECK(buffers[i]->is_cuda() && buffers[i]->device()==m.device() && buffers[i]->scalar_type()==torch::kUInt8 && buffers[i]->is_contiguous() && size_t(buffers[i]->numel())>=sizes[i],"invalid forward cache");
    auto dm=torch::zeros_like(m),dc=torch::zeros_like(c),ds=torch::zeros_like(scales),dr=torch::zeros_like(rot);
    // 始终先产生 packed covariance 梯度，因为尺度/旋转链式求导也以它为中间量。
    // 公开 autograd 会按实际输入路径返回对应梯度，未用分支返回 None。
    auto dS=torch::zeros({p.count,6},m.options());
    Gradients g{dm.data_ptr<float>(),dc.data_ptr<float>(),dS.data_ptr<float>(),ds.data_ptr<float>(),dr.data_ptr<float>()};
    cudaStream_t stream=c10::cuda::getCurrentCUDAStream(m.get_device()).stream();
    Rasterizer::backward(p,instances,reinterpret_cast<char*>(geometry.data_ptr<uint8_t>()),reinterpret_cast<char*>(binning.data_ptr<uint8_t>()),reinterpret_cast<char*>(image.data_ptr<uint8_t>()),incoming.data_ptr<float>(),g,stream);
    if(debug) cudaCheck(cudaStreamSynchronize(stream));
    return {dm,dc,ds,dr,dS};
}
