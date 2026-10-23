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

#include "backward.h"
#include "auxiliary.h"
using namespace CudaRasterizer;
// 【像素损失 -> Gaussian 参数梯度】
// 每线程读取一个像素的上游梯度 g，以及同次 forward 保存的 D 和预测 I。
// 对于 I=(sum(w*c)+tau*b)/D，有 dL/dc=g*w/D，dL/dw=g*(c-I)/D。
// 同一 Gaussian 可被多个像素访问，最终参数梯度必须用 atomicAdd 累加。
__global__ void backwardCUDA(Parameters p,GeometryState geom,BinningState bin,ImageState img,const float* incoming,Gradients g) {
    int x=blockIdx.x*kTileWidth+threadIdx.x,y=blockIdx.y*kTileHeight+threadIdx.y;
    if(x>=p.width||y>=p.height) return;
    int pix=y*p.width+x;
    double point[3]; pixelPoint(p.slice_to_world,x,y,point);
    // scale=g/D。这里的 D 和 I 来自 forward 缓存，避免重复归一化造成不一致。
    // 背景、tau、矩阵和截断超参数在本版本均固定，不返回其梯度。
    double scale=incoming[pix]/img.denom[pix],pred=img.prediction[pix];
    uint2 range=img.ranges[blockIdx.y*gridDim.x+blockIdx.x];
    for(uint32_t j=range.x;j<range.y;++j) {
        uint32_t i=bin.list[j]; double d[3],v[3],w,slope;
        for(int a=0;a<3;++a) d[a]=point[a]-p.means[3*i+a];
        // 重复使用 forward 的二次型、支撑边界和 taper 函数。
        // 不能前向用平滑窗、反向却只用未加窗 Gaussian 的导数。
        if(!kernelWeight(quadratic(geom.precision+6*i,d,v),p,w,slope)) continue;
        // kernelWeight 返回 slope=-2*dw/dQ。
        // 因此 beta=(dL/dw)*slope；硬截断内部 slope=w，恢复文档中的 beta=w*g*(c-I)/D。
        // 设 v=S^-1(p-mu)：dL/dmu=beta*v，dL/dS=(beta/2)*v*v^T。
        double beta=scale*(p.intensities[i]-pred)*slope;
        atomicAdd(g.intensities+i,(float)(scale*w));
        for(int a=0;a<3;++a) atomicAdd(g.means+3*i+a,(float)(beta*v[a]));
        // packed 协方差的 xy/xz/yz 同时影响矩阵中的两个对称元素，
        // 所以它们的导数比单个矩阵元素导数多一倍。
        // 循环顺序恰好生成 xx,xy,xz,yy,yz,zz；不能随意改变存储顺序。
        int index=0;
        for(int a=0;a<3;++a) for(int b=a;b<3;++b)
            atomicAdd(g.covariance+6*i+index++,(float)(beta*v[a]*v[b]*(a==b?0.5:1.0)));
    }
}
// 【协方差梯度 -> 尺度和旋转梯度】
// 只在 scales+rotations 路径执行。预计算 covariance 路径直接把 packed 梯度返回上游。
// 本函数每线程独占一个 Gaussian 的尺度与四元数结果，因此不需要 atomicAdd。
__global__ void covarianceBackward(Parameters p,Gradients g) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=p.count) return;
    const float* c=g.covariance+6*i;
    // 先把 packed 梯度还原为矩阵梯度 G：非对角项需除以二。
    // 否则在下面的矩阵链式求导中又计算两次，会造成尺度/旋转梯度翻倍。
    double G[9]={c[0],c[1]*0.5,c[2]*0.5,c[1]*0.5,c[3],c[4]*0.5,c[2]*0.5,c[4]*0.5,c[5]};
    // S=R diag(s²) R^T，G 对称时有 dL/dR=2 G R diag(s²)。
    // 尺度梯度为 dL/ds_k=2*s_k*(R^T G R)kk。固定 jitter 不改变这两式。
    double R[9],dR[9]={0}; rotation(p.rotations+4*i,R);
    for(int a=0;a<3;++a) for(int b=0;b<3;++b)
        for(int k=0;k<3;++k) {
            double s=p.scales[3*i+b]; dR[3*a+b]+=2*G[3*a+k]*R[3*k+b]*s*s;
        }
    for(int b=0;b<3;++b) {
        double v=0; for(int a=0;a<3;++a) for(int k=0;k<3;++k) v+=R[3*a+b]*G[3*a+k]*R[3*k+b];
        g.scales[3*i+b]=(float)(2*p.scales[3*i+b]*v);
    }
    const float* q=p.rotations+4*i; double w=q[0],x=q[1],y=q[2],z=q[3];
    // Derivatives of the row-major R with respect to normalized (w,x,y,z).
    // Python performs normalization outside the custom Function, so autograd
    // supplies the normalization Jacobian exactly once.
    // J 的四个连续 9 元素块依次是 dR/dw、dR/dx、dR/dy、dR/dz。
    // 这里把归一化后的四元数当输入变量；Python 在 custom Function 外完成归一化，
    // PyTorch 会继续将本处梯度传到原始四元数，不应在 CUDA 再重复归一化求导。
    const double J[36]={
        0,-2*z,2*y, 2*z,0,-2*x, -2*y,2*x,0,
        0,2*y,2*z, 2*y,-4*x,-2*w, 2*z,2*w,-4*x,
        -4*y,2*x,2*w, 2*x,0,2*z, -2*w,2*z,-4*y,
        -4*z,-2*w,2*x, 2*w,-4*z,2*y, 2*x,2*y,0};
    for(int a=0;a<4;++a) {
        double v=0; for(int k=0;k<9;++k) v+=dR[k]*J[9*a+k];
        g.rotations[4*i+a]=(float)v;
    }
}
// 反向仍遍历当前 tile 的全部有效候选，没有旧 alpha 合成的逆序回放。
// Gaussian 顺序对数学结果无影响，但浮点累积次序可能带来小差异。
void BACKWARD::render(const Parameters& p,GeometryState g,BinningState b,ImageState im,const float* incoming,Gradients grads,dim3 grid,cudaStream_t s) {
    backwardCUDA<<<grid,dim3(kTileWidth,kTileHeight),0,s>>>(p,g,b,im,incoming,grads);
}
void BACKWARD::preprocess(const Parameters& p,Gradients g,cudaStream_t s) {
    if(p.count && !p.covariance) covarianceBackward<<<(p.count+255)/256,256,0,s>>>(p,g);
}
