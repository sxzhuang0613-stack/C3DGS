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

// 每像素仅反传二维中心、对称精度与 log 幅值；含平滑支撑窗的完整导数。
__global__ void backwardCUDA(Parameters p,GeometryState geom,BinningState bin,ImageState img,const float* incoming,Gradients g) {
    int x=blockIdx.x*kTileWidth+threadIdx.x,y=blockIdx.y*kTileHeight+threadIdx.y;
    if(x>=p.width||y>=p.height) return;
    double upstream=incoming[y*p.width+x];
    uint2 range=img.ranges[blockIdx.y*gridDim.x+blockIdx.x];
    for(uint32_t j=range.x;j<range.y;++j) {
        uint32_t i=bin.list[j];
        double dx=x-geom.means2D[i].x,dy=y-geom.means2D[i].y;
        const double* h=geom.precision+3*i;
        double Q=h[0]*dx*dx+2*h[1]*dx*dy+h[2]*dy*dy-2*geom.log_amplitude[i];
        double response,slope;
        if(!kernelWeight(Q,p,response,slope)) continue;
        double beta=upstream*p.intensities[i]*slope;
        atomicAdd(g.intensities+i,(float)(upstream*response));
        float* f=g.footprint+6*i;
        atomicAdd(f,(float)(beta*(h[0]*dx+h[1]*dy)));
        atomicAdd(f+1,(float)(beta*(h[1]*dx+h[2]*dy)));
        atomicAdd(f+2,(float)(-0.5*beta*dx*dx));
        atomicAdd(f+3,(float)(-beta*dx*dy));
        atomicAdd(f+4,(float)(-0.5*beta*dy*dy));
        atomicAdd(f+5,(float)beta);
    }
}

// 每 Gaussian 独占三维梯度，完整反传 m=H^-1 J^T P d 和稳定 q、logD。
// PSF 向量固定：dS_eff=dS；幅值产生 0.5*(S^-1-P) 项，不能只对扩大的核求导。
__global__ void footprintBackward(Parameters p,GeometryState geom,Gradients g) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=p.count || !geom.tiles_touched[i]) return;
    double S[9],K[9],E[9],P[9],logD;
    if(!gaussianMatrices(p,i,S,K,E,P,logD)) return; // forward 已校验相同参数。
    double J[3][2],d[3],n[3]; sliceGeometry(p,i,J,d,n);
    const double* h=geom.precision+3*i;
    double V[3]; inverse2D(h[0],h[1],h[2],V);
    const float* f=g.footprint+6*i;
    double t[2]={V[0]*f[0]+V[1]*f[1],V[1]*f[0]+V[2]*f[1]};
    double m[2]={geom.means2D[i].x,geom.means2D[i].y};
    // packed xy 同时作用于两个矩阵位置，在矩阵链式求导前各分一半。
    double GH[2][2]={{f[2],0.5*f[3]},{0.5*f[3],f[4]}};
    for(int a=0;a<2;++a) for(int b=0;b<2;++b)
        GH[a][b]-=0.5*(t[a]*m[b]+m[a]*t[b]);
    double jt[3]={0},nd=0,nen=0,GP[9]={0};
    for(int a=0;a<3;++a) {
        for(int k=0;k<2;++k) jt[a]+=J[a][k]*t[k];
        nd+=n[a]*d[a];
        for(int b=0;b<3;++b) nen+=n[a]*E[3*a+b]*n[b];
    }
    for(int a=0;a<3;++a) for(int b=0;b<3;++b) {
        for(int k=0;k<2;++k) for(int l=0;l<2;++l)
            GP[3*a+b]+=J[a][k]*GH[k][l]*J[b][l];
        GP[3*a+b]+=0.5*(jt[a]*d[b]+d[a]*jt[b]);
    }
    for(int a=0;a<3;++a) {
        double dm=-double(f[5])*nd/nen*n[a];
        for(int b=0;b<3;++b) dm+=P[3*a+b]*jt[b];
        g.means[3*i+a]=(float)dm;
    }
    double ratio=nd/nen; int index=0;
    for(int a=0;a<3;++a) for(int b=a;b<3;++b) {
        double gradient=0;
        for(int k=0;k<3;++k) for(int l=0;l<3;++l)
            gradient-=P[3*a+k]*GP[3*k+l]*P[3*l+b];
        gradient+=0.5*double(f[5])*(K[3*a+b]-P[3*a+b]+ratio*ratio*n[a]*n[b]);
        g.covariance[6*i+index++]=(float)(gradient*(a==b?1:2));
    }
}

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

void BACKWARD::render(const Parameters& p,GeometryState g,BinningState b,ImageState im,const float* incoming,Gradients grads,dim3 grid,cudaStream_t s) {
    backwardCUDA<<<grid,dim3(kTileWidth,kTileHeight),0,s>>>(p,g,b,im,incoming,grads);
}
void BACKWARD::preprocess(const Parameters& p,GeometryState geom,Gradients g,cudaStream_t s) {
    if(!p.count) return;
    footprintBackward<<<(p.count+255)/256,256,0,s>>>(p,geom,g);
    if(!p.covariance) covarianceBackward<<<(p.count+255)/256,256,0,s>>>(p,g);
}
