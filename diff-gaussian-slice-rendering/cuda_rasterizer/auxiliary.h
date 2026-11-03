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
#include "config.h"
#include "rasterizer.h"
#include <cmath>
#include <stdexcept>
#include <string>
// 统一检查 CUDA runtime/CUB 返回状态。这里抛出的 C++ 异常由 PyBind 传回 Python。
// 它不是 GPU 数值正确性的证明，debug 同步也不能替代数值梯度验证。
inline void cudaCheck(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
// 四元数约定是 (w,x,y,z)，输出为 row-major 旋转矩阵。
// 这组标量公式替代原 glm::mat3/glm::vec4 操作，避免依赖 GLM 的构造/存储约定。
// 函数假设 q 已单位化；公开 Python 包装会可微归一化，native 层也会检查范数。
__device__ inline void rotation(const float* q,double* r) {
    const double w=q[0],x=q[1],y=q[2],z=q[3];
    r[0]=1-2*(y*y+z*z); r[1]=2*(x*y-w*z); r[2]=2*(x*z+w*y);
    r[3]=2*(x*y+w*z); r[4]=1-2*(x*x+z*z); r[5]=2*(y*z-w*x);
    r[6]=2*(x*z-w*y); r[7]=2*(y*z+w*x); r[8]=1-2*(x*x+y*y);
}
// Cholesky solves; no determinant+epsilon inverse. The identical effective
// covariance (including jitter) defines the bound, forward and backward.
// 使用 S=L L^T 的 Cholesky 分解求逆：先前代解 Ly=e_c，再回代解 L^T x=y。
// 对 c=0,1,2 分别求解，得到逆矩阵的三列。所有中间量使用 double。
// 正定性检查针对加入 jitter 后的有效协方差；失败返回 false，不静默修复矩阵。
// 因此输入本身应合理；不能依靠很大的 jitter 将任意不定矩阵变成有意义的协方差。
__device__ inline bool inverseSPD(const double* a,double* k) {
    double l[9]={0};
    for(int i=0;i<3;++i) for(int j=0;j<=i;++j) {
        double v=a[3*i+j];
        for(int t=0;t<j;++t) v-=l[3*i+t]*l[3*j+t];
        if(i==j) { if(!(v>0)||!isfinite(v)) return false; l[3*i+j]=sqrt(v); }
        else l[3*i+j]=v/l[3*j+j];
    }
    // 逐列求逆，而不是写带任意 epsilon 的行列式/伴随矩阵闭式表达式。
    // 矩阵维数固定为 3，局部数组长度也固定，避免动态矩阵库依赖。
    for(int c=0;c<3;++c) {
        double y[3],x[3];
        for(int i=0;i<3;++i) {
            double v=(i==c?1.0:0.0);
            for(int j=0;j<i;++j) v-=l[3*i+j]*y[j];
            y[i]=v/l[3*i+i];
        }
        for(int i=2;i>=0;--i) {
            double v=y[i]; for(int j=i+1;j<3;++j) v-=l[3*j+i]*x[j];
            x[i]=v/l[3*i+i];
        }
        for(int i=0;i<3;++i) k[3*i+c]=x[i];
    }
    for(int i=0;i<9;++i) if(!isfinite(k[i])) return false;
    return true;
}
// S 包含固定 jitter；PSF 是 rank-one 协方差 vv^T，不是逐点可学习属性。
// 幅值和有效核使用同一个 S；v=0 时精确复用点采样的逆矩阵。
__device__ inline bool gaussianMatrices(const CudaRasterizer::Parameters& p,int i,
    double* S,double* K,double* E,double* P,double& logD) {
    for(int a=0;a<9;++a) S[a]=0;
    if(p.covariance) {
        const float* c=p.covariance+6*i;
        S[0]=c[0]; S[1]=S[3]=c[1]; S[2]=S[6]=c[2];
        S[4]=c[3]; S[5]=S[7]=c[4]; S[8]=c[5];
    } else {
        double R[9]; rotation(p.rotations+4*i,R);
        for(int a=0;a<3;++a) for(int b=0;b<3;++b) for(int c=0;c<3;++c) {
            double scale=p.scales[3*i+c];
            S[3*a+b]+=R[3*a+c]*R[3*b+c]*scale*scale;
        }
    }
    S[0]+=p.jitter; S[4]+=p.jitter; S[8]+=p.jitter;
    if(!inverseSPD(S,K)) return false;
    double vkv=0; bool point=true;
    for(int a=0;a<3;++a) {
        point=point && p.psf_vector[a]==0;
        for(int b=0;b<3;++b) {
            double va=p.psf_vector[a],vb=p.psf_vector[b];
            E[3*a+b]=S[3*a+b]+va*vb;
            vkv+=va*K[3*a+b]*vb;
        }
    }
    logD=log1p(vkv);
    if(!isfinite(logD)) return false;
    if(point) { for(int a=0;a<9;++a) P[a]=K[a]; return true; }
    return inverseSPD(E,P);
}
// J 为像素列/行的三维位移，d=mu-o；几何法向仅用于稳定计算最小距离。
__device__ inline void sliceGeometry(const CudaRasterizer::Parameters& p,int i,
    double J[3][2],double* d,double* n) {
    for(int a=0;a<3;++a) {
        J[a][0]=p.slice_to_world[4*a]; J[a][1]=p.slice_to_world[4*a+1];
        d[a]=double(p.means[3*i+a])-p.slice_to_world[4*a+3];
    }
    n[0]=J[1][0]*J[2][1]-J[2][0]*J[1][1];
    n[1]=J[2][0]*J[0][1]-J[0][0]*J[2][1];
    n[2]=J[0][0]*J[1][1]-J[1][0]*J[0][1];
}
// 固定 2x2 SPD Cholesky；不加经验 epsilon 修改有效核。
__device__ inline bool inverse2D(double h00,double h01,double h11,double* V) {
    if(!(h00>0)) return false;
    double l=sqrt(h00),r=h01/l,schur=h11-r*r;
    if(!(schur>0)||!isfinite(schur)) return false;
    V[2]=1/schur; V[1]=-r/l*V[2]; V[0]=1/h00+(r/l)*(r/l)*V[2];
    return isfinite(V[0]) && isfinite(V[1]) && isfinite(V[2]);
}
// Returns w and -2 dw/dQ. For transition=0 this is the documented hard
// cutoff. For transition>0 a C1 compact taper removes its jump.
// 统一核函数同时被 forward/backward 调用。
// 硬截断模式：Q<Qmax 时 w=exp(-Q/2)。平滑模式：在最外一段乘 C1 三次窗。
// 返回值 slope=-2 dw/dQ，便于复用 Gaussian 均值/协方差的链式公式。
// 这里判断的是未加窗 Q；对加窗后的 w 再设阈值会重新产生不连续边界。
__device__ inline bool kernelWeight(double Q,const CudaRasterizer::Parameters& p,double& w,double& slope) {
    if(!isfinite(Q)||Q>=p.qmax) return false;
    Q=fmax(Q,0.0);
    double h=1,dh=0;
    if(p.transition>0 && Q>p.qmax-p.transition) {
        double t=(Q-(p.qmax-p.transition))/p.transition;
        // h=1-3t²+2t³，在区间两端一阶导数为零。
        // dh 是对 Q 的导数，故必须除以 transition，而不是仅对 t 求导。
        h=(1-t)*(1-t)*(1+2*t); dh=-6*t*(1-t)/p.transition;
    }
    double e=exp(-0.5*Q); w=e*h; slope=e*(h-2*dh); return true;
}
