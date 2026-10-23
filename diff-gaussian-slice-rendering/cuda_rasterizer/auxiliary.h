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
// 将对称 3×3 矩阵的六参数表示展开为 row-major 九元素表示。
// 当前主路径直接读取 packed precision；本工具保留作显式矩阵操作辅助。
__device__ inline void unpack(const double* s,double* a) {
    a[0]=s[0]; a[1]=a[3]=s[1]; a[2]=a[6]=s[2];
    a[4]=s[3]; a[5]=a[7]=s[4]; a[8]=s[5];
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
// T 按 C-contiguous row-major 布局读取。
// 第 r 行给出 p_r=T[r,0]*column+T[r,1]*row+T[r,3]。
// 第三列乘以 0，所以这里没有层厚、投影深度或沿法向积分。
__device__ inline void pixelPoint(const float* t,int j,int k,double* p) {
    for(int a=0;a<3;++a) p[a]=double(t[4*a])*j+double(t[4*a+1])*k+t[4*a+3];
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
// K 为 packed precision，先算 v=Kd，再返回 Q=d^T v。
// 同一函数同时提供核值和反向所需 v，保证两者使用相同的矩阵及索引约定。
__device__ inline double quadratic(const double* packed,const double* d,double* v) {
    v[0]=packed[0]*d[0]+packed[1]*d[1]+packed[2]*d[2];
    v[1]=packed[1]*d[0]+packed[3]*d[1]+packed[4]*d[2];
    v[2]=packed[2]*d[0]+packed[4]*d[1]+packed[5]*d[2];
    return d[0]*v[0]+d[1]*v[1]+d[2]*v[2];
}
