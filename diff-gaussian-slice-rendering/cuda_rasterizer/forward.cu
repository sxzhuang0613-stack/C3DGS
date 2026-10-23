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

#include "forward.h"
#include "auxiliary.h"
#include <climits>
using namespace CudaRasterizer;
// 【阶段一：每个线程负责一个 Gaussian】
// 输入是三维均值、强度，以及两种协方差参数化之一；这里不计算最终像素强度。
// 本阶段构造有效协方差及其逆，计算 Gaussian 与当前切片平面的交截核，
// 再求包含该核截断支撑域的 tile 矩形。只有矩形内 tiles 才获得该 Gaussian 的候选记录。
// 这里计算的是三维核在平面上的限制，不是沿视线积分，也不是相机投影。
__global__ void preprocessCUDA(Parameters p,GeometryState g,int* radii,dim3 grid) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=p.count) return;
    // 先写入“无候选”的安全状态。任何提前 return 都不会留下未初始化的 tile 计数。
    // radii 是包围半径诊断量，不是实际对像素的贡献，也不参与梯度。
    radii[i]=0; g.tiles_touched[i]=0; g.rectangles[i]=make_uint4(0,0,0,0);
    double S[9]={0},K[9];
    // 分支 A：直接读取 packed covariance，顺序为 xx,xy,xz,yy,yz,zz。
    // 分支 B：由 R diag(s²) R^T 构造协方差；scales 是标准差，不是方差或 log scale。
    // 两条路径随后使用相同的 jitter、逆矩阵和切片计算。
    if(p.covariance) {
        const float* c=p.covariance+6*i;
        S[0]=c[0]; S[1]=S[3]=c[1]; S[2]=S[6]=c[2];
        S[4]=c[3]; S[5]=S[7]=c[4]; S[8]=c[5];
    } else {
        double R[9]; rotation(p.rotations+4*i,R);
        for(int a=0;a<3;++a) for(int b=0;b<3;++b)
            for(int c=0;c<3;++c) {
                double s=p.scales[3*i+c]; S[3*a+b]+=R[3*a+c]*R[3*b+c]*s*s;
            }
    }
    // 固定对角稳定项的单位是模型坐标的平方。
    // 它加入真实核使用的协方差，而不是仅扩大包围盒；因此三阶段必须一致。
    // 固定 jitter 对原协方差的导数为恒等映射，不额外产生可学习参数。
    S[0]+=p.jitter; S[4]+=p.jitter; S[8]+=p.jitter;
    if(!inverseSPD(S,K)) { atomicExch(g.invalid,1); return; }
    // 只缓存逆协方差的六个独立元素，供 forward/backward 共同读取。
    // 使用 double 缓存降低求逆和二次型计算中的舍入误差；代价是更多带宽与显存。
    double* packed=g.precision+6*i;
    packed[0]=K[0]; packed[1]=K[1]; packed[2]=K[2];
    packed[3]=K[4]; packed[4]=K[5]; packed[5]=K[8];
    // T 的第 0/1 列分别是增加一个列/行索引时的模型空间位移，第四列是首像素中心 o。
    // 记 A=[a,b]、d=mu-o、K=S^-1。切片坐标 u=[column,row] 中的条件核满足：
    // H=A^T K A，h=A^T K d，V=H^-1，m=Vh。
    // m 已经是像素索引，不能再乘 W/2 或 H/2。
    double a[3],b[3],d[3],n[3];
    for(int j=0;j<3;++j) {
        a[j]=p.slice_to_world[4*j]; b[j]=p.slice_to_world[4*j+1];
        d[j]=double(p.means[3*i+j])-p.slice_to_world[4*j+3];
    }
    // 法向 n=a×b 不必单位化，后面的距离商会消去 n 的长度。
    // 这样即使像素间距不同或平面基向量带缩放，几何定义仍一致。
    n[0]=a[1]*b[2]-a[2]*b[1]; n[1]=a[2]*b[0]-a[0]*b[2]; n[2]=a[0]*b[1]-a[1]*b[0];
    double nsn=0,nd=0,H00=0,H01=0,H11=0,h0=0,h1=0;
    for(int j=0;j<3;++j) {
        nd+=n[j]*d[j];
        for(int k=0;k<3;++k) {
            nsn+=n[j]*S[3*j+k]*n[k];
            H00+=a[j]*K[3*j+k]*a[k]; H01+=a[j]*K[3*j+k]*b[k];
            H11+=b[j]*K[3*j+k]*b[k];
            h0+=a[j]*K[3*j+k]*d[k]; h1+=b[j]*K[3*j+k]*d[k];
        }
    }
    // q 是整张无限平面上最小的 Mahalanobis 平方距离。
    // 用法向商避免 d^T K d - h^T H^-1 h 的大数相减。
    // 若 q>=Qmax，平面上任何位置都不可能进入该 Gaussian 的截断支撑域。
    double q=nd*nd/nsn; // stable distance to plane, no subtraction of large terms
    if(!(nsn>0)||!isfinite(q)||!(H00>0)) { atomicExch(g.invalid,1); return; }
    if(q>=p.qmax) return;
    // 对二维 H 使用 Cholesky/Schur 补求逆，获得像素坐标中的 V。
    // 若矩阵退化或数值上不再正定，记录 invalid，主机端会报错。
    // 不以任意 epsilon 修改分母来掩盖不合法的几何。
    double l0=sqrt(H00),l10=H01/l0,schur=H11-l10*l10;
    if(!(schur>0)||!isfinite(schur)) { atomicExch(g.invalid,1); return; }
    double V11=1/schur,V01=-l10/l0*V11,V00=1/H00+(l10/l0)*(l10/l0)*V11;
    double mx=V00*h0+V01*h1,my=V01*h0+V11*h1;
    // 椭圆支撑域满足 (u-m)^T V^-1 (u-m)<Qmax-q。
    // 其外接圆半径为 sqrt(lambda_max(V)*(Qmax-q))；向上取整并加 1 像素余量。
    // 该圆用于保守分桶，不会替代像素处实际计算的三维核。
    double lambda=0.5*(V00+V11+hypot(V00-V11,2*V01));
    double r=ceil(sqrt(lambda*(p.qmax-q)))+1;
    if(!isfinite(r)||!isfinite(mx)||!isfinite(my)) { atomicExch(g.invalid,1); return; }
    // Compute and clamp in double before converting to integers, including
    // Gaussians centered outside the image with a large support radius.
    // 先在 double 中计算和裁剪 tile 边界，再转成无符号整数，
    // 避免中心位于图像外很远时直接 float->int 溢出。
    // 矩形采用半开区间 [x0,x1)×[y0,y1)，复制记录和计数必须使用同一约定。
    unsigned x0=(unsigned)fmin(double(grid.x),fmax(0.0,floor((mx-r)/kTileWidth)));
    unsigned y0=(unsigned)fmin(double(grid.y),fmax(0.0,floor((my-r)/kTileHeight)));
    unsigned x1=(unsigned)fmin(double(grid.x),fmax(0.0,floor((mx+r)/kTileWidth)+1));
    unsigned y1=(unsigned)fmin(double(grid.y),fmax(0.0,floor((my+r)/kTileHeight)+1));
    if(x1<=x0||y1<=y0) return;
    g.means2D[i]=make_double2(mx,my); g.rectangles[i]=make_uint4(x0,y0,x1,y1);
    g.tiles_touched[i]=uint64_t(x1-x0)*(y1-y0);
    radii[i]=(int)fmin(r,double(INT_MAX));
}
// 【阶段二：每个线程负责一个像素】
// 一个 CUDA block 对应 16×16 像素的 tile；只遍历属于该 tile 的候选 Gaussian。
// 当前实现直接读取全局内存，不使用旧版本的 shared-memory 分批加载。
// 这是实现上的简化；虽然保留 tile 候选裁剪，却不能据此推断性能与原版等价。
__global__ void renderCUDA(Parameters p,GeometryState g,BinningState bin,ImageState img,float* out,float* weights) {
    int x=blockIdx.x*kTileWidth+threadIdx.x,y=blockIdx.y*kTileHeight+threadIdx.y;
    if(x>=p.width||y>=p.height) return;
    int pix=y*p.width+x; uint2 range=img.ranges[blockIdx.y*gridDim.x+blockIdx.x];
    // 按同一张 T 得到该像素中心的三维位置，和预处理的 a,b,o 完全一致。
    // 随后使用原三维 Gaussian 二次型，而不是把条件二维核当成另一套强度模型。
    double point[3]; pixelPoint(p.slice_to_world,x,y,point);
    // sw 累积有效权重，swc 累积强度加权和；不包含背景伪样本。
    // 无 alpha、不更新透射率、不因遮挡提前终止；所有有效候选均有机会贡献。
    double sw=0,swc=0;
    for(uint32_t j=range.x;j<range.y;++j) {
        uint32_t i=bin.list[j]; double d[3],v[3],w,slope;
        for(int a=0;a<3;++a) d[a]=point[a]-p.means[3*i+a];
        double Q=quadratic(g.precision+6*i,d,v);
        if(!kernelWeight(Q,p,w,slope)) continue;
        sw+=w; swc+=w*p.intensities[i];
    }
    // 归一化混合：D=tau+sum(w)，I=(sum(w*c)+tau*b)/D。
    // 拆成两项计算，避免单独形成 tau*b 的不必要溢出。
    // 空候选时 sw=swc=0，因此图像恰为 background；tau>0 保证分母不为零。
    double den=p.tau+sw,pred=swc/den+(p.tau/den)*p.background;
    // 保存同一次前向的 double 分母与预测值，供商法则反向使用。
    // 公开 weight_sum 不含 tau，且被 Python 标记为不可微；它不能代替内部 denom。
    img.denom[pix]=den; img.prediction[pix]=pred;
    out[pix]=(float)pred; weights[pix]=(float)sw;
}
// 主机侧薄封装：显式传递 PyTorch 当前 stream，不默认切换到 stream 0。
// 零 Gaussian 不发射零维 grid；像素 render 仍需要执行，以写出背景图。
void FORWARD::preprocess(const Parameters& p,GeometryState g,int* radii,dim3 grid,cudaStream_t stream) {
    if(p.count) preprocessCUDA<<<(p.count+255)/256,256,0,stream>>>(p,g,radii,grid);
}
void FORWARD::render(const Parameters& p,GeometryState g,BinningState b,ImageState im,float* out,float* w,dim3 grid,cudaStream_t s) {
    renderCUDA<<<grid,dim3(kTileWidth,kTileHeight),0,s>>>(p,g,b,im,out,w);
}
