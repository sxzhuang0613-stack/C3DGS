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

// 每 Gaussian 完成厚度积分及补平方，像素阶段不再计算三维二次型。
__global__ void preprocessCUDA(Parameters p,GeometryState g,int* radii,dim3 grid) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=p.count) return;
    radii[i]=0; g.tiles_touched[i]=0; g.rectangles[i]=make_uint4(0,0,0,0);
    double S[9],K[9],E[9],P[9],logD;
    if(!gaussianMatrices(p,i,S,K,E,P,logD)) { atomicExch(g.invalid,1); return; }
    double J[3][2],d[3],n[3]; sliceGeometry(p,i,J,d,n);
    double H[3]={0},rhs[2]={0},nd=0,nen=0;
    for(int a=0;a<3;++a) {
        nd+=n[a]*d[a];
        for(int b=0;b<3;++b) {
            H[0]+=J[a][0]*P[3*a+b]*J[b][0];
            H[1]+=J[a][0]*P[3*a+b]*J[b][1];
            H[2]+=J[a][1]*P[3*a+b]*J[b][1];
            rhs[0]+=J[a][0]*P[3*a+b]*d[b];
            rhs[1]+=J[a][1]*P[3*a+b]*d[b];
            nen+=n[a]*E[3*a+b]*n[b];
        }
    }
    if(!(nen>0)) { atomicExch(g.invalid,1); return; }
    // q 避免 d^T P d-g^T H^-1 g 的大数相减；logD 保留归一化 PSF 的幅值修正。
    double q=nd*nd/nen,base=q+logD,V[3];
    if(!isfinite(base)||!inverse2D(H[0],H[1],H[2],V)) { atomicExch(g.invalid,1); return; }
    if(base>=p.qmax) return;
    double mx=V[0]*rhs[0]+V[1]*rhs[1],my=V[1]*rhs[0]+V[2]*rhs[1];
    double lambda=0.5*(V[0]+V[2]+hypot(V[0]-V[2],2*V[1]));
    double radius=ceil(sqrt(lambda*(p.qmax-base)))+1;
    if(!isfinite(radius)||!isfinite(mx)||!isfinite(my)) { atomicExch(g.invalid,1); return; }
    unsigned x0=(unsigned)fmin(double(grid.x),fmax(0.0,floor((mx-radius)/kTileWidth)));
    unsigned y0=(unsigned)fmin(double(grid.y),fmax(0.0,floor((my-radius)/kTileHeight)));
    unsigned x1=(unsigned)fmin(double(grid.x),fmax(0.0,floor((mx+radius)/kTileWidth)+1));
    unsigned y1=(unsigned)fmin(double(grid.y),fmax(0.0,floor((my+radius)/kTileHeight)+1));
    if(x1<=x0||y1<=y0) return;
    g.means2D[i]=make_double2(mx,my);
    for(int a=0;a<3;++a) g.precision[3*i+a]=H[a];
    g.log_amplitude[i]=-0.5*base;
    g.rectangles[i]=make_uint4(x0,y0,x1,y1);
    g.tiles_touched[i]=uint64_t(x1-x0)*(y1-y0);
    radii[i]=(int)fmin(radius,double(INT_MAX));
}

// 背景仅加一次；weight_sum 是不含有符号强度的响应和，不是归一化分母。
__global__ void renderCUDA(Parameters p,GeometryState g,BinningState bin,ImageState img,float* out,float* weights) {
    int x=blockIdx.x*kTileWidth+threadIdx.x,y=blockIdx.y*kTileHeight+threadIdx.y;
    if(x>=p.width||y>=p.height) return;
    uint2 range=img.ranges[blockIdx.y*gridDim.x+blockIdx.x];
    double prediction=p.background,sum=0;
    for(uint32_t j=range.x;j<range.y;++j) {
        uint32_t i=bin.list[j];
        double dx=x-g.means2D[i].x,dy=y-g.means2D[i].y;
        const double* h=g.precision+3*i;
        double Q=h[0]*dx*dx+2*h[1]*dx*dy+h[2]*dy*dy-2*g.log_amplitude[i];
        double response,slope;
        if(!kernelWeight(Q,p,response,slope)) continue;
        prediction+=p.intensities[i]*response; sum+=response;
    }
    int pixel=y*p.width+x;
    out[pixel]=(float)prediction; weights[pixel]=(float)sum;
}
void FORWARD::preprocess(const Parameters& p,GeometryState g,int* radii,dim3 grid,cudaStream_t s) {
    if(p.count) preprocessCUDA<<<(p.count+255)/256,256,0,s>>>(p,g,radii,grid);
}
void FORWARD::render(const Parameters& p,GeometryState g,BinningState b,ImageState im,float* out,float* w,dim3 grid,cudaStream_t s) {
    renderCUDA<<<grid,dim3(kTileWidth,kTileHeight),0,s>>>(p,g,b,im,out,w);
}
