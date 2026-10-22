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
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// ----------------------------------------------------------
// 新增的辅助结构体和函数 (用于3D高斯计算)
// ----------------------------------------------------------
struct float3x3 { float m[3][3]; };

// 矩阵向量乘法
__device__ __forceinline__ float3 mat_vec_mult(const float3x3& M, const float3& v)
{
    return { M.m[0][0] * v.x + M.m[0][1] * v.y + M.m[0][2] * v.z,
             M.m[1][0] * v.x + M.m[1][1] * v.y + M.m[1][2] * v.z,
             M.m[2][0] * v.x + M.m[2][1] * v.y + M.m[2][2] * v.z };
}

// 向量点乘
__device__ __forceinline__ float dot(const float3& v1, const float3& v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

// 3x3 矩阵求逆 (解析法)
__device__ __forceinline__ float3x3 invert_3x3(const float* cov3D)
{
    float3x3 inv;
    float a = cov3D[0], b = cov3D[1], c = cov3D[2];
    float d = cov3D[3], e = cov3D[4], f = cov3D[5];

    float det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    float inv_det = 1.0f / (det + 1e-12f);

    inv.m[0][0] = (d * f - e * e) * inv_det;
    inv.m[0][1] = (c * e - b * f) * inv_det;
    inv.m[0][2] = (b * e - c * d) * inv_det;
    inv.m[1][0] = (c * e - b * f) * inv_det;
    inv.m[1][1] = (a * f - c * c) * inv_det;
    inv.m[1][2] = (b * c - a * e) * inv_det;
    inv.m[2][0] = (b * e - c * d) * inv_det;
    inv.m[2][1] = (b * c - a * e) * inv_det;
    inv.m[2][2] = (a * d - b * b) * inv_det;

    return inv;
}

// 新增：根据轴方向计算条件均值和协方差
__device__ __forceinline__ void compute_conditional_params(
    const float3& mu3D, const float* cov3D, const glm::vec3& cam_pos, int axis,
    float& mu_u, float& mu_v, float& slice_coord, float3& cov_2d)
{
    float t = cam_pos.x; // 默认为X轴
    if (axis == 1) t = cam_pos.y; // Y轴
    else if (axis == 2) t = cam_pos.z; // Z轴
    
    if (axis == 0) { // X轴切片：YZ平面
        slice_coord = mu3D.x;
        float t_minus_mu = t - mu3D.x;
        float sigma_xx = cov3D[0];
        float sigma_xy = cov3D[1]; 
        float sigma_xz = cov3D[2];
        float sigma_yy = cov3D[3];
        float sigma_yz = cov3D[4];
        float sigma_zz = cov3D[5];
        
        float inv_sigma_xx = 1.0f / (sigma_xx + 1e-8f);
        mu_u = mu3D.y + sigma_xy * t_minus_mu * inv_sigma_xx;
        mu_v = mu3D.z + sigma_xz * t_minus_mu * inv_sigma_xx;
        
        cov_2d.x = sigma_yy - (sigma_xy * sigma_xy / (sigma_xx + 1e-6f));
        cov_2d.y = sigma_yz - (sigma_xy * sigma_xz / (sigma_xx + 1e-6f));
        cov_2d.z = sigma_zz - (sigma_xz * sigma_xz / (sigma_xx + 1e-6f));
    }
    else if (axis == 1) { // Y轴切片：XZ平面
        slice_coord = mu3D.y;
        float t_minus_mu = t - mu3D.y;
        float sigma_xx = cov3D[0];
        float sigma_xy = cov3D[1];
        float sigma_xz = cov3D[2];
        float sigma_yy = cov3D[3];
        float sigma_yz = cov3D[4];
        float sigma_zz = cov3D[5];
        
        float inv_sigma_yy = 1.0f / (sigma_yy + 1e-8f);
        mu_u = mu3D.x + sigma_xy * t_minus_mu * inv_sigma_yy;
        mu_v = mu3D.z + sigma_yz * t_minus_mu * inv_sigma_yy;
        
        cov_2d.x = sigma_xx - (sigma_xy * sigma_xy / (sigma_yy + 1e-6f));
        cov_2d.y = sigma_xz - (sigma_xy * sigma_yz / (sigma_yy + 1e-6f));
        cov_2d.z = sigma_zz - (sigma_yz * sigma_yz / (sigma_yy + 1e-6f));
    }
    else { // Z轴切片：XY平面 (原始逻辑)
        slice_coord = mu3D.z;
        float t_minus_mu = t - mu3D.z;
        float sigma_xx = cov3D[0];
        float sigma_xy = cov3D[1];
        float sigma_xz = cov3D[2];
        float sigma_yy = cov3D[3];
        float sigma_yz = cov3D[4];
        float sigma_zz = cov3D[5];
        
        float inv_sigma_zz = 1.0f / (sigma_zz + 1e-8f);
        mu_u = mu3D.x + sigma_xz * t_minus_mu * inv_sigma_zz;
        mu_v = mu3D.y + sigma_yz * t_minus_mu * inv_sigma_zz;
        
        cov_2d.x = sigma_xx - (sigma_xz * sigma_xz / (sigma_zz + 1e-6f));
        cov_2d.y = sigma_xy - (sigma_xz * sigma_yz / (sigma_zz + 1e-6f));
        cov_2d.z = sigma_yy - (sigma_yz * sigma_yz / (sigma_zz + 1e-6f));
    }
}

// 新增：根据轴方向计算3D世界坐标
__device__ __forceinline__ float3 compute_3d_point(float x_world, float y_world, const glm::vec3& cam_pos, int axis)
{
    if (axis == 0) { // X轴切片：YZ平面
        return {cam_pos.x, x_world, y_world};
    }
    else if (axis == 1) { // Y轴切片：XZ平面
        return {x_world, cam_pos.y, y_world};
    }
    else { // Z轴切片：XY平面
        return {x_world, y_world, cam_pos.z};
    }
}

// ----------------------------------------------------------
// 辅助函数 (无需修改)
// ----------------------------------------------------------
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// 使用条件概率公式进行简化，符合我们的模型，保持不变
	glm::mat3 cov = glm::mat3(0.0f);
	float cov3d_5_safe = cov3D[5] + 1e-6;
	cov[0][0] = cov3D[0] - (cov3D[2] * cov3D[2] / cov3d_5_safe);
	cov[0][1] = cov3D[1] - (cov3D[2] * cov3D[4] / cov3d_5_safe);
	cov[1][0] = cov[0][1];
	cov[1][1] = cov3D[3] - (cov3D[4] * cov3D[4] / cov3d_5_safe);

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;

	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// ----------------------------------------------------------
// 1. preprocessCUDA 核函数 (支持多轴)
// ----------------------------------------------------------
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	int axis)  // 新增轴方向参数
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P) 
		return;

	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// if (cov3D_precomp == nullptr)
	// {
	// 	computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
	// }
	// const float* cov3D = (cov3D_precomp != nullptr) ? cov3D_precomp + idx * 6 : cov3Ds + idx * 6;
	constexpr float kCovarianceJitter = 1e-8f;

	// 将两种输入形式统一写入内部协方差缓冲区
	float* cov3D = cov3Ds + idx * 6;

	if (cov3D_precomp != nullptr)
	{
		#pragma unroll
		for (int k = 0; k < 6; ++k)
			cov3D[k] = cov3D_precomp[idx * 6 + k];
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3D);
	}

	// 对协方差本身加对角正则化，
	// 而不是在求逆时修改行列式。
	cov3D[0] += kCovarianceJitter;  // xx
	cov3D[3] += kCovarianceJitter;  // yy
	cov3D[5] += kCovarianceJitter;  // zz

	const float a = cov3D[0];
	const float b = cov3D[1];
	const float c = cov3D[2];
	const float d = cov3D[3];
	const float e = cov3D[4];
	const float f = cov3D[5];

	const float minor2 = a * d - b * b;
	const float det3 = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);

	// Sylvester 判据：协方差必须正定
	if (!(a > 0.0f) || !(minor2 > 0.0f) || !(det3 > 0.0f) || !isfinite(det3))
	{
		return;
	}

	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	
	// 根据轴方向计算条件均值和协方差
	float mu_u, mu_v, slice_coord;
	float3 cov_norm;
	compute_conditional_params(p_orig, cov3D, *cam_pos, axis, mu_u, mu_v, slice_coord, cov_norm);

    // 2. 将其重构为2x2矩阵
    glm::mat2 cov_norm_mat = glm::mat2(cov_norm.x, cov_norm.y, cov_norm.y, cov_norm.z);

    // 3. 定义从归一化空间到像素空间的缩放矩阵
	glm::mat2 scale_mat = glm::mat2(W / 2.0f, 0.0f, 0.0f, H / 2.0f);
    
    // 4. 执行变换，得到像素空间下的协方差矩阵: Sigma_pix = S * Sigma_norm * S
    glm::mat2 cov_pixel_mat = scale_mat * cov_norm_mat * scale_mat;
    
    // -- START: 在此处添加正则化项 --
    cov_pixel_mat[0][0] += 0.3f;
    cov_pixel_mat[1][1] += 0.3f;
    // -- END: 在此处添加正则化项 --

    // 5. 提取变换后的协方差分量，用于后续计算
    float3 cov = {cov_pixel_mat[0][0], cov_pixel_mat[0][1], cov_pixel_mat[1][1]};

	// 后续的计算现在使用的是像素空间下的协方差
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f) 
		return;
    float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	// float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

    // --- START: 新增代码：根据距离对半径和深度进行衰减 ---
    // 1. 计算高斯中心到切片的距离
    float t = cam_pos->x;
    if (axis == 1) t = cam_pos->y;
    else if (axis == 2) t = cam_pos->z;
    float dist = abs(t - slice_coord);

    // 2. 获取切片轴向的方差
    float sigma_gauss;
    if (axis == 0) { // X-axis
        sigma_gauss = cov3D[0]; // sigma_xx
    } else if (axis == 1) { // Y-axis
        sigma_gauss = cov3D[3]; // sigma_yy
    } else { // Z-axis
        sigma_gauss = cov3D[5]; // sigma_zz
    }
    
    // // 3. 计算衰减因子 p(z)
    // float p_z = expf(-0.5f * dist * dist / (sigma_gauss + 1e-6f));

    // // 4. 计算加权后的半径
	// // float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)) * p_z);
    
    // // --- START: 添加安全检查 ---
    // float base_radius = 3.f * sqrt(max(lambda1, lambda2));
    // float weighted_radius = base_radius * sqrt(p_z);
    
    // // 如果加权后的半径太小，直接跳过这个高斯点
    // if (weighted_radius < 1.0f) {
    //     return;
    // }
    
    // float my_radius = ceil(weighted_radius);
    // // --- END: 添加安全检查 ---
    
    // // 5. 在此处计算深度
    // depths[idx] = -dist;
    // // --- END: 新增代码 ---


    // 将世界坐标系下的条件均值转换到像素空间
    float2 point_image = { ((mu_u + 1.0f) * W - 1.0f) * 0.5f, 
                           ((mu_v + 1.0f) * H - 1.0f) * 0.5f };

	uint2 rect_min, rect_max;
	// 1. 确保法向方差有效
	if (!(sigma_gauss > 1e-12f) || !isfinite(sigma_gauss)) { return; }

	// 2. Gaussian 中心在目标切片上的法向衰减
	const float p_z = expf( -0.5f * dist * dist / sigma_gauss );

	// 3. Gaussian 在该切片上的最大未截断 alpha
	const float peak_alpha = opacities[idx] * p_z;

	// 与 renderCUDA 的 alpha threshold 保持一致
	constexpr float alpha_threshold = 1.0f / 255.0f;

	// 该 Gaussian 在整张切片上都没有有效贡献
	if (!(peak_alpha > alpha_threshold)) {
		return;
	}

	// 4. 求 alpha 有效支撑区域的 Mahalanobis 阈值
	const float cutoff_squared = 2.0f * logf( peak_alpha / alpha_threshold );

	// 5. 沿最大 covariance 特征值方向计算保守半径
	const float max_eigenvalue =
		fmaxf(lambda1, lambda2);

	if (!(max_eigenvalue > 0.0f) ||
		!isfinite(max_eigenvalue)) {
		return;
	}

	// 增加 1 pixel 安全余量，减少取整导致的遗漏
	const int my_radius = max(
		1,
		static_cast<int>(
			ceilf(
				sqrtf(
					max_eigenvalue * cutoff_squared
				)
			) + 1.0f
		)
	);

	// 6. 保留原来的切片距离排序策略
	depths[idx] = -dist;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) 
		return;

	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

    // 根据轴方向计算深度 (这部分代码已被移到前面)
    /*
    float t = cam_pos->x; // 默认为X轴
    if (axis == 1) t = cam_pos->y; // Y轴
    else if (axis == 2) t = cam_pos->z; // Z轴
    
    float dist = abs(t - slice_coord);
	depths[idx] = -dist;    
    */
	
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// ----------------------------------------------------------
// 2. renderCUDA 核函数 (支持多轴)
// ----------------------------------------------------------
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
    // --- 用于G3D计算的参数 ---
    const float* __restrict__ means3D,
    const float* __restrict__ cov3Ds,
	const glm::vec3* __restrict__ cam_pos,
	int axis,  // 新增轴方向参数
    // --- 结束 ---
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };
	
	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;
	
	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;
	
	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	// 新增的共享内存数组 (用于G3D计算)
	__shared__ float3 collected_means3D[BLOCK_SIZE];
	__shared__ float collected_cov3D[BLOCK_SIZE * 6];
	
	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[NUM_CHANNELS] = { 0 };
	
	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{		
	// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;
		
		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			if (coll_id >= 0) {
				collected_id[block.thread_rank()] = coll_id;
				collected_xy[block.thread_rank()] = points_xy_image[coll_id];
				collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
				// --- 在这里从全局内存读取3D高斯参数 ---
				collected_means3D[block.thread_rank()] = {means3D[coll_id * 3], means3D[coll_id * 3 + 1], means3D[coll_id * 3 + 2]};
				for (int j = 0; j < 6; j++)
					collected_cov3D[block.thread_rank() * 6 + j] = cov3Ds[coll_id * 6 + j];
			} else {
				collected_id[block.thread_rank()] = -1;
			}
		}
		else
		{
			collected_id[block.thread_rank()] = -1;
		}
		block.sync();
		
		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// 检查collected_id是否有效
			if (collected_id[j] < 0)
				continue;
				
			// Keep track of current position in range
			contributor++;

			// =================== G3D Alpha 计算逻辑 (支持多轴) ===================
			// 1. 将像素坐标转换回世界坐标系中的切片平面上的点
			float x_world = (2.0f * pixf.x + 1.0f) / W - 1.0f;
			float y_world = (2.0f * pixf.y + 1.0f) / H - 1.0f;
			float3 p_slice = compute_3d_point(x_world, y_world, *cam_pos, axis);

			// 2. 从共享内存获取高斯参数
			const float3 mu = collected_means3D[j];
			const float* cov3D = &collected_cov3D[j * 6];

			// 3. 计算 3D 偏差向量
			float3 d_3D = {p_slice.x - mu.x, p_slice.y - mu.y, p_slice.z - mu.z};

			// 4. 计算 3D 高斯 power
			float3x3 inv_cov = invert_3x3(cov3D);
			float3 v = mat_vec_mult(inv_cov, d_3D);
			float power = -0.5f * dot(d_3D, v);

			if (power > 0.0f) 
				continue;

			// 5. 计算最终 Alpha
			float G3D = expf(power);
			const float opacity = collected_conic_opacity[j].w;
			float alpha = min(0.99f, opacity * G3D);
			// ==========================================================

			if (alpha < 1.0f / 255.0f) 
				continue;
			float test_T = T * (1.f - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < NUM_CHANNELS; ch++)
				C[ch] += features[collected_id[j] * NUM_CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}
	
	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < NUM_CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

// ----------------------------------------------------------
// 3. FORWARD 命名空间中的包装函数 (支持多轴)
// ----------------------------------------------------------

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	int axis)  // 新增轴方向参数
{
	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P, D, M,
		means3D,
		scales, 
		scale_modifier, 
		rotations, 
		opacities, 
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		focal_x, focal_y, 
		tan_fovx, tan_fovy,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		axis);  // 传递轴方向参数
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* points_xy_image,
	const float* features,
	const float4* conic_opacity,
	const float* means3D,
	const float* cov3Ds,
	const glm::vec3* cam_pos,
	int axis,  // 新增轴方向参数
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS><<<grid, block>>>(
		ranges,
		point_list,
		W, H,
		points_xy_image,
		features,
		conic_opacity,
		means3D,
		cov3Ds,
		cam_pos,
		axis,  // 传递轴方向参数
		final_T,
		n_contrib,
		bg_color,
		out_color);
}