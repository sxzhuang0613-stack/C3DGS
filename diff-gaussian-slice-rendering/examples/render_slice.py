"""Usage example only; requires building the CUDA extension first."""
import torch
from diff_gaussian_slice_rendering import GaussianRasterizationSettings, rasterize_gaussians

# 示例刻意创建极少量 Gaussian，展示参数单位与可微输入路径。
# 这是调用示例，不是完整训练器或已通过的正确性测试。
xyz = torch.tensor([[0., 0., 0.], [.3, -.2, .1]], device="cuda", requires_grad=True)
intensity = torch.tensor([[.4], [.9]], device="cuda", requires_grad=True)
log_scales = torch.full((2, 3), -2., device="cuda", requires_grad=True)
quaternion = torch.tensor([[1., 0., 0., 0.], [1., .1, 0., 0.]], device="cuda", requires_grad=True)
settings = GaussianRasterizationSettings.from_axis(
    image_height=128, image_width=128, axis=2, slice_idx=4, num_slices=9,
    bounds=(1., 1., 1.), background=0., background_weight=.01,
    kernel_cutoff=.001, cutoff_transition=2., covariance_jitter=1e-8,
)
image, radii, weight_sum = rasterize_gaussians(
    xyz, intensity, settings, scales=log_scales.exp(), rotations=quaternion,
)
# loss 只通过 image 传播；返回的 radii 和 weight_sum 不可微。
# 真实训练应将 image 与同一几何、同一强度单位的目标切片比较。
image.square().mean().backward()
print(image.shape, radii.shape, weight_sum.shape)
print("mean/intensity/log-scale/quaternion gradients:",
      xyz.grad, intensity.grad, log_scales.grad, quaternion.grad)

# Stage 2: packed covariance comes from your differentiable motion model.
# Do NOT detach it when constructing this tensor.
variance = torch.full((2, 3), .02, device="cuda", requires_grad=True)
z = torch.zeros_like(variance[:, 0])
covariance = torch.stack((variance[:, 0], z, z, variance[:, 1], z, variance[:, 2]), dim=1)
image2, _, _ = rasterize_gaussians(xyz, intensity, settings, cov3Ds_precomp=covariance)
image2.mean().backward()
print("Stage 2 variance gradients:", variance.grad)
