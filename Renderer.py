
import torch
import math
import torch.nn.functional as F
from diff_gaussian_slice_rendering import rasterize_gaussians, GaussianRasterizationSettings


def covariance_to_vector(covariance):
    """将 [N, 3, 3] 对称协方差压缩为 CUDA 渲染器使用的 [N, 6]。"""
    if covariance.ndim == 2 and covariance.shape[-1] == 6:
        return covariance
    if covariance.ndim != 3 or covariance.shape[-2:] != (3, 3):
        raise ValueError("covariance 必须为 [N, 3, 3] 或 [N, 6]")
    return torch.stack([
        covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 0, 2],
        covariance[:, 1, 1], covariance[:, 1, 2], covariance[:, 2, 2],
    ], dim=-1)


def render(Gaussians, img_size, slice_idx, axis, num_slices,
           xyz=None, colors=None, opacities=None, scales=None,
           rotations=None, covariance=None):

    device = Gaussians.get_xyz.device
    dtype = Gaussians.get_xyz.dtype
    h, w = img_size

    settings = GaussianRasterizationSettings.from_axis(
                    image_height=h,
                    image_width=w,
                    axis=2,
                    slice_idx=slice_idx,
                    num_slices=num_slices,
                    bounds=(1.0, 1.0, 1.0),
                )

    xyz = Gaussians.get_xyz if xyz is None else xyz
    colors = Gaussians.get_color if colors is None else colors

    if covariance is None:
        scales = Gaussians.get_scaling if scales is None else scales
        rotations = Gaussians.get_rotation if rotations is None else rotations
        covariance_precomp = torch.empty(0, dtype=dtype, device=device)
    else:
        # GaussianEvolution 直接产生 Sigma(t)，避免再次分解为尺度与四元数。
        covariance_precomp = covariance_to_vector(covariance).contiguous()
        scales = torch.empty(0, dtype=dtype, device=device)
        rotations = torch.empty(0, dtype=dtype, device=device)

    rendered_image, radii, weight_sum = rasterize_gaussians(
                means3D=xyz,
                intensities=colors,
                raster_settings=settings,
                scales=scales,
                rotations=rotations,
            )

    return {"render": rendered_image[0],
            "visibility_filter" : radii > 0,
            "radii": radii,}


