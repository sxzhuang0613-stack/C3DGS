
import torch
import random

# 按当前 Python 环境的正常导入规则使用 pip 安装的渲染库。
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
           rotations=None, covariance=None, bounds=None, raster_options=None):

    h, w = img_size

    settings = GaussianRasterizationSettings.from_axis(
                    image_height=h,
                    image_width=w,
                    axis=axis,
                    slice_idx=slice_idx,
                    num_slices=num_slices,
                    bounds=bounds if bounds is not None else getattr(Gaussians, "bounds", (1., 1., 1.)),
                    **(raster_options or {}),
                )

    xyz = Gaussians.get_xyz if xyz is None else xyz
    colors = Gaussians.get_color if colors is None else colors

    # 两种协方差参数化互斥，不能向新接口传空尺度占位。
    parameters = {}
    if covariance is None:
        parameters['scales'] = Gaussians.get_scaling if scales is None else scales
        # 库入口负责唯一一次可微单位化，避免模型属性与库包装重复构建归一化图。
        if rotations is None:
            rotations = Gaussians._rotation if hasattr(Gaussians,'_rotation') else Gaussians.get_rotation
        parameters['rotations'] = rotations
    else:
        parameters['cov3Ds_precomp'] = covariance_to_vector(covariance).contiguous()

    rendered_image, radii, weight_sum = rasterize_gaussians(
        means3D=xyz, intensities=colors, raster_settings=settings, **parameters
    )

    # rendered_image = torch.rand((1, h, w),requires_grad=True).cuda()
    # radii = xyz[:,0] ** 2


    return {"render": rendered_image[0],
            "visibility_filter" : radii > 0,
            "radii": radii,}


