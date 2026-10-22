
import torch
import math
import torch.nn.functional as F
from diff_gaussian_rasterization_slice import rasterize_gaussians, GaussianRasterizationSettings

def get_camera_position(slice_position, axis_idx, total_slices):
    """Calculate camera position based on axis direction"""
    # Normalize slice position to [-1, 1] range
    normalized_pos = (slice_position / max(1, total_slices - 1)) * 2.0 - 1.0
    
    if axis_idx == 0:  # X-axis slice
        return torch.tensor([normalized_pos, 0.0, 0.0])
    elif axis_idx == 1:  # Y-axis slice
        return torch.tensor([0.0, normalized_pos, 0.0])
    else:  # Z-axis slice
        return torch.tensor([0.0, 0.0, normalized_pos])

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
    background = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)
    dummy_matrix = torch.eye(4, dtype=dtype, device=device)
    cam_pos = get_camera_position(slice_idx, axis, num_slices).to(device=device, dtype=dtype)

    raster_settings = GaussianRasterizationSettings(
                        image_height=img_size[0],
                        image_width=img_size[1],
                        tanfovx=1.0,
                        tanfovy=1.0,
                        bg=background,
                        scale_modifier=1.0,
                        viewmatrix=dummy_matrix,
                        projmatrix=dummy_matrix,
                        sh_degree=0,
                        campos=cam_pos,
                        prefiltered=False,
                        debug=False,
                        antialiasing=False,
                        axis=axis
                    )

    xyz = Gaussians.get_xyz if xyz is None else xyz
    colors = Gaussians.get_color if colors is None else colors
    opacities = Gaussians.get_opacity if opacities is None else opacities
    screenspace_points = torch.zeros_like(xyz, requires_grad=True) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if covariance is None:
        scales = Gaussians.get_scaling if scales is None else scales
        rotations = Gaussians.get_rotation if rotations is None else rotations
        covariance_precomp = torch.empty(0, dtype=dtype, device=device)
    else:
        # GaussianEvolution 直接产生 Sigma(t)，避免再次分解为尺度与四元数。
        covariance_precomp = covariance_to_vector(covariance).contiguous()
        scales = torch.empty(0, dtype=dtype, device=device)
        rotations = torch.empty(0, dtype=dtype, device=device)

    empty_tensor = torch.empty(0, dtype=dtype, device=device)

    # Call rasterize_gaussians
    rendered_image, radii, _ = rasterize_gaussians(
        xyz,
        screenspace_points,  # means2D (not used)
        empty_tensor,              # sh
        colors,                             # colors_precomp
        opacities,
        scales,
        rotations,
        covariance_precomp,
        raster_settings
    )
    radii_full = radii

    return {"render": rendered_image[0],
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii_full > 0,
            "radii": radii_full,}


