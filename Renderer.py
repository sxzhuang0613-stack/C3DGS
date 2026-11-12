"""Stage1 的归一化高斯切片包装；参数仍保存在病例模型坐标中。"""
import math
import numbers

import torch

from Gaussians.utils import covariance_from_scaling_rotation


def covariance_to_vector(covariance):
    """将 [N,3,3] 对称协方差压缩为 [xx,xy,xz,yy,yz,zz]。"""
    if covariance.ndim == 2 and covariance.shape[-1] == 6:
        return covariance
    if covariance.ndim != 3 or covariance.shape[-2:] != (3, 3):
        raise ValueError("covariance 必须为 [N,3,3] 或 [N,6]")
    return torch.stack([
        covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 0, 2],
        covariance[:, 1, 1], covariance[:, 1, 2], covariance[:, 2, 2],
    ], dim=-1)


def render(Gaussians, img_size, slice_idx, axis, num_slices,
           xyz=None, colors=None, opacities=None, scales=None,
           rotations=None, covariance=None, bounds=None, raster_options=None):
    # 正常导入已安装包；延迟导入使 CPU 几何与结构检查无需加载 CUDA 扩展。
    from diff_gaussian_rasterization_slice import GaussianRasterizationSettings, GaussianRasterizer

    h, w = img_size
    if (isinstance(axis, bool) or not isinstance(axis, numbers.Integral)
            or axis not in (0, 1, 2)):
        raise ValueError('axis 必须为 0、1 或 2')
    if (isinstance(num_slices, bool) or not isinstance(num_slices, numbers.Integral)
            or num_slices < 1):
        raise ValueError('num_slices 必须为正整数')
    if (isinstance(slice_idx, bool) or not isinstance(slice_idx, numbers.Real)
            or not math.isfinite(slice_idx) or not 0 <= slice_idx <= num_slices-1):
        raise ValueError('slice_idx 必须为 [0,num_slices-1] 内的有限连续索引')
    bounds = bounds if bounds is not None else getattr(Gaussians, 'bounds', (1., 1., 1.))
    if len(bounds) != 3 or any(not math.isfinite(value) or value <= 0 for value in bounds):
        raise ValueError('bounds 必须为三个有限正半边长')
    options = dict(raster_options or {})
    unsupported = options.keys() - {'background', 'debug'}
    if unsupported:
        raise ValueError(f'归一化切片不支持这些旧渲染选项：{sorted(unsupported)}')

    xyz = Gaussians.get_xyz if xyz is None else xyz
    colors = Gaussians.get_color if colors is None else colors
    if colors.ndim != 2 or colors.shape != (len(xyz), 1):
        raise ValueError('Stage1 强度必须为 [N,1]')
    if covariance is None:
        scales = Gaussians.get_scaling if scales is None else scales
        if rotations is None:
            rotations = Gaussians._rotation if hasattr(Gaussians, '_rotation') else Gaussians.get_rotation
        packed = covariance_from_scaling_rotation(scales, rotations)
    else:
        packed = covariance_to_vector(covariance)

    # 非等比盒子归一化必须变换完整协方差，不能只缩放三个主轴尺度。
    a, b, c = (1./value for value in bounds)
    inverse_bounds = xyz.new_tensor([a, b, c])
    factors = xyz.new_tensor([a*a, a*b, a*c, b*b, b*c, c*c])
    means = (xyz * inverse_bounds).contiguous()
    packed = (packed * factors).contiguous()
    plane = [0., 0., 0.]
    plane[axis] = 0. if num_slices == 1 else -1.+2.*slice_idx/(num_slices-1)
    identity = torch.eye(4, device=xyz.device, dtype=xyz.dtype)
    settings = GaussianRasterizationSettings(
        image_height=h, image_width=w, tanfovx=1., tanfovy=1.,
        bg=xyz.new_full((3,), options.get('background', 0.)), scale_modifier=1.,
        viewmatrix=identity, projmatrix=identity, sh_degree=0, campos=xyz.new_tensor(plane),
        prefiltered=False, debug=options.get('debug', False), antialiasing=False, axis=axis,
    )
    image, radii, _ = GaussianRasterizer(settings)(
        means3D=means, means2D=torch.zeros_like(means),
        colors_precomp=colors.expand(-1, 3).contiguous(),
        opacities=xyz.new_ones((len(xyz), 1)), cov3D_precomp=packed,
    )
    # opacity 仅是兼容参数；radii 表示候选范围，不是实际归一化参与量。
    return {'render': image[0], 'visibility_filter': radii > 0, 'radii': radii}
