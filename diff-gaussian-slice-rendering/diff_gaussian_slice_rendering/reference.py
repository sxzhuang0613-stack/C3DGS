"""小规模 PyTorch 数学参考：完整解析核与可选有限支撑，用于核对 CUDA。"""
import math
import torch


def covariance_from_scales(scales, rotations):
    """标准差和 wxyz 四元数 -> [N,3,3]；保留自动求导链。"""
    w, x, y, z = (rotations / rotations.norm(dim=-1, keepdim=True)).unbind(-1)
    rotation = torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ), dim=-1).reshape(-1, 3, 3)
    return (rotation * scales.square().unsqueeze(-2)) @ rotation.transpose(-1, -2)


def unpack_covariance(packed):
    xx, xy, xz, yy, yz, zz = packed.unbind(-1)
    return torch.stack((xx, xy, xz, xy, yy, yz, xz, yz, zz), -1).reshape(-1, 3, 3)


def effective_covariance(covariance, psf_vector, jitter=0.):
    identity = torch.eye(3, dtype=covariance.dtype, device=covariance.device)
    base = covariance + jitter * identity
    vector = torch.as_tensor(psf_vector, dtype=base.dtype, device=base.device)
    effective = base + vector[:, None] * vector[None, :]
    # 行列式引理避免直接计算两个小行列式的比值。
    log_d = torch.log1p((vector * torch.linalg.solve(base, vector.expand(len(base), 3))).sum(-1))
    return base, effective, log_d


def response_3d(points, means, covariance, psf_vector, jitter=0.):
    """完整厚度响应 [P,N]；PSF 向量固定且与 means 处于同一坐标系。"""
    _, effective, log_d = effective_covariance(covariance, psf_vector, jitter)
    delta = points[:, None, :] - means[None, :, :]
    solved = torch.linalg.solve(effective, delta.permute(1, 2, 0))
    distance = (delta.permute(1, 0, 2) * solved.transpose(1, 2)).sum(-1).transpose(0, 1)
    return torch.exp(-0.5 * (distance + log_d))


def footprint_2d(means, covariance, affine, psf_vector, jitter=0.):
    """返回像素坐标中心、二维精度和对数幅值，不含 intensity。"""
    _, effective, log_d = effective_covariance(covariance, psf_vector, jitter)
    axes, origin = affine[:3, :2], affine[:3, 3]
    solved_axes = torch.linalg.solve(effective, axes.expand(len(means), 3, 2))
    precision = axes.T @ solved_axes
    delta = means - origin
    rhs = (solved_axes.transpose(-1, -2) @ delta.unsqueeze(-1)).squeeze(-1)
    center = torch.linalg.solve(precision, rhs)
    normal = torch.linalg.cross(axes[:, 0], axes[:, 1])
    # 与补平方公式等价，避免离平面距离的大数相减。
    q = (delta @ normal).square() / torch.einsum('i,nij,j->n', normal, effective, normal)
    return center, precision, -0.5 * (q + log_d)


def rasterize_reference(means, intensities, covariance, settings, *, finite_support=True):
    """等效二维加性渲染；有限支撑是在完整解析积分之后施加的近似。"""
    affine = settings.slice_to_world.to(means)
    center, precision, log_amplitude = footprint_2d(
        means, covariance, affine, settings.psf_vector, settings.covariance_jitter)
    rows, columns = torch.meshgrid(
        torch.arange(settings.image_height, device=means.device, dtype=means.dtype),
        torch.arange(settings.image_width, device=means.device, dtype=means.dtype), indexing='ij')
    pixels = torch.stack((columns, rows), -1).reshape(-1, 2)
    delta = pixels[:, None, :] - center[None, :, :]
    q = torch.einsum('pni,nij,pnj->pn', delta, precision, delta) - 2 * log_amplitude
    response = torch.exp(-0.5 * q)
    if finite_support:
        qmax = -2 * math.log(settings.kernel_cutoff)
        width = settings.cutoff_transition
        if width > 0:
            t = ((q - (qmax - width)) / width).clamp(0, 1)
            response = response * (1 - t).square() * (1 + 2 * t)
        response = torch.where(q < qmax, response, 0.)
    image = settings.background + (response @ intensities).reshape(
        1, settings.image_height, settings.image_width)
    return image
