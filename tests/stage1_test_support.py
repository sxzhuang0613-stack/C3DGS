"""Stage1 测试专用参考与显式临时扩展加载；生产代码没有后备渲染路径。"""
import importlib.util
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_test_extension():
    path = os.environ.get('STAGE1_NORMALIZED_EXTENSION')
    if not path:
        return importlib.util.find_spec('diff_gaussian_rasterization_slice') is not None
    name = 'diff_gaussian_rasterization_slice'
    native_name = name+'._C_G3D'
    spec = importlib.util.spec_from_file_location(native_name, path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    sys.modules[native_name] = native
    package_path = ROOT/'diff-gaussian-rasterization_slice'/name
    spec = importlib.util.spec_from_file_location(name, package_path/'__init__.py',
                                                 submodule_search_locations=[str(package_path)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return True


NATIVE_AVAILABLE = load_test_extension()


def unpack_covariance(packed):
    a, b, c, d, e, f = packed.unbind(-1)
    return torch.stack((a, b, c, b, d, e, c, e, f), -1).reshape(-1, 3, 3)


def activated_covariance(log_scales, rotations):
    from Gaussians.utils import covariance_from_scaling_rotation
    return unpack_covariance(covariance_from_scaling_rotation(log_scales.double().exp(), rotations.double()))


def reference_render(model, img_size, slice_idx, axis, num_slices, **kwargs):
    # 独立 CPU float64 场采样，忽略 tile 包围近似；只用于几何/线性算子参考。
    means = kwargs.get('xyz', model.get_xyz)
    colors = kwargs.get('colors', model.get_intensity)
    covariance = kwargs.get('covariance')
    if covariance is None:
        from Gaussians.utils import covariance_from_scaling_rotation
        covariance = unpack_covariance(covariance_from_scaling_rotation(
            kwargs.get('scales', model.get_scaling).double(),
            kwargs.get('rotations', model._rotation).double()))
    elif covariance.ndim == 2:
        covariance = unpack_covariance(covariance)
    bounds = kwargs.get('bounds', model.bounds)
    height, width = img_size
    u, v = ((1, 2), (0, 2), (0, 1))[axis]
    y, x = torch.meshgrid(
        ((torch.arange(height, dtype=torch.float64)+.5)*2/height-1)*bounds[v],
        ((torch.arange(width, dtype=torch.float64)+.5)*2/width-1)*bounds[u], indexing='ij')
    plane = 0. if num_slices == 1 else (-1+2*slice_idx/(num_slices-1))*bounds[axis]
    coordinates = [x, y]
    coordinates.insert(axis, torch.full_like(x, plane))
    points = torch.stack(coordinates, -1)
    delta = points[None] - means.to(device='cpu', dtype=torch.float64)[:, None, None]
    covariance = covariance.to(device='cpu', dtype=torch.float64)
    power = -.5*torch.einsum('nhwi,nij,nhwj->nhw',delta,torch.linalg.inv(covariance),delta)
    weights = power.exp()
    weights = torch.where(weights >= 1/255, weights, 0.)
    total = weights.sum(0)
    numerator = torch.einsum('nhw,nc->chw',weights,colors.to(device='cpu', dtype=torch.float64))[0]
    background = kwargs.get('raster_options', {}).get('background', 0.)
    image = torch.where(total > 1e-8, numerator/total.clamp_min(1e-8), background)
    visible = weights.flatten(1).sum(-1) > 0
    return {'render':image.to(means), 'visibility_filter':visible.to(means.device)}
