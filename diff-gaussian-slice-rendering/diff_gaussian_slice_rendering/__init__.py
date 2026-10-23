"""Differentiable scalar Gaussian fields sampled on fixed affine slice planes.

Public outputs: image [1,H,W], radii [N], weight_sum [1,H,W].
Only image is differentiable. Geometry and settings are fixed observations.
"""
from dataclasses import dataclass, replace
import math
import numbers
import torch
from torch import nn
from torch.autograd.function import once_differentiable

# 扩展尚未编译时仍允许导入并构造切片几何；真正调用渲染才报错。
# 这使 CPU 上准备 DICOM/轴向矩阵不必立即依赖可运行的 CUDA 扩展。
try:
    from . import _C_G3D as _C
    _extension_error = None
except ImportError as exc:
    _C = None
    _extension_error = str(exc)


# 避免把 bool（Python 中属于 int 子类）误当作图像尺寸或切片数。
def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


# 固定几何统一复制为 CPU float64，防止输入对象被外部修改后悄悄改变 Settings。
# 本版本不支持对切片矩阵求梯度，因此显式拒绝 requires_grad=True。
def _fixed_tensor(value, shape, name):
    if isinstance(value, torch.Tensor) and value.requires_grad:
        raise ValueError(f"{name} is fixed geometry; gradients are not supported")
    t = torch.as_tensor(value, dtype=torch.float64, device="cpu").clone()
    if tuple(t.shape) != shape or not bool(torch.isfinite(t).all()):
        raise ValueError(f"{name} must be a finite {shape} matrix")
    return t


# 校验仿射并重算几何法向。面内两列包含方向和像素间距，不能先各自归一化。
# 第三列仅存 a×b 的单位方向；真实像素点始终使用 [column,row,0,1]。
def _plane_matrix(value):
    t = _fixed_tensor(value, (4, 4), "slice_to_world")
    if not torch.equal(t[3], torch.tensor([0., 0., 0., 1.], dtype=t.dtype)):
        raise ValueError("slice_to_world bottom row must be [0,0,0,1]")
    a, b = t[:3, 0], t[:3, 1]
    n = torch.linalg.cross(a, b)
    aa, bb, nnorm = a.norm(), b.norm(), n.norm()
    if not (aa > 0 and bb > 0 and nnorm > 1e-6 * aa * bb):
        raise ValueError("pixel axes must be nonzero and not nearly collinear")
    t[:3, 2] = n / nnorm
    return t


# Settings 是固定观测条件，不是可学习参数集合。
# frozen 防止字段重新赋值，但 Tensor 内容仍可原位改变；forward 还会另存独立矩阵快照。
# background_weight=tau 是背景伪样本权重，不是逐 Gaussian opacity。
@dataclass(frozen=True)
class GaussianRasterizationSettings:
    image_height: int
    image_width: int
    slice_to_world: torch.Tensor
    background: float = 0.0
    background_weight: float = 0.01
    kernel_cutoff: float = 0.001
    debug: bool = False
    # Width in squared Mahalanobis distance Q; zero reproduces v1.1 hard cutoff.
    cutoff_transition: float = 2.0
    # Additive covariance diagonal, in squared model-coordinate units.
    covariance_jitter: float = 1e-8

    # 集中校验标量范围与矩阵约定，三个几何入口最终都会执行这里。
    # transition 的单位是 Mahalanobis 平方距离 Q，必须小于 Qmax。
    def __post_init__(self):
        for name in ("image_height", "image_width"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in ("background", "background_weight", "kernel_cutoff", "cutoff_transition", "covariance_jitter"):
            v = float(getattr(self, name))
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, v)
        if self.background_weight <= 0 or not 0 < self.kernel_cutoff < 1:
            raise ValueError("background_weight > 0 and 0 < kernel_cutoff < 1 are required")
        if not 0 <= self.cutoff_transition < -2 * math.log(self.kernel_cutoff):
            raise ValueError("cutoff_transition must be in [0, -2 log(kernel_cutoff))")
        if self.covariance_jitter < 0:
            raise ValueError("covariance_jitter must be nonnegative")
        object.__setattr__(self, "slice_to_world", _plane_matrix(self.slice_to_world))

    # 无 DICOM 时的三轴便捷入口，仅在 Python 生成矩阵，不向 CUDA 传 axis。
    # 模型空间为以零为中心、半边长 bounds 的盒子；这不是自动恢复真实患者坐标。
    @classmethod
    def from_axis(cls, image_height, image_width, *, axis=2, slice_idx=0,
                  num_slices=1, bounds=(1., 1., 1.), **kwargs):
        h = _positive_int(image_height, "image_height")
        w = _positive_int(image_width, "image_width")
        ns = _positive_int(num_slices, "num_slices")
        if isinstance(axis, bool) or not isinstance(axis, numbers.Integral) or axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        if isinstance(slice_idx, bool) or not isinstance(slice_idx, numbers.Integral) or not 0 <= slice_idx < ns:
            raise ValueError("slice_idx must be an integer in [0,num_slices)")
        bounds = torch.as_tensor(bounds, dtype=torch.float64)
        if bounds.shape != (3,) or not bool(torch.isfinite(bounds).all()) or not bool((bounds > 0).all()):
            raise ValueError("bounds must contain three positive model-space half-extents")
        # 明确图像列/行各对应哪条模型轴，避免 axis=1 时混淆 y 法向与图像行。
        # 面内像素取单元中心，法向层位置包含两端；num_slices=1 时层位置为零。
        u, v = ((1, 2), (0, 2), (0, 1))[axis]
        t = torch.eye(4, dtype=torch.float64)
        t[:3, :3] = 0
        t[u, 0], t[v, 1] = 2*bounds[u]/w, 2*bounds[v]/h
        t[u, 3], t[v, 3] = -bounds[u]+bounds[u]/w, -bounds[v]+bounds[v]/h
        t[axis, 3] = 0. if ns == 1 else -bounds[axis]+2*bounds[axis]*slice_idx/(ns-1)
        return cls(h, w, t, **kwargs)

    # DICOM 入口：读取真实首像素中心、方向余弦与行/列间距。
    # SAX/LAX 在这里不需要不同 CUDA 分支；不同视图只是提供不同的同型矩阵。
    # 若几何标签不足，不把缺失的三维位置静默猜成 axis=2；应显式用 from_axis 或已知矩阵。
    @classmethod
    def from_dicom(cls, dicom_ds, *, model_from_patient=None, frame_index=None,
                   pixel_to_source=None, image_height=None, image_width=None, **kwargs):
        """Build geometry in LPS mm, optionally transformed into model space.

        Supports classic single-frame and enhanced per-frame/shared groups.
        frame_index is zero based. No slice location is inferred from thickness.
        A resize/crop requires an explicit new-pixel -> original-pixel 3x3 affine.
        """
        ds = dicom_ds
        nframes = int(getattr(ds, "NumberOfFrames", 1))
        if nframes > 1 and frame_index is None:
            raise ValueError("multi-frame DICOM requires explicit frame_index")
        frame = 0 if frame_index is None else frame_index
        if isinstance(frame, bool) or not isinstance(frame, numbers.Integral) or not 0 <= frame < nframes:
            raise ValueError("invalid zero-based frame_index")
        per = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
        shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
        if per is not None and len(per) != nframes:
            raise ValueError("PerFrameFunctionalGroupsSequence length mismatch")
        # 增强型 DICOM 优先读取当前帧 functional group，再读取 shared group，最后读取顶层。
        # 多帧位置另要求逐帧显式存在，避免仅凭一个顶层位置错误定位其他帧。
        sources = ([per[frame]] if per is not None else []) + ([shared[0]] if shared else [])

        # 只读取几何标签，不读取 PixelData；强度校准、归一化、时间配准由上层负责。
        def attribute(sequence, tag):
            for source in sources:
                seq = getattr(source, sequence, None)
                if seq and hasattr(seq[0], tag):
                    return getattr(seq[0], tag)
            if hasattr(ds, tag):
                return getattr(ds, tag)
            raise ValueError(f"DICOM missing {tag}; supply an explicit slice_to_world instead")

        if nframes > 1:
            pos = getattr(per[frame], "PlanePositionSequence", None) if per is not None else None
            if not pos or not hasattr(pos[0], "ImagePositionPatient"):
                raise ValueError("multi-frame geometry requires per-frame ImagePositionPatient; no spacing inference is performed")
        ipp = torch.as_tensor([float(x) for x in attribute("PlanePositionSequence", "ImagePositionPatient")], dtype=torch.float64)
        iop = torch.as_tensor([float(x) for x in attribute("PlaneOrientationSequence", "ImageOrientationPatient")], dtype=torch.float64)
        spacing = torch.as_tensor([float(x) for x in attribute("PixelMeasuresSequence", "PixelSpacing")], dtype=torch.float64)
        if ipp.shape != (3,) or iop.shape != (6,) or spacing.shape != (2,):
            raise ValueError("invalid DICOM geometry tag dimensions")
        if not all(bool(torch.isfinite(x).all()) for x in (ipp, iop, spacing)) or not bool((spacing > 0).all()):
            raise ValueError("invalid DICOM geometry values")
        col_dir, row_dir = iop[:3], iop[3:]
        if abs(float(col_dir.norm())-1) > 1e-3 or abs(float(row_dir.norm())-1) > 1e-3 or abs(float(col_dir @ row_dir)) > 1e-3:
            raise ValueError("DICOM orientation vectors must be orthonormal direction cosines")
        t = torch.eye(4, dtype=torch.float64)
        # DICOM PixelSpacing 的顺序是 [row_spacing,column_spacing]。
        # IOP 第一向量随列索引增长，第二向量随行索引增长，所以 spacing 的使用顺序不能交换。
        t[:3, 0] = col_dir * spacing[1]  # increasing column uses column spacing
        t[:3, 1] = row_dir * spacing[0]  # increasing row uses row spacing
        t[:3, 3] = ipp                  # first pixel CENTER, not corner
        t = _plane_matrix(t)
        # 患者 LPS mm -> 模型坐标的显式仿射。只变换切片；Gaussian 均值/协方差由上层同步变换。
        # 非等比缩放或 shear 后重新由面内轴求法向，不能直接使用旧法向的普通向量变换结果。
        if model_from_patient is not None:
            transform = _fixed_tensor(model_from_patient, (4, 4), "model_from_patient")
            if not torch.equal(transform[3], torch.tensor([0.,0.,0.,1.], dtype=transform.dtype)) or abs(float(torch.linalg.det(transform[:3,:3]))) == 0:
                raise ValueError("model_from_patient must be an invertible affine")
            t = _plane_matrix(transform @ t)  # recompute normal after nonuniform scale/shear
        h, w = int(ds.Rows), int(ds.Columns)
        result = cls(h, w, t, **kwargs)
        out_h = h if image_height is None else _positive_int(image_height, "image_height")
        out_w = w if image_width is None else _positive_int(image_width, "image_width")
        if pixel_to_source is None:
            if (out_h, out_w) != (h, w):
                raise ValueError("changed image dimensions require pixel_to_source")
            return result
        return result.with_pixel_transform(pixel_to_source, image_height=out_h, image_width=out_w)

    # 裁剪/缩放后，新像素应映射回原像素再映射到模型坐标。
    # 因此这里接收 new_pixel -> source_pixel 的矩阵，而不是相反方向。
    # 新原点为 o+A*offset，新面内基为 A*M2x2，最后重新计算法向。
    def with_pixel_transform(self, pixel_to_source, *, image_height, image_width):
        """Map new [column,row,1] to the old pixel grid (crop/resize/flip)."""
        m = _fixed_tensor(pixel_to_source, (3, 3), "pixel_to_source")
        if not torch.equal(m[2], torch.tensor([0.,0.,1.], dtype=m.dtype)):
            raise ValueError("pixel_to_source must be a 2D affine")
        t = self.slice_to_world.clone()
        old_axes = t[:3, :2].clone()
        t[:3, :2] = old_axes @ m[:2, :2]
        t[:3, 3] += old_axes @ m[:2, 2]
        return replace(self, image_height=image_height, image_width=image_width, slice_to_world=t)


# 必须重新编译本包，旧库即使模块名字相同也不兼容新参数和缓存布局。
def _require_extension():
    if _C is None:
        raise RuntimeError("CUDA extension is unavailable; build this source package first. " + str(_extension_error))
    if getattr(_C, "API_VERSION", None) != 2:
        raise RuntimeError("Loaded an old _C_G3D binary. Remove the old build/binary and rebuild this package.")


# 自动微分边界的六个输入：means、intensities、scales、rotations、covariance、settings。
# CUDA 返回的缓存保存在 ctx；公开输出只有 image、radii、weight_sum。
# 反向必须返回六项，最后一项固定为 None，未使用的参数分支也返回 None。
class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, intensities, scales, rotations, covariance, settings):
        _require_extension()
        # Snapshot both scalar settings and affine. Mutating a Settings tensor
        # after forward must not change the geometry used by backward.
        # 转换后 clone，确保 backward 使用与 forward 相同的数值与设备上的矩阵快照。
        # 不能在 backward 中重新从外部 Settings 读取可能已被更改的 Tensor。
        affine = settings.slice_to_world.to(device=means.device, dtype=torch.float32).contiguous().clone()
        args = (settings.image_height, settings.image_width, settings.background,
                settings.background_weight, settings.kernel_cutoff,
                settings.cutoff_transition, settings.covariance_jitter, settings.debug)
        n, image, radii, geom, binning, imgstate, weights = _C.rasterize_gaussians(
            means, intensities, scales, rotations, covariance, affine, *args)
        ctx.args, ctx.instances = args, n
        ctx.precomputed = covariance.ndim == 2
        # 保存 Tensor 会启用 PyTorch 的版本检查，防止反向前原位修改模型输入而悄悄产生错误梯度。
        # 同次前向的缓存由 ctx 持有，其存储在反向结束前保持有效。
        ctx.save_for_backward(means, intensities, scales, rotations, covariance,
                              affine, geom, binning, imgstate)
        # 半径与权重和仅作诊断。本版本没有 weight_sum 的独立反向传播路径。
        # set_materialize_grads(False) 允许未用输出的上游梯度保持 None，而不是分配全零大图。
        ctx.mark_non_differentiable(radii, weights)
        ctx.set_materialize_grads(False)
        return image, radii, weights

    @staticmethod
    # 显式限定一阶反向，不承诺支持二阶导数或 Hessian。
    # 这里的 CUDA 原子累加没有实现可供再次自动微分的计算图。
    @once_differentiable
    def backward(ctx, grad_image, _grad_radii, _grad_weights):
        if grad_image is None:
            return (None,) * 6
        means, intensities, scales, rotations, cov, affine, geom, binning, imgstate = ctx.saved_tensors
        dm, dc, ds, dr, dS = _C.rasterize_gaussians_backward(
            means, intensities, scales, rotations, cov, affine, *ctx.args,
            grad_image.contiguous(), ctx.instances, geom, binning, imgstate)
        if ctx.precomputed:
            return dm, dc, None, None, dS, None
        return dm, dc, ds, dr, None, None


# 推荐的唯一函数式公开入口。负责互斥输入、shape、设备和 dtype 校验。
# 在 custom Function 外进行 contiguous 和四元数归一化，使这些操作保留 PyTorch 梯度链。
def rasterize_gaussians(means3D, intensities, raster_settings, *,
                        scales=None, rotations=None, cov3Ds_precomp=None):
    """Sample a scalar field. Supply exactly one covariance parameterization.

    scales: positive activated standard deviations, not log-scales.
    rotations: wxyz quaternions, normalized here with differentiable PyTorch ops.
    cov3Ds_precomp: [N,6], packed xx,xy,xz,yy,yz,zz (not precision).
    """
    if not isinstance(raster_settings, GaussianRasterizationSettings):
        raise TypeError("raster_settings must be GaussianRasterizationSettings")
    if cov3Ds_precomp is None:
        if scales is None or rotations is None:
            raise ValueError("supply both scales and rotations, or cov3Ds_precomp")
    elif scales is not None or rotations is not None:
        raise ValueError("covariance and scale/rotation paths are mutually exclusive")
    tensors = {"means3D": means3D, "intensities": intensities,
               "scales": scales, "rotations": rotations, "cov3Ds_precomp": cov3Ds_precomp}
    for name, value in tensors.items():
        if value is not None and (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not value.is_cuda):
            raise TypeError(f"{name} must be a CUDA float32 Tensor")
        if value is not None and value.device != means3D.device:
            raise ValueError("all Gaussian tensors must use the same CUDA device")
    if means3D.ndim != 2 or means3D.shape[1] != 3:
        raise ValueError("means3D must have shape [N,3]")
    n = means3D.shape[0]
    if intensities.shape != (n, 1):
        raise ValueError("intensities must have shape [N,1]")
    empty = means3D.new_empty((0,))
    if cov3Ds_precomp is None:
        if scales.shape != (n, 3) or rotations.shape != (n, 4):
            raise ValueError("scales/rotations must have shapes [N,3]/[N,4]")
        # 使用 double 求四元数范数，降低 float32 平方和的溢出风险。
        # 零四元数没有有效方向，直接报错；非零四元数先单位化再转换回 float32。
        # CUDA 返回对单位四元数的梯度，PyTorch 再通过除以范数的运算传播到原始参数。
        norms = rotations.double().norm(dim=1, keepdim=True)
        if not bool(torch.isfinite(norms).all()) or bool((norms < 1e-12).any()):
            raise ValueError("rotations must be finite nonzero quaternions")
        rotations = (rotations.double() / norms).float().contiguous()
        scales = scales.contiguous()
        cov = empty
    else:
        if cov3Ds_precomp.shape != (n, 6):
            raise ValueError("cov3Ds_precomp must have shape [N,6]")
        scales = rotations = empty
        cov = cov3Ds_precomp.contiguous()
    return _RasterizeGaussians.apply(means3D.contiguous(), intensities.contiguous(),
                                    scales, rotations, cov, raster_settings)


# 面向对象薄包装，只保存 Settings 并调用同一函数，不维护第二套渲染逻辑。
# 模型中的强度、均值、尺度等参数仍属于上层模块，而不是这个 rasterizer。
class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self, means3D, intensities, *, scales=None, rotations=None, cov3Ds_precomp=None):
        return rasterize_gaussians(means3D, intensities, self.raster_settings,
                                  scales=scales, rotations=rotations, cov3Ds_precomp=cov3Ds_precomp)


__all__ = ["GaussianRasterizationSettings", "GaussianRasterizer", "rasterize_gaussians"]
