"""合成场景短基准：不读取病例、不优化模型，输出计时与峰值显存 JSON。"""
import argparse
import importlib
import json
from pathlib import Path
import statistics
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--extension-dir', type=Path)
parser.add_argument('--points', type=int, default=50000)
parser.add_argument('--size', type=int, default=128)
parser.add_argument('--mode', choices=['point', 'gaussian'], default='point')
parser.add_argument('--sigma', type=float, default=0.)
parser.add_argument('--warmup', type=int, default=3)
parser.add_argument('--repeats', type=int, default=10)
parser.add_argument('--parameterization', choices=['covariance','scales'], default='covariance')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
sys.path.insert(0, str(args.package_root))
import torch
import diff_gaussian_slice_rendering as package

if args.extension_dir:
    package.__path__.append(str(args.extension_dir / 'diff_gaussian_slice_rendering'))
    package._C = importlib.import_module('diff_gaussian_slice_rendering._C_G3D')
package._require_extension()
version = package._C.API_VERSION
torch.manual_seed(args.seed)
means = (torch.rand(args.points, 3, device='cuda') * 2 - 1).requires_grad_()
intensities = (torch.randn(args.points, 1, device='cuda') * .01).requires_grad_()
scales = (torch.rand(args.points, 3, device='cuda') * .025 + .015).requires_grad_()
rotations = torch.randn(args.points, 4, device='cuda', requires_grad=True)
parameters = {'scales': scales, 'rotations': rotations}
inputs = (means, intensities, scales, rotations)
if args.parameterization == 'covariance':
    # 两版本使用同一个 packed 协方差输入，独立比较渲染核而不混入四元数归一化。
    q = rotations.detach().cpu().double()
    q = q / q.norm(dim=-1, keepdim=True)
    w,x,y,z = q.unbind(-1)
    rotation = torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                            2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                            2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)), -1).reshape(-1,3,3)
    covariance = (rotation * scales.detach().cpu().double().square()[:,None,:]) @ rotation.transpose(-1,-2)
    packed = torch.stack((covariance[:,0,0],covariance[:,0,1],covariance[:,0,2],
                          covariance[:,1,1],covariance[:,1,2],covariance[:,2,2]), -1).float().cuda().requires_grad_()
    parameters, inputs = {'cov3Ds_precomp': packed}, (means, intensities, packed)
options = {} if version == 2 else dict(response_mode=args.mode, psf_sigma=args.sigma)
settings = package.GaussianRasterizationSettings.from_axis(
    args.size, args.size, axis=2, slice_idx=0, num_slices=1, **options)
weight = torch.linspace(-1, 1, args.size ** 2, device='cuda').reshape(1, args.size, args.size)
forward_times, backward_times, forward_wall, backward_wall, peaks = [], [], [], [], []
for iteration in range(args.warmup + args.repeats):
    for value in inputs: value.grad = None
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start, middle, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    before = time.perf_counter()
    start.record()
    image, radii, response = package.rasterize_gaussians(
        means, intensities, settings, **parameters)
    middle.record()
    torch.cuda.synchronize()
    after_forward = time.perf_counter()
    (image * weight).mean().backward()
    end.record()
    torch.cuda.synchronize()
    after_backward = time.perf_counter()
    if not torch.isfinite(image).all() or not all(torch.isfinite(v.grad).all() for v in inputs):
        raise FloatingPointError('基准中出现非有限输出或梯度')
    if iteration >= args.warmup:
        forward_times.append(start.elapsed_time(middle))
        backward_times.append(middle.elapsed_time(end))
        forward_wall.append((after_forward-before)*1000)
        backward_wall.append((after_backward-after_forward)*1000)
        peaks.append((torch.cuda.max_memory_allocated()-baseline)/1024**2)
    visible = int((radii > 0).sum())
    mean_radius = float(radii[radii > 0].float().mean()) if visible else 0.
    del image, radii, response
print(json.dumps(dict(
    api_version=version, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
    points=args.points, size=args.size, seed=args.seed, parameterization=args.parameterization, mode=args.mode, sigma=args.sigma,
    kernel_cutoff=settings.kernel_cutoff, cutoff_transition=settings.cutoff_transition,
    warmup=args.warmup, repeats=args.repeats, visible_gaussians=visible,
    mean_radius_pixels=mean_radius,
    forward_gpu_ms=statistics.median(forward_times), backward_gpu_ms=statistics.median(backward_times),
    forward_wall_ms=statistics.median(forward_wall), backward_wall_ms=statistics.median(backward_wall),
    peak_increment_mib=max(peaks),
), indent=2))
