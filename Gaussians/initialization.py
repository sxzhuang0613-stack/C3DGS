"""spacing 驱动的初值；固定几何系数求解与正常 Adam 训练相互独立。"""
import hashlib
import math

import torch
import torch.nn.functional as F
from torch import nn

from Renderer import render


def _sample_positions(images, spacing, bounds, count, args, generator, guided=True):
    slices, height, width = images.shape
    device = images.device
    uniform_count = round(count*args.init_uniform_fraction) if guided else count
    foreground_count = min(round(count*args.init_foreground_fraction),count-uniform_count) if guided else 0
    edge_count = count-uniform_count-foreground_count
    positions = []
    # 每层分配均匀份额；矩形网格用随机子集，避免尾行偏置。
    for layer in range(slices):
        n = uniform_count//slices + int(layer < uniform_count%slices)
        if not n:
            continue
        nx = math.ceil(math.sqrt(n*width/height))
        ny = math.ceil(n/nx)
        cells = torch.randperm(nx*ny,device=device,generator=generator)[:n]
        jitter = (torch.rand(n,2,device=device,generator=generator)*2-1)*args.init_grid_jitter
        u = -.5+(cells%nx+.5+jitter[:,0])*width/nx
        v = -.5+(cells//nx+.5+jitter[:,1])*height/ny
        positions.append(torch.stack((torch.full_like(u,layer),v,u),1))
    if foreground_count or edge_count:
        low,high = torch.quantile(images.flatten(),images.new_tensor(args.init_sampling_quantiles))
        # 分位数图只产生采样权重，训练强度始终保持 dataloader 原输出。
        normalized = ((images-low)/(high-low)).clamp(0,1) if high > low else torch.zeros_like(images)
        foreground = (normalized > args.init_foreground_threshold).float()[:,None]
        radius = args.init_foreground_dilation
        foreground = F.max_pool2d(foreground,2*radius+1,stride=1,padding=radius)[:,0]
        radius = math.ceil(args.init_edge_truncate*args.init_edge_sigma)
        coordinate = torch.arange(-radius,radius+1,device=device,dtype=images.dtype)
        kernel = torch.exp(-.5*(coordinate/args.init_edge_sigma).square())
        kernel /= kernel.sum()
        window = (kernel[:,None]*kernel[None,:])[None,None]
        smooth = F.conv2d(F.pad(normalized[:,None],(radius,)*4,mode='replicate'),window)[:,0]
        padded = F.pad(smooth[:,None],(1,1,1,1),mode='replicate')[:,0]
        gx = (padded[:,1:-1,2:]-padded[:,1:-1,:-2])/2
        gy = (padded[:,2:,1:-1]-padded[:,:-2,1:-1])/2
        edge = torch.sqrt(gx.square()+gy.square())*foreground
        valid = edge[foreground.bool()]
        threshold = torch.quantile(valid,args.init_edge_quantile) if valid.numel() else edge.new_zeros(())
        edge *= edge >= threshold
        for n,weights in ((foreground_count,foreground),(edge_count,edge)):
            if not n:
                continue
            weights = weights.flatten()
            if not bool(weights.sum() > 0):
                weights = torch.ones_like(weights)
            indices = torch.multinomial(weights,n,replacement=True,generator=generator)
            jitter = (torch.rand(n,2,device=device,generator=generator)*2-1)*args.init_grid_jitter
            positions.append(torch.stack((indices//(height*width),
                (indices//width)%height+jitter[:,1],indices%width+jitter[:,0]),1))
    locations = torch.cat(positions)
    z = (locations[:,0]-(slices-1)/2)*spacing[2]
    lower = (z-args.init_normal_jitter*spacing[2]).clamp_min(-bounds[2])
    upper = (z+args.init_normal_jitter*spacing[2]).clamp_max(bounds[2])
    z = lower+(upper-lower)*torch.rand(count,device=device,generator=generator)
    return torch.stack(((locations[:,2]+.5-width/2)*spacing[0],
                        (locations[:,1]+.5-height/2)*spacing[1],z),1)


class IntensityOperator:
    """正式 Renderer 的线性强度算子及伴随；不构造像素数乘高斯数的矩阵。"""
    def __init__(self, model, images, rendering):
        self.model, self.images = model, images
        self.rendering = dict(rendering,raster_options=dict(rendering['raster_options'],background=0.))
        self.parameters = dict(xyz=model.get_xyz.detach(),scales=model.get_scaling.detach(),
                               rotations=model.get_rotation.detach())

    def view(self, coefficients, index):
        return render(self.model,list(self.images.shape[-2:]),index,2,len(self.images),
                      colors=coefficients[:,None],**self.parameters,**self.rendering)

    @torch.no_grad()
    def forward(self, coefficients, visibility=False):
        coefficients = coefficients.to(self.images)
        outputs = [self.view(coefficients,i) for i in range(len(self.images))]
        prediction = torch.stack([output['render'] for output in outputs]).double()
        if visibility:
            return prediction,torch.stack([o['visibility_filter'] for o in outputs]).any(0)
        return prediction

    def adjoint(self, values):
        # W=1/(S*H*W)，与全病例均方误差归一化一致；只保留系数梯度。
        with torch.enable_grad():
            coefficients = self.model._intensity.detach().flatten().clone().requires_grad_()
            result = coefficients.new_zeros(coefficients.shape,dtype=torch.float64)
            for index in range(len(self.images)):
                prediction = self.view(coefficients,index)['render']
                incoming = (values[index]/self.images.numel()).to(prediction)
                result += torch.autograd.grad(prediction,coefficients,grad_outputs=incoming)[0].double()
        return result


@torch.no_grad()
def initialize_from_images(model, images, geometry, args, rendering, case_name):
    if not torch.isfinite(images).all():
        raise ValueError('初始化图像含非有限值')
    seed = int.from_bytes(hashlib.blake2b(
        f'{args.random_seed}:{case_name}'.encode(),digest_size=8).digest(),'little')
    generator = torch.Generator(device=images.device).manual_seed(seed)
    spacing = images.new_tensor(geometry['spacing_xyz'])/geometry['scale_mm']
    bounds = images.new_tensor(geometry['bounds'])
    count = args.num_pts
    points = _sample_positions(images,spacing,bounds,count,args,generator)
    scales = images.new_empty(count,3)
    scales[:,:2] = spacing[:2]*args.init_fine_sigma_pixels
    coarse = torch.randperm(count,device=images.device,generator=generator)[:round(count*args.init_coarse_fraction)]
    scales[coarse,:2] = spacing[:2]*args.init_coarse_sigma_pixels
    # PSF 是固定观测向量；允许其本身宽于覆盖目标，不反向缩小 PSF。
    options = rendering['raster_options']
    vector = options['psf_std_vector']
    normal_psf = options['psf_sigma'] if vector is None else abs(vector[2])
    floor = args.init_normal_floor_pixels*spacing[:2].min()
    scales[:,2] = torch.sqrt(((args.init_normal_sigma_spacing*spacing[2])**2
                              -normal_psf**2-options['covariance_jitter']).clamp_min(floor**2))
    scales = torch.maximum(scales,scales.max(1).values[:,None]/args.init_max_axis_ratio)
    logs = scales.log()
    if not bool(((logs >= args.log_scale_min)&(logs <= args.log_scale_max)).all()):
        raise ValueError('初始化物理尺度超出配置的 log_scale 范围')
    rotations = images.new_zeros(count,4)
    rotations[:,0] = 1
    model._xyz = nn.Parameter(points)
    model._scaling = nn.Parameter(logs)
    model._rotation = nn.Parameter(rotations)
    model._intensity = nn.Parameter(images.new_ones(count,1))
    model.geometry_reference_count = count
    operator = IntensityOperator(model,images,rendering)
    for attempt in range(args.init_support_retries+1):
        field,visible = operator.forward(torch.ones(count,device=images.device),visibility=True)
        if bool(visible.all()):
            break
        if attempt == args.init_support_retries:
            raise ValueError('初始化仍存在无观测支持的高斯，请检查 PSF、cutoff 和初始尺度')
        model._xyz[~visible] = _sample_positions(images,spacing,bounds,int((~visible).sum()),args,generator,guided=False)
    target = images.double()-options['background']
    field_norm = field.square().mean()
    if not bool(torch.isfinite(field_norm) & (field_norm > 0)):
        raise FloatingPointError('初始化的单位系数响应为零或非有限')
    common = (field*target).mean()/field_norm
    baseline = common.expand(count).clone()
    initial_prediction = operator.forward(baseline)
    ridge = args.init_ridge_relative*field_norm/count
    residual = operator.adjoint(target-initial_prediction)
    direction = residual.clone()
    residual_norm = residual@residual
    initial_norm = residual_norm.clone()
    stopping_norm = residual_norm*args.init_cg_rtol**2
    delta = baseline.new_zeros(count)
    steps = 0
    for _ in range(args.init_fit_steps):
        if bool(residual_norm <= stopping_norm):
            break
        product = operator.adjoint(operator.forward(direction))+ridge*direction
        curvature = direction@product
        if not bool(torch.isfinite(curvature) & (curvature > 0)):
            raise FloatingPointError('初始化线性求解出现非正或非有限曲率')
        alpha = residual_norm/curvature
        delta += alpha*direction
        residual -= alpha*product
        next_norm = residual@residual
        direction = residual+(next_norm/residual_norm)*direction
        residual_norm = next_norm
        steps += 1
    coefficients = (baseline+delta).to(images)
    if not bool(torch.isfinite(coefficients).all()):
        raise FloatingPointError('初始化强度求解出现 NaN/Inf')
    model._intensity.copy_(coefficients[:,None])
    prediction = operator.forward(coefficients)
    report = dict(common_intensity=float(common),fit_steps=steps,
                  relative_residual=float(torch.sqrt(residual_norm/initial_norm)) if initial_norm > 0 else 0.,
                  mse_before=float((initial_prediction-target).square().mean()),
                  mse_after=float((prediction-target).square().mean()),
                  count=count)
    print(f'[Initialization] {report}')
    return report
