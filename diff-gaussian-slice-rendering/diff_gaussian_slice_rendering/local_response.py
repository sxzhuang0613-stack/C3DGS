"""少量 Gaussian 的稀疏观测响应；复用主 Renderer 的激活及 PSF 定义。"""
from dataclasses import dataclass
import math

import torch
from . import normalize_rotations
from .reference import footprint_2d


@dataclass
class ResponsePatch:
    indices: torch.Tensor
    values: torch.Tensor

    @property
    def nbytes(self):
        return self.indices.numel() * self.indices.element_size() + self.values.numel() * self.values.element_size()


def activated_covariance(log_scales, raw_rotations):
    # Renderer 直接传原始四元数，局部响应复用库入口的一次单位化及舍入。
    q = normalize_rotations(raw_rotations).double()
    w, x, y, z = q.unbind(-1)
    rotation = torch.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ), -1).reshape(-1, 3, 3)
    scales = log_scales.exp().double()
    return (rotation * scales.square()[:, None]) @ rotation.transpose(-1, -2)


def kernel_response(q, cutoff, transition):
    qmax = -2 * math.log(cutoff)
    response = torch.exp(-0.5 * q.clamp_min(0))
    if transition > 0:
        t = ((q - qmax + transition) / transition).clamp(0, 1)
        response = response * (1-t).square() * (1+2*t)
    return torch.where(q < qmax, response, 0.)


@torch.no_grad()
def response_patches(means, covariance, settings, pixel_budget):
    """返回每个 Gaussian 跨全部观测的稀疏块；超预算返回 None，绝不截短实际支撑。

    indices 使用 view*H*W+row*W+column；局部坐标始终引用原切片全局像素。
    """
    count, device = len(means), means.device
    height, width = settings[0].image_height, settings[0].image_width
    footprints, boxes = [], []
    for view in settings:
        # CUDA 输入几何先转 float32，再在 kernel 中使用 double。
        affine = view.slice_to_world.to(device=device, dtype=torch.float32).double()
        vector = view.psf_vector.to(device=device, dtype=torch.float32).double()
        center, precision, log_amp = footprint_2d(means.double(), covariance, affine, vector, view.covariance_jitter)
        variance = torch.linalg.solve(precision, torch.eye(2,device=device,dtype=precision.dtype).expand(count,2,2))
        largest = torch.linalg.eigvalsh(variance)[:, -1]
        budget = -2*math.log(view.kernel_cutoff) + 2*log_amp
        radius = torch.ceil(torch.sqrt(largest * budget.clamp_min(0))) + 1
        lower = torch.floor(center-radius[:, None]).clamp_min(0)
        upper = torch.ceil(center+radius[:, None]+1)
        lower[:, 0].clamp_max_(width); lower[:, 1].clamp_max_(height)
        upper[:, 0].clamp_(0, width); upper[:, 1].clamp_(0, height)
        box = torch.stack((lower[:, 0], lower[:, 1], upper[:, 0], upper[:, 1]), -1).long()
        box[budget <= 0] = 0
        boxes.append(box)
        footprints.append((center, precision, log_amp))
    # 所有 footprint 边界只批量回传一次；不在逐 Gaussian 坐标循环中 .item()。
    boxes = torch.stack(boxes, 1).cpu().tolist()
    result = []
    for index in range(count):
        area = sum((x1-x0)*(y1-y0) for x0,y0,x1,y1 in boxes[index])
        if area > pixel_budget:
            result.append(None)
            continue
        indices, values = [], []
        for v, (x0,y0,x1,y1) in enumerate(boxes[index]):
            if x1 <= x0 or y1 <= y0:
                continue
            rows, columns = torch.meshgrid(torch.arange(y0,y1,device=device),
                                           torch.arange(x0,x1,device=device), indexing='ij')
            center, precision, log_amp = footprints[v]
            delta = torch.stack((columns, rows), -1).double() - center[index]
            q = torch.einsum('...i,ij,...j->...', delta, precision[index], delta) - 2*log_amp[index]
            indices.append((v*height*width+rows*width+columns).flatten())
            values.append(kernel_response(q, settings[v].kernel_cutoff, settings[v].cutoff_transition).flatten())
        result.append(ResponsePatch(
            torch.cat(indices) if indices else torch.empty(0,device=device,dtype=torch.long),
            torch.cat(values) if values else torch.empty(0,device=device,dtype=torch.float64)))
    return result
