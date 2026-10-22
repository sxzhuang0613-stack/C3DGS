import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from random import randint


def l1_loss(network_output, gt, mode='mean'):
    if mode =='mean':
        return torch.abs((network_output - gt)).mean()
    elif mode =='sum':
        return torch.abs((network_output - gt)).sum()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def slice_tv_loss(images):
    """约束相邻切片的强度变化，输入的第 0 维表示切片维度。"""
    if images.shape[0] < 2:
        return images.new_zeros(())
    return torch.abs(images[1:] - images[:-1]).mean()

def anisotropic_scale_loss(scales, max_ratio=6.0, eps=1e-6):
    """仅惩罚超过允许范围的高斯长短轴比例。"""
    if scales.numel() == 0:
        return scales.new_zeros(())

    max_scale = torch.max(scales, dim=-1).values
    min_scale = torch.min(scales, dim=-1).values.clamp_min(eps)
    scale_ratio = max_scale / min_scale
    return torch.mean(F.relu(scale_ratio / max_ratio - 1.0) ** 2)

# ============ ssim =========================

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=5, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def penalize_outside_range(tensor, lower_bound=0.0, upper_bound=1.0):
    error = 0
    below_lower_bound = tensor[tensor < lower_bound]
    above_upper_bound = tensor[tensor > upper_bound]
    if below_lower_bound.numel():
        error += torch.mean(torch.abs(below_lower_bound - lower_bound))
    if above_upper_bound.numel():
        error += torch.mean(torch.abs(above_upper_bound - upper_bound))
    return error 

def psnr(img1, img2, alpha=20):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return alpha * torch.log10(1.0 / torch.sqrt(mse))