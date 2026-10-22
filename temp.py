import os
import numpy as np
import torch
import torch.nn.functional as F
from Gaussians.GaussianModel import GaussianModel
from Dataloader import dataset
from CostFunctions import l1_loss, ssim, slice_tv_loss, penalize_outside_range, psnr, anisotropic_scale_loss
from Renderer import render
import yaml
import argparse
import random
import nibabel as nib
from tools import subplots
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pydicom



def overlay_mask(gray_image, mask, save_path, alpha=0.5, color=(255, 0, 0)):
    """
    将二值掩码 B 半透明地覆盖在灰度图 A 上，并保存结果。

    参数:
        gray_image: 灰度图 A。可以是 numpy 数组、PIL Image 或图像文件路径。
        mask:       二值掩码 B。可以是 numpy 数组、PIL Image 或图像文件路径。
                    非零像素被视为掩码区域。
        save_path:  保存路径（含文件名）。
        alpha:      掩码的透明度，范围 [0, 1]。0 表示完全透明，1 表示完全不透明。
        color:      掩码覆盖的颜色 (R, G, B)。

    返回:
        保存的叠加图像 (numpy 数组，RGB 格式)。
    """
    # --- 统一读入为 numpy 数组 ---
    if isinstance(gray_image, str):
        gray_image = np.array(Image.open(gray_image).convert('L'))
    elif isinstance(gray_image, Image.Image):
        gray_image = np.array(gray_image.convert('L'))
    else:
        gray_image = np.asarray(gray_image)

    if isinstance(mask, str):
        mask = np.array(Image.open(mask).convert('L'))
    elif isinstance(mask, Image.Image):
        mask = np.array(mask.convert('L'))
    else:
        mask = np.asarray(mask)

    # --- 尺寸对齐 ---
    if gray_image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"图像尺寸不一致: A={gray_image.shape[:2]}, B={mask.shape[:2]}"
        )

    # --- 归一化灰度图到 [0, 1] ---
    gray = gray_image.astype(np.float32)
    if gray.max() > 1.0:
        gray /= 255.0

    # 把灰度图扩展成 RGB
    rgb = np.stack([gray, gray, gray], axis=-1)  # H x W x 3

    # --- 生成掩码 alpha 图 ---
    mask_bool = mask > 0  # 二值化
    mask_alpha = np.zeros(gray.shape, dtype=np.float32)
    mask_alpha[mask_bool] = alpha  # 掩码区域 alpha 值

    # --- 颜色层 ---
    color_layer = np.zeros_like(rgb)
    color_layer[..., 0] = color[0] / 255.0
    color_layer[..., 1] = color[1] / 255.0
    color_layer[..., 2] = color[2] / 255.0

    # --- 半透明混合: out = (1-a)*base + a*color ---
    a = mask_alpha[..., None]  # H x W x 1
    out = (1 - a) * rgb + a * color_layer
    out = np.clip(out, 0, 1)

    # --- 保存 ---
    out_uint8 = (out * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(out_uint8, mode='RGB').save(save_path)

    return out_uint8


# Dataset = torch.utils.data.DataLoader(
#         dataset(mode="train", datasets='shengyi', align=False, augmentation=False),
#         batch_size=1, 
#         num_workers=0, 
#         drop_last=True,
#         pin_memory=True,
#     )

# for sample_idx, data in enumerate(Dataset):
    
#     # if os.path.exists(os.path.join(r'output/acdc_experiment', f'sample_{sample_idx:03d}')):
#     #     continue
#     imgs = data['cmr'][0,:,0]    # initial time step
#     lbls = data['cmr_lbl']
#     p_name = data['info'][0].split('.')[0]
#     imgs[imgs != 1] = 0
#     out_path = "/data1/zhuangshuxin/codes/3DGS-m/overlay_result.png"
#     overlay_mask(imgs[-1], lbls[0,-1,0], out_path, alpha=0.5, color=(255, 0, 0))
pth1 = r'/data1/zhuangshuxin/datasets/shengyi_all/cropped/images/cmr'
pth2 = r'/data1/zhuangshuxin/datasets/shengyi_all/cropped/labels/cmr_Myo_mask'

def adjust_frames(arr, target_frames=40):
    """将数组第一个维度调整为 target_frames。"""
    T = arr.shape[0]

    if T == 0:
        raise ValueError("Empty sequence!")

    if T < target_frames:
        # 复制最后一帧
        padding = np.repeat(
            arr[-1:],
            target_frames - T,
            axis=0
        )
        arr = np.concatenate([arr, padding], axis=0)

    elif T > target_frames:
        # 截取前40帧
        arr = arr[:target_frames]

    return arr


save = r'/data1/zhuangshuxin/datasets/shengyi_all/cropCMRnpz'
os.makedirs(save, exist_ok=True)

for p in tqdm(sorted(os.listdir(pth1))):
    ppth1 = os.path.join(pth1, p)
    ppth2 = os.path.join(pth2, p)

    slcs = [f for f in os.listdir(ppth2) if f.endswith(".nii.gz")]
    slcs = sorted(
        slcs,
        key=lambda x: float(x.replace(".nii.gz", ""))
    )

    cmrs = []
    lbls = []

    for s in slcs:
        img_path = os.path.join(ppth1, s)
        lbl_path = os.path.join(ppth2, s)

        # ===== CMR =====
        img = nib.load(img_path).get_fdata()

        vmin = np.min(img)
        vmax = np.max(img)

        normalized = (img - vmin) / (vmax - vmin + 1e-8)
        normalized = (normalized * 255).astype(np.uint8)

        # 调整时间维度为40
        normalized = adjust_frames(normalized, 40)

        # ===== Label =====
        lbl = nib.load(lbl_path).get_fdata()

        # 调整时间维度为40
        lbl = adjust_frames(lbl, 40)

        cmrs.append(normalized)
        lbls.append(lbl)

    data = {
        'cmr': np.stack(cmrs, axis=0),
        'cmr_lbl': np.stack(lbls, axis=0),
        'info': p,
    }

    spth = os.path.join(save, p + '.npz')
    np.savez(spth, **data)