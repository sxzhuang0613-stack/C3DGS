"""显式运行的数据转换入口；导入本模块不会访问或覆盖病例文件。"""
import argparse
import os
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm
from config_utils import load_config


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


def preprocess(args):
    os.makedirs(args.preprocess_output_path, exist_ok=True)
    for patient in tqdm(sorted(os.listdir(args.preprocess_image_path))):
        image_dir = os.path.join(args.preprocess_image_path, patient)
        label_dir = os.path.join(args.preprocess_label_path, patient)
        output = os.path.join(args.preprocess_output_path, patient + '.npz')
        if os.path.exists(output) and not args.preprocess_overwrite:
            raise FileExistsError(f'已有结果未覆盖：{output}')
        slices = sorted((f for f in os.listdir(label_dir) if f.endswith('.nii.gz')),
                        key=lambda f: float(f.removesuffix('.nii.gz')))
        images, labels, affines, spacings, original_frames = [], [], [], [], []
        for name in slices:
            source = nib.load(os.path.join(image_dir, name))
            label_source = nib.load(os.path.join(label_dir, name))
            image = np.moveaxis(source.get_fdata(), args.preprocess_time_axis, 0)
            label = np.moveaxis(label_source.get_fdata(), args.preprocess_time_axis, 0)
            if image.shape != label.shape:
                raise ValueError(f'{patient}/{name}: 图像与标签 shape 不一致')
            original_frames.append(image.shape[0])
            if args.preprocess_normalization == 'slice_uint8':
                # 默认保持原有逐切片归一化协议；可显式选择保留浮点原始强度。
                image = (image - image.min()) / (image.max() - image.min() + args.preprocess_epsilon)
                image = (image * 255).astype(np.uint8)
            elif args.preprocess_normalization != 'none':
                raise ValueError('preprocess_normalization 必须为 slice_uint8 或 none')
            images.append(adjust_frames(image, args.preprocess_target_frames))
            labels.append(adjust_frames(label, args.preprocess_target_frames))
            affines.append(source.affine)
            spacings.append(source.header.get_zooms())
        # 保留原 NIfTI 几何；不要将其误认为归一化训练坐标的 affine。
        np.savez_compressed(
            output, cmr=np.stack(images), cmr_lbl=np.stack(labels), info=patient,
            source_affine=np.stack(affines), source_spacing=np.asarray(spacings),
            source_time_axis=args.preprocess_time_axis, original_frames=np.asarray(original_frames),
            normalization=args.preprocess_normalization,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_train.yaml')
    preprocess(load_config(parser.parse_args().config))
