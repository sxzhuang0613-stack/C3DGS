"""Stage1 重建评估：只加载参考相位高斯，不进行拟合或参数更新。"""
import argparse
import csv
import os

import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config_utils import load_config, setup_runtime, dataset_options, ssim_options
from CostFunctions import l1_loss, ssim
from Dataloader import dataset
from Gaussians.GaussianModel_noop import GaussianModel
from Renderer import render
from case_geometry import make_geometry, check_geometry, attach_geometry, read_model_geometry


def evaluate(args):
    setup_runtime(args)
    similarity = ssim_options(args)
    loader = torch.utils.data.DataLoader(
        dataset(mode=args.dataset_mode, require_labels=True, **dataset_options(args)),
        batch_size=1, num_workers=args.num_workers, pin_memory=args.pin_memory,
    )
    os.makedirs(args.output_path, exist_ok=True)
    rows = []
    for data in tqdm(loader, desc='Evaluating stage1'):
        case = os.path.splitext(data['info']['filename'][0])[0]
        sequence = data['cmr'][0]
        images = sequence[:, args.reference_phase].to(args.device)
        s, h, w = images.shape
        model = GaussianModel()
        ply_path = args.stage1_ply_path.format(case=case)
        record = read_model_geometry(ply_path)
        geometry = record['geometry'] if record else make_geometry(
            data['info']['spacing_xyz'][0],images.shape,args,legacy=True)
        check_geometry(geometry,make_geometry(data['info']['spacing_xyz'][0],images.shape,args,
                                             legacy=geometry['version']==0))
        if record is None:
            tqdm.write(f'[{case}] 无几何记录的旧 PLY：使用配置中的 legacy_box。')
        attach_geometry(model,geometry)
        rendering = geometry['rendering']
        model.load_ply(ply_path,
                       legacy_color_policy=args.legacy_color_policy, device=args.device)
        case_dir = os.path.join(args.output_path, case)
        os.makedirs(case_dir, exist_ok=True)
        arrays = []
        with torch.no_grad():
            for index in range(s):
                pred = render(model, [h, w], index, 2, s, **rendering)['render']
                if args.clamp_predictions:
                    pred = pred.clamp(0, args.data_range)
                gt = images[index]
                mse = ((pred - gt) ** 2).mean()
                psnr = (float('inf') if mse.item() == 0 else
                        float(10 * torch.log10(args.data_range ** 2 / mse)))
                rows.append(dict(case=case, phase=args.reference_phase, slice=index,
                                 l1=l1_loss(pred, gt).item(), psnr=psnr,
                                 ssim=ssim(pred[None, None], gt[None, None],
                                           **similarity).item()))
                image = pred.cpu().numpy()
                if args.save_arrays:
                    arrays.append(image)
                if args.save_images:
                    plt.imsave(os.path.join(case_dir, f'slice_{index:03d}.png'),
                               image, cmap='gray', vmin=0, vmax=args.data_range)
        if args.save_arrays:
            np.savez_compressed(os.path.join(case_dir, 'reconstruction.npz'),
                                image=np.stack(arrays), phase=args.reference_phase)
        del model
    if not rows:
        raise ValueError('测试集为空，没有可报告的结果')
    with open(os.path.join(args.output_path, 'metrics.csv'), 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_test.yaml')
    evaluate(load_config(parser.parse_args().config))
