"""训练与测试共用的配置、设备及数据参数。"""
import argparse
import os
from pathlib import Path

import yaml


def load_config(path):
    # 拒绝重复键，避免同一实验参数被后面的值静默覆盖。
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if key in result:
                raise ValueError(f'配置键重复：{key}')
            result[key] = loader.construct_object(value_node)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    config_path = Path(path).resolve()
    with config_path.open(encoding='utf-8') as stream:
        values = yaml.load(stream, Loader=UniqueLoader)
    values.setdefault('dataset_name', 'acdc')
    if not 0 <= values['train_fraction'] <= 1 or values['reference_phase'] < 0:
        raise ValueError('病例划分比例或参考相位非法')
    if len(values['bounds']) != 3 or any(value <= 0 for value in values['bounds']):
        raise ValueError('bounds 必须为三个正半边长')
    if 'iters' in values:
        for key in ('iters', 'checkpoint_interval', 'loss_plot_interval', 'status_interval',
                    'real_slices_per_iter', 'num_pts', 'interp_step_size',
                    'stage2_iters', 'stage2_save_interval', 'stage2_loss_plot_interval',
                    'initial_z_scale', 'initial_distance_floor'):
            if values[key] <= 0:
                raise ValueError(f'{key} 必须大于零')
        if not 1 <= values['structural_min_points'] <= values['num_pts'] <= values['max_num_pts']:
            raise ValueError('结构容量需要 1 <= min_points <= num_pts <= max_num_pts')
        if not 0 < values['structural_split_fraction'] < 1:
            raise ValueError('structural_split_fraction 必须在 (0,1) 内')
        if not values['structural_interp_nodes'] or any(not 0 < n < 1 for n in values['structural_interp_nodes']):
            raise ValueError('结构插层积分节点需要非空且位于 (0,1)')
        for key in ('structural_candidates','structural_max_accept','structural_peak_window',
                    'structural_refit_steps','structural_patch_budget','structural_cache_mb'):
            if values[key] <= 0:
                raise ValueError(f'{key} 必须大于零')
        if values['structural_neighbors'] < 0 or values['structural_count_weight'] < 0 or values['structural_amplitude_weight'] < 0:
            raise ValueError('结构邻居数及正则权重不能为负')
        if (values['lambda_l1'] < 0 or values['lambda_ssim'] < 0
                or values['lambda_l1']+values['lambda_ssim'] <= 0):
            raise ValueError('结构评价需要非负且不全为零的 L1/SSIM 权重')
        if values['ssim_window_size'] <= 0 or values['ssim_window_size'] % 2 != 1:
            raise ValueError('局部 SSIM 结构评价需要正奇数窗口')
        if not 0 <= values['structural_merge_correlation'] <= 1:
            raise ValueError('响应相关性阈值必须位于 [0,1]')
        for key in ('structural_rank_rtol','structural_min_gain','structural_verify_atol','structural_verify_rtol'):
            if values[key] < 0:
                raise ValueError(f'{key} 不能为负')
        if not (0 <= values['interpolation_probability'] <= 1
                and 0 <= values['interpolation_alpha_min'] <= values['interpolation_alpha_max'] <= 1):
            raise ValueError('插层概率或 alpha 范围非法')
    # 相对路径以配置文件为基准，不依赖启动命令所在目录。
    for key, value in values.items():
        if key.endswith(('_path', '_dir', '_root')) and isinstance(value, str) and value:
            value = os.path.expandvars(os.path.expanduser(value))
            values[key] = str((config_path.parent / value).resolve())
    values['dataset_paths'] = {
        name: str((config_path.parent / os.path.expandvars(os.path.expanduser(value))).resolve())
        for name, value in values['dataset_paths'].items()
    }
    return argparse.Namespace(**values)


def setup_runtime(args):
    import random
    import numpy as np
    import torch

    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('当前渲染扩展要求 CUDA 设备')
    torch.cuda.set_device(device)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)


def dataset_options(args):
    # 主训练/重建按完整病例工作；固定使用加载器的 [S,T,H,W] 输出。
    name = getattr(args, 'dataset_name', 'acdc').lower()
    name = {'mm':'mnms','mms':'mnms','m&ms':'mnms'}.get(name,name)
    return dict(
        datasets=name, data_path=args.dataset_paths[name], dim='3D',
        train_fraction=args.train_fraction,
        normalization_epsilon=args.normalization_epsilon, config=args,
    )


def render_options(args):
    return dict(bounds=args.bounds, raster_options={
        'background': args.background,
        'response_mode': args.response_mode,
        'psf_sigma': args.psf_sigma,
        'psf_std_vector': args.psf_std_vector,
        'kernel_cutoff': args.kernel_cutoff,
        'cutoff_transition': args.cutoff_transition,
        'covariance_jitter': args.covariance_jitter,
        'debug': args.raster_debug,
    })


def ssim_options(args):
    return dict(window_size=args.ssim_window_size, sigma=args.ssim_sigma,
                k1=args.ssim_k1, k2=args.ssim_k2, data_range=args.data_range)
