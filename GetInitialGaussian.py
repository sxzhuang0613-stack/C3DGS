import os
import torch
from Gaussians.GaussianModel_structural import GaussianModel
from Dataloader import dataset
from CostFunctions import l1_loss, ssim, anisotropic_scale_loss
from Renderer import render
import argparse
import random
import numpy as np
from config_utils import load_config, setup_runtime, dataset_options, ssim_options
from case_geometry import make_geometry, check_geometry, attach_geometry, save_model_ply, read_model_geometry
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# ==================== [功能模块 1: 动态自适应子损失函数趋势图] ====================
def update_loss_plot(out_dir, iters_list, loss_dict, dpi=150):
    plt.figure(figsize=(10, 6))
    
    # 预设一条色板，自动分配给不同的子损失函数
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22', '#1abc9c']
    
    # 优先将 Total Loss 画在最上层，粗体红色
    if "Total Loss" in loss_dict:
        plt.plot(iters_list, loss_dict["Total Loss"], label='Total Loss', color=colors[0], linewidth=2.0)
        
    color_idx = 1  
    for loss_name, loss_values in loss_dict.items():
        if loss_name == "Total Loss":
            continue
        # 给其他子损失自动分配颜色，使用虚线
        color = colors[color_idx % len(colors)]
        plt.plot(iters_list, loss_values, label=loss_name, color=color, linestyle='--', alpha=0.8, linewidth=1.5)
        color_idx += 1
        
    plt.title("Training Loss Trajectory (Dynamic Sub-Losses)", fontsize=14, fontweight='bold')
    plt.xlabel("Iteration Step", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    
    # 静默覆盖保存
    plot_path = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(plot_path, dpi=dpi)
    plt.close('all')

# ==================== [功能模块 2: 周期性全序列快照重建 (含插帧联动)] ====================
def reconstruct_full_case(Gaussians, s, h, w, out_dir, iter_num, include_interp, step_size=0.5, **render_kwargs):
    """重建当前模型在整个 Case 下的所有切片。如果已进入插帧期，将自动采用亚像素步长进行细分插帧"""
    rec_dir = os.path.join(out_dir, "reconstructions", f"iter_{iter_num:06d}")
    os.makedirs(rec_dir, exist_ok=True)
    
    with torch.no_grad():
        if include_interp:
            # 自动计算插帧序列：例如 s=10, step=0.5，则生成 0.0, 0.5, 1.0 ... 9.0
            num_steps = int((s - 1) / step_size) + 1
            slice_indices = [min(i * step_size, s - 1) for i in range(num_steps)]
            if slice_indices[-1] < s - 1:
                slice_indices.append(float(s - 1))
        else:
            # 基础序列：只重现原始的整数切片 0, 1, 2...
            slice_indices = list(range(s))
            
        
        for idx in slice_indices:
            render_pkg = render(Gaussians, img_size=[h, w], slice_idx=idx, axis=2, num_slices=s, **render_kwargs)
            pred_tensor = render_pkg['render'].clamp(0, 1).cpu().numpy()
            
            # 命名支持浮点数：如 slice_000.00.png, slice_000.50.png
            file_name = f"slice_{idx:05.2f}.png"
            plt.imsave(os.path.join(rec_dir, file_name), pred_tensor, cmap='gray', vmin=0, vmax=1)
            
    plt.close('all')
    torch.cuda.empty_cache()

def save_training_checkpoint(path, gaussians, iteration, history_iters, history_losses, args):
    """保存拓扑、Adam 动量、增殖统计和随机状态；PLY 只用于交换模型。"""
    state = {
        'iteration': iteration, 'config': vars(args),
        'geometry': getattr(gaussians,'geometry',None),
        'geometry_reference_count': getattr(gaussians,'geometry_reference_count',len(gaussians.get_xyz)),
        'parameters': {name: getattr(gaussians, name).detach() for name in
                       ('_xyz', '_intensity', '_scaling', '_rotation')},
        'optimizer': gaussians.optimizer.state_dict(),
        'statistics': {name: getattr(gaussians, name) for name in
                       ('xyz_gradient_accum', 'denom', 'max_radii2D')},
        'history_iters': history_iters, 'history_losses': history_losses,
        'random': random.getstate(), 'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
    }
    torch.save(state, path + '.tmp')
    os.replace(path + '.tmp', path)

def restore_training_checkpoint(state, gaussians, args):
    if state.get('geometry') is not None:
        attach_geometry(gaussians,state['geometry'])
    for name, tensor in state['parameters'].items():
        setattr(gaussians, name, torch.nn.Parameter(tensor.to(args.device)))
    gaussians.geometry_reference_count = state.get('geometry_reference_count',len(gaussians.get_xyz))
    gaussians.training_setup(args)
    gaussians.optimizer.load_state_dict(state['optimizer'])
    for name, tensor in state['statistics'].items():
        setattr(gaussians, name, tensor.to(args.device))
    random.setstate(state['random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])

def check_resume_config(saved, args):
    # 旧结构搜索器与批量统计的单位不同，不能把旧状态解释为精确续训。
    if saved.get('structural_algorithm') != args.structural_algorithm:
        raise ValueError('结构算法已改变，不能完整续训；请使用独立输出目录和 PLY 参数热启动')
    # 允许延长训练和改变输出频率；优化目标或 Adam 参数改变时应参数热启动。
    keys = ('position_lr_init','position_lr_final','position_lr_max_steps','position_lr_delay_mult',
            'intensity_lr','scaling_lr','rotation_lr','adam_eps','adam_betas',
            'log_scale_min','log_scale_max','real_slices_per_iter','interpolation',
            'interpolate_iter','interpolation_probability','interpolation_alpha_min',
            'interpolation_alpha_max','anisotropic_max_ratio','anisotropic_epsilon',
            'densify_from_iter','densify_until_iter','densification_interval','max_num_pts',
            'lambda_l1','lambda_ssim','lambda_anisotropic','lambda_interp_l1','lambda_interp_ssim')
    keys = set(keys)|{key for key in vars(args) if key.startswith('structural_')}
    changed = sorted(key for key in keys if key in saved and saved[key] != getattr(args,key))
    if changed:
        raise ValueError(f'续训配置改变了优化条件 {changed}；请使用独立实验或 PLY 参数热启动')

def training(args):
    setup_runtime(args)
    loader = torch.utils.data.DataLoader(
        dataset(mode=args.dataset_mode, **dataset_options(args)),
        batch_size=1, num_workers=args.num_workers, pin_memory=args.pin_memory,
    )
    similarity = ssim_options(args)
    for sample_idx, data in enumerate(loader):
        sample_name = os.path.splitext(data['info']['filename'][0])[0]
        case_dir = os.path.join(args.output_path, sample_name)
        checkpoint_path = os.path.join(case_dir, 'training_state.pth')
        final_ply = os.path.join(case_dir, 'point_cloud_final.ply')
        state = None
        if (args.stage1_resume or args.stage1_skip_existing) and os.path.isfile(checkpoint_path):
            # 只加载本地可信训练文件；完整随机状态包含 NumPy 对象。
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        skip_existing = (args.stage1_skip_existing and state is not None
                         and state['iteration'] >= args.iters
                         and (not args.save_ply or os.path.isfile(final_ply)))
        if skip_existing and args.save_ply and state.get('geometry') is not None:
            final_record = read_model_geometry(final_ply)
            skip_existing = final_record is not None and final_record.get('iteration') == state['iteration']
            if skip_existing:
                check_geometry(state['geometry'],final_record['geometry'])
        if state is not None and (args.stage1_resume or skip_existing):
            check_resume_config(state['config'],args)
        sequence = data['cmr'][0]
        imgs = sequence[:, args.reference_phase].to(args.device)
        s, h, w = imgs.shape
        record = None
        if state is not None and (args.stage1_resume or skip_existing):
            geometry = state.get('geometry')
            if geometry is None:
                # 无几何记录的旧 checkpoint 继续使用旧盒子，不解释为 spacing 坐标。
                old_args = argparse.Namespace(**state['config'])
                geometry = make_geometry(data['info']['spacing_xyz'][0],imgs.shape,old_args,legacy=True)
                tqdm.write(f'[{sample_name}] 旧 checkpoint：保留 legacy_box 坐标。')
        elif args.warm_start_ply_path:
            record = read_model_geometry(args.warm_start_ply_path)
            geometry = record['geometry'] if record else make_geometry(
                data['info']['spacing_xyz'][0],imgs.shape,args,legacy=True)
            if record is None:
                tqdm.write(f'[{sample_name}] 无几何记录的旧 PLY：保留 legacy_box 坐标。')
        else:
            geometry = make_geometry(data['info']['spacing_xyz'][0],imgs.shape,args)
        current = make_geometry(data['info']['spacing_xyz'][0],imgs.shape,args,legacy=geometry['version']==0)
        check_geometry(geometry,current)
        rendering = geometry['rendering']
        if skip_existing:
            tqdm.write(f'[{sample_name}] 已完成 {state["iteration"]} 步，跳过。')
            continue
        os.makedirs(case_dir, exist_ok=True)
        gaussians = GaussianModel()
        attach_geometry(gaussians,geometry)
        history_iters, history_losses = [], {}
        start_iteration = 0
        if args.stage1_resume and state is not None:
            tqdm.write(f'[{sample_name}] 续训已有参数，初始化不重新执行。')
            restore_training_checkpoint(state, gaussians, args)
            start_iteration = state['iteration']
            history_iters = state['history_iters']
            history_losses = state['history_losses']
        else:
            if args.warm_start_ply_path:
                gaussians.load_ply(args.warm_start_ply_path,
                                   legacy_color_policy=args.legacy_color_policy, device=args.device)
                if record is not None:
                    gaussians.geometry_reference_count = record['geometry_reference_count']
            else:
                gaussians.initialize_from_images(imgs,geometry,args,rendering,sample_name)
            gaussians.training_setup(args)

        with torch.no_grad():
            gaussians.prune_points(gaussians._invalid_point_mask())
        if gaussians.get_xyz.shape[0] == 0:
            raise FloatingPointError(f'{sample_name}: 没有有效高斯点，无法开始优化')
        pbar = tqdm(range(start_iteration + 1, args.iters + 1),
                    desc=f'Case [{sample_idx + 1}/{len(loader)}] {sample_name}')
        for iteration in pbar:
            gaussians.update_learning_rate(iteration)
            # 无放回均匀采样，单层和双层病例也能获得真实监督。
            indices = random.sample(range(s), min(s, args.real_slices_per_iter))
            outputs = [render(gaussians, [h, w], index, 2, s, **rendering)
                       for index in indices]
            predictions = torch.stack([output['render'] for output in outputs])[:, None]
            targets = imgs[indices, None]
            loss_l1 = l1_loss(predictions, targets)
            loss_ssim = 1 - ssim(predictions, targets, **similarity)
            loss_aniso = anisotropic_scale_loss(
                gaussians.get_scaling, args.anisotropic_max_ratio, args.anisotropic_epsilon,
                reference_count=gaussians.geometry_reference_count)
            loss_interp_l1 = imgs.new_zeros(())
            loss_interp_ssim = imgs.new_zeros(())
            if args.interpolation and iteration >= args.interpolate_iter and s > 1:
                if random.random() < args.interpolation_probability:
                    alpha = random.uniform(args.interpolation_alpha_min,
                                           args.interpolation_alpha_max)
                    left = random.randrange(max(1, s - 2))
                    intervals = range(left, min(left + 2, s - 1))
                    interp_outputs, interp_targets = [], []
                    for index in intervals:
                        interp_outputs.append(render(
                            gaussians, [h, w], index + alpha, 2, s, **rendering)['render'])
                        interp_targets.append((1 - alpha) * imgs[index] + alpha * imgs[index + 1])
                    interp_pred = torch.stack(interp_outputs)[:, None]
                    interp_gt = torch.stack(interp_targets)[:, None]
                else:
                    index = random.randrange(s)
                    interp_pred = render(gaussians, [h, w], index, 2, s,
                                         **rendering)['render'][None, None]
                    interp_gt = imgs[index][None, None]
                loss_interp_l1 = l1_loss(interp_pred, interp_gt)
                loss_interp_ssim = 1 - ssim(interp_pred, interp_gt, **similarity)
            loss = (args.lambda_l1 * loss_l1 + args.lambda_ssim * loss_ssim
                    + args.lambda_anisotropic * loss_aniso
                    + args.lambda_interp_l1 * loss_interp_l1
                    + args.lambda_interp_ssim * loss_interp_ssim)
            # if not torch.isfinite(loss):
            #     raise FloatingPointError(f'{sample_name}: 第 {iteration} 步损失非有限')
            loss.backward()
            with torch.no_grad():
                if iteration < args.densify_until_iter:
                    visible = torch.stack([output['visibility_filter'] for output in outputs]).any(dim=0)
                    gaussians.add_densification_stats_3d(visible)
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                gaussians._scaling.clamp_(args.log_scale_min, args.log_scale_max)
                gaussians.prune_points(gaussians._invalid_point_mask())
                if gaussians.get_xyz.shape[0] == 0:
                    raise FloatingPointError(f'{sample_name}: 优化后无有效高斯点')
                if (args.densify_from_iter <= iteration < args.densify_until_iter
                        and args.densification_interval > 0
                        and iteration % args.densification_interval == 0):
                    gaussians.structural_step(imgs, iteration, rendering, similarity)
            current = {'Total Loss': loss.item(), 'L1 Loss': loss_l1.item(),
                       'SSIM Loss': loss_ssim.item(), 'Aniso': loss_aniso.item(),
                       'L1 interpo': loss_interp_l1.item(), 'ssim inter': loss_interp_ssim.item()}
            history_iters.append(iteration)
            for name, value in current.items():
                history_losses.setdefault(name, []).append(value)
            if iteration % args.status_interval == 0:
                pbar.set_postfix(loss=current['Total Loss'], points=gaussians.get_xyz.shape[0])
            if iteration % args.loss_plot_interval == 0 or iteration == args.iters:
                update_loss_plot(case_dir, history_iters, history_losses, args.plot_dpi)
            if (args.reconstruct_interval > 0 and iteration % args.reconstruct_interval == 0):
                reconstruct_full_case(
                    gaussians, s, h, w, case_dir, iteration,
                    args.interpolation and iteration >= args.interpolate_iter,
                    args.interp_step_size, **rendering)
            if args.save_ply and iteration in args.save_iterations:
                save_model_ply(gaussians,os.path.join(case_dir, f'point_cloud_iter_{iteration:06d}.ply'),iteration)
            if iteration % args.checkpoint_interval == 0 or iteration == args.iters:
                save_training_checkpoint(checkpoint_path, gaussians, iteration,
                                         history_iters, history_losses, args)
        if args.save_ply:
            save_model_ply(gaussians,final_ply,max(start_iteration,args.iters))
        del gaussians, state
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_train.yaml')
    training(load_config(parser.parse_args().config))
