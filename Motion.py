import os
import random
import re
import csv
import numpy as np
import torch
import yaml
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Gaussians.GaussianModel import GaussianModel
from Gaussians.GaussianEvolution import GaussianEvolutionNetwork
from Dataloader import dataset
from CostFunctions import l1_loss, ssim
from Renderer import render


MOTION_CODE_VERSION = "20260917_control_graph_slice_training_modes_v9"


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


class TimeSliceSampler:
    """采样非参考时间，并按训练模式返回一张或全部切片。"""

    def __init__(self, time_steps, num_slices, mode='balanced',
                 slice_training_mode='random_slice'):
        self.time_steps = int(time_steps)
        self.num_slices = int(num_slices)
        self.mode = str(mode).lower()
        self.slice_training_mode = str(slice_training_mode).lower()
        if self.mode not in ['balanced', 'random']:
            raise ValueError("stage2_sampling_mode 仅支持 'balanced' 或 'random'")
        if self.slice_training_mode not in ['random_slice', 'all_slices']:
            raise ValueError(
                "stage2_slice_training_mode 仅支持 "
                "'random_slice' 或 'all_slices'"
            )
        if self.num_slices <= 0:
            raise ValueError("num_slices 必须大于 0")
        self._samples = []
        self._cursor = 0
        self._reshuffle()

    def _reshuffle(self):
        phase_indices = list(range(1, self.time_steps))
        if not phase_indices:
            phase_indices = [0]
        if self.slice_training_mode == 'random_slice':
            self._samples = [
                (phase_idx, slice_idx)
                for phase_idx in phase_indices
                for slice_idx in range(self.num_slices)
            ]
        else:
            self._samples = [(phase_idx, None) for phase_idx in phase_indices]
        random.shuffle(self._samples)
        self._cursor = 0

    def sample(self):
        if self.mode == 'random':
            phase_idx = random.randint(1, self.time_steps - 1) \
                if self.time_steps > 1 else 0
            if self.slice_training_mode == 'random_slice':
                slice_idx = random.randint(0, self.num_slices - 1)
                return phase_idx, [slice_idx]
            return phase_idx, list(range(self.num_slices))
        if self._cursor >= len(self._samples):
            self._reshuffle()
        phase_idx, slice_idx = self._samples[self._cursor]
        self._cursor += 1
        if self.slice_training_mode == 'random_slice':
            return phase_idx, [slice_idx]
        return phase_idx, list(range(self.num_slices))


def get_sample_name(data, sample_idx):
    if 'name' in data:
        sample_name = data['name']
        if isinstance(sample_name, (list, tuple)):
            sample_name = sample_name[0]
        return str(sample_name)
    return f"sample_{sample_idx:03d}"


def find_latest_stage1_ply(case_directories):
    """优先读取迭代次数最大的 Stage 1 PLY，并兼容最终模型命名。"""
    iteration_candidates = []
    final_candidates = []

    for case_directory in dict.fromkeys(case_directories):
        if not os.path.isdir(case_directory):
            continue

        final_path = os.path.join(case_directory, "point_cloud_final.ply")
        if os.path.exists(final_path):
            final_candidates.append(final_path)

        for file_name in os.listdir(case_directory):
            matched = re.fullmatch(r"point_cloud_iter_(\d+)\.ply", file_name)
            if matched is None:
                continue
            iteration_candidates.append((
                int(matched.group(1)),
                os.path.join(case_directory, file_name),
            ))
    if final_candidates:
        return final_candidates[0]
    if iteration_candidates:
        return max(iteration_candidates, key=lambda item: item[0])[1]
    return None


def update_loss_plot(out_dir, iters_list, loss_dict):
    """将当前 sample 的全部 Stage 2 损失更新到同一张走势图中。"""
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "loss_curves.png")
    temporary_plot_path = os.path.join(out_dir, "loss_curves.tmp.png")
    figure = plt.figure(figsize=(10, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6']

    if not iters_list:
        plt.text(
            0.5, 0.5, "Waiting for the first iteration...",
            horizontalalignment='center', verticalalignment='center',
            transform=plt.gca().transAxes
        )
    elif "Total Loss" in loss_dict:
        plt.plot(
            iters_list, loss_dict["Total Loss"], label="Total Loss",
            color=colors[0], linewidth=2.0
        )

    color_idx = 1
    for loss_name, loss_values in loss_dict.items():
        if loss_name == "Total Loss":
            continue
        plt.plot(
            iters_list, loss_values, label=loss_name,
            color=colors[color_idx % len(colors)],
            linestyle='--', alpha=0.8, linewidth=1.2
        )
        color_idx += 1

    plt.title("GaussianEvolution Loss Trajectory", fontsize=14, fontweight='bold')
    plt.xlabel("Iteration Step", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    if iters_list:
        plt.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    figure.savefig(temporary_plot_path, dpi=150)
    plt.close(figure)
    os.replace(temporary_plot_path, plot_path)
    return plot_path


def update_loss_history_csv(out_dir, iters_list, loss_dict):
    """将逐 iteration 损失写入 CSV，便于核对曲线数据是否持续更新。"""
    if not iters_list:
        return None

    csv_path = os.path.join(out_dir, "loss_history.csv")
    temporary_csv_path = os.path.join(out_dir, "loss_history.tmp.csv")
    loss_names = list(loss_dict.keys())
    with open(temporary_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Iteration"] + loss_names)
        for history_idx, iteration in enumerate(iters_list):
            writer.writerow([
                iteration,
                *[loss_dict[loss_name][history_idx] for loss_name in loss_names],
            ])
    os.replace(temporary_csv_path, csv_path)
    return csv_path


def prepare_cmr_sequence(cmr, cmr_layout='STHW'):
    """将 DataLoader 输出统一整理为 [T, S, H, W]。"""
    if cmr.ndim == 5:
        cmr = cmr[0]
    if cmr.ndim != 4:
        raise ValueError(f"cmr 应为 [S,T,H,W] 或 [T,S,H,W]，实际为 {tuple(cmr.shape)}")

    cmr_layout = cmr_layout.upper()
    if cmr_layout == 'STHW':
        cmr = cmr.permute(1, 0, 2, 3)
    elif cmr_layout != 'TSHW':
        raise ValueError("cmr_layout 仅支持 'STHW' 或 'TSHW'")
    return cmr.contiguous().cuda(non_blocking=True)


def build_evolution_network(args):
    return GaussianEvolutionNetwork(
        hidden_dim=getattr(args, 'evolution_hidden_dim', 128),
        kinematics_depth=getattr(args, 'kinematics_depth', 4),
        optics_depth=getattr(args, 'optics_depth', 3),
        residual_fourier_bands=getattr(args, 'residual_fourier_bands', 3),
        residual_scale=getattr(args, 'residual_scale', 1.0),
        motion_dimension=getattr(args, 'motion_dimension', '2D'),
        through_plane_max_displacement=getattr(
            args, 'through_plane_max_displacement', 0.1
        ),
        transverse_max_log_scale=getattr(args, 'transverse_max_log_scale', 0.7),
        control_deformation_mode=getattr(
            args, 'control_deformation_mode', 'kernel'
        ),
        control_coverage_radius=getattr(
            args, 'control_coverage_radius', 0.12
        ),
        control_min_points=getattr(args, 'control_min_points', 32),
        control_max_points=getattr(args, 'control_max_points', 512),
        control_neighbors=getattr(args, 'control_neighbors', 8),
        control_kernel_bandwidth=getattr(
            args, 'control_kernel_bandwidth', 0.0
        ),
        control_mls_regularization=getattr(
            args, 'control_mls_regularization', 1e-4
        ),
        control_knn_chunk_size=getattr(
            args, 'control_knn_chunk_size', 32768
        ),
    ).cuda()


def build_optimizer(evolution_net, args):
    kinematics_parameters = list(evolution_net.kinematics_trunk.parameters())
    if evolution_net.motion_dimension == '2D':
        kinematics_parameters += list(evolution_net.transverse_trunk.parameters())

    optimizer = torch.optim.Adam([
        {
            'params': kinematics_parameters,
            'lr': getattr(args, 'kinematics_lr', 1e-4),
        },
        {
            'params': evolution_net.optics_trunk.parameters(),
            'lr': getattr(args, 'optics_lr', 5e-5),
            'weight_decay': getattr(args, 'optics_weight_decay', 0.0),
        },
    ], eps=getattr(args, 'stage2_adam_eps', 1e-15))
    return optimizer


def save_checkpoint(path, evolution_net, optimizer, scheduler, iteration, args,
                    history_iters, history_losses_dict):
    torch.save({
        'iteration': iteration,
        'model': evolution_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'config': vars(args),
        'loss_history': {
            'iterations': history_iters,
            'losses': history_losses_dict,
        },
    }, path)


def load_checkpoint(path, evolution_net, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, map_location='cuda')
    if 'model' not in checkpoint:
        evolution_net.load_state_dict(checkpoint)
        return 0, [], {}

    evolution_net.load_state_dict(checkpoint['model'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    loss_history = checkpoint.get('loss_history', {})
    history_iters = list(loss_history.get('iterations', []))
    history_losses_dict = {
        loss_name: list(loss_values)
        for loss_name, loss_values in loss_history.get('losses', {}).items()
    }
    return int(checkpoint.get('iteration', 0)), history_iters, history_losses_dict


def export_motion_sequence(evolution_net, base_xyz, base_scaling, base_rotation,
                           time_steps, num_ode_steps, save_path):
    """导出高斯轨迹，并区分运动 Jacobian 与高斯支撑体积比。"""
    evolution_net.eval()
    xyz_sequence = []
    displacement_sequence = []
    flow_jacobian_sequence = []
    support_volume_sequence = []
    transverse_stretch_sequence = []

    for phase_idx in tqdm(range(time_steps), desc="Exporting motion", leave=False):
        target_time_t = float(phase_idx) / max(1, time_steps - 1)
        with torch.enable_grad():
            evolved = evolution_net(
                base_xyz, base_scaling, base_rotation,
                target_time_t=target_time_t,
                num_ode_steps=num_ode_steps,
                create_graph=False,
            )
        xyz_sequence.append(evolved['xyz'].detach().cpu().numpy())
        displacement_sequence.append(evolved['displacement'].detach().cpu().numpy())
        flow_jacobian_sequence.append(
            evolved['flow_jacobian_det'].detach().cpu().numpy()
        )
        support_volume_sequence.append(
            evolved['support_volume_ratio'].detach().cpu().numpy()
        )
        transverse_stretch_sequence.append(
            evolved['transverse_stretch'].detach().cpu().numpy()
        )

    np.savez_compressed(
        save_path,
        time=np.linspace(0.0, 1.0, time_steps, dtype=np.float32),
        reference_xyz=base_xyz.detach().cpu().numpy(),
        xyz=np.stack(xyz_sequence, axis=0),
        displacement=np.stack(displacement_sequence, axis=0),
        flow_jacobian_det=np.stack(flow_jacobian_sequence, axis=0),
        support_volume_ratio=np.stack(support_volume_sequence, axis=0),
        transverse_stretch=np.stack(transverse_stretch_sequence, axis=0),
        motion_dimension=np.asarray(evolution_net.motion_dimension),
        control_deformation_mode=np.asarray(
            evolution_net.control_deformation_mode
        ),
        control_indices=evolution_net.control_indices.detach().cpu().numpy(),
        reference_control_xyz=(
            evolution_net.reference_control_xyz.detach().cpu().numpy()
        ),
        through_plane_max_displacement=np.asarray(
            evolution_net.through_plane_max_displacement, dtype=np.float32
        ),
    )


def export_dense_flow(evolution_net, time_steps, num_slices, height, width,
                      num_ode_steps, chunk_size, save_path):
    """在成像切片网格上查询参考相位到所有相位的稠密拉格朗日位移。"""
    device = next(evolution_net.parameters()).device
    dtype = next(evolution_net.parameters()).dtype
    coordinate_z = torch.linspace(-1.0, 1.0, num_slices, device=device, dtype=dtype)
    coordinate_y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    coordinate_x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_z, grid_y, grid_x = torch.meshgrid(
        coordinate_z, coordinate_y, coordinate_x, indexing='ij'
    )
    reference_grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    chunk_size = max(1, int(chunk_size))

    pixel_flow_sequence = []
    through_plane_sequence = []
    evolution_net.eval()
    for phase_idx in tqdm(range(time_steps), desc="Exporting dense flow", leave=False):
        target_time_t = float(phase_idx) / max(1, time_steps - 1)
        phase_displacement = []
        with torch.no_grad():
            evolved_control_xyz = None
            if evolution_net.uses_control_points:
                evolved_control_xyz = evolution_net.integrate_control_positions(
                    target_time_t, num_ode_steps=num_ode_steps
                )
            for start_idx in range(0, reference_grid.shape[0], chunk_size):
                query_points = reference_grid[start_idx:start_idx + chunk_size]
                evolved_points = evolution_net.deform_query_points(
                    query_points,
                    target_time_t,
                    num_ode_steps=num_ode_steps,
                    evolved_control_xyz=evolved_control_xyz,
                )
                phase_displacement.append((evolved_points - query_points).cpu())

        phase_displacement = torch.cat(phase_displacement, dim=0)
        phase_displacement = phase_displacement.reshape(num_slices, height, width, 3)
        pixel_flow = torch.stack([
            phase_displacement[..., 0] * max(1, width - 1) / 2.0,
            phase_displacement[..., 1] * max(1, height - 1) / 2.0,
        ], dim=1)
        pixel_flow_sequence.append(pixel_flow.numpy())
        through_plane_sequence.append(phase_displacement[..., 2].numpy())

    np.savez_compressed(
        save_path,
        time=np.linspace(0.0, 1.0, time_steps, dtype=np.float32),
        flow_pixel=np.stack(pixel_flow_sequence, axis=0),
        through_plane_normalized=np.stack(through_plane_sequence, axis=0),
        motion_dimension=np.asarray(evolution_net.motion_dimension),
        control_deformation_mode=np.asarray(
            evolution_net.control_deformation_mode
        ),
        control_indices=evolution_net.control_indices.detach().cpu().numpy(),
        through_plane_max_displacement=np.asarray(
            evolution_net.through_plane_max_displacement, dtype=np.float32
        ),
    )


def training(args):
    set_random_seed(getattr(args, 'random_seed', 42))
    tqdm.write(f"Motion code version: {MOTION_CODE_VERSION}")
    slice_training_mode = str(getattr(
        args, 'stage2_slice_training_mode', 'random_slice'
    )).lower()
    if slice_training_mode not in ['random_slice', 'all_slices']:
        raise ValueError(
            "stage2_slice_training_mode 仅支持 "
            "'random_slice' 或 'all_slices'"
        )
    tqdm.write(f"Stage 2 切片训练模式: {slice_training_mode}")

    Dataset = torch.utils.data.DataLoader(
        dataset(
            mode=getattr(args, 'stage2_dataset_mode', 'val'),
            datasets=getattr(args, 'stage2_dataset_name', 'acdc'),
            align=False,
            augmentation=False,
        ),
        batch_size=1, num_workers=0, drop_last=True, pin_memory=True,
    )

    stage2_out_dir = os.path.join(args.output_path, "stage2_experiments")
    os.makedirs(stage2_out_dir, exist_ok=True)
    total_cases = len(Dataset)

    for sample_idx, data in enumerate(Dataset):
        # sample_name = get_sample_name(data, sample_idx)
        sample_name = data['info'][0].split('.')[0]
        case_out_dir = os.path.join(stage2_out_dir, sample_name)
        os.makedirs(case_out_dir, exist_ok=True)
        training_artifact_dir = os.path.join(
            case_out_dir, f"training_{slice_training_mode}"
        )
        os.makedirs(training_artifact_dir, exist_ok=True)
        # 不同切片训练模式使用独立结果文件，避免某一模式的既有结果
        # 导致另一模式被 stage2_skip_existing 错误跳过。
        motion_path = os.path.join(
            case_out_dir, f"motion_sequence_{slice_training_mode}.npz"
        )
        dense_flow_path = os.path.join(
            case_out_dir, f"dense_flow_{slice_training_mode}.npz"
        )

        expected_outputs = []
        if getattr(args, 'stage2_export_motion', False):
            expected_outputs.append(motion_path)
        if getattr(args, 'stage2_export_dense_flow', False):
            expected_outputs.append(dense_flow_path)
        outputs_complete = expected_outputs and all(
            os.path.exists(path) for path in expected_outputs
        )
        if getattr(args, 'stage2_skip_existing', True) and outputs_complete:
            tqdm.write(f"[{sample_name}] 已存在完整运动结果，跳过。")
            continue

        tqdm.write("\n==================================================")
        tqdm.write(f"开始处理 Case [{sample_idx + 1}/{total_cases}]: {sample_name}")

        cmr_4d = prepare_cmr_sequence(
            data['cmr'], cmr_layout=getattr(args, 'cmr_layout', 'STHW')
        )
        time_steps, num_slices, height, width = cmr_4d.shape

        stage1_case_directories = [
            os.path.join(args.output_path, sample_name),
        ]
        if 'info' in data:
            info_name = data['info'][0] if isinstance(data['info'], (list, tuple)) else data['info']
            stage1_case_directories.append(
                os.path.join(args.output_path, str(info_name))
            )
        stage1_ply_path = find_latest_stage1_ply(stage1_case_directories)
        if stage1_ply_path is None:
            tqdm.write(f"[警告] 找不到第一阶段高斯模型，跳过 {sample_name}。")
            continue

        BaseGaussians = GaussianModel()
        tqdm.write(f"读取第一阶段高斯检查点: {stage1_ply_path}")
        BaseGaussians.load_ply(stage1_ply_path)
        tqdm.write(f"成功加载第一阶段高斯: {BaseGaussians.get_xyz.shape[0]} 个点。")

        base_xyz = BaseGaussians.get_xyz.detach()
        base_scaling = BaseGaussians._scaling.detach()
        base_rotation = BaseGaussians._rotation.detach()
        base_colors = BaseGaussians.get_color.detach()
        base_opacities = BaseGaussians.get_opacity.detach()

        evolution_net = build_evolution_network(args)
        control_count = evolution_net.configure_control_points(base_xyz)
        tqdm.write(
            f"Stage 2 运动模式: {evolution_net.motion_dimension}; "
            f"传播方法: {evolution_net.control_deformation_mode}; "
            f"控制点: {control_count}/{base_xyz.shape[0]}; "
            f"切片训练: {slice_training_mode}"
        )
        optimizer_net = build_optimizer(evolution_net, args)
        with torch.no_grad():
            base_covariance = evolution_net.build_base_covariance(
                base_scaling, base_rotation
            )
            image_coordinate_features = evolution_net.build_image_coordinate_features(
                height, width, base_xyz.device, base_xyz.dtype
            )
        stage2_iters = getattr(args, "stage2_iters", 20000)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_net, T_max=max(1, stage2_iters), eta_min=0.0
        )

        checkpoint_path = os.path.join(
            training_artifact_dir,
            (
                f"net_latest_{evolution_net.control_deformation_mode}_"
                f"{slice_training_mode}.pth"
            ),
        )
        start_iteration = 0
        history_iters = []
        history_losses_dict = {}
        if getattr(args, 'stage2_resume', True) and os.path.exists(checkpoint_path):
            start_iteration, history_iters, history_losses_dict = load_checkpoint(
                checkpoint_path, evolution_net, optimizer_net, scheduler
            )
            tqdm.write(f"从第 {start_iteration} 次迭代继续训练。")

        loss_plot_path = update_loss_plot(
            training_artifact_dir, history_iters, history_losses_dict
        )
        if history_iters:
            update_loss_history_csv(
                training_artifact_dir, history_iters, history_losses_dict
            )
        tqdm.write(f"损失曲线文件: {os.path.abspath(loss_plot_path)}")

        pbar = tqdm(
            range(start_iteration + 1, stage2_iters + 1),
            desc=f"Training motion on {sample_name}", leave=False
        )
        num_ode_steps = getattr(args, 'num_ode_steps', 4)
        lambda_volume = getattr(args, 'lambda_volume', 0.1)
        lambda_smoothness = getattr(args, 'lambda_smoothness', 0.01)
        lambda_folding = getattr(args, 'lambda_folding', 0.0)
        lambda_control_anchor = getattr(args, 'lambda_control_anchor', 1.0)
        folding_epsilon = getattr(args, 'folding_epsilon', 0.0)
        ssim_weight = getattr(args, 'stage2_ssim_weight', 0.0)
        time_slice_sampler = TimeSliceSampler(
            time_steps,
            num_slices,
            mode=getattr(args, 'stage2_sampling_mode', 'balanced'),
            slice_training_mode=slice_training_mode,
        )

        for iter_i in pbar:
            evolution_net.train()
            optimizer_net.zero_grad(set_to_none=True)

            # Stage 1 已监督 t=0。运动只演化一次；随后按配置渲染一张
            # 随机切片或当前目标时间的全部切片。
            phase_idx, slice_indices = time_slice_sampler.sample()
            target_time_t = float(phase_idx) / max(1, time_steps - 1)

            evolved = evolution_net(
                base_xyz, base_scaling, base_rotation,
                target_time_t=target_time_t,
                num_ode_steps=num_ode_steps,
                create_graph=True,
                base_covariance=base_covariance,
            )

            # 与文章一致：L = L_R + lambda_V L_V + lambda_S L_S。
            # 支撑体积用于准不可压缩；真实运动 Jacobian 单独用于折叠约束。
            loss_volume = torch.mean(
                (evolved['support_volume_ratio'] - 1.0) ** 2
            )
            loss_smoothness = evolved['spatial_smoothness']
            loss_folding = torch.mean(
                torch.relu(
                    folding_epsilon - evolved['flow_jacobian_det']
                ) ** 2
            )
            negative_jacobian_ratio = torch.mean(
                (evolved['flow_jacobian_det'] <= 0).float()
            )
            regularization_loss = (
                lambda_volume * loss_volume
                + lambda_smoothness * loss_smoothness
                + lambda_folding * loss_folding
                + lambda_control_anchor * evolved['control_anchor_loss']
            )

            image_residual = evolution_net.image_residual_from_features(
                image_coordinate_features, height, width, target_time_t
            )
            loss_render = evolved['xyz'].new_zeros(())
            loss_ssim = evolved['xyz'].new_zeros(())
            for slice_idx in slice_indices:
                render_pkg = render(
                    BaseGaussians,
                    img_size=[height, width],
                    slice_idx=slice_idx,
                    axis=2,
                    num_slices=num_slices,
                    xyz=evolved['xyz'],
                    covariance=evolved['covariance'],
                    colors=base_colors,
                    opacities=base_opacities,
                )
                pred_image = render_pkg['render'] + image_residual
                gt_image = cmr_4d[phase_idx, slice_idx]
                loss_render = loss_render + l1_loss(pred_image, gt_image)
                if ssim_weight > 0:
                    loss_ssim = loss_ssim + 1.0 - ssim(
                        pred_image[None, None], gt_image[None, None]
                    )

            # 两种模式都使用切片平均损失，保持损失尺度和权重一致。
            rendered_slice_count = len(slice_indices)
            loss_render = loss_render / rendered_slice_count
            if ssim_weight > 0:
                loss_ssim = loss_ssim / rendered_slice_count
            total_loss = (
                loss_render
                + regularization_loss
                + ssim_weight * loss_ssim
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"{sample_name} iteration {iter_i} 出现非有限损失。"
                )

            total_loss.backward()

            grad_clip = getattr(args, 'stage2_grad_clip', 0.0)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(evolution_net.parameters(), max_norm=grad_clip)
            optimizer_net.step()
            scheduler.step()

            current_losses = {
                "Total Loss": total_loss.item(),
                "Reconstruction Loss": loss_render.item(),
                "Volume Loss": loss_volume.item(),
                "Smoothness Loss": loss_smoothness.item(),
                "Folding Loss": loss_folding.item(),
                "Control Anchor Loss": evolved['control_anchor_loss'].item(),
                "Negative Jacobian Ratio": negative_jacobian_ratio.item(),
                "SSIM Loss": loss_ssim.item(),
            }
            history_iters.append(iter_i)
            for loss_name, loss_value in current_losses.items():
                if loss_name not in history_losses_dict:
                    history_losses_dict[loss_name] = []
                history_losses_dict[loss_name].append(loss_value)

            loss_plot_interval = max(
                1, int(getattr(args, 'stage2_loss_plot_interval', 1000))
            )
            update_loss_curve = (
                iter_i == start_iteration + 1
                or iter_i % loss_plot_interval == 0
                or iter_i == stage2_iters
            )
            if update_loss_curve:
                loss_plot_path = update_loss_plot(
                    training_artifact_dir, history_iters, history_losses_dict
                )
                loss_history_path = update_loss_history_csv(
                    training_artifact_dir, history_iters, history_losses_dict
                )
                tqdm.write(
                    f"[{sample_name}] Iteration {iter_i}: "
                    f"损失曲线已更新至 {os.path.abspath(loss_plot_path)}；"
                    f"数据已更新至 {os.path.abspath(loss_history_path)}"
                )

            slice_status = (
                slice_indices[0]
                if len(slice_indices) == 1
                else f"all({len(slice_indices)})"
            )
            pbar.set_postfix({
                "Loss": f"{current_losses['Total Loss']:.5f}",
                "Rec": f"{current_losses['Reconstruction Loss']:.5f}",
                "Vol": f"{current_losses['Volume Loss']:.5f}",
                "Smooth": f"{current_losses['Smoothness Loss']:.5f}",
                "Jmot": f"{evolved['flow_jacobian_det'].mean().item():.4f}",
                "Jsup": f"{evolved['support_volume_ratio'].mean().item():.4f}",
                "NegJ": f"{negative_jacobian_ratio.item():.4f}",
                "Ctrl": control_count,
                "T": phase_idx,
                "S": slice_status,
            })

            save_interval = getattr(args, 'stage2_save_interval', 1000)
            if iter_i % save_interval == 0 or iter_i == stage2_iters:
                save_checkpoint(
                    checkpoint_path, evolution_net, optimizer_net,
                    scheduler, iter_i, args,
                    history_iters, history_losses_dict
                )

        if start_iteration >= stage2_iters:
            tqdm.write(f"[{sample_name}] 网络训练已经完成，直接导出运动结果。")

        if getattr(args, 'stage2_export_motion', False):
            export_motion_sequence(
                evolution_net, base_xyz, base_scaling, base_rotation,
                time_steps=time_steps, num_ode_steps=num_ode_steps,
                save_path=motion_path,
            )
            tqdm.write(f"运动结果已保存至: {motion_path}")

        if getattr(args, 'stage2_export_dense_flow', False):
            export_dense_flow(
                evolution_net,
                time_steps=time_steps, num_slices=num_slices,
                height=height, width=width,
                num_ode_steps=num_ode_steps,
                chunk_size=getattr(args, 'dense_flow_chunk_size', 32768),
                save_path=dense_flow_path,
            )
            tqdm.write(f"稠密运动场已保存至: {dense_flow_path}")

        del BaseGaussians, evolution_net, optimizer_net, scheduler
        del base_covariance, image_coordinate_features
        torch.cuda.empty_cache()


def load_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        config_dict = yaml.safe_load(file)
    return argparse.Namespace(**config_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument(
        '--control-deformation-mode',
        choices=['none', 'kernel', 'mls'],
        default=None,
        help=(
            "none=原始全高斯ODE；kernel=控制点核传播；"
            "mls=控制点局部MLS/仿射变形图"
        ),
    )
    parser.add_argument(
        '--control-coverage-radius', type=float, default=None,
        help="自适应FPS的覆盖半径（与高斯坐标使用相同单位）",
    )
    parser.add_argument(
        '--control-neighbors', type=int, default=None,
        help="每个高斯连接的邻近控制点数量",
    )
    parser.add_argument(
        '--slice-training-mode',
        choices=['random_slice', 'all_slices'],
        default=None,
        help=(
            "random_slice=每个iter随机时间和随机切片；"
            "all_slices=每个iter随机时间并渲染全部切片"
        ),
    )
    cli_args = parser.parse_args()
    config_args = load_config(cli_args.config)
    if cli_args.control_deformation_mode is not None:
        config_args.control_deformation_mode = cli_args.control_deformation_mode
    if cli_args.control_coverage_radius is not None:
        config_args.control_coverage_radius = cli_args.control_coverage_radius
    if cli_args.control_neighbors is not None:
        config_args.control_neighbors = cli_args.control_neighbors
    if cli_args.slice_training_mode is not None:
        config_args.stage2_slice_training_mode = cli_args.slice_training_mode
    training(config_args)
