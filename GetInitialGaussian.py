import os
import torch
import torch.nn.functional as F
from Gaussians.GaussianModel_noop import GaussianModel
from Dataloader import dataset
from CostFunctions import l1_loss, ssim, slice_tv_loss, penalize_outside_range, psnr, anisotropic_scale_loss
from Renderer import render
import yaml
import argparse
import random
from tools import subplots
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# ==================== [功能模块 1: 动态自适应子损失函数趋势图] ====================
def update_loss_plot(out_dir, iters_list, loss_dict):
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
    plt.savefig(plot_path, dpi=150)
    plt.close('all')  # 必须清理显存，防止内存泄漏


# ==================== [功能模块 2: 周期性全序列快照重建 (含插帧联动)] ====================
def reconstruct_full_case(Gaussians, s, h, w, out_dir, iter_num, include_interp, step_size=0.5):
    """重建当前模型在整个 Case 下的所有切片。如果已进入插帧期，将自动采用亚像素步长进行细分插帧"""
    rec_dir = os.path.join(out_dir, f"reconstructions")
    os.makedirs(rec_dir, exist_ok=True)
    
    with torch.no_grad():
        if include_interp:
            # 自动计算插帧序列：例如 s=10, step=0.5，则生成 0.0, 0.5, 1.0 ... 9.0
            num_steps = int((s - 1) / step_size) + 1
            slice_indices = [i * step_size for i in range(num_steps)]
            mode_str = f"插帧细分模式 (步长={step_size}, 共 {len(slice_indices)} 帧)"
        else:
            # 基础序列：只重现原始的整数切片 0, 1, 2...
            slice_indices = list(range(s))
            mode_str = f"真实切片模式 (共 {len(slice_indices)} 帧)"
            
        # tqdm.write(f"\n📸 [Iteration {iter_num}] 正在执行全序列重建 -> {mode_str} ...")
        
        for idx in slice_indices:
            render_pkg = render(Gaussians, img_size=[h, w], slice_idx=idx, axis=2, num_slices=s)
            pred_tensor = render_pkg['render'].clamp(0, 1).cpu().numpy()
            
            # 命名支持浮点数：如 slice_000.00.png, slice_000.50.png
            file_name = f"slice_{idx:05.2f}.png"
            plt.imsave(os.path.join(rec_dir, file_name), pred_tensor, cmap='gray')
            
    # tqdm.write(f"✅ 重建快照完成！图像已保存至: {rec_dir}")
    plt.close('all')
    torch.cuda.empty_cache()


# ==================== [主训练流程] ====================
def training(args):
    dname = 'acdc'
    Dataset = torch.utils.data.DataLoader(
        dataset(mode="val", datasets=dname, align=False, augmentation=False),
        batch_size=1, 
        num_workers=0, 
        drop_last=True,
        pin_memory=True,
    )

    os.makedirs(args.output_path, exist_ok=True)
    total_cases = len(Dataset)

    for sample_idx, data in enumerate(Dataset):
        
        # if os.path.exists(os.path.join(r'output/acdc_experiment', f'sample_{sample_idx:03d}')):
        #     continue
        imgs = data['cmr'][0,:,0].cuda()    # initial time step
        lbls = data['cmr_lbl'].cuda()
        p_name = data['info'][0].split('.')[0]
        s, h, w = imgs.shape
        if os.path.exists(os.path.join(f'output/{dname}_experiment', f'{p_name}')):
            continue

        sample_name = data.get('name', [f"{p_name}"])[0] 
        sample_out_dir = os.path.join(args.output_path, str(sample_name))
        os.makedirs(sample_out_dir, exist_ok=True)
        
        Gaussians = GaussianModel()
        
        # --- 智能检查点与初始化 ---
        if getattr(args, "start_checkpoint", "") and os.path.exists(args.start_checkpoint):
            tqdm.write(f"[{sample_name}] Loading Gaussian checkpoint: {args.start_checkpoint}")
            Gaussians.load_ply(args.start_checkpoint)
        else:
            tqdm.write(f"[{sample_name}] Initializing Gaussian...")
            
            # [安全修复] 动态解包 ROI
            case_roi = None
            if 'roi_mask' in data and torch.is_tensor(data['roi_mask']):
                case_roi = data['roi_mask'][0].cuda()
                tqdm.write(f"[{sample_name}] -> 已挂载 ROI 掩码引导。")
                
            Gaussians.InitializeGaussians(
                num_pts = args.num_pts,         
                bounds = args.bounds, 
                roi_mask = case_roi                                
            )
            
        Gaussians.training_setup(args)

        # [新增模块]: 准备动态自适应 Loss 记录器
        history_iters = []
        history_losses_dict = {}

        # 进度条 UI 初始化
        desc_str = f"Case [{sample_idx + 1}/{total_cases}] ({sample_name})"
        status_bar = tqdm(total=0, position=0, bar_format='{desc}', leave=True)
        pbar = tqdm(range(1, args.iters + 1), desc=desc_str, position=1, dynamic_ncols=True, leave=True)
        status_bar.set_description_str(f"📈 [Real time] Gaussian num_pts: {Gaussians.get_xyz.shape[0]:,} Pts.")

        for iter in pbar:
            outputs = {}
            outputs_inter = {}
            interpolation = 0
            Gaussians.update_learning_rate(iter)
            idx_s = random.randint(1, s - 2)

            # --- 渲染真实切片 ---
            render_pkg = render(Gaussians, img_size=[h,w], slice_idx=idx_s-1, axis=2, num_slices=s)            
            render_pkg['gts'] = imgs[idx_s - 1]
            outputs['t1'] = render_pkg
            
            render_pkg = render(Gaussians, img_size=[h,w], slice_idx=idx_s+1, axis=2, num_slices=s)            
            render_pkg['gts'] = imgs[idx_s + 1]
            outputs['t2'] = render_pkg

            # --- 渲染插帧切片 (当跨越 densify_until_iter 后自动开启) ---
            if args.interpolation and iter >= args.densify_until_iter:
                interpolation = 1
                if torch.rand(1) > 0.5:
                    alpha = random.uniform(0.2, 0.8)
                    idx_1 = idx_s - 1 + alpha  
                    idx_2 = idx_s + alpha  
                    gt_1 = (1 - alpha) * imgs[idx_s-1] + alpha * imgs[idx_s]
                    gt_2 = (1 - alpha) * imgs[idx_s] + alpha * imgs[idx_s+1]

                    render_pkg_interp_1 = render(Gaussians, img_size=[h,w], slice_idx=idx_1, axis=2, num_slices=s)  
                    render_pkg_interp_1['gts'] = gt_1
                    outputs_inter['itp1'] = render_pkg_interp_1

                    render_pkg_interp_2 = render(Gaussians, img_size=[h,w], slice_idx=idx_2, axis=2, num_slices=s)  
                    render_pkg_interp_2['gts'] = gt_2
                    outputs_inter['itp2'] = render_pkg_interp_2

                else:
                    render_pkg_interp = render(Gaussians, img_size=[h,w], slice_idx=idx_s, axis=2, num_slices=s)  
                    render_pkg_interp['gts'] = imgs[idx_s]
                    outputs_inter['itp'] = render_pkg_interp

            # 拼合视角梯度与特征字典
            data_dict = {} 
            data_inter = {}
            for k in outputs["t1"].keys():
                # if k == "viewspace_points":
                #     data_dict[k] = [output[k] for output in outputs.values()]
                if k in ["visibility_filter", "radii"]:
                    data_dict[k] = [output[k] for output in outputs.values()]
                elif k in ["render", "gts", "mask"]:
                    data_dict[k] = torch.stack([output[k] for output in outputs.values()], dim=0)
                    if interpolation == 1:
                        data_inter[k] = torch.stack([output[k] for output in outputs_inter.values()], dim=0)
                elif k in ['consist_loss']:
                    data_dict[k] = sum([output[k] for output in outputs.values()])
            data_dict['mask_t'] = torch.stack(data_dict['visibility_filter'], dim=-1).any(dim=1)
            
            # ==================== Loss ====================
            Ll1 = l1_loss(data_dict["render"], data_dict["gts"])
            ssim_loss = 1.0 - ssim(data_dict['render'], data_dict['gts'])
            aniso_loss = anisotropic_scale_loss(Gaussians.get_scaling, max_ratio=6.0)
            if interpolation == 1:    #插帧
                Ll1_inter = l1_loss(data_inter["render"], data_inter["gts"])
                ssim_inter = 1.0 - ssim(data_inter['render'], data_inter['gts'])
            else:
                Ll1_inter = torch.tensor(0.0)
                ssim_inter = torch.tensor(0.0)

            
            weighted_l1 = 2.0 * Ll1
            weighted_ssim = 0.25 * ssim_loss
            weighted_aniso = 0.1 * aniso_loss
            # weighted_tv = tv_loss
            loss = weighted_l1 + weighted_ssim + weighted_aniso + Ll1_inter + 0.1 * ssim_inter
            loss.backward()

            # [扩展预留点] 未来只要在这往 current_losses 字典里加新的 loss (比如 "TV Loss": tv_loss.item())，图表上就会自动多一条线！
            current_losses = {
                "Total Loss": loss.item(),
                "L1 Loss": Ll1.item(),
                "SSIM Loss": ssim_loss.item(),
                "L1 interpo": Ll1_inter.item(),
                "ssim inter": ssim_inter.item(),
                "Aniso": aniso_loss.item(),
            }
            
            # 将当前状态记录进全局历史字典
            history_iters.append(iter)
            for k, v in current_losses.items():
                if k not in history_losses_dict:
                    history_losses_dict[k] = []
                history_losses_dict[k].append(v)

            # 每 100 步后台画图并保存一次
            plot_interval = getattr(args, "loss_plot_interval", 500)
            if iter % plot_interval == 0 or iter == args.iters:
                update_loss_plot(sample_out_dir, history_iters, history_losses_dict)
            # =================================================================

            # 上方状态行每 10 步刷新一次最新点数
            if iter % 10 == 0:
                pts_count = Gaussians.get_xyz.shape[0]
                status_bar.set_description_str(f"📈 [Real time] Gaussian num_pts: {pts_count:,} Pts | at iter {iter}")

            # 下方主进度条尾部刷新 Total Loss
            pbar.set_postfix({"Loss": f"{loss.item():.5f}"})
            
            # ==================== Optimization & Densification ====================

            densify_until = getattr(args, "densify_until_iter", 6000)

            if iter < densify_until:
                with torch.no_grad():
                    radii_batch = torch.stack(data_dict["radii"], dim=-1).max(dim=-1).values
                    visibility_mask = data_dict["mask_t"]
                    Gaussians.max_radii2D[visibility_mask] = torch.maximum(
                        Gaussians.max_radii2D[visibility_mask],
                        radii_batch[visibility_mask],
                    )
                    Gaussians.add_densification_stats_3d(visibility_mask)

            # 2. update Parameter before reset
            Gaussians.optimizer.step()
            # with torch.no_grad():
            #     Gaussians._scaling.clamp_(-10.0, 10.0)
            Gaussians.optimizer.zero_grad(set_to_none=True)

            # 3. control Gaussian
            if iter < densify_until:
                with torch.no_grad():
                    if (
                        iter >= args.densify_from_iter
                        and args.densification_interval > 0
                        and iter % args.densification_interval == 0
                    ):
                        # size_threshold = (args.max_screen_size if iter > args.opacity_reset_interval else None)
                        size_threshold = (args.max_screen_size if getattr(args, 'enable_size_pruning', False)
                                                    and iter > getattr(args, 'size_prune_from_iter', 2500)
                                                    else None)
                        # Gaussians.densify_and_prune(
                        #     max_grad=args.densify_grad_threshold,
                        #     min_opacity=args.min_opacity,
                        #     clone_scale_threshold=args.clone_scale_threshold,
                        #     split_scale_threshold=args.split_scale_threshold,
                        #     max_screen_size=size_threshold,
                        # )
                        Gaussians.densify_and_prune(
                            max_grad=args.densify_grad_threshold,
                            clone_scale_threshold=args.clone_scale_threshold,
                            split_scale_threshold=args.split_scale_threshold,
                            max_screen_size=size_threshold,
                        )


                    # if (
                    #     args.opacity_reset_interval > 0
                    #     and (
                    #         iter % args.opacity_reset_interval == 0
                    #         or iter == args.densify_from_iter
                    #     )
                    #     and iter < args.opacity_reset_untile_iter
                    # ):
                    #     Gaussians.reset_opacity()

            # ========================================================================

            # Gaussians.get_scaling.data.clamp_(-10, 10)
            # Gaussians.optimizer.step()
            # Gaussians.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                Gaussians._scaling.clamp_(-10.0, 10.0)

            # --- [周期重建] 检查并触发每隔 2000 步的全 Case 重建切片 ---
            rec_interval = getattr(args, "reconstruct_interval", 2000)
            if iter % rec_interval == 0 or iter == args.iters:
                # 动态判断是否处于插帧活跃期
                is_interp_active = args.interpolation and iter >= args.densify_until_iter
                step_val = getattr(args, "interp_step_size", 0.5)
                reconstruct_full_case(
                    Gaussians=Gaussians, s=s, h=h, w=w, 
                    out_dir=sample_out_dir, iter_num=iter, 
                    include_interp=is_interp_active, step_size=step_val
                )

            # 周期保存 PLY 模型点云检查点
            if getattr(args, "save_ply", True) and (iter in getattr(args, "save_iterations", [])):
                ply_path = os.path.join(sample_out_dir, f"point_cloud_iter_{iter:06d}.ply")
                tqdm.write(f"\n[Iteration {iter}] 达到设定节点，正在保存检查点至: {ply_path}")
                Gaussians.save_ply(ply_path)

        # 闭合终端进度条排版
        status_bar.close()
        pbar.close()

        # 结束时执行最后一次保存
        if getattr(args, "save_ply", True):
            final_ply_path = os.path.join(sample_out_dir, "point_cloud_final.ply")
            tqdm.write(f"\n---> 样本 [{sample_name}] 训练完成！保存最终结果至: {final_ply_path}\n")
            Gaussians.save_ply(final_ply_path)
        
        # 彻底清理历史记录与计算图，拥抱干净的下一次循环
        del Gaussians, outputs, data_dict, render_pkg, loss, Ll1, ssim_loss, imgs, lbls
        del history_iters, history_losses_dict
        torch.cuda.empty_cache()


def load_config(yaml_path): 
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    args = argparse.Namespace(**config_dict)
    return args


if __name__ == "__main__":
    args = load_config("config.yaml")
    training(args)
