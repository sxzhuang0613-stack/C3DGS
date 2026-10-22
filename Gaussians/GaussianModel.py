import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from plyfile import PlyData, PlyElement
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from .utils import (
    inverse_sigmoid, get_expon_lr_func, build_rotation, RGB2SH, SH2RGB,
    strip_symmetric, build_scaling_rotation
)
from simple_knn._C import distCUDA2


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


class GaussianModel:

    def __init__(self, ):
        # activations of Gaussian attributes
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.sigmoid
        self.m_activation = torch.softmax

        # Gaussian attributes
        self._xyz = torch.empty(0)
        self._color = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        # self.sigma = torch.empty(0)  # [SIGMA-DISABLED]
        self.time_func = torch.empty(0)
        self.use_dff = False 

    @property
    def get_rotation(self):
        return F.normalize(self._rotation, p=2, dim=-1)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    # [SIGMA-DISABLED]
    # @property
    # def get_sigma(self):
    #     return self.scaling_activation(self.sigma)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_time(self):
        if self.use_dff:
            return F.softmax(self.time_func.squeeze(), dim=0)
        else:
            return self.time_func

    @property
    def get_color(self):
        return self._color

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def InitializeGaussians(self, num_pts=50000, bounds=[1.0, 1.0, 1.0], roi_mask=None):
        """
        参数:
            num_pts (int): 初始高斯点云的总数量 (常量控制，彻底杜绝初始化显存爆炸)
            bounds (list/tuple): [x_bound, y_bound, z_bound] 正数边界，将自动映射为 [-bound, +bound]
            roi_mask (torch.Tensor, optional): 
                - 若传入 3D 二值张量 (H, W, D)：将在掩码 > 0 的物理坐标范围内采样。
                - 若传入已提取的空间点云坐标 (M, 3)：将直接从中采样并加扰动。
                - 若传入 None：降级为长方体包围盒全局均匀随机采样。
        """
        # 1. 对称边界自动解析
        x_b, y_b, z_b = bounds
        low = np.array([-x_b, -y_b, -z_b], dtype=np.float32)
        high = np.array([x_b, y_b, z_b], dtype=np.float32)

        # 2. 空间坐标采样 (带 ROI 引导与优雅降级)
        if roi_mask is not None and torch.is_tensor(roi_mask) and roi_mask.numel() > 0:
            print(f"--> [初始化] 检测到有效 ROI 引导，正在目标区域内生成 {num_pts:,} 个高斯种子...")
            
            # 分支 A1: 如果传入的是 (M, 3) 的空间坐标张量
            if roi_mask.ndim == 2 and roi_mask.shape[1] == 3:
                valid_coords = roi_mask.detach().float()
            # 分支 A2: 如果传入的是 (H, W, D) 的 3D 分割/前景掩码
            else:
                # 获取所有非零体素的索引，并归一化映射到 [-bound, +bound] 物理坐标系
                nonzero_indices = torch.nonzero(roi_mask > 0).float()
                if nonzero_indices.shape[0] == 0:
                    raise ValueError("传入的 roi_mask 中没有发现非零像素！")
                
                grid_size = torch.tensor(roi_mask.shape, device=nonzero_indices.device).float()
                # 归一化到 [0, 1] 后映射到 [low, high]
                norm_coords = nonzero_indices / (grid_size - 1.0)
                low_t = torch.from_numpy(low).to(norm_coords.device)
                high_t = torch.from_numpy(high).to(norm_coords.device)
                valid_coords = low_t + norm_coords * (high_t - low_t)

            # 有放回或无放回地随机采样 num_pts 个点
            M = valid_coords.shape[0]
            replace = True if M < num_pts else False
            select_idx = torch.randint(0, M, (num_pts,), device=valid_coords.device) if replace else torch.randperm(M, device=valid_coords.device)[:num_pts]
            base_points = valid_coords[select_idx]

            # [关键技术]: 添加亚体素均匀抖动 (Sub-voxel Jitter)，避免所有点死板地重叠在体素中心
            jitter_span = (high - low) / (np.cbrt(num_pts) * 2.0)
            jitter = torch.empty_like(base_points).uniform_(-1.0, 1.0) * torch.from_numpy(jitter_span).to(base_points.device)
            points = torch.clamp(base_points + jitter, min=torch.from_numpy(low).to(base_points.device), max=torch.from_numpy(high).to(base_points.device))
            points = points.cuda()
            
        else:
            # 分支 B: 未提供 ROI 或 ROI 无效，降级为自定义长方体均匀随机采样
            print(f"--> [初始化] 未提供 ROI，在全包围盒 X:±{x_b}, Y:±{y_b}, Z:±{z_b} 内均匀分布 {num_pts:,} 个高斯...")
            xyz_rand = np.random.uniform(low=low, high=high, size=(num_pts, 3))
            points = torch.from_numpy(xyz_rand).float().cuda()

        N = points.shape[0]

        # 3. 颜色初始化 (使用温和的中性灰调，提高早期渲染和求导的稳定性)
        shs = np.ones((N, 1), dtype=np.float32) * 0.5 
        shs = np.concatenate((shs, shs, shs), axis=1)
        colors = SH2RGB(shs)
        
        # 4. 随机四元数旋转
        rotation = torch.logit(torch.rand((N, 4), dtype=torch.float, device="cuda"))
        
        # 5. 基于自适应邻域距离的缩放尺度 (S0)
        # distCUDA2 会直接根据目前这 num_pts 个点的真实空间密度，精确计算邻域距离
        dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
        base_scales = torch.log(torch.sqrt(dist2))[..., None]
        
        # [进阶优化]: 针对医疗成像切片 Z 轴层间距偏大的各向异性，对 Z 轴赋予初始拉伸补偿
        scale_x = base_scales
        scale_y = base_scales
        scale_z = base_scales + torch.log(torch.tensor(2.0, device="cuda")) # 让初始椭球在层间自然衔接
        scales = torch.cat([scale_x, scale_y, scale_z], dim=1)
        
        # 6. 初始不透明度 (设定安全的低不透明度，配合后续的克隆与修剪机制)
        opacities = inverse_sigmoid(0.1 * torch.ones((N, 1), dtype=torch.float, device="cuda"))

        # 7. 绑定为 PyTorch 可优化参数
        self._xyz = nn.Parameter(points.requires_grad_(True))
        self._color = nn.Parameter(torch.tensor(colors, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

    def training_setup(self, args):
        # self.percent_dense 相关的变量不再用于判定，保留以便兼容其它流程
        self.percent_dense = getattr(args, 'percent_dense', 0.01) 
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # [修复]: 将 sigma 注册进优化器
        l = [
            {'params': [self._xyz], 'lr': args.position_lr_init, "name": "xyz"},
            {'params': [self._color], 'lr': args.color_lr, "name": "color"},
            {'params': [self._rotation], 'lr': args.rotation_lr, "name": "rotation"},
            {'params': [self._scaling], 'lr': args.scaling_lr, "name": "scaling"},
            {'params': [self._opacity], 'lr': args.opacity_lr, "name": "opacity"},
            # {'params': [self.sigma], 'lr': args.scaling_lr, "name": "sigma"},  # [SIGMA-DISABLED]
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=args.position_lr_init,
            lr_final=args.position_lr_final,
            lr_delay_mult=args.position_lr_delay_mult,
            max_steps=args.position_lr_max_steps
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def reset_opacity(self):
        '''reset opacity at intervals'''
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                clean_tensor = torch.nan_to_num(tensor.detach().contiguous(), nan=-10.0, neginf=-10.0, posinf=10.0)
                
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(clean_tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(clean_tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(clean_tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(clean_tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
                
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == 'time_func':
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._color = optimizable_tensors["color"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self.sigma = optimizable_tensors["sigma"]  # [SIGMA-DISABLED]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == 'time_func':
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # [SIGMA-DISABLED] 移除了形参 new_sigma
    def densification_postfix(self, new_xyz, new_color, new_opacities, new_scaling, new_rotation): # , new_sigma):
        d = {"xyz": new_xyz,
             "color": new_color,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             # "sigma": new_sigma,  # [SIGMA-DISABLED]
             }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._color = optimizable_tensors["color"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # self.sigma = optimizable_tensors["sigma"]  # [SIGMA-DISABLED]

        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        n_new = new_xyz.shape[0]
        device = self.get_xyz.device

        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros((n_new, 1), device=device),], dim=0)
        self.denom = torch.cat([self.denom, torch.zeros((n_new, 1), device=device),], dim=0)
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(n_new, device=device),], dim=0)

    def densify_and_split(self, grads, grad_threshold, split_scale_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        # 条件1：梯度大 且 尺度超过上限
        grad_condition = torch.logical_and(
            padded_grad >= grad_threshold,
            torch.max(self.get_scaling, dim=1).values > split_scale_threshold
        )
        
        # [核心修复 2]: 将强制分裂的容忍倍数从 1.5 倍稍微放宽至 2.5 倍
        # 防止过大的球在尚未收敛到正确坐标前就被集体无效打碎
        force_split_condition = torch.max(self.get_scaling, dim=1).values > (split_scale_threshold * 2.5)
        
        selected_pts_mask = grad_condition | force_split_condition

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        
        # [核心修复 3]: 将分裂尺度因子由 / (0.8 * N) [即/1.6] 改为 / (0.63 * N) [即/1.26]
        # 数学原理：让 1.26^3 ≈ 2，确保 1 变 2 分裂之后的“总体积”刚好与原来的 1 个大球完美等值，不留空洞！
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.63 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_color = self._color[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_color, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    # [核心修改]: 使用绝对尺度 clone_scale_threshold
    def densify_and_clone(self, grads, grad_threshold, clone_scale_threshold):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        
        # 仅当高斯体本身已经非常小，且重建不充分（高梯度）时，才触发克隆
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= clone_scale_threshold
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_color = self._color[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        # new_sigma = self.sigma[selected_pts_mask]  # [SIGMA-DISABLED]

        # [SIGMA-DISABLED] 移除了传参 new_sigma
        self.densification_postfix(new_xyz, new_color, new_opacities, new_scaling, new_rotation) # , new_sigma)

    # [核心修改]: 接口不再传入 extent，转而传入具体的克隆和分裂物理绝对阈值
    
    @torch.no_grad()
    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        clone_scale_threshold,
        split_scale_threshold,
        max_screen_size=None,
    ):
        n_before = self.get_xyz.shape[0]

        grads = torch.nan_to_num(
            self.xyz_gradient_accum / self.denom.clamp_min(1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # 1. Clone
        self.densify_and_clone(grads, max_grad, clone_scale_threshold)
        n_after_clone = self.get_xyz.shape[0]
        # 2. Split
        self.densify_and_split(grads, max_grad, split_scale_threshold)
        n_after_split = self.get_xyz.shape[0]
        # 3. Prune
        low_opacity = (self.get_opacity.squeeze(-1) < min_opacity)
        prune_mask = low_opacity.clone()

        screen_prune = 0
        world_prune = 0

        if max_screen_size is not None:
            big_screen = self.max_radii2D > max_screen_size
            big_world = (
                self.get_scaling.max(dim=1).values
                > split_scale_threshold * 5.0
            )

            screen_prune = int(big_screen.sum().item())
            world_prune = int(big_world.sum().item())

            prune_mask |= big_screen | big_world

        n_low_opacity = int(low_opacity.sum().item())
        n_pruned = int(prune_mask.sum().item())

        self.prune_points(prune_mask)

        n_after = self.get_xyz.shape[0]

        print(
            f"[Densification] "
            f"Before={n_before}, "
            f"Cloned={n_after_clone - n_before}, "
            f"Split(net)={n_after_split - n_after_clone}, "
            f"LowOpacity={n_low_opacity}, "
            f"ScreenCandidates={screen_prune}, "
            f"WorldCandidates={world_prune}, "
            f"Pruned={n_pruned}, "
            f"After={n_after}"
        )

        # 开始新的统计周期
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()
        self.max_radii2D.zero_()


    def add_densification_stats(self, xyscreen, update_filter_batch):
        grads_scale = [d.grad[update_filter_batch, :2].detach() * 1 for d in xyscreen]
        grads_norm = [d.norm(dim=-1, keepdim=True) for d in grads_scale]
        grads_mean = torch.stack(grads_norm, dim=0).mean(dim=0)
        grads = grads_mean

        self.xyz_gradient_accum[update_filter_batch] += grads
        self.denom[update_filter_batch] += 1

    @torch.no_grad()
    def add_densification_stats_3d(self, visibility_filter):
        grad = self._xyz.grad

        if grad is None:
            return

        grad = grad.detach()

        # 采用三维位置梯度的 L2 norm
        grad_norm = torch.linalg.vector_norm(
            grad[visibility_filter],
            dim=-1,
            keepdim=True,
        )

        self.xyz_gradient_accum[visibility_filter] += grad_norm
        self.denom[visibility_filter] += 1

    # ======================= save / load ======================================

    def construct_attributes(self):
        """构建顶点 (vertex) 元素在 PLY 文件中的属性列名称列表"""
        l = ['x', 'y', 'z']

        color_dim = self._color.view(self._color.shape[0], -1).shape[1]
        for i in range(color_dim):
            l.append(f'color_{i}')
            
        l.append('opacity')
        
        scale_dim = self._scaling.view(self._scaling.shape[0], -1).shape[1]
        for i in range(scale_dim):
            l.append(f'scale_{i}')
            
        # Rotation
        rot_dim = self._rotation.view(self._rotation.shape[0], -1).shape[1]
        for i in range(rot_dim):
            l.append(f'rot_{i}')
            
        # [SIGMA-DISABLED] sigma  (N, 1, 1)
        # l.append('sigma')
        
        return l

    def save_ply(self, path):
        """将模型的高斯点参数和全局时序参数保存为 .ply 文件"""
        folder_path = os.path.dirname(path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 1. 提取所有点云属性并展平为 2D 矩阵 (N, C)
        xyz = self._xyz.detach().cpu().numpy()
        # 将 color 处理为 2D 数组
        color = self._color.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        # [SIGMA-DISABLED] 处理 (N, 1, 1) 的 sigma 变为 (N, 1)
        # sigma = self.sigma.detach().flatten(start_dim=1).cpu().numpy()

        # [SIGMA-DISABLED] 拼接所有 attributes 为 (N, Total_Channels)，移除了 sigma
        attributes = np.concatenate((xyz, color, opacities, scale, rotation), axis=1) # , sigma)

        # 构建类型结构
        dtype_full = [(attribute, 'f4') for attribute in self.construct_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        # 2. 处理全局的时序/形变参数 time_func
        # 如果 time_func 尚未初始化或为空，给一个默认的零矩阵防止报错
        if self.time_func.numel() == 0:
            time_tensor = torch.zeros((1, 1))
        else:
            time_tensor = self.time_func.detach().cpu().view(1, -1)
            
        time_dims = time_tensor.shape[1]
        dtype_time = [(f'time_func_{i}', 'f4') for i in range(time_dims)]
        
        time_arr = np.empty(1, dtype=dtype_time)
        time_arr[:] = list(map(tuple, time_tensor.numpy()))
        time_el = PlyElement.describe(time_arr, 'time')

        # 写入双元素 (vertex + time) PLY 文件
        PlyData([el, time_el]).write(path)
        print(f"成功保存模型到: {path}")

    def load_ply(self, path):
        """从 .ply 文件读取参数并加载到高斯模型中"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到指定的 PLY 文件: {path}")

        plydata = PlyData.read(path)
        vertex = plydata['vertex']
        
        # 1. 解析基础坐标 (N, 3)
        xyz = np.stack((np.asarray(vertex["x"]), 
                        np.asarray(vertex["y"]), 
                        np.asarray(vertex["z"])), axis=1)
        
        # 2. 解析颜色 Color
        color_names = [p.name for p in vertex.properties if p.name.startswith("color_")]
        color = np.zeros((xyz.shape[0], len(color_names)))
        for idx, attr_name in enumerate(color_names):
            color[:, idx] = np.asarray(vertex[attr_name])
        # 如果你初始化时使用 (N, 3)，需保证维度适配
        
        # 3. 解析 Opacity (N, 1)
        opacities = np.asarray(vertex["opacity"])[..., np.newaxis]
        
        # 4. 解析 Scale (N, 3)
        scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(vertex[attr_name])
            
        # 5. 解析 Rotation (N, 1 - 对应单轴旋转)
        rot_names = [p.name for p in vertex.properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(vertex[attr_name])
            
        # [SIGMA-DISABLED] 6. 解析 Sigma 并恢复为特定的 (N, 1, 1) 维度
        # sigma = np.asarray(vertex["sigma"])[..., np.newaxis, np.newaxis]

        # 转换为需要梯度的 PyTorch nn.Parameter
        self._xyz = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._color = torch.nn.Parameter(torch.tensor(color, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = torch.nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = torch.nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        # self.sigma = torch.nn.Parameter(torch.tensor(sigma, dtype=torch.float, device="cuda").requires_grad_(True))  # [SIGMA-DISABLED]

        # 7. 解析全局时序属性 `time_func`
        if 'time' in plydata:
            time_el = plydata['time']
            time_names = [p.name for p in time_el.properties if p.name.startswith("time_func_")]
            time_val = np.zeros((len(time_names), 1))
            for idx, attr_name in enumerate(time_names):
                time_val[idx, 0] = np.asarray(time_el[attr_name])[0]
            self.time_func = torch.nn.Parameter(torch.tensor(time_val, dtype=torch.float, device="cuda").requires_grad_(True))
        else:
            self.time_func = torch.empty(0, device="cuda")

        print(f"成功加载 {xyz.shape[0]} 个高斯顶点参数。")


