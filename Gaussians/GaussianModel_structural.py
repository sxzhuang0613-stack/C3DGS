import math
import os
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from .utils import get_expon_lr_func
from diff_gaussian_slice_rendering import normalize_rotations


class GaussianModel:
    parameter_names = {'xyz': '_xyz', 'intensity': '_intensity',
                       'rotation': '_rotation', 'scaling': '_scaling'}

    def __init__(self):
        self._xyz = torch.empty(0, 3)
        self._intensity = torch.empty(0, 1)
        self._scaling = torch.empty(0, 3)
        self._rotation = torch.empty(0, 4)
        self.time_func = torch.empty(0)
        self.use_dff = False

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_intensity(self):
        return self._intensity

    @property
    def get_color(self):
        """兼容共享 Renderer 的灰度属性名。"""
        return self._intensity

    @property
    def get_scaling(self):
        return self._scaling.exp()

    @property
    def get_rotation(self):
        return normalize_rotations(self._rotation)

    @property
    def get_time(self):
        return F.softmax(self.time_func.squeeze(), dim=0) if self.use_dff else self.time_func

    def initialize_from_images(self, images, geometry, args, rendering, case_name):
        from .initialization import initialize_from_images
        return initialize_from_images(self, images, geometry, args, rendering, case_name)

    def training_setup(self, args):
        if len(self._xyz) > args.max_num_pts:
            raise ValueError('已有点数超过 max_num_pts，请先增大容量上限')
        self.structural_args = args
        if not hasattr(self, 'geometry_reference_count'):
            self.geometry_reference_count = len(self._xyz)
        self._reset_statistics()
        self._structural_spacing = None
        if hasattr(self, 'geometry'):
            self._voxel_spacing()
        groups = [dict(params=[getattr(self, name)], lr=getattr(args, lr), name=key)
                  for key, name, lr in (
                      ('xyz', '_xyz', 'position_lr_init'),
                      ('intensity', '_intensity', 'intensity_lr'),
                      ('rotation', '_rotation', 'rotation_lr'),
                      ('scaling', '_scaling', 'scaling_lr'))]
        self.optimizer = torch.optim.Adam(groups, lr=0., eps=args.adam_eps,
                                          betas=tuple(args.adam_betas))
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=args.position_lr_init, lr_final=args.position_lr_final,
            lr_delay_mult=args.position_lr_delay_mult, max_steps=args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        lr = self.xyz_scheduler_args(iteration)
        for group in self.optimizer.param_groups:
            if group['name'] == 'xyz':
                group['lr'] = lr
        return lr

    def _reset_statistics(self):
        self.xyz_gradient_accum = self._xyz.new_zeros((len(self._xyz), 1))
        self.denom = self._xyz.new_zeros((len(self._xyz), 1))
        # 仅保留旧 checkpoint 的字段；批量结构操作不统计屏幕半径。
        self.max_radii2D = self._xyz.new_zeros(len(self._xyz))

    def _voxel_spacing(self):
        if getattr(self, '_structural_spacing', None) is not None:
            return self._structural_spacing
        geometry = getattr(self, 'geometry', None)
        if geometry is None:
            raise ValueError('批量结构操作需要病例 geometry；请先 attach_geometry')
        if geometry['version'] != 0:
            spacing = self._xyz.new_tensor(geometry['spacing_xyz']) / geometry['scale_mm']
        else:
            s, h, w = geometry['shape_shw']
            spacing = 2 * self._xyz.new_tensor(geometry['bounds']) / self._xyz.new_tensor([w, h, max(s-1, 1)])
        # 病例几何固定；避免逐训练步重复创建 CPU->GPU 的 spacing Tensor。
        self._structural_spacing = spacing
        return spacing

    @staticmethod
    def _rotation_matrix(raw):
        # 与渲染器使用同一 wxyz 激活；不依赖旧模型或硬编码 CUDA 的 utils.build_rotation。
        w, x, y, z = normalize_rotations(raw).unbind(-1)
        return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                            2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                            2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)

    def _invalid_point_mask(self):
        invalid = torch.zeros(len(self._xyz), device=self._xyz.device, dtype=torch.bool)
        for value in (self._xyz, self._intensity, self._scaling, self._rotation):
            invalid |= ~torch.isfinite(value).flatten(1).all(-1)
        scales = self.get_scaling
        invalid |= ~torch.isfinite(scales).all(-1) | (scales <= 0).any(-1)
        components = tuple(component.double() for component in self._rotation.unbind(-1))
        invalid |= torch.sqrt(sum(value.square() for value in components)) < 1e-12
        return invalid

    @torch.no_grad()
    def add_densification_stats_3d(self, visibility_filter):
        if self._xyz.grad is None:
            return
        # g_voxel = dL/dx_model * 一个体素的模型长度；不改变实际训练梯度。
        gradient = self._xyz.grad.detach() * self._voxel_spacing()
        values = torch.linalg.vector_norm(gradient, dim=-1, keepdim=True)
        values = torch.where(visibility_filter[:, None], values, 0.)
        self.xyz_gradient_accum.add_(values)
        self.denom.add_(visibility_filter[:, None])

    @torch.no_grad()
    def _replace_structure(self, survivors, sources, data, reset=None):
        """每组参数只提交一次；保留旧行动量，新行动量清零，沿用 Adam 组级时钟。"""
        reset = {} if reset is None else reset
        for group in self.optimizer.param_groups:
            key = group['name']
            old = group['params'][0]
            parameter = nn.Parameter(data[key])
            state = self.optimizer.state.pop(old, None)
            if state is not None:
                state = dict(state)
                for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    if name in state:
                        moment = torch.cat((state[name][survivors], torch.zeros_like(data[key][len(survivors):])), 0)
                        if key in reset:
                            moment[:len(survivors)][reset[key]] = 0
                        state[name] = moment
                self.optimizer.state[parameter] = state
            group['params'][0] = parameter
            setattr(self, self.parameter_names[key], parameter)

    @torch.no_grad()
    def prune_points(self, mask):
        if mask.dtype != torch.bool or mask.shape != (len(self._xyz),):
            raise ValueError('prune mask 必须是 bool [N]')
        if not bool(mask.any()):
            return
        survivors = torch.where(~mask)[0]
        sources = survivors[:0]
        data = {key: getattr(self, name)[survivors] for key, name in self.parameter_names.items()}
        self._replace_structure(survivors, sources, data)
        for name in ('xyz_gradient_accum', 'denom', 'max_radii2D'):
            setattr(self, name, getattr(self, name)[survivors])

    @torch.no_grad()
    def structural_step(self, images, iteration, render_kwargs=None, similarity_kwargs=None):
        """阈值选点并批量调整；images 及 kwargs 保留调用兼容，不执行任何额外渲染。"""
        args = self.structural_args
        before = len(self._xyz)
        invalid = self._invalid_point_mask()
        valid = ~invalid
        if int(valid.sum()) < args.structural_min_points:
            raise FloatingPointError('有效点少于 structural_min_points，无法安全调整结构')
        # 无效点不参与旋转/尺度计算，统一在本轮提交时删除。
        valid_ids = torch.where(valid)[0]
        scales = self.get_scaling[valid_ids]
        rotation = self._rotation_matrix(self._rotation[valid_ids])
        spacing = self._voxel_spacing()
        # 面内投影协方差的最大特征值，单位为输出像素平方；无需逐点 eigh。
        projected = rotation[:, :2, :] * scales[:, None, :] / spacing[None, :2, None]
        xx, yy = projected[:, 0].square().sum(-1), projected[:, 1].square().sum(-1)
        xy = (projected[:, 0] * projected[:, 1]).sum(-1)
        sigma_pixels = ((xx+yy+torch.sqrt((xx-yy).square()+4*xy.square()))*.5).clamp_min(0).sqrt()
        observed = self.denom[valid_ids, 0] >= args.structural_min_observations
        gradients = self.xyz_gradient_accum[valid_ids, 0] / self.denom[valid_ids, 0].clamp_min(1)
        eligible = observed & torch.isfinite(gradients)
        positive = gradients[eligible & (gradients > 0)]
        median = positive.median() if positive.numel() else gradients.new_zeros(())
        threshold = torch.maximum(gradients.new_tensor(args.structural_grad_threshold),
                                  median * args.structural_grad_multiplier)
        high = eligible & (gradients > 0) & (gradients >= threshold)
        # 不生成超出既有 log_scale 下界的子点，避免下一步 clamp 破坏分裂尺度。
        split_axes = projected.square().sum(1).argmax(-1)
        shrink = math.sqrt(1-args.structural_split_fraction**2)
        axis_logs = self._scaling[valid_ids].gather(1, split_axes[:, None])[:, 0]
        can_split = axis_logs + math.log(shrink) >= args.log_scale_min
        high &= (sigma_pixels <= args.structural_split_sigma_pixels) | can_split
        # 弱幅值裁剪默认关闭；开启后只裁掉已充分观测的点，不把强度视作 opacity。
        weak = observed & (self._intensity[valid_ids, 0].abs() < args.structural_prune_peak)
        max_prune = len(valid_ids)-args.structural_min_points
        weak_ids = torch.where(weak)[0]
        if len(weak_ids) > max_prune:
            weak.zero_()
            weak[weak_ids[torch.randperm(len(weak_ids), device=self._xyz.device)[:max_prune]]] = True
        # 阈值选出的点只在超容量/增长预算时随机限流，不按梯度 Top-K 排序。
        candidates = torch.where(high & ~weak)[0]
        above_threshold = len(candidates)
        kept_count = len(valid_ids)-int(weak.sum())
        capacity = max(0, args.max_num_pts-kept_count)
        growth_limit = int(kept_count*args.structural_growth_fraction)
        budget = min(capacity, growth_limit)
        if len(candidates) > budget:
            candidates = candidates[torch.randperm(len(candidates), device=self._xyz.device)[:budget]]
        split_local = candidates[sigma_pixels[candidates] > args.structural_split_sigma_pixels]
        clone_local = candidates[sigma_pixels[candidates] <= args.structural_split_sigma_pixels]
        split_ids, clone_ids = valid_ids[split_local], valid_ids[clone_local]
        removed = invalid.clone()
        removed[valid_ids[weak]] = True
        removed[split_ids] = True
        survivors = torch.where(~removed)[0]
        sources = torch.cat((clone_ids, split_ids.repeat_interleave(2)))
        pruned_amplitude = self._intensity[valid_ids[weak]].abs().sum()
        changed = bool(removed.any()) or len(sources) > 0
        if changed:
            data = {key: torch.cat((getattr(self, name)[survivors], getattr(self, name)[sources]), 0)
                    for key, name in self.parameter_names.items()}
            clone_rows = torch.zeros(len(survivors), device=self._xyz.device, dtype=torch.bool)
            rows = torch.searchsorted(survivors, clone_ids)
            clone_rows[rows] = True
            # Clone 对父/子施加对称小扰动并均分强度；无扰动时严格保持加性场。
            if len(clone_ids):
                clone_rotation = rotation[clone_local]
                offset = torch.randn_like(self._xyz[clone_ids]) * scales[clone_local] * args.structural_clone_jitter
                offset = (clone_rotation @ offset[:, :, None]).squeeze(-1)
                # survivors 与 clone_ids 顺序不同，显式映射避免随机限流后配错父点。
                data['xyz'][rows] -= offset
                data['xyz'][len(survivors):len(survivors)+len(clone_ids)] += offset
                data['intensity'][rows] *= .5
                data['intensity'][len(survivors):len(survivors)+len(clone_ids)] *= .5
            if len(split_ids):
                # 选择对面内 footprint 贡献最大的局部轴，避免仅因层厚大而沿法向反复分裂。
                axes = split_axes[split_local]
                rows = torch.arange(len(split_ids), device=self._xyz.device)
                direction = rotation[split_local][rows, :, axes]
                distance = scales[split_local][rows, axes] * args.structural_split_fraction
                signs = self._xyz.new_tensor([-1., 1.])
                offset = (direction*distance[:, None])[:, None, :] * signs[None, :, None]
                tail = len(survivors)+len(clone_ids)
                data['xyz'][tail:] += offset.reshape(-1, 3)
                # 一轴矩匹配：sigma_child² + displacement² = sigma_parent²。
                child_logs = data['scaling'][tail:].reshape(-1, 2, 3)
                child_logs[rows[:, None], torch.arange(2, device=self._xyz.device)[None, :], axes[:, None]] += math.log(shrink)
                # 保持未加 jitter、未截断内禀 Gaussian 的有符号积分；并非逐像素不变。
                data['intensity'][tail:] /= 2*shrink
            self._replace_structure(survivors, sources, data,
                                    reset={'xyz': clone_rows, 'intensity': clone_rows})
        self._reset_statistics()
        report = dict(before=before, after=len(self._xyz), clone=len(clone_ids), split=len(split_ids),
                      pruned=int(removed.sum())-len(split_ids), eligible=int(eligible.sum()),
                      above_threshold=above_threshold, selected=len(candidates), budget=budget,
                      grad_median=float(median), grad_threshold=float(threshold),
                      sigma_pixels_median=float(sigma_pixels.median()),
                      pruned_abs_amplitude=float(pruned_amplitude))
        print(f'[Structure {iteration}] {report}')
        return report

    def InitializeGaussians(self, num_pts=50000, bounds=(1.,1.,1.), roi_mask=None,
                            intensity_init=0.6410473958869391, roi_layout='SHW',
                            roi_jitter_factor=.5, initial_z_scale=2., initial_distance_floor=1e-7):
        device = torch.device('cuda', torch.cuda.current_device())
        low = np.negative(np.asarray(bounds, dtype=np.float32))
        high = np.asarray(bounds, dtype=np.float32)
        if roi_mask is None:
            points = torch.from_numpy(np.random.uniform(low, high, (num_pts, 3))).float().to(device)
        else:
            coordinates = roi_mask.detach().to(device).float()
            if coordinates.ndim != 2:
                layout = roi_layout.upper().translate(str.maketrans({'S':'Z','D':'Z','H':'Y','W':'X'}))
                if coordinates.ndim != 3 or sorted(layout) != ['X','Y','Z']:
                    raise ValueError('roi_layout 必须为三维 mask 的 XYZ 轴序')
                order = [layout.index(axis) for axis in 'XYZ']
                shape = coordinates.new_tensor([coordinates.shape[i] for i in order])
                indices = torch.nonzero(coordinates > 0).float()[:, order]
                normalized = torch.where(shape > 1, indices/(shape-1).clamp_min(1), .5)
                coordinates = normalized * torch.as_tensor(high-low,device=device) + torch.as_tensor(low,device=device)
            if coordinates.shape[-1] != 3 or not len(coordinates):
                raise ValueError('ROI 需要非空的模型空间 XYZ 点或非空三维 mask')
            if len(coordinates) < num_pts:
                selected = torch.randint(len(coordinates),(num_pts,),device=device)
            else:
                selected = torch.randperm(len(coordinates),device=device)[:num_pts]
            points = coordinates[selected]
            jitter = points.new_tensor(high-low)/np.cbrt(num_pts)*roi_jitter_factor
            points = points + torch.empty_like(points).uniform_(-1,1)*jitter
            points = points.clamp(min=points.new_tensor(low),max=points.new_tensor(high))
        if not math.isfinite(intensity_init):
            raise ValueError('intensity_init 必须有限')
        rotation = torch.logit(torch.rand(num_pts,4,device=device))
        from simple_knn._C import distCUDA2
        distance = distCUDA2(points).clamp_min(initial_distance_floor)
        log_scale = distance.sqrt().log()[:,None].repeat(1,3)
        log_scale[:,2] += math.log(initial_z_scale)
        self._xyz = nn.Parameter(points)
        self._intensity = nn.Parameter(torch.full((num_pts,1),intensity_init,device=device))
        self._scaling = nn.Parameter(log_scale)
        self._rotation = nn.Parameter(rotation)


    def construct_attributes(self):
        return ['x', 'y', 'z', 'intensity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]

    def save_ply(self, path):
        """将模型的高斯点参数和全局时序参数保存为 .ply 文件"""
        folder_path = os.path.dirname(path)
        if folder_path and not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 1. 提取所有点云属性并展平为 2D 矩阵 (N, C)
        xyz = self._xyz.detach().cpu().numpy()
        # 单通道强度直接写入 intensity；不写入 opacity。
        intensity = self._intensity.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # [SIGMA-DISABLED] 处理 (N, 1, 1) 的 sigma 变为 (N, 1)
        # sigma = self.sigma.detach().flatten(start_dim=1).cpu().numpy()

        # [SIGMA-DISABLED] 拼接所有 attributes 为 (N, Total_Channels)，移除了 sigma
        attributes = np.concatenate((xyz, intensity, scale, rotation), axis=1) # , sigma)

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

    def load_ply(self, path, legacy_color_policy="strict", device="cuda"):
        """从 .ply 文件读取参数并加载到高斯模型中"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到指定的 PLY 文件: {path}")

        plydata = PlyData.read(path)
        vertex = plydata['vertex']

        # 1. 解析基础坐标 (N, 3)
        xyz = np.stack((np.asarray(vertex["x"]),
                        np.asarray(vertex["y"]),
                        np.asarray(vertex["z"])), axis=1)

        # 新 PLY 优先读取 intensity。旧 opacity 字段故意忽略。
        names = {p.name for p in vertex.properties}
        if 'intensity' in names:
            intensity = np.asarray(vertex['intensity'], dtype=np.float32)[:, None]
        else:
            color_names = sorted((n for n in names if n.startswith('color_')), key=lambda n: int(n.split('_')[-1]))
            if not color_names:
                raise ValueError("PLY has neither intensity nor legacy color_* fields")
            color = np.stack([np.asarray(vertex[n]) for n in color_names], axis=1)
            if legacy_color_policy == 'first':
                intensity = color[:, :1]
            elif legacy_color_policy == 'mean':
                intensity = color.mean(axis=1, keepdims=True)
            elif legacy_color_policy == 'strict':
                if not np.allclose(color, color[:, :1], rtol=1e-5, atol=1e-6):
                    raise ValueError("Legacy color channels differ: explicitly select legacy_color_policy='first' or 'mean'")
                intensity = color[:, :1]
            else:
                raise ValueError("legacy_color_policy must be strict, first or mean")
            warnings.warn(f"Loaded legacy color_* as intensity using {legacy_color_policy}; opacity is ignored", UserWarning)

        # 4. 解析 Scale (N, 3)
        scale_names = sorted([p.name for p in vertex.properties if p.name.startswith("scale_")], key=lambda n: int(n.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(vertex[attr_name])

        # 解析原始 wxyz 四元数 (N, 4)
        rot_names = sorted([p.name for p in vertex.properties if p.name.startswith("rot_")], key=lambda n: int(n.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(vertex[attr_name])

        # [SIGMA-DISABLED] 6. 解析 Sigma 并恢复为特定的 (N, 1, 1) 维度
        # sigma = np.asarray(vertex["sigma"])[..., np.newaxis, np.newaxis]

        if scales.shape[1] != 3 or rots.shape[1] != 4:
            raise ValueError("PLY requires scale_0..2 and rot_0..3")
        if not all(np.isfinite(a).all() for a in (xyz, intensity, scales, rots)):
            raise ValueError("PLY contains nonfinite attributes")
        # 转换为需要梯度的 PyTorch nn.Parameter
        self._xyz = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True))
        self._intensity = torch.nn.Parameter(torch.tensor(intensity, dtype=torch.float, device=device).requires_grad_(True))
        self._scaling = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = torch.nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))
        # self.sigma = torch.nn.Parameter(torch.tensor(sigma, dtype=torch.float, device=device).requires_grad_(True))  # [SIGMA-DISABLED]

        # 7. 解析全局时序属性 `time_func`
        if 'time' in plydata:
            time_el = plydata['time']
            time_names = [p.name for p in time_el.properties if p.name.startswith("time_func_")]
            time_val = np.zeros((len(time_names), 1))
            for idx, attr_name in enumerate(time_names):
                time_val[idx, 0] = np.asarray(time_el[attr_name])[0]
            self.time_func = torch.nn.Parameter(torch.tensor(time_val, dtype=torch.float, device=device).requires_grad_(True))
        else:
            self.time_func = torch.empty(0, device=device)

        print(f"成功加载 {xyz.shape[0]} 个高斯顶点参数。")
