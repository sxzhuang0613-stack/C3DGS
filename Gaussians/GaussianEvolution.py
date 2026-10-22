import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_encoding(coordinates, num_bands, include_input=True):
    """对连续坐标进行截断 Fourier 位置编码。"""
    features = [coordinates] if include_input else []
    if num_bands > 0:
        frequencies = 2.0 ** torch.arange(
            num_bands, device=coordinates.device, dtype=coordinates.dtype
        )
        angles = math.pi * coordinates.unsqueeze(-1) * frequencies
        features.extend([
            torch.sin(angles).flatten(start_dim=-2),
            torch.cos(angles).flatten(start_dim=-2),
        ])
    return torch.cat(features, dim=-1)


def quaternion_to_matrix(quaternions):
    """将实部在前的四元数转换为旋转矩阵。"""
    quaternions = F.normalize(quaternions, p=2, dim=-1, eps=1e-8)
    r, i, j, k = torch.unbind(quaternions, dim=-1)
    two_s = 2.0
    matrix = torch.stack([
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ], dim=-1)
    return matrix.reshape(quaternions.shape[:-1] + (3, 3))


class GaussianEvolutionNetwork(nn.Module):
    """连续拉格朗日高斯演化与可选控制点形变图。

    ``control_deformation_mode`` 支持：
    ``none``（原始全高斯 ODE）、``kernel``（控制点核传播）和
    ``mls``（控制点局部 MLS/仿射变形图）。
    """

    def __init__(self, hidden_dim=128, kinematics_depth=4, optics_depth=3,
                 residual_fourier_bands=3, residual_scale=1.0,
                 motion_dimension='2D', through_plane_max_displacement=0.1,
                 transverse_max_log_scale=0.7,
                 control_deformation_mode='none',
                 control_coverage_radius=0.12,
                 control_min_points=32,
                 control_max_points=512,
                 control_neighbors=8,
                 control_kernel_bandwidth=0.0,
                 control_mls_regularization=1e-4,
                 control_knn_chunk_size=32768):
        super().__init__()
        self.residual_fourier_bands = int(residual_fourier_bands)
        self.residual_scale = float(residual_scale)
        self.motion_dimension = str(motion_dimension).upper()
        self.through_plane_max_displacement = float(through_plane_max_displacement)
        self.transverse_max_log_scale = float(transverse_max_log_scale)
        self.control_deformation_mode = str(control_deformation_mode).lower()
        self.control_coverage_radius = float(control_coverage_radius)
        self.control_min_points = int(control_min_points)
        self.control_max_points = int(control_max_points)
        self.control_neighbors = int(control_neighbors)
        self.control_kernel_bandwidth = float(control_kernel_bandwidth)
        self.control_mls_regularization = float(control_mls_regularization)
        self.control_knn_chunk_size = int(control_knn_chunk_size)

        if self.motion_dimension not in ['2D', '3D']:
            raise ValueError("motion_dimension 仅支持 '2D' 或 '3D'")
        if self.control_deformation_mode not in ['none', 'kernel', 'mls']:
            raise ValueError(
                "control_deformation_mode 仅支持 'none'、'kernel' 或 'mls'"
            )
        if self.through_plane_max_displacement < 0:
            raise ValueError("through_plane_max_displacement 必须大于或等于 0")
        if self.transverse_max_log_scale < 0:
            raise ValueError("transverse_max_log_scale 必须大于或等于 0")
        if self.control_coverage_radius <= 0:
            raise ValueError("control_coverage_radius 必须大于 0")
        if self.control_min_points <= 0:
            raise ValueError("control_min_points 必须大于 0")
        if self.control_max_points != 0 and self.control_max_points < self.control_min_points:
            raise ValueError("control_max_points 必须为 0 或不小于 control_min_points")
        if self.control_neighbors <= 0:
            raise ValueError("control_neighbors 必须大于 0")
        if self.control_mls_regularization <= 0:
            raise ValueError("control_mls_regularization 必须大于 0")

        kinematics_dim = 4
        image_dim = 4 * self.residual_fourier_bands + 1
        self.kinematics_trunk = self._build_mlp(
            kinematics_dim, hidden_dim, 3, kinematics_depth
        )
        self.transverse_trunk = self._build_mlp(
            kinematics_dim, hidden_dim, 1, kinematics_depth
        )
        self.optics_trunk = self._build_mlp(
            image_dim, hidden_dim, 1, optics_depth
        )

        nn.init.zeros_(self.kinematics_trunk[-1].weight)
        nn.init.zeros_(self.kinematics_trunk[-1].bias)
        nn.init.zeros_(self.transverse_trunk[-1].weight)
        nn.init.zeros_(self.transverse_trunk[-1].bias)
        nn.init.zeros_(self.optics_trunk[-1].weight)
        nn.init.zeros_(self.optics_trunk[-1].bias)

        # 由每个病例的 Stage-1 高斯确定，重载 checkpoint 时可确定性重建。
        self.register_buffer('_control_indices', torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer('_reference_control_xyz', torch.empty(0, 3), persistent=False)
        self.register_buffer(
            '_base_neighbor_indices', torch.empty(0, 0, dtype=torch.long), persistent=False
        )
        self.register_buffer('_base_neighbor_weights', torch.empty(0, 0), persistent=False)
        self.register_buffer('_base_weight_gradients', torch.empty(0, 0, 3), persistent=False)
        self.register_buffer('_base_mls_projector', torch.empty(0, 4, 0), persistent=False)

    @property
    def uses_control_points(self):
        return self.control_deformation_mode != 'none'

    @property
    def control_indices(self):
        return self._control_indices

    @property
    def reference_control_xyz(self):
        return self._reference_control_xyz

    @property
    def control_count(self):
        return int(self._control_indices.numel())

    @staticmethod
    def _build_mlp(input_dim, hidden_dim, output_dim, depth):
        layers = []
        current_dim = input_dim
        for _ in range(max(1, depth)):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.SiLU()])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def _time_coordinates(self, target_time_t, count, device, dtype):
        target_time_t = torch.as_tensor(target_time_t, device=device, dtype=dtype)
        return target_time_t.reshape(1, 1).expand(count, 1)

    def _kinematics_features(self, xyz, time_t):
        time_coordinates = self._time_coordinates(
            time_t, xyz.shape[0], xyz.device, xyz.dtype
        )
        return torch.cat([xyz, time_coordinates], dim=-1)

    def velocity(self, xyz, time_t):
        raw_velocity = self.kinematics_trunk(self._kinematics_features(xyz, time_t))
        planar_velocity = raw_velocity[:, :2]
        if self.motion_dimension == '2D':
            through_plane_velocity = raw_velocity[:, 2:3] * 0.0
        else:
            through_plane_velocity = (
                self.through_plane_max_displacement * torch.tanh(raw_velocity[:, 2:3])
            )
        return torch.cat([planar_velocity, through_plane_velocity], dim=-1)

    def transverse_log_stretch(self, reference_xyz, target_time_t):
        raw_log_stretch = self.transverse_trunk(
            self._kinematics_features(reference_xyz, target_time_t)
        )
        time_scale = torch.as_tensor(
            target_time_t, device=reference_xyz.device, dtype=reference_xyz.dtype
        ).reshape(1, 1)
        return (
            time_scale * self.transverse_max_log_scale
            * torch.tanh(raw_log_stretch)
        )

    def _velocity_and_gradient(self, xyz, time_t, create_graph=True):
        if not xyz.requires_grad:
            xyz = xyz.detach().requires_grad_(True)
        velocity = self.velocity(xyz, time_t)
        gradients = []
        components = 2 if self.motion_dimension == '2D' else 3
        for output_idx in range(components):
            gradients.append(torch.autograd.grad(
                velocity[:, output_idx].sum(), xyz,
                create_graph=create_graph, retain_graph=True
            )[0])
        if self.motion_dimension == '2D':
            gradients.append(torch.zeros_like(gradients[0]))
        return velocity, torch.stack(gradients, dim=1)

    def _rk4_integrate(self, reference_xyz, target_time_t, num_ode_steps):
        target_time = float(target_time_t)
        num_ode_steps = max(1, int(num_ode_steps))
        if target_time == 0.0:
            return reference_xyz
        xyz = reference_xyz
        step_size = target_time / num_ode_steps
        for step_idx in range(num_ode_steps):
            time_0 = step_idx * step_size
            time_half = time_0 + 0.5 * step_size
            time_1 = time_0 + step_size
            k1 = self.velocity(xyz, time_0)
            k2 = self.velocity(xyz + 0.5 * step_size * k1, time_half)
            k3 = self.velocity(xyz + 0.5 * step_size * k2, time_half)
            k4 = self.velocity(xyz + step_size * k3, time_1)
            xyz = xyz + step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return xyz

    @torch.no_grad()
    def _adaptive_farthest_point_indices(self, points):
        """用覆盖半径停止的确定性 FPS，自适应决定每个病例的控制点数。"""
        count = int(points.shape[0])
        if count == 0:
            raise ValueError("Stage-1 高斯为空，无法选择控制点")
        min_points = min(count, self.control_min_points)
        max_points = count if self.control_max_points <= 0 else min(
            count, self.control_max_points
        )
        centroid = points.mean(dim=0, keepdim=True)
        current_idx = int(torch.argmin(
            torch.sum((points - centroid) ** 2, dim=1)
        ).item())
        minimum_distance_squared = torch.full(
            (count,), float('inf'), device=points.device, dtype=points.dtype
        )
        selected = []
        radius_squared = self.control_coverage_radius ** 2
        while len(selected) < max_points:
            selected.append(current_idx)
            distance_squared = torch.sum(
                (points - points[current_idx:current_idx + 1]) ** 2, dim=1
            )
            minimum_distance_squared = torch.minimum(
                minimum_distance_squared, distance_squared
            )
            maximum_distance_squared, next_idx = torch.max(
                minimum_distance_squared, dim=0
            )
            if len(selected) >= min_points and maximum_distance_squared <= radius_squared:
                break
            if maximum_distance_squared <= torch.finfo(points.dtype).eps:
                break
            current_idx = int(next_idx.item())
        return torch.as_tensor(selected, device=points.device, dtype=torch.long)

    @torch.no_grad()
    def _knn_to_controls(self, points):
        if self.reference_control_xyz.numel() == 0:
            raise RuntimeError("控制点尚未配置")
        neighbors = min(self.control_neighbors, self.reference_control_xyz.shape[0])
        index_chunks = []
        distance_chunks = []
        chunk_size = max(1, self.control_knn_chunk_size)
        for start_idx in range(0, points.shape[0], chunk_size):
            point_chunk = points[start_idx:start_idx + chunk_size]
            distances = torch.cdist(point_chunk, self.reference_control_xyz)
            values, indices = torch.topk(
                distances, k=neighbors, dim=1, largest=False, sorted=True
            )
            index_chunks.append(indices)
            distance_chunks.append(values ** 2)
        return torch.cat(index_chunks, dim=0), torch.cat(distance_chunks, dim=0)

    @torch.no_grad()
    def _kernel_context(self, points, neighbor_indices=None,
                        neighbor_distance_squared=None):
        if neighbor_indices is None or neighbor_distance_squared is None:
            neighbor_indices, neighbor_distance_squared = self._knn_to_controls(points)
        neighbor_control_xyz = self.reference_control_xyz[neighbor_indices]
        delta = points[:, None, :] - neighbor_control_xyz
        if self.control_kernel_bandwidth > 0:
            bandwidth = points.new_full(
                (points.shape[0], 1), self.control_kernel_bandwidth
            )
        else:
            # 使用固定参考带宽，使下面的权重空间导数与插值公式一致。
            bandwidth = points.new_full(
                (points.shape[0], 1), self.control_coverage_radius
            ).clamp_min(1e-4)
        inverse_bandwidth_squared = bandwidth.reciprocal() ** 2
        logits = -0.5 * neighbor_distance_squared * inverse_bandwidth_squared
        weights = torch.softmax(logits, dim=1)
        log_weight_gradients = -delta * inverse_bandwidth_squared.unsqueeze(-1)
        mean_log_gradient = torch.sum(
            weights.unsqueeze(-1) * log_weight_gradients, dim=1, keepdim=True
        )
        weight_gradients = weights.unsqueeze(-1) * (
            log_weight_gradients - mean_log_gradient
        )
        return neighbor_indices, weights, weight_gradients

    @torch.no_grad()
    def _build_mls_projector(self, points, neighbor_indices, weights):
        neighbor_control_xyz = self.reference_control_xyz[neighbor_indices]
        design = torch.cat([
            neighbor_control_xyz,
            torch.ones_like(neighbor_control_xyz[..., :1]),
        ], dim=-1)
        weighted_design = design * weights.unsqueeze(-1)
        normal_matrix = torch.bmm(design.transpose(1, 2), weighted_design)
        identity = torch.eye(
            4, device=points.device, dtype=points.dtype
        ).unsqueeze(0)
        normal_matrix = normal_matrix + self.control_mls_regularization * identity
        right_hand_side = design.transpose(1, 2) * weights.unsqueeze(1)
        return torch.linalg.solve(normal_matrix, right_hand_side)

    @torch.no_grad()
    def configure_control_points(self, base_xyz):
        """从 Stage-1 参考高斯选择并固定控制点，预计算稀疏传播图。"""
        if not self.uses_control_points:
            self._control_indices = torch.empty(
                0, device=base_xyz.device, dtype=torch.long
            )
            self._reference_control_xyz = torch.empty(
                0, 3, device=base_xyz.device, dtype=base_xyz.dtype
            )
            return 0
        reference_xyz = base_xyz.detach()
        self._control_indices = self._adaptive_farthest_point_indices(reference_xyz)
        self._reference_control_xyz = reference_xyz[self._control_indices].contiguous()
        neighbor_indices, neighbor_distances = self._knn_to_controls(reference_xyz)
        neighbor_indices, weights, weight_gradients = self._kernel_context(
            reference_xyz, neighbor_indices, neighbor_distances
        )
        self._base_neighbor_indices = neighbor_indices
        self._base_neighbor_weights = weights
        self._base_weight_gradients = weight_gradients
        if self.control_deformation_mode == 'mls':
            self._base_mls_projector = self._build_mls_projector(
                reference_xyz, neighbor_indices, weights
            )
        else:
            self._base_mls_projector = torch.empty(
                0, 4, 0, device=base_xyz.device, dtype=base_xyz.dtype
            )
        return self.control_count

    def _force_2d_flow(self, reference_xyz, xyz, flow_jacobian):
        if self.motion_dimension != '2D':
            return xyz, flow_jacobian
        xyz = torch.cat([xyz[:, :2], reference_xyz[:, 2:3]], dim=1)
        identity_row = torch.cat([
            reference_xyz.new_zeros((reference_xyz.shape[0], 2)),
            reference_xyz.new_ones((reference_xyz.shape[0], 1)),
        ], dim=1)
        flow_jacobian = torch.stack([
            flow_jacobian[:, 0], flow_jacobian[:, 1], identity_row
        ], dim=1)
        return xyz, flow_jacobian

    def _propagate_kernel(self, reference_xyz, evolved_control_xyz,
                          neighbor_indices, weights, weight_gradients):
        control_displacement = evolved_control_xyz - self.reference_control_xyz
        local_displacement = control_displacement[neighbor_indices]
        displacement = torch.sum(weights.unsqueeze(-1) * local_displacement, dim=1)
        xyz = reference_xyz + displacement
        identity = torch.eye(
            3, device=reference_xyz.device, dtype=reference_xyz.dtype
        ).unsqueeze(0).expand(reference_xyz.shape[0], -1, -1)
        displacement_gradient = torch.einsum(
            'nko,nki->noi', local_displacement, weight_gradients
        )
        return self._force_2d_flow(
            reference_xyz, xyz, identity + displacement_gradient
        )

    def _propagate_mls(self, reference_xyz, evolved_control_xyz,
                       neighbor_indices, mls_projector):
        # 拟合控制点位移而非绝对坐标，使 t=0 时严格得到恒等映射。
        control_displacement = evolved_control_xyz - self.reference_control_xyz
        local_control_displacement = control_displacement[neighbor_indices]
        affine_displacement = torch.bmm(
            mls_projector, local_control_displacement
        )
        homogeneous_reference = torch.cat([
            reference_xyz,
            reference_xyz.new_ones((reference_xyz.shape[0], 1)),
        ], dim=1)
        displacement = torch.einsum(
            'ni,nio->no', homogeneous_reference, affine_displacement
        )
        xyz = reference_xyz + displacement
        identity = torch.eye(
            3, device=reference_xyz.device, dtype=reference_xyz.dtype
        ).unsqueeze(0).expand(reference_xyz.shape[0], -1, -1)
        flow_jacobian = (
            identity + affine_displacement[:, :3, :].transpose(1, 2)
        )
        return self._force_2d_flow(reference_xyz, xyz, flow_jacobian)

    def _propagate_points(self, reference_xyz, evolved_control_xyz,
                          neighbor_indices, weights, weight_gradients,
                          mls_projector=None):
        if self.control_deformation_mode == 'mls':
            if mls_projector is None:
                mls_projector = self._build_mls_projector(
                    reference_xyz, neighbor_indices, weights
                )
            return self._propagate_mls(
                reference_xyz, evolved_control_xyz,
                neighbor_indices, mls_projector
            )
        return self._propagate_kernel(
            reference_xyz, evolved_control_xyz,
            neighbor_indices, weights, weight_gradients
        )

    def integrate_control_positions(self, target_time_t, num_ode_steps=4):
        if not self.uses_control_points or self.control_count == 0:
            raise RuntimeError("当前网络没有配置控制点")
        return self._rk4_integrate(
            self.reference_control_xyz, target_time_t, num_ode_steps
        )

    def deform_query_points(self, reference_xyz, target_time_t,
                            num_ode_steps=4, evolved_control_xyz=None):
        """用训练时相同的控制形变场查询任意参考点。"""
        if not self.uses_control_points:
            return self._rk4_integrate(reference_xyz, target_time_t, num_ode_steps)
        if self.control_count == 0:
            raise RuntimeError("调用 deform_query_points 前必须配置控制点")
        if evolved_control_xyz is None:
            evolved_control_xyz = self.integrate_control_positions(
                target_time_t, num_ode_steps
            )
        neighbor_indices, distances = self._knn_to_controls(reference_xyz)
        neighbor_indices, weights, weight_gradients = self._kernel_context(
            reference_xyz, neighbor_indices, distances
        )
        mls_projector = None
        if self.control_deformation_mode == 'mls':
            mls_projector = self._build_mls_projector(
                reference_xyz, neighbor_indices, weights
            )
        xyz, _ = self._propagate_points(
            reference_xyz, evolved_control_xyz,
            neighbor_indices, weights, weight_gradients,
            mls_projector=mls_projector,
        )
        return xyz

    def integrate_positions(self, base_xyz, target_time_t, num_ode_steps=4):
        if self.uses_control_points:
            return self.deform_query_points(base_xyz, target_time_t, num_ode_steps)
        return self._rk4_integrate(base_xyz, target_time_t, num_ode_steps)

    @staticmethod
    def _control_log_stretch_to_points(control_log_stretch,
                                       neighbor_indices, weights):
        return torch.sum(
            weights.unsqueeze(-1) * control_log_stretch[neighbor_indices], dim=1
        )

    def integrate_flow(self, base_xyz, target_time_t, num_ode_steps=4,
                       create_graph=True):
        """返回材料中心流、真实运动 Jacobian 和高斯支撑变换。"""
        reference_xyz = base_xyz.detach()
        control_anchor_loss = reference_xyz.new_zeros(())
        if self.uses_control_points:
            if self.control_count == 0:
                raise RuntimeError("控制模式下必须先调用 configure_control_points")
            evolved_control_xyz = self.integrate_control_positions(
                target_time_t, num_ode_steps
            )
            xyz, flow_jacobian = self._propagate_points(
                reference_xyz, evolved_control_xyz,
                self._base_neighbor_indices,
                self._base_neighbor_weights,
                self._base_weight_gradients,
                mls_projector=(
                    self._base_mls_projector
                    if self.control_deformation_mode == 'mls' else None
                ),
            )
            control_anchor_loss = torch.mean(
                (xyz[self.control_indices] - evolved_control_xyz) ** 2
            )
            smoothness_xyz = evolved_control_xyz
        else:
            reference_xyz = reference_xyz.requires_grad_(True)
            xyz = self._rk4_integrate(reference_xyz, target_time_t, num_ode_steps)
            gradient_rows = []
            for output_idx in range(3):
                gradient_rows.append(torch.autograd.grad(
                    xyz[:, output_idx].sum(), reference_xyz,
                    create_graph=create_graph, retain_graph=True
                )[0])
            flow_jacobian = torch.stack(gradient_rows, dim=1)
            xyz, flow_jacobian = self._force_2d_flow(
                reference_xyz, xyz, flow_jacobian
            )
            smoothness_xyz = xyz

        _, velocity_gradient = self._velocity_and_gradient(
            smoothness_xyz, target_time_t, create_graph=create_graph
        )
        spatial_smoothness = torch.sum(
            velocity_gradient ** 2, dim=(1, 2)
        ).mean()

        transverse_stretch = reference_xyz.new_ones((reference_xyz.shape[0], 1))
        support_transform = flow_jacobian
        if self.motion_dimension == '2D':
            if self.uses_control_points:
                control_reference = self.reference_control_xyz.detach().requires_grad_(True)
                control_log_stretch = self.transverse_log_stretch(
                    control_reference, target_time_t
                )
                log_transverse_stretch = self._control_log_stretch_to_points(
                    control_log_stretch,
                    self._base_neighbor_indices,
                    self._base_neighbor_weights,
                )
                transverse_gradient = torch.autograd.grad(
                    control_log_stretch.sum(), control_reference,
                    create_graph=create_graph, retain_graph=True
                )[0]
            else:
                log_transverse_stretch = self.transverse_log_stretch(
                    reference_xyz, target_time_t
                )
                transverse_gradient = torch.autograd.grad(
                    log_transverse_stretch.sum(), reference_xyz,
                    create_graph=create_graph, retain_graph=True
                )[0]
            transverse_stretch = torch.exp(log_transverse_stretch)
            spatial_smoothness = spatial_smoothness + torch.sum(
                transverse_gradient ** 2, dim=1
            ).mean()
            support_transverse_row = torch.cat([
                reference_xyz.new_zeros((reference_xyz.shape[0], 2)),
                transverse_stretch,
            ], dim=1)
            support_transform = torch.stack([
                flow_jacobian[:, 0],
                flow_jacobian[:, 1],
                support_transverse_row,
            ], dim=1)

        return {
            'xyz': xyz,
            'flow_jacobian': flow_jacobian,
            'support_transform': support_transform,
            'spatial_smoothness': spatial_smoothness,
            'transverse_stretch': transverse_stretch,
            'control_anchor_loss': control_anchor_loss,
        }

    def image_residual(self, height, width, target_time_t, device, dtype):
        coordinate_features = self.build_image_coordinate_features(
            height, width, device, dtype
        )
        return self.image_residual_from_features(
            coordinate_features, height, width, target_time_t
        )

    def build_image_coordinate_features(self, height, width, device, dtype):
        coordinate_y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        coordinate_x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(coordinate_y, coordinate_x, indexing='ij')
        coordinates = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        return fourier_encoding(
            coordinates, self.residual_fourier_bands, include_input=False
        )

    def image_residual_from_features(self, coordinate_features, height, width,
                                     target_time_t):
        time_coordinates = self._time_coordinates(
            target_time_t, coordinate_features.shape[0],
            coordinate_features.device, coordinate_features.dtype
        )
        features = torch.cat([coordinate_features, time_coordinates], dim=-1)
        residual = self.optics_trunk(features).reshape(height, width)
        return self.residual_scale * float(target_time_t) * residual

    @staticmethod
    def build_base_covariance(base_scaling, base_rotation):
        base_scales = torch.exp(base_scaling)
        base_rotation_matrix = quaternion_to_matrix(base_rotation)
        return torch.bmm(
            base_rotation_matrix * (base_scales ** 2).unsqueeze(1),
            base_rotation_matrix.transpose(1, 2)
        )

    def forward(self, base_xyz, base_scaling, base_rotation, target_time_t,
                num_ode_steps=4, create_graph=True, base_covariance=None):
        flow = self.integrate_flow(
            base_xyz, target_time_t,
            num_ode_steps=num_ode_steps,
            create_graph=create_graph,
        )
        if base_covariance is None:
            base_covariance = self.build_base_covariance(
                base_scaling, base_rotation
            )
        support_transform = flow['support_transform']
        covariance = torch.bmm(
            torch.bmm(support_transform, base_covariance),
            support_transform.transpose(1, 2),
        )
        covariance = 0.5 * (covariance + covariance.transpose(1, 2))

        flow_jacobian_det = torch.linalg.det(flow['flow_jacobian'])
        support_volume_ratio = torch.linalg.det(support_transform)
        return {
            'xyz': flow['xyz'],
            'displacement': flow['xyz'] - base_xyz,
            'covariance': covariance,
            'flow_jacobian': flow['flow_jacobian'],
            'support_transform': support_transform,
            'flow_jacobian_det': flow_jacobian_det,
            'support_volume_ratio': support_volume_ratio,
            'spatial_smoothness': flow['spatial_smoothness'],
            'transverse_stretch': flow['transverse_stretch'],
            'control_anchor_loss': flow['control_anchor_loss'],
            # 兼容旧调用；新代码应使用两个明确命名的量。
            'deformation_gradient': support_transform,
            'jacobian_det': support_volume_ratio,
            'control_count': self.control_count,
        }
