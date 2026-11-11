"""批量高斯操作回归；CPU 数值/状态检查与短 CUDA 集成分开执行。"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from config_utils import load_config
from case_geometry import make_geometry, attach_geometry
from Gaussians.GaussianModel_structural import GaussianModel
from diff_gaussian_slice_rendering import GaussianRasterizationSettings as Settings
from diff_gaussian_slice_rendering.local_response import activated_covariance
from diff_gaussian_slice_rendering.reference import rasterize_reference


def reference_render(model,img_size,slice_idx,axis,num_slices,**kwargs):
    cfg = Settings.from_axis(*img_size,axis=axis,slice_idx=slice_idx,num_slices=num_slices,
                             bounds=kwargs['bounds'],**kwargs['raster_options'])
    cfg = Settings(cfg.image_height,cfg.image_width,cfg.slice_to_world.float().double(),
                   background=cfg.background,covariance_jitter=cfg.covariance_jitter,
                   response_mode=cfg.response_mode,psf_std_vector=cfg.psf_vector.float().double(),
                   kernel_cutoff=cfg.kernel_cutoff,cutoff_transition=cfg.cutoff_transition)
    covariance = activated_covariance(model._scaling,model._rotation)
    image = rasterize_reference(model.get_xyz.double(),model.get_intensity.double(),covariance,cfg).to(model.get_xyz.dtype)
    return {'render':image[0]}


class StructuralTests(unittest.TestCase):
    def setUp(self):
        self.args = load_config(ROOT/'config_train.yaml')
        self.args.max_num_pts = 32
        self.args.structural_min_observations = 1
        self.args.structural_growth_fraction = 1.
        self.args.structural_grad_multiplier = 0.
        self.args.structural_grad_threshold = .1
        self.args.structural_clone_jitter = 0.
        self.images = torch.zeros(3, 32, 32)

    def model(self, scales=(.05, .05, .05), count=4, device='cpu'):
        model = GaussianModel()
        model._xyz = nn.Parameter(torch.zeros(count, 3, device=device))
        model._scaling = nn.Parameter(torch.tensor(scales, device=device).log().repeat(count, 1))
        model._rotation = nn.Parameter(torch.tensor([[1., 0., 0., 0.]], device=device).repeat(count, 1))
        values = torch.arange(1, count+1, device=device).float()
        values[::2] *= -1
        model._intensity = nn.Parameter(values[:, None])
        attach_geometry(model, make_geometry([1., 1., 8.], self.images.shape, self.args))
        model.training_setup(self.args)
        model.xyz_gradient_accum.fill_(1.)
        model.denom.fill_(1.)
        # 模拟已训练的 Adam 状态，逐行不同以检验索引对应关系。
        for group in model.optimizer.param_groups:
            parameter = group['params'][0]
            moment = torch.arange(1, count+1, device=device).reshape(count, *([1]*(parameter.ndim-1))).expand_as(parameter).clone().float()
            model.optimizer.state[parameter] = dict(step=torch.tensor(7.), exp_avg=moment,
                                                     exp_avg_sq=moment.square())
        return model

    def step(self, model):
        return model.structural_step(self.images.to(model.get_xyz.device), 1000)

    def test_standalone_and_no_candidate_render_or_solver(self):
        self.assertEqual(GaussianModel.__bases__, (object,))
        model = self.model()
        with mock.patch('Renderer.render', side_effect=AssertionError('不应渲染')), \
                mock.patch('torch.linalg.svd', side_effect=AssertionError('不应拟合')), \
                mock.patch('torch.topk', side_effect=AssertionError('不应 Top-K')):
            report = self.step(model)
        self.assertEqual(report['clone'], 4)
        self.assertEqual(report['after'], 8)

    def test_clone_preserves_signed_field_and_resets_changed_moments(self):
        model = self.model()
        xyz = model.get_xyz.detach().clone()
        intensity = model.get_intensity.detach().clone()
        scaling_moment = model.optimizer.state[model._scaling]['exp_avg'].clone()
        cfg = Settings.from_axis(9, 11, num_slices=1, bounds=model.bounds)
        before = rasterize_reference(model.get_xyz.double(), model.get_intensity.double(),
                                     activated_covariance(model._scaling, model._rotation), cfg)
        report = self.step(model)
        self.assertEqual(report['split'], 0)
        torch.testing.assert_close(model.get_xyz, torch.cat((xyz, xyz)))
        torch.testing.assert_close(model.get_intensity, torch.cat((intensity, intensity))*.5)
        after = rasterize_reference(model.get_xyz.double(), model.get_intensity.double(),
                                    activated_covariance(model._scaling, model._rotation), cfg)
        torch.testing.assert_close(after, before, atol=1e-12, rtol=1e-12)
        for name in ('_xyz', '_intensity'):
            self.assertEqual(torch.count_nonzero(model.optimizer.state[getattr(model, name)]['exp_avg']), 0)
        torch.testing.assert_close(model.optimizer.state[model._scaling]['exp_avg'][:4], scaling_moment)
        self.assertEqual(model.geometry_reference_count, 4)
        self.assertEqual(model.denom.sum(), 0)
        self.assertEqual(model.optimizer.state[model._xyz]['step'], 7)

    def test_split_conserves_intrinsic_signed_mass_center_and_covariance(self):
        model = self.model(scales=(.3, .1, .5))
        with torch.no_grad():
            model._rotation.copy_(torch.tensor([[.8, .2, .3, .4]]).repeat(4, 1))
        parent_xyz = model.get_xyz.detach().double().clone()
        parent_cov = activated_covariance(model._scaling, model._rotation).clone()
        mass = model.get_intensity[:, 0].double()*model.get_scaling.double().prod(-1)
        report = self.step(model)
        self.assertEqual(report['split'], 4)
        positions = model.get_xyz.double().reshape(4, 2, 3)
        torch.testing.assert_close(positions.mean(1), parent_xyz)
        child_mass = (model.get_intensity[:, 0].double()*model.get_scaling.double().prod(-1)).reshape(4, 2)
        torch.testing.assert_close(child_mass.sum(1), mass, rtol=1e-6, atol=1e-8)
        cov = activated_covariance(model._scaling, model._rotation).reshape(4, 2, 3, 3)
        offset = positions-parent_xyz[:, None]
        torch.testing.assert_close((cov+offset[..., :, None]*offset[..., None, :]).mean(1),
                                   parent_cov, rtol=2e-6, atol=1e-8)
        for group in model.optimizer.param_groups:
            self.assertEqual(torch.count_nonzero(model.optimizer.state[group['params'][0]]['exp_avg']), 0)

    def test_layer_spacing_does_not_force_normal_split(self):
        model = self.model(scales=(.05, .05, .8))
        self.assertEqual(self.step(model)['clone'], 4)

    def test_capacity_random_selection_and_symmetric_clone_mapping(self):
        self.args.max_num_pts = 6
        self.args.structural_clone_jitter = .1
        model = self.model()
        intensity = model.get_intensity.detach().clone()
        xyz = model.get_xyz.detach().clone()
        report = self.step(model)
        self.assertEqual((report['clone'], report['after']), (2, 6))
        children = model.get_intensity[4:, 0]
        parents = torch.stack([torch.where(intensity[:, 0] == value*2)[0][0] for value in children])
        torch.testing.assert_close(model.get_xyz[parents]+model.get_xyz[4:], 2*xyz[parents])
        for key in ('_xyz', '_intensity'):
            state = model.optimizer.state[getattr(model, key)]['exp_avg']
            self.assertEqual(torch.count_nonzero(state[parents]), 0)
            untouched = torch.ones(4, dtype=torch.bool); untouched[parents] = False
            self.assertGreater(torch.count_nonzero(state[:4][untouched]), 0)

    def test_prune_grace_minimum_and_state_alignment(self):
        self.args.structural_prune_peak = .1
        self.args.structural_growth_fraction = 0.
        model = self.model()
        with torch.no_grad():model._intensity[:2] = 0.
        model.denom[1] = 0  # 未充分观测的新点不参与弱贡献裁剪。
        report = self.step(model)
        self.assertEqual(report['pruned'], 1)
        self.assertEqual(report['after'], 3)
        expected = torch.tensor([[2., 2., 2.], [3., 3., 3.], [4., 4., 4.]])
        torch.testing.assert_close(model.optimizer.state[model._xyz]['exp_avg'], expected)
        model._intensity.data.zero_();model.denom.fill_(1.)
        self.assertEqual(self.step(model)['after'], 1)

    def test_invalid_point_is_removed_without_normalizing_bad_rotation(self):
        model = self.model()
        self.args.structural_growth_fraction = 0.
        with torch.no_grad():model._rotation[0] = 0.
        self.assertEqual(self.step(model)['pruned'], 1)
        self.assertTrue(torch.isfinite(model.get_rotation).all())
        with torch.no_grad():model._rotation.zero_()
        with self.assertRaises(FloatingPointError):self.step(model)

    def test_gradient_units_relative_threshold_and_noop_parameter_identity(self):
        model = self.model()
        model.xyz_gradient_accum.zero_();model.denom.zero_()
        model._xyz.grad = torch.ones_like(model._xyz)
        model.add_densification_stats_3d(torch.tensor([True, False, True, False]))
        expected = model._voxel_spacing().norm()
        torch.testing.assert_close(model.xyz_gradient_accum[:, 0], torch.tensor([expected, 0., expected, 0.]))
        model.xyz_gradient_accum[:, 0] = torch.tensor([1., 2., 4., 8.])
        model.denom.fill_(1.)
        self.args.structural_grad_multiplier = 2.
        self.args.structural_growth_fraction = 0.
        parameter = model._xyz
        report = self.step(model)
        self.assertEqual(report['grad_threshold'], 4.)
        self.assertIs(model._xyz, parameter)
        self.assertEqual(model.denom.sum(), 0)

    def test_checkpoint_resume_after_batch_structure_preserves_rng_and_adam(self):
        from GetInitialGaussian import save_training_checkpoint, restore_training_checkpoint, check_resume_config
        model = self.model();self.step(model)
        self.args.device = 'cpu'
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(torch.cuda, 'get_rng_state_all', return_value=[]), \
                mock.patch.object(torch.cuda, 'set_rng_state_all'):
            path = str(Path(directory)/'state.pth')
            save_training_checkpoint(path, model, 1000, [], {}, self.args)
            expected = torch.rand(3)
            state = torch.load(path, weights_only=False)
            other = GaussianModel();restore_training_checkpoint(state, other, self.args)
            torch.testing.assert_close(torch.rand(3), expected)
            torch.testing.assert_close(other._xyz, model._xyz)
            torch.testing.assert_close(other.optimizer.state[other._scaling]['exp_avg'],
                                       model.optimizer.state[model._scaling]['exp_avg'])
            self.assertEqual(other.geometry_reference_count, 4)
            check_resume_config(state['config'], self.args)
            state['config'].pop('structural_algorithm')
            with self.assertRaisesRegex(ValueError, '结构算法'):check_resume_config(state['config'], self.args)

    def test_voxel_gradient_and_pixel_scale_are_coordinate_scale_invariant(self):
        first, second = self.model(), self.model()
        geometry = dict(second.geometry, scale_mm=second.geometry['scale_mm']/2,
                        bounds=[value*2 for value in second.bounds])
        attach_geometry(second, geometry)
        with torch.no_grad():
            second._xyz.mul_(2)
            second._scaling.add_(torch.tensor(2.).log())
        second.training_setup(self.args)
        first.xyz_gradient_accum.zero_();first.denom.zero_()
        first._xyz.grad = torch.ones_like(first._xyz)
        second._xyz.grad = torch.ones_like(second._xyz)*.5
        visible = torch.ones(4, dtype=torch.bool)
        first.add_densification_stats_3d(visible)
        second.add_densification_stats_3d(visible)
        torch.testing.assert_close(first.xyz_gradient_accum, second.xyz_gradient_accum)
        first_report, second_report = self.step(first), self.step(second)
        self.assertAlmostEqual(first_report['sigma_pixels_median'], second_report['sigma_pixels_median'], places=6)
        self.assertEqual(first_report['selected'], second_report['selected'])

    def test_mixed_clone_split_prune_commits_aligned_states_once(self):
        self.args.max_num_pts = 6
        self.args.structural_prune_peak = .1
        model = self.model()
        with torch.no_grad():
            model._scaling[[1, 3]] = torch.tensor([.3, .1, .2]).log()
            model._intensity[0] = 0.
        with mock.patch.object(model, '_replace_structure', wraps=model._replace_structure) as commit:
            report = self.step(model)
        self.assertEqual(commit.call_count, 1)
        self.assertEqual((report['clone'], report['split'], report['pruned'], report['after']), (1, 2, 1, 6))
        torch.testing.assert_close(model.optimizer.state[model._scaling]['exp_avg'][:, 0],
                                   torch.tensor([3., 0., 0., 0., 0., 0.]))
        self.assertEqual(len(model.optimizer.state), 4)
        for group in model.optimizer.param_groups:
            parameter = group['params'][0]
            self.assertEqual(parameter.shape, model.optimizer.state[parameter]['exp_avg'].shape)
        # 新参数下一步可继续使用 Adam，不能留下旧参数引用。
        sum(group['params'][0].square().sum() for group in model.optimizer.param_groups).backward()
        model.optimizer.step()

    def test_split_respects_log_scale_floor(self):
        self.args.log_scale_min = -1.21
        model = self.model(scales=(.3, .3, .3))
        self.assertEqual(self.step(model)['selected'], 0)

    def test_short_cuda_render_update_structure_update(self):
        if not torch.cuda.is_available():self.skipTest('需要 CUDA')
        from Renderer import render
        model = self.model(scales=(.3, .1, .2), device='cuda')
        for iteration in range(4):
            loss = (render(model, [17, 19], 1, 2, 3, **model.geometry['rendering'])['render']-.1).square().mean()
            loss.backward()
            model.add_densification_stats_3d(torch.ones(len(model.get_xyz), device='cuda', dtype=torch.bool))
            model.optimizer.step();model.optimizer.zero_grad(set_to_none=True)
            if iteration == 1:
                model.xyz_gradient_accum.fill_(1.);model.denom.fill_(1.)
                self.step(model)
        for group in model.optimizer.param_groups:
            self.assertTrue(torch.isfinite(group['params'][0]).all())
            self.assertEqual(group['params'][0].shape, model.optimizer.state[group['params'][0]]['exp_avg'].shape)
        torch.cuda.synchronize()


if __name__ == '__main__':
    unittest.main()
