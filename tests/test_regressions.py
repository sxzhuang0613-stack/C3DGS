"""配置和接口回归可在 CPU 运行；真实张量及 CUDA 验证按依赖明确跳过。"""
import ast
import importlib.util
import math
import numbers
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source_function(filename, name, namespace, class_name=None):
    tree = ast.parse((ROOT / filename).read_text())
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    function = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


@unittest.skipUnless(importlib.util.find_spec('yaml'), '需要 PyYAML')
class ConfigTests(unittest.TestCase):
    def test_configs_have_numeric_optimizer_and_renderer_values(self):
        from config_utils import load_config
        args = load_config(ROOT / 'config_train.yaml')
        for key in ('adam_eps', 'stage2_adam_eps', 'covariance_jitter', 'initial_distance_floor'):
            self.assertIsInstance(getattr(args, key), float)
        self.assertEqual(args.interpolate_iter, 12500)
        self.assertGreaterEqual(args.max_num_pts, args.num_pts)

    def test_test_config_only_contains_stage1_parameters(self):
        from config_utils import load_config
        args = load_config(ROOT / 'config_test.yaml')
        self.assertFalse(any(key.startswith('stage2') for key in vars(args)))
        self.assertFalse(hasattr(args, 'iters'))
        self.assertIn('{case}', args.stage1_ply_path)

    def test_duplicate_keys_are_rejected(self):
        from config_utils import load_config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('device: cuda:0\ndevice: cuda:1\n')
            with self.assertRaisesRegex(ValueError, '重复'):
                load_config(path)

    def test_config_paths_do_not_depend_on_cwd(self):
        from config_utils import load_config
        args = load_config(ROOT / 'config_test.yaml')
        self.assertEqual(Path(args.output_path), ROOT / 'output/evaluation')

    def test_active_entry_fields_are_all_configured(self):
        from config_utils import load_config
        for config, files in (
            ('config_train.yaml', ['GetInitialGaussian.py', 'Motion.py', 'temp.py',
                                   'Gaussians/GaussianModel_noop.py', 'config_utils.py']),
            ('config_test.yaml', ['evaluate.py', 'config_utils.py']),
        ):
            args = load_config(ROOT / config)
            used = set()
            for filename in files:
                tree = ast.parse((ROOT / filename).read_text())
                used.update(n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                            and isinstance(n.value, ast.Name) and n.value.id == 'args')
            self.assertFalse(used - set(vars(args)))


class RendererTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        class Value:
            device, dtype = 'cuda:0', 'float32'
            def contiguous(self): return self
            def __gt__(self, value): return True
        self.value = Value()
        def rasterize(**kwargs):
            self.calls.append(kwargs)
            return [self.value], self.value, self.value
        settings = types.SimpleNamespace(from_axis=lambda **kwargs: kwargs)
        namespace = {'GaussianRasterizationSettings': settings,
                     'covariance_to_vector': lambda value: value,
                     'rasterize_gaussians': rasterize}
        self.render = source_function('Renderer.py', 'render', namespace)
        self.model = types.SimpleNamespace(
            get_xyz=self.value, get_color=self.value, get_scaling=self.value,
            get_rotation=self.value, bounds=[2., 3., 4.])

    def test_precomputed_covariance_is_forwarded_exclusively(self):
        self.render(self.model, [16, 20], 0.25, 1, 3, covariance=self.value)
        call = self.calls[-1]
        self.assertIs(call['cov3Ds_precomp'], self.value)
        self.assertNotIn('scales', call)
        self.assertNotIn('rotations', call)

    def test_axis_bounds_and_settings_are_preserved(self):
        self.render(self.model, [16, 20], 0.25, 0, 3,
                    raster_options={'background_weight': 0.2})
        settings = self.calls[-1]['raster_settings']
        self.assertEqual(settings['axis'], 0)
        self.assertEqual(settings['bounds'], self.model.bounds)
        self.assertEqual(settings['slice_idx'], 0.25)
        self.assertEqual(settings['background_weight'], 0.2)

    def test_scale_rotation_path_still_works(self):
        result = self.render(self.model, [16, 20], 0, 2, 3)
        self.assertIn('scales', self.calls[-1])
        self.assertNotIn('cov3Ds_precomp', self.calls[-1])
        self.assertEqual(set(result), {'render', 'radii', 'visibility_filter'})

    def test_continuous_index_validation(self):
        # 运行原始校验；在进入 Tensor 几何构造时停止，不冒充几何数值测试。
        class GeometryReached(Exception): pass
        def reached(*args, **kwargs): raise GeometryReached
        namespace = {'numbers': numbers, 'math': math,
                     'torch': types.SimpleNamespace(as_tensor=reached, float64=None)}
        source_function('diff-gaussian-slice-rendering/diff_gaussian_slice_rendering/__init__.py',
                        '_positive_int', namespace)
        axis = source_function(
            'diff-gaussian-slice-rendering/diff_gaussian_slice_rendering/__init__.py',
            'from_axis', namespace, class_name='GaussianRasterizationSettings')
        for index, slices in [(0., 1), (0.25, 3), (2., 3)]:
            with self.subTest(index=index), self.assertRaises(GeometryReached):
                axis(None, 16, 16, slice_idx=index, num_slices=slices)
        for index in (-0.1, 2.1, float('nan'), float('inf'), True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                axis(None, 16, 16, slice_idx=index, num_slices=3)


HAS_TENSORS = all(importlib.util.find_spec(name) is not None for name in ('torch', 'numpy', 'plyfile', 'yaml'))


@unittest.skipUnless(HAS_TENSORS, '需要 PyTorch、NumPy、plyfile 和 PyYAML')
class TensorTests(unittest.TestCase):
    def model(self, scale=0.1, count=4, cap=6):
        import torch
        from config_utils import load_config
        from Gaussians.GaussianModel_noop import GaussianModel
        args = load_config(ROOT / 'config_train.yaml')
        args.device, args.max_num_pts = 'cpu', cap
        model = GaussianModel()
        for name, value in (
            ('_xyz', torch.zeros(count, 3)), ('_intensity', torch.full((count, 1), 0.5)),
            ('_scaling', torch.full((count, 3), math.log(scale))),
            ('_rotation', torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1)),
        ):
            setattr(model, name, torch.nn.Parameter(value))
        model.training_setup(args)
        return model, args

    def test_clone_and_split_obey_capacity(self):
        import torch
        for scale in (0.005, 0.2):
            model, args = self.model(scale=scale)
            model.xyz_gradient_accum.fill_(1)
            model.denom.fill_(1)
            model.densify_and_prune(0.0005, 0.01, 0.05)
            self.assertEqual(model.get_xyz.shape[0], args.max_num_pts)
            self.assertEqual(model.denom.shape[0], args.max_num_pts)
            self.assertTrue(torch.isfinite(model.get_scaling).all())

    def test_new_ply_round_trip(self):
        import torch
        from Gaussians.GaussianModel_noop import GaussianModel
        model, _ = self.model()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'model.ply')
            model.save_ply(path)
            other = GaussianModel()
            other.load_ply(path, device='cpu')
        self.assertTrue(torch.equal(model.get_intensity, other.get_intensity))
        self.assertTrue(torch.equal(model.get_xyz, other.get_xyz))

    def test_training_checkpoint_restores_optimizer_and_randomness(self):
        import torch
        from GetInitialGaussian import save_training_checkpoint, restore_training_checkpoint
        model, args = self.model()
        model.get_xyz.sum().backward()
        model.optimizer.step()
        model.xyz_gradient_accum.fill_(3)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(torch.cuda, 'get_rng_state_all', return_value=[]), \
                mock.patch.object(torch.cuda, 'set_rng_state_all'):
            path = str(Path(directory) / 'state.pth')
            save_training_checkpoint(path, model, 123, [123], {'loss': [0.5]}, args)
            expected_random = random.random()
            state = torch.load(path, weights_only=False)
            other, _ = self.model(count=2)
            restore_training_checkpoint(state, other, args)
            self.assertEqual(random.random(), expected_random)
            self.assertTrue(torch.equal(model.get_xyz, other.get_xyz))
            self.assertTrue(torch.equal(other.xyz_gradient_accum, model.xyz_gradient_accum))
            self.assertEqual(state['iteration'], 123)
            self.assertTrue(torch.equal(model.optimizer.state[model._xyz]['exp_avg'],
                                        other.optimizer.state[other._xyz]['exp_avg']))

    def test_ssim_identity(self):
        import torch
        from CostFunctions import ssim
        image = torch.rand(2, 1, 16, 16)
        self.assertAlmostEqual(ssim(image, image).item(), 1., places=6)

    def test_cuda_covariance_path_has_gradients(self):
        import torch
        sys.path.insert(0, str(ROOT / 'diff-gaussian-slice-rendering'))
        from diff_gaussian_slice_rendering import GaussianRasterizationSettings, rasterize_gaussians, _C
        if not torch.cuda.is_available() or _C is None:
            self.skipTest('需要可用 CUDA 和已编译的当前渲染扩展')
        means = torch.tensor([[0., 0., 0.]], device='cuda', requires_grad=True)
        intensity = torch.tensor([[0.5]], device='cuda', requires_grad=True)
        covariance = torch.tensor([[0.1, 0., 0., 0.1, 0., 0.1]], device='cuda', requires_grad=True)
        settings = GaussianRasterizationSettings.from_axis(16, 16, slice_idx=0.25, num_slices=2)
        image, _, _ = rasterize_gaussians(means, intensity, settings, cov3Ds_precomp=covariance)
        image.sum().backward()
        for value in (means, intensity, covariance):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
        self.assertGreater(covariance.grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
