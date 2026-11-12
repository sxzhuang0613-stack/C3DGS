"""归一化 Stage1 包装、梯度、背景及恢复协议回归；CPU 模拟器仅用于测试。"""
import copy
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from Renderer import render, covariance_to_vector
from config_utils import load_config
from case_geometry import make_geometry, attach_geometry, check_geometry
from Gaussians.GaussianModel_structural import GaussianModel
from Gaussians.initialization import initialize_from_images
from Gaussians.utils import covariance_from_scaling_rotation
from stage1_test_support import reference_render, unpack_covariance, NATIVE_AVAILABLE


class RendererTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(52)
        self.calls = []
        calls = self.calls
        class Rasterizer:
            def __init__(self, settings):
                self.settings = settings
            def __call__(self, **inputs):
                settings = self.settings
                calls.append((settings,inputs))
                means = inputs['means3D']
                view = SimpleNamespace(get_xyz=means,get_intensity=inputs['colors_precomp'][:,:1],
                                       bounds=[1.,1.,1.])
                image = reference_render(view,[settings.image_height,settings.image_width],
                    (settings.campos[settings.axis].item()+1)/2,settings.axis,2,
                    covariance=unpack_covariance(inputs['cov3D_precomp']),
                    raster_options={'background':settings.bg[0].item()})['render']
                return image[None].expand(3,-1,-1),means.new_ones(len(means)),image[None]*0
        package = SimpleNamespace(GaussianRasterizationSettings=SimpleNamespace,
                                  GaussianRasterizer=Rasterizer)
        patch = mock.patch.dict(sys.modules,{'diff_gaussian_rasterization_slice':package})
        patch.start();self.addCleanup(patch.stop)
        self.model = GaussianModel()
        self.model.bounds = [1.7,.9,2.3]
        self.model._xyz = torch.nn.Parameter(torch.tensor([[.1,.15,-.2],[-.2,.2,.15]],dtype=torch.float64))
        self.model._intensity = torch.nn.Parameter(torch.tensor([[.2],[.8]],dtype=torch.float64))
        self.model._scaling = torch.nn.Parameter(torch.tensor([[.9,.8,.7],[.6,.9,1.]],dtype=torch.float64).log())
        self.model._rotation = torch.nn.Parameter(torch.tensor([[.8,.2,.3,.4],[1.,.1,-.2,.3]],dtype=torch.float64))

    def test_nonuniform_coordinates_covariance_and_gradients(self):
        for axis in range(3):
            with self.subTest(axis=axis):
                options = dict(bounds=self.model.bounds,raster_options={'background':.12})
                image = render(self.model,[7,9],.75,axis,3,**options)['render']
                reference = reference_render(self.model,[7,9],.75,axis,3,**options)['render']
                torch.testing.assert_close(image,reference,atol=1e-12,rtol=1e-12)
                parameters=(self.model._xyz,self.model._intensity,self.model._scaling,self.model._rotation)
                upstream=torch.linspace(-1,1,image.numel()).reshape_as(image)
                actual=torch.autograd.grad((image*upstream).sum(),parameters)
                expected=torch.autograd.grad((reference*upstream).sum(),parameters)
                for a,b in zip(actual,expected):
                    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)
                settings,inputs=self.calls[-1]
                torch.testing.assert_close(settings.campos[axis],image.new_tensor(-.25))
                self.assertNotIn('scales',inputs)
                self.assertFalse(inputs['opacities'].requires_grad)

    def test_precomputed_covariance_is_differentiable(self):
        covariance=torch.tensor([[[.7,.05,.1],[.05,.8,.02],[.1,.02,.9]]],dtype=torch.float64).repeat(2,1,1).requires_grad_()
        full=render(self.model,[7,9],0,2,1,covariance=covariance)['render']
        packed=render(self.model,[7,9],0,2,1,covariance=covariance_to_vector(covariance))['render']
        torch.testing.assert_close(full,packed)
        full.sum().backward()
        self.assertTrue(torch.isfinite(covariance.grad).all())
        self.assertGreater(covariance.grad.abs().sum(),0)
        self.assertEqual(self.calls[-1][0].campos[2],0)

    def test_shared_covariance_preserves_multislice_gradients(self):
        covariance=covariance_from_scaling_rotation(self.model.get_scaling,self.model._rotation)
        shared=sum(render(self.model,[7,9],index,2,3,covariance=covariance)['render'].square().mean()
                   for index in [0.5,1.25])
        independent=sum(render(self.model,[7,9],index,2,3)['render'].square().mean()
                        for index in [0.5,1.25])
        parameters=(self.model._xyz,self.model._intensity,self.model._scaling,self.model._rotation)
        left=torch.autograd.grad(shared,parameters)
        right=torch.autograd.grad(independent,parameters)
        for a,b in zip(left,right):
            torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-10)

    def test_opacity_placeholder_and_public_contract(self):
        first=render(self.model,[5,7],0,2,1,opacities=torch.zeros(2,1))
        second=render(self.model,[5,7],0,2,1,opacities=torch.ones(2,1))
        torch.testing.assert_close(first['render'],second['render'])
        self.assertEqual(set(first),{'render','radii','visibility_filter'})
        self.assertEqual(first['render'].shape,(5,7))
        self.assertEqual(self.calls[-1][1]['colors_precomp'].shape,(2,3))

    def test_invalid_geometry_and_obsolete_options(self):
        for index,axis,count in [(float('nan'),2,3),(.1,2,1),(0,3,1),(0,2,0)]:
            with self.subTest(index=index,axis=axis,count=count),self.assertRaises(ValueError):
                render(self.model,[5,7],index,axis,count)
        with self.assertRaisesRegex(ValueError,'旧渲染选项'):
            render(self.model,[5,7],0,2,1,raster_options={'psf_sigma':0.})


class ProtocolAndInitializationTests(unittest.TestCase):
    def setUp(self):
        self.args=load_config(ROOT/'config_train.yaml')

    def test_config_rejects_old_renderer_and_amplitude_prune(self):
        original=yaml.safe_load((ROOT/'config_train.yaml').read_text())
        for values in [dict(original,psf_sigma=0.),dict(original,structural_prune_peak=.001),
                       dict(original,structural_algorithm='batch_v1')]:
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'config.yaml';path.write_text(yaml.safe_dump(values))
                with self.assertRaises(ValueError):load_config(path)

    def test_metadata_and_checkpoint_semantics(self):
        from GetInitialGaussian import check_resume_config,restore_training_checkpoint
        geometry=make_geometry([1.,1.2,8.],[3,7,9],self.args)
        self.assertEqual(geometry['intensity_semantics'],'local_value')
        old=copy.deepcopy(geometry);old.pop('render_model')
        with self.assertRaisesRegex(ValueError,'几何'):check_geometry(old,geometry)
        old_config=dict(vars(self.args));old_config.pop('render_model')
        with self.assertRaisesRegex(ValueError,'渲染模型'):check_resume_config(old_config,self.args)
        with self.assertRaisesRegex(ValueError,'归一化渲染记录'):
            restore_training_checkpoint({'config':vars(self.args),'geometry':old},GaussianModel(),self.args)

    def test_nonzero_background_is_only_subtracted_outside_coverage(self):
        self.args.num_pts=1
        self.args.background=.3
        self.args.init_fit_steps=3
        images=torch.full((1,9,13),.7)
        geometry=make_geometry([1.,1.5,8.],images.shape,self.args)
        model=GaussianModel();attach_geometry(model,geometry)
        with mock.patch('Gaussians.initialization.render',side_effect=reference_render):
            initialize_from_images(model,images,geometry,self.args,geometry['rendering'],'constant')
        torch.testing.assert_close(model.get_intensity,torch.full((1,1),.7),atol=1e-6,rtol=1e-6)


if __name__=='__main__':
    unittest.main()
