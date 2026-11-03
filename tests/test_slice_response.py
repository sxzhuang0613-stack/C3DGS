"""加性场、解析 PSF、二维 footprint 及 CUDA 完整反向的回归测试。"""
import importlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import types
import unittest

if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('需要 PyTorch')
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'diff-gaussian-slice-rendering'))
import diff_gaussian_slice_rendering as package
from diff_gaussian_slice_rendering import GaussianRasterizationSettings as Settings, rasterize_gaussians
from diff_gaussian_slice_rendering.reference import (
    covariance_from_scales, footprint_2d, rasterize_reference, response_3d, unpack_covariance,
)

# 可在独立 build_ext 目录验证新二进制，无需安装或覆盖已安装版本。
if os.environ.get('C3DGS_EXTENSION_DIR'):
    package.__path__.append(str(Path(os.environ['C3DGS_EXTENSION_DIR']) / 'diff_gaussian_slice_rendering'))
    package._C = importlib.import_module('diff_gaussian_slice_rendering._C_G3D')


def scene(dtype=torch.float64, device='cpu'):
    generator = torch.Generator(device=device).manual_seed(71)
    means = torch.randn(4, 3, generator=generator, dtype=dtype, device=device) * .2
    scales = torch.tensor([[.1, .3, .6], [.4, .15, .2], [.2, .5, .12], [.3, .2, .4]], dtype=dtype, device=device)
    rotations = torch.randn(4, 4, generator=generator, dtype=dtype, device=device)
    intensity = torch.tensor([[.7], [-.6], [0.], [.2]], dtype=dtype, device=device)
    return tuple(value.requires_grad_() for value in (means, scales, rotations, intensity))


def settings(mode='gaussian', sigma=.15, **kwargs):
    affine = torch.tensor([[.08, .015, 0., -.5], [.01, .075, 0., -.5],
                           [.025, -.02, 1., .03], [0., 0., 0., 1.]], dtype=torch.float64)
    return Settings(13, 17, affine, response_mode=mode, psf_sigma=sigma,
                    background=-.17, **kwargs)


class AnalyticTests(unittest.TestCase):
    def test_point_zero_sigma_and_signed_additivity(self):
        means, scales, rotations, intensity = scene()
        cov = covariance_from_scales(scales, rotations)
        point = settings('point', 0.)
        zero = settings('gaussian', 0.)
        self.assertTrue(torch.equal(rasterize_reference(means, intensity, cov, point),
                                    rasterize_reference(means, intensity, cov, zero)))
        total = rasterize_reference(means, intensity, cov, point)
        parts = [rasterize_reference(means[i:i+1], intensity[i:i+1], cov[i:i+1], point)
                 - point.background for i in range(len(means))]
        torch.testing.assert_close(total, point.background + sum(parts))
        self.assertLess(rasterize_reference(means[:1], -intensity[:1], cov[:1], point).min().item(), point.background)
        empty = rasterize_reference(means[:0], intensity[:0], cov[:0], point)
        torch.testing.assert_close(empty, torch.full_like(empty, point.background))

    def test_analytic_matches_high_precision_numerical_integral(self):
        means, scales, rotations, _ = scene()
        cov = covariance_from_scales(scales, rotations)
        points = torch.tensor([[.1,.2,.0],[-.3,.1,.2]], dtype=torch.float64)
        vector = torch.tensor([.1,-.07,.18], dtype=torch.float64)
        # t 是标准正态；在 [-10,10] 上逐步加密，独立核对无限域解析积分。
        estimates = []
        for count in (2001, 8001):
            t = torch.linspace(-10,10,count,dtype=torch.float64)
            shifted = (points[:,None] + t[None,:,None]*vector).reshape(-1,3)
            values = response_3d(shifted, means, cov, torch.zeros(3)).reshape(2,count,4)
            normal = torch.exp(-t.square()/2) / math.sqrt(2*math.pi)
            estimates.append(torch.trapezoid(values * normal[None,:,None], t, dim=1))
        exact = response_3d(points, means, cov, vector)
        torch.testing.assert_close(estimates[0], estimates[1], rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(exact, estimates[-1], rtol=1e-10, atol=1e-12)

    def test_3d_and_2d_outputs_and_all_parameter_gradients(self):
        inputs = scene()
        means, scales, rotations, intensity = inputs
        cfg = settings(covariance_jitter=0.)
        covariance = covariance_from_scales(scales, rotations)
        rows,cols = torch.meshgrid(torch.arange(13,dtype=means.dtype),torch.arange(17,dtype=means.dtype),indexing='ij')
        pixels = torch.stack((cols,rows),-1).reshape(-1,2)
        points = pixels @ cfg.slice_to_world[:3,:2].T + cfg.slice_to_world[:3,3]
        direct = cfg.background + (response_3d(points,means,covariance,cfg.psf_vector) @ intensity).reshape(1,13,17)
        projected = rasterize_reference(means,intensity,covariance,cfg,finite_support=False)
        torch.testing.assert_close(direct,projected,rtol=1e-11,atol=1e-12)
        weight=torch.linspace(-1,1,direct.numel()).reshape_as(direct)
        first=torch.autograd.grad((direct*weight).sum(),inputs,retain_graph=True)
        second=torch.autograd.grad((projected*weight).sum(),inputs)
        for a,b in zip(first,second): torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)

    def test_amplitude_covariance_term_and_finite_difference(self):
        cov=torch.diag(torch.tensor([.04,.08,.12],dtype=torch.float64))[None].requires_grad_()
        mean=torch.zeros(1,3,dtype=torch.float64,requires_grad=True)
        vector=torch.tensor([0.,0.,.4],dtype=torch.float64)
        output=response_3d(torch.zeros(1,3,dtype=torch.float64),mean,cov,vector).sum()
        gradient=torch.autograd.grad(output,cov)[0]
        self.assertGreater(gradient[0,2,2].item(),0.)
        epsilon=1e-6
        plus=cov.detach().clone();minus=plus.clone()
        plus[0,2,2]+=epsilon;minus[0,2,2]-=epsilon
        difference=(response_3d(mean.detach(),mean.detach(),plus,vector)-response_3d(mean.detach(),mean.detach(),minus,vector))/(2*epsilon)
        self.assertAlmostEqual(gradient[0,2,2].item(),difference.item(),places=8)

    def test_cutoff_error_is_separate_from_exact_integral(self):
        means,scales,rotations,intensity=scene()
        covariance=covariance_from_scales(scales,rotations)
        for transition in (0.,2.):
            cfg=settings(kernel_cutoff=1e-5,cutoff_transition=transition)
            exact=rasterize_reference(means,intensity,covariance,cfg,finite_support=False)
            clipped=rasterize_reference(means,intensity,covariance,cfg)
            bound=intensity.abs().sum()*cfg.kernel_cutoff*math.exp(transition/2)
            self.assertLessEqual((exact-clipped).abs().max().item(),bound.item()+1e-12)

    def test_dicom_physical_psf_and_affine_invariance(self):
        ds=types.SimpleNamespace(Rows=13,Columns=17,ImagePositionPatient=[0.,0.,0.],
                                 ImageOrientationPatient=[1.,0.,0.,0.,1.,0.],PixelSpacing=[.8,.7],SliceThickness=9.)
        transform=torch.tensor([[.1,.02,.03,.2],[0.,.2,.04,-.1],[0.,0.,.3,.05],[0.,0.,0.,1.]],dtype=torch.float64)
        physical=Settings.from_dicom(ds,response_mode='gaussian',psf_sigma_mm=.6,covariance_jitter=0.)
        model=Settings.from_dicom(ds,response_mode='gaussian',psf_sigma_mm=.6,model_from_patient=transform,covariance_jitter=0.)
        torch.testing.assert_close(model.psf_vector,transform[:3,:3]@physical.psf_vector)
        self.assertFalse(torch.allclose(torch.linalg.cross(model.psf_vector,model.slice_to_world[:3,2]),torch.zeros(3,dtype=torch.float64)))
        resized=model.with_pixel_transform([[2,0,.5],[0,2,.5],[0,0,1]],image_height=6,image_width=8)
        torch.testing.assert_close(resized.psf_vector,model.psf_vector)
        means,scales,rotations,intensity=scene()
        covariance=covariance_from_scales(scales,rotations)
        mapped=means@transform[:3,:3].T+transform[:3,3]
        mapped_cov=transform[:3,:3]@covariance@transform[:3,:3].T
        torch.testing.assert_close(rasterize_reference(means,intensity,covariance,physical,finite_support=False),
                                   rasterize_reference(mapped,intensity,mapped_cov,model,finite_support=False),rtol=1e-10,atol=1e-12)
        self.assertTrue(torch.equal(Settings.from_dicom(ds).psf_vector,torch.zeros(3,dtype=torch.float64)))

    def test_fixed_geometry_contract_and_zero_sigma(self):
        with self.assertRaises(ValueError): settings('point',.1)
        with self.assertRaises(ValueError): settings('gaussian',-.1)
        with self.assertRaises(ValueError): settings('gaussian',torch.tensor(.1,requires_grad=True))
        with self.assertRaises(ValueError): settings(psf_std_vector=torch.ones(3,requires_grad=True))
        for index in (0.,.25,1.):
            Settings.from_axis(8,8,slice_idx=index,num_slices=2,response_mode='gaussian',psf_sigma=0.)


@unittest.skipUnless(torch.cuda.is_available() and package._C is not None, '需要 CUDA 和 API_VERSION=3 扩展')
class CudaTests(unittest.TestCase):
    def compare(self,cfg,precomputed=False):
        inputs=scene(torch.float32,'cuda')
        means,scales,rotations,intensity=inputs
        covariance=covariance_from_scales(scales,rotations)
        if precomputed:
            packed=torch.stack((covariance[:,0,0],covariance[:,0,1],covariance[:,0,2],
                                covariance[:,1,1],covariance[:,1,2],covariance[:,2,2]),-1).detach().requires_grad_()
            inputs=(means,intensity,packed)
            covariance=unpack_covariance(packed)
            actual,radii,weights=rasterize_gaussians(means,intensity,cfg,cov3Ds_precomp=packed)
        else:
            actual,radii,weights=rasterize_gaussians(means,intensity,cfg,scales=scales,rotations=rotations)
        expected=rasterize_reference(means,intensity,covariance,cfg)
        torch.testing.assert_close(actual,expected,rtol=4e-5,atol=3e-6)
        weight=torch.linspace(-1,1,actual.numel(),device='cuda').reshape_as(actual)
        cg=torch.autograd.grad((actual*weight).sum(),inputs,retain_graph=True)
        rg=torch.autograd.grad((expected*weight).sum(),inputs)
        for a,b in zip(cg,rg):
            self.assertTrue(torch.isfinite(a).all())
            torch.testing.assert_close(a,b,rtol=4e-4,atol=5e-4)
        self.assertFalse(weights.requires_grad)
        self.assertFalse(radii.requires_grad)
        self.assertGreater(weights.max().item(),0.)

    def test_forward_backward_both_covariance_paths(self):
        for mode,sigma in [('point',0.),('gaussian',0.),('gaussian',.15),('gaussian',.6)]:
            for transition in (0.,2.):
                for precomputed in (False,True):
                    with self.subTest(mode=mode,sigma=sigma,transition=transition,packed=precomputed):
                        self.compare(settings(mode,sigma,cutoff_transition=transition),precomputed)

    def test_cuda_central_differences_for_original_attributes(self):
        means,scales,rotation,intensity=scene(torch.float32,'cuda')
        cfg=settings(kernel_cutoff=1e-7,cutoff_transition=2.)
        inputs=(means,scales,rotation,intensity)
        weight=torch.linspace(-1,1,13*17,device='cuda').reshape(1,13,17)
        def objective():
            image,_,_=rasterize_gaussians(means,intensity,cfg,scales=scales,rotations=rotation)
            return (image*weight).mean()
        gradients=torch.autograd.grad(objective(),inputs)
        epsilon=1e-3
        for value,gradient in zip(inputs,gradients):
            for component in range(value.shape[1]):
                with torch.no_grad():
                    original=value[0,component].item()
                    value[0,component]=original+epsilon
                    plus=objective().item()
                    value[0,component]=original-epsilon
                    minus=objective().item()
                    value[0,component]=original
                numerical=(plus-minus)/(2*epsilon)
                self.assertAlmostEqual(gradient[0,component].item(),numerical,
                                       delta=2e-4+abs(numerical)*.01)

    def test_general_psf_vector(self):
        self.compare(settings('gaussian',0.,psf_std_vector=[.1,-.05,.2]),True)

    def test_zero_sigma_bitwise_point_equivalence(self):
        means,scales,rotations,intensity=scene(torch.float32,'cuda')
        values=[]
        for cfg in (settings('point',0.),settings('gaussian',0.)):
            image,_,_=rasterize_gaussians(means,intensity,cfg,scales=scales,rotations=rotations)
            grads=torch.autograd.grad(image.sum(),(means,scales,rotations,intensity))
            values.append((image,grads))
        self.assertTrue(torch.equal(values[0][0],values[1][0]))
        for a,b in zip(values[0][1],values[1][1]):
            # 原子累加顺序可能不同，不要求反向逐位一致。
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)

    def test_large_batch_quaternion_normalization(self):
        # 大点数真实训练会触发 float64 行归约的不同 kernel，必须覆盖小场景之外的路径。
        count=50000
        means=torch.full((count,3),100.,device='cuda',requires_grad=True)
        scales=torch.full((count,3),.1,device='cuda',requires_grad=True)
        rotations=(torch.randn(count,4,device='cuda')*1e20).requires_grad_()
        intensity=torch.zeros(count,1,device='cuda',requires_grad=True)
        image,_,_=rasterize_gaussians(means,intensity,settings(),scales=scales,rotations=rotations)
        image.sum().backward()
        self.assertTrue(torch.isfinite(rotations.grad).all())
        self.assertEqual(rotations.grad.abs().max().item(),0.)

    def test_thickness_recovers_off_plane_gaussian(self):
        means=torch.tensor([[0.,0.,.5]],device='cuda',requires_grad=True)
        intensity=torch.tensor([[-.7]],device='cuda',requires_grad=True)
        scales=torch.tensor([[.15,.15,.02]],device='cuda',requires_grad=True)
        rotation=torch.tensor([[1.,0.,0.,0.]],device='cuda',requires_grad=True)
        outputs=[]
        for mode,sigma in [('point',0.),('gaussian',.4)]:
            cfg=Settings.from_axis(17,17,num_slices=1,response_mode=mode,psf_sigma=sigma)
            outputs.append(rasterize_gaussians(means,intensity,cfg,scales=scales,rotations=rotation))
        self.assertEqual(outputs[0][1].item(),0)
        self.assertGreater(outputs[1][1].item(),0)
        self.assertLess(outputs[1][0].min().item(),0.)
        outputs[1][0].sum().backward()
        self.assertGreater(means.grad.abs().sum().item(),0.)

    def test_empty_no_candidates_and_nondefault_stream(self):
        for count,z in [(0,0.),(2,100.)]:
            means=torch.full((count,3),z,device='cuda',requires_grad=True)
            intensity=torch.zeros(count,1,device='cuda',requires_grad=True)
            covariance=torch.tensor([[.1,0.,0.,.1,0.,.1]],device='cuda').repeat(count,1).requires_grad_()
            cfg=settings()
            with torch.cuda.stream(torch.cuda.Stream()):
                image,radii,weights=rasterize_gaussians(means,intensity,cfg,cov3Ds_precomp=covariance)
                image.sum().backward()
                torch.cuda.current_stream().synchronize()
            torch.testing.assert_close(image,torch.full_like(image,cfg.background))
            self.assertEqual(weights.sum().item(),0.)
            self.assertEqual(means.grad.abs().sum().item(),0.)

    @unittest.skipUnless(importlib.util.find_spec('yaml'), '需要 PyYAML 读取训练/测试配置')
    def test_train_and_test_render_configuration(self):
        from config_utils import load_config,render_options
        from Renderer import render
        means,scales,rotation,intensity=scene(torch.float32,'cuda')
        model=types.SimpleNamespace(get_xyz=means,get_color=intensity,get_scaling=scales,get_rotation=rotation)
        for filename in ('config_train.yaml','config_test.yaml'):
            args=load_config(ROOT/filename)
            for mode,sigma in [('point',0.),('gaussian',.1)]:
                args.response_mode,args.psf_sigma=mode,sigma
                result=render(model,[13,17],0.,2,2,**render_options(args))
                gradients=torch.autograd.grad(result['render'].mean(),(means,scales,rotation,intensity))
                self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_settings_snapshot_and_training_wrapper(self):
        from Renderer import render
        means,scales,rotation,intensity=scene(torch.float32,'cuda')
        model=types.SimpleNamespace(get_xyz=means,get_color=intensity,get_scaling=scales,get_rotation=rotation)
        for covariance in (None,covariance_from_scales(scales,rotation)):
            result=render(model,[13,17],0.,2,2,covariance=covariance,
                          raster_options={'response_mode':'gaussian','psf_sigma':.15})
            torch.autograd.grad(result['render'].sum(),(means,scales,rotation,intensity),retain_graph=True)
        cfg=settings('gaussian',0.,psf_std_vector=[0.,0.,.1])
        image,_,_=rasterize_gaussians(means,intensity,cfg,scales=scales,rotations=rotation)
        original=torch.autograd.grad(image.sum(),means,retain_graph=True)[0]
        cfg.slice_to_world[0,3]+=1
        cfg.psf_std_vector[2]=2
        torch.testing.assert_close(torch.autograd.grad(image.sum(),means)[0],original)


if __name__=='__main__': unittest.main()
