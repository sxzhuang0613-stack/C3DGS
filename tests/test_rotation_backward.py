"""覆盖实际点数下的四元数梯度和连续多切片反向，避免只测零梯度场景。"""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if importlib.util.find_spec('torch') is None:
    raise unittest.SkipTest('需要已有 PyTorch')
import torch
import diff_gaussian_slice_rendering as package
from Gaussians.GaussianModel_structural import GaussianModel
from Renderer import render
from config_utils import load_config,render_options


@unittest.skipUnless(torch.cuda.is_available() and package._C is not None,'需要 CUDA 与已安装渲染扩展')
class RotationBackwardTests(unittest.TestCase):
    def test_native_forward_and_backward_still_reject_nonunit_quaternion(self):
        means = torch.zeros(1,3,device='cuda')
        intensity = torch.ones(1,1,device='cuda')
        scales = torch.full((1,3),.1,device='cuda')
        rotation = torch.tensor([[1.,0.,0.,0.]],device='cuda')
        covariance = torch.empty(0,device='cuda')
        cfg = package.GaussianRasterizationSettings.from_axis(3,3)
        affine = cfg.slice_to_world.cuda().float().contiguous()
        psf = cfg.psf_vector.cuda().float().contiguous()
        args = (3,3,0.,cfg.kernel_cutoff,cfg.cutoff_transition,cfg.covariance_jitter,False)
        bad = rotation*2
        with self.assertRaisesRegex(RuntimeError,'native API expects normalized'):
            package._C.rasterize_gaussians(means,intensity,scales,bad,covariance,affine,psf,*args)
        count,image,_,geometry,binning,pixels,_ = package._C.rasterize_gaussians(
            means,intensity,scales,rotation,covariance,affine,psf,*args)
        with self.assertRaisesRegex(RuntimeError,'native API expects normalized'):
            package._C.rasterize_gaussians_backward(means,intensity,scales,bad,covariance,
                affine,psf,*args,torch.ones_like(image),count,geometry,binning,pixels)

    def test_large_batch_activation_matches_cpu_forward_and_gradient(self):
        generator = torch.Generator().manual_seed(71)
        raw = torch.randn(50000,4,generator=generator)
        reference = raw.clone().requires_grad_()
        # CPU double 行归约作为独立数学参考；保留两次激活的 float32 舍入。
        first = (reference.double()/reference.double().norm(dim=1,keepdim=True)).float()
        expected = (first.double()/first.double().norm(dim=1,keepdim=True)).float()
        weights = torch.linspace(-1,1,raw.numel()).reshape_as(raw)
        (expected*weights).mean().backward()
        model = GaussianModel()
        model._rotation = torch.nn.Parameter(raw.cuda())
        actual = package.normalize_rotations(model.get_rotation)
        (actual*weights.cuda()).mean().backward()
        torch.testing.assert_close(actual.cpu(),expected.detach(),atol=2e-7,rtol=2e-6)
        torch.testing.assert_close(model._rotation.grad.cpu(),reference.grad,atol=1e-9,rtol=2e-4)
        self.assertGreater(model._rotation.grad.abs().max().item(),0.)

    def test_50000_points_two_views_600_backward_updates(self):
        args = load_config(ROOT/'config_train.yaml')
        generator = torch.Generator(device='cuda').manual_seed(42)
        count = 50000
        model = GaussianModel()
        model._xyz = torch.nn.Parameter(torch.rand(count,3,device='cuda',generator=generator)*2-1)
        model._scaling = torch.nn.Parameter(torch.tensor([[.05,.03,.04]],device='cuda').log().repeat(count,1))
        model._rotation = torch.nn.Parameter(torch.randn(count,4,device='cuda',generator=generator))
        model._intensity = torch.nn.Parameter(torch.full((count,1),.01,device='cuda'))
        model.bounds = args.bounds
        model.training_setup(args)
        for _ in range(600):
            images = [render(model,[17,19],index,2,3,**render_options(args))['render'] for index in (0,1)]
            loss = (torch.stack(images)-.1).square().mean()
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
        for parameter in (model._xyz,model._scaling,model._rotation,model._intensity):
            self.assertTrue(torch.isfinite(parameter).all())
        self.assertTrue(torch.isfinite(loss))


if __name__=='__main__':unittest.main()
