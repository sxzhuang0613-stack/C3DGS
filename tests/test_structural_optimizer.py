"""小规模合成结构回归；不读取实际病例、不运行完整训练或性能基准。"""
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if not all(importlib.util.find_spec(name) for name in ('torch','numpy','yaml','plyfile')):
    raise unittest.SkipTest('需要已有 PyTorch、NumPy、PyYAML 和 plyfile')
import numpy as np
import torch
from torch import nn
from config_utils import load_config,render_options,ssim_options,dataset_options
from Gaussians.GaussianModel_structural import GaussianModel,GaussianStructuralOptimizer,Primitive,Proposal
from diff_gaussian_slice_rendering import GaussianRasterizationSettings as Settings
from diff_gaussian_slice_rendering.local_response import activated_covariance,response_patches
from diff_gaussian_slice_rendering.reference import rasterize_reference
import diff_gaussian_slice_rendering as package



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
        self.args.structural_neighbors = 0
        self.args.structural_amplitude_weight = 0.
        self.args.structural_count_weight = 1e-5
        self.args.lambda_anisotropic = 0.
        self.args.interpolation = False
        self.args.structural_candidates = 2
        self.args.structural_max_accept = 3
        self.args.max_num_pts = 32
        self.render_patch = mock.patch('Gaussians.GaussianModel_structural.render',side_effect=reference_render)
        self.render_patch.start()
        self.addCleanup(self.render_patch.stop)

    def model(self,points=None,intensity=None):
        points = [[0.,0.,0.],[.4,.3,0.]] if points is None else points
        intensity = [.5,.2] if intensity is None else intensity
        result = GaussianModel()
        result._xyz = nn.Parameter(torch.tensor(points,dtype=torch.float32))
        result._scaling = nn.Parameter(torch.full((len(points),3),np.log(.3),dtype=torch.float32))
        result._rotation = nn.Parameter(torch.tensor([[1.,0.,0.,0.]]).repeat(len(points),1))
        result._intensity = nn.Parameter(torch.tensor(intensity,dtype=torch.float32)[:,None])
        result.bounds = self.args.bounds
        result.training_setup(self.args)
        return result

    @torch.no_grad()
    def images(self,model,size=17,slices=3):
        return torch.stack([reference_render(model,[size,size],index,2,slices,**render_options(self.args))['render'] for index in range(slices)])

    @torch.no_grad()
    def engine(self,model,target=None):
        target = self.images(model) if target is None else target
        result = GaussianStructuralOptimizer(model,target,render_options(self.args),ssim_options(self.args),1000)
        result.prediction = result.render_all()
        return result

    def test_local_response_matches_formal_reference_with_psf(self):
        model = self.model([[0.,0.,.2],[.2,-.1,0.]],[-.4,.8])
        covariance = activated_covariance(model._scaling,model._rotation)
        for mode,sigma in [('point',0.),('gaussian',.3)]:
            cfg = Settings.from_axis(17,17,num_slices=1,response_mode=mode,psf_sigma=sigma)
            patches = response_patches(model.get_xyz,covariance,[cfg],10000)
            actual = torch.zeros(17*17,dtype=torch.float64)
            for index,patch in enumerate(patches):
                actual[patch.indices] += patch.values*model.get_intensity[index,0]
            expected = rasterize_reference(model.get_xyz.double(),model.get_intensity.double(),covariance,cfg)
            torch.testing.assert_close(actual.reshape_as(expected),expected,atol=2e-7,rtol=2e-6)
            self.assertIsNone(response_patches(model.get_xyz[:1],covariance[:1],[cfg],1)[0])

    @torch.no_grad()
    def test_local_l1_ssim_delta_including_image_edges(self):
        engine = self.engine(self.model())
        indices = torch.tensor([0,1,16,17,144,288,289,300,500,866])
        delta = torch.linspace(-.2,.3,len(indices),dtype=torch.float64)
        expected = engine.prediction.clone()
        expected.flatten()[indices] += delta
        torch.testing.assert_close(engine.image_delta(indices,delta),
                                   engine.image_loss(expected)-engine.image_loss(engine.prediction),atol=1e-12,rtol=1e-10)

    @torch.no_grad()
    def test_weighted_refit_lasso_and_rank_deficient_columns(self):
        engine = self.engine(self.model())
        indices = torch.arange(50)
        basis = torch.linspace(.1,.9,50,dtype=torch.float64)[:,None]
        weights = engine._pixel_weights(indices)
        target = -2*basis[:,0]
        self.args.structural_amplitude_weight = 1e-3
        coefficient,norm = engine.refit(indices,basis,target,torch.zeros(1))
        correlation = basis[:,0]@(weights*target)
        expected = correlation.sign()*(correlation.abs()-self.args.structural_amplitude_weight*norm[0]).clamp_min(0)/(norm[0]**2)
        torch.testing.assert_close(coefficient[0],expected)
        self.args.structural_amplitude_weight = 0.
        repeated = basis.repeat(1,2)
        coefficient,_ = engine.refit(indices,repeated,target,torch.zeros(2))
        self.assertTrue(torch.isfinite(coefficient).all())
        torch.testing.assert_close(repeated@coefficient,target)

    @torch.no_grad()
    def test_split_positive_definiteness_and_signed_moments(self):
        model = self.model(intensity=[-.8,.2])
        engine = self.engine(model)
        proposal = engine.split(0)
        self.assertIsNotNone(proposal)
        covariance = engine.covariance(0)+self.args.covariance_jitter*torch.eye(3)
        mass = engine.coefficient(0)*torch.linalg.det(covariance).sqrt()
        children = [engine.primitive(i) for i in proposal.added]
        matrices = torch.stack([engine.covariance(i) for i in proposal.added])+self.args.covariance_jitter*torch.eye(3)
        masses = torch.stack([p.intensity[0].double() for p in children])*torch.linalg.det(matrices).sqrt()
        centers = torch.stack([p.xyz.double() for p in children])
        torch.testing.assert_close(masses.sum(),mass,rtol=2e-6,atol=1e-8)
        center = (centers*masses[:,None]).sum(0)/masses.sum()
        torch.testing.assert_close(center,model.get_xyz[0].double(),atol=1e-7,rtol=1e-6)
        offset = centers-center
        matched = ((matrices+offset[:,:,None]*offset[:,None,:])*masses[:,None,None]).sum(0)/masses.sum()
        torch.testing.assert_close(matched,covariance,atol=1e-7,rtol=3e-6)
        self.assertTrue((torch.linalg.eigvalsh(matrices)>0).all())

    @torch.no_grad()
    def test_merge_same_negative_sign_and_reject_opposite(self):
        engine = self.engine(self.model([[-.03,0.,0.],[.03,0.,0.]],[-.3,-.7]))
        proposal = engine.merge(0,1)
        self.assertIsNotNone(proposal)
        self.assertLess(engine.coefficient(proposal.added[0]).item(),0.)
        engine.coefficients[1] = torch.tensor(.7)
        self.assertIsNone(engine.merge(0,1))

    @torch.no_grad()
    def test_prune_zero_and_joint_opposite_cancellation(self):
        model = self.model([[0.,0.,0.],[0.,0.,0.],[.4,.2,0.]],[1.,-1.,.2])
        engine = self.engine(model)
        self.assertFalse(engine.evaluate(Proposal('prune',(0,))))
        self.assertTrue(engine.evaluate(Proposal('prune',(0,1))))
        self.assertEqual(engine.count,1)
        self.assertFalse(engine.evaluate(Proposal('prune',(2,))))

    @torch.no_grad()
    def test_birth_refit_and_swap_at_capacity(self):
        model = self.model([[.8,.8,0.],[-.4,-.3,0.]],[0.,.2])
        teacher = self.model([[-.4,-.3,0.],[.3,.1,0.]],[.2,.7])
        target = self.images(teacher)
        engine = self.engine(model,target)
        point = Primitive(teacher._xyz[1].detach(),teacher._scaling[1].detach(),
                          teacher._rotation[1].detach(),torch.zeros(1))
        index = engine.register(point)
        self.args.max_num_pts = 2
        self.assertFalse(engine.evaluate(Proposal('birth',(),(index,))))
        self.assertTrue(engine.evaluate(Proposal('swap',(0,),(index,))))
        self.assertEqual(engine.count,2)
        self.assertAlmostEqual(engine.coefficient(index).item(),.7,places=5)
        self.assertFalse(engine.evaluate(Proposal('prune',(0,))))

    @torch.no_grad()
    def test_birth_commit_and_formal_verification(self):
        model = self.model([[.8,.8,0.]],[0.])
        teacher = self.model([[.1,.2,0.]],[.7])
        engine = self.engine(model,self.images(teacher))
        point = Primitive(teacher._xyz[0].detach(),teacher._scaling[0].detach(),teacher._rotation[0].detach(),torch.zeros(1))
        index = engine.register(point)
        engine.proposals = lambda:[Proposal('birth',(),(index,))]
        report = engine.run()
        self.assertEqual(report['accepted']['birth'],1)
        self.assertEqual(len(model.get_xyz),2)
        self.assertLess(report['delta_j'],0.)
        self.assertFalse(report['rolled_back'])

    @torch.no_grad()
    def test_split_and_merge_are_accepted_by_observations(self):
        model = self.model([[0.,0.,0.]],[.8])
        engine = self.engine(model)
        proposal = engine.split(0)
        target = torch.zeros_like(engine.targets)
        for index in proposal.added:
            patch = engine.response(index)
            target.flatten()[patch.indices] += patch.values*engine.coefficient(index)
        engine.targets = target
        self.assertTrue(engine.evaluate(proposal))
        duplicate = self.model([[0.,0.,0.],[0.,0.,0.]],[.3,.5])
        engine = self.engine(duplicate)
        self.assertTrue(engine.evaluate(engine.merge(0,1)))
        self.assertEqual(engine.count,1)

    def test_batch_optimizer_mapping_and_rollback(self):
        model = self.model([[0.,0.,0.],[.2,.1,0.],[.4,.3,0.]],[.3,.4,.5])
        sum(value.sum() for value in (model._xyz,model._intensity,model._scaling,model._rotation)).backward()
        model.optimizer.step();model.optimizer.zero_grad(set_to_none=True)
        engine = self.engine(model)
        old = {group['name']:(group['params'][0],model.optimizer.state[group['params'][0]]) for group in model.optimizer.param_groups}
        old_statistics = model.denom
        point = engine.primitive(0)
        new = engine.register(Primitive(point.xyz.clone(),point.scaling.clone(),point.rotation.clone(),point.intensity.clone()))
        engine.removed = {0};engine.active_new = {new};engine.coefficients = {1:torch.tensor(.1)}
        with torch.no_grad(): snapshot = model._commit_structure(engine)
        for group in model.optimizer.param_groups:
            parameter = group['params'][0]
            state = model.optimizer.state[parameter]
            previous,old_state = old[group['name']]
            torch.testing.assert_close(state['exp_avg'][1],old_state['exp_avg'][2])
            self.assertTrue(torch.equal(state['exp_avg'][-1],torch.zeros_like(state['exp_avg'][-1])))
            self.assertEqual(state['step'].item(),old_state['step'].item())
            if group['name']=='intensity':
                self.assertEqual(state['exp_avg'][0].item(),0.)
            else:
                torch.testing.assert_close(state['exp_avg'][0],old_state['exp_avg'][1])
        model._rollback_structure(snapshot)
        self.assertIs(model.denom,old_statistics)
        for group in model.optimizer.param_groups:
            self.assertIs(group['params'][0],old[group['name']][0])
            self.assertIs(model.optimizer.state[group['params'][0]],old[group['name']][1])

    @torch.no_grad()
    def test_failed_verification_restores_all_references(self):
        model = self.model(intensity=[0.,.2])
        engine = self.engine(model)
        engine.proposals = lambda:[Proposal('prune',(0,))]
        old = model._xyz
        baseline_render = engine.render_all
        calls = [0]
        def inconsistent():
            calls[0] += 1
            result = baseline_render()
            return result if calls[0]==1 else result+.1
        engine.render_all = inconsistent
        report = engine.run()
        self.assertTrue(report['rolled_back'])
        self.assertIs(model._xyz,old)
        self.assertEqual(sum(report['accepted'].values()),0)

    @torch.no_grad()
    def test_response_budget_and_no_training_rng_consumption(self):
        model = self.model(intensity=[0.,.2])
        engine = self.engine(model)
        engine.proposals = lambda:[Proposal('prune',(0,))]
        py,np_state,torch_state = random.getstate(),np.random.get_state(),torch.get_rng_state()
        engine.run()
        self.assertEqual(random.getstate(),py)
        self.assertTrue(np.array_equal(np.random.get_state()[1],np_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(),torch_state))
        engine = self.engine(self.model())
        engine.args.structural_patch_budget = 1
        self.assertFalse(engine.evaluate(Proposal('prune',(0,))))
        self.assertEqual(engine.budget_rejections,1)

    @torch.no_grad()
    def test_formal_render_exception_rolls_back_commit(self):
        model = self.model(intensity=[0.,.2])
        engine = self.engine(model)
        engine.proposals = lambda:[Proposal('prune',(0,))]
        original = model._xyz
        baseline = engine.render_all()
        with mock.patch.object(engine,'render_all',side_effect=[baseline,RuntimeError('synthetic render error')]):
            with self.assertRaisesRegex(RuntimeError,'synthetic render error'):
                engine.run()
        self.assertIs(model._xyz,original)
        self.assertIs(model.optimizer.param_groups[0]['params'][0],original)

    @torch.no_grad()
    def test_interpolation_bank_matches_expected_view_weights(self):
        self.args.interpolation = True
        model = self.model()
        images = self.images(model,slices=4)
        engine = GaussianStructuralOptimizer(model,images,render_options(self.args),ssim_options(self.args),self.args.interpolate_iter)
        self.assertEqual(engine.real_count,4)
        self.assertAlmostEqual(engine.l1_weights.sum().item(),self.args.lambda_l1+self.args.lambda_interp_l1,places=6)
        self.assertAlmostEqual(engine.ssim_weights.sum().item(),self.args.lambda_ssim+self.args.lambda_interp_ssim,places=6)

    @torch.no_grad()
    def test_real_proposal_cycle_has_all_five_kinds(self):
        model = self.model([[0.,0.,0.],[0.,0.,0.],[.4,.2,0.]],[.3,.4,0.])
        teacher = self.model([[.2,.2,0.]],[.8])
        engine = self.engine(model,self.images(teacher))
        self.args.structural_neighbors = 2
        proposals = engine.proposals()
        self.assertEqual(set(op.kind for op in proposals),set(engine.kinds))
        report = engine.run()
        self.assertLessEqual(report['delta_j'],0.)
        self.assertGreaterEqual(report['after'],1)
        self.assertLessEqual(report['after'],self.args.max_num_pts)


@unittest.skipUnless(torch.cuda.is_available() and package._C is not None, '需要 CUDA 与当前渲染扩展')
class StructuralCudaTests(unittest.TestCase):
    def model(self,points,intensities):
        result = GaussianModel()
        result._xyz = nn.Parameter(torch.tensor(points,device='cuda',dtype=torch.float32))
        result._scaling = nn.Parameter(torch.tensor([[-1.5,-1.,-.7]],device='cuda').repeat(len(points),1))
        result._rotation = nn.Parameter(torch.tensor([[.7,-.2,.3,.4]],device='cuda').repeat(len(points),1))
        result._intensity = nn.Parameter(torch.tensor(intensities,device='cuda')[:,None])
        result.bounds = self.args.bounds
        result.training_setup(self.args)
        return result

    def setUp(self):
        self.args = load_config(ROOT/'config_train.yaml')
        self.args.interpolation = False
        self.args.structural_amplitude_weight = 0.
        self.args.structural_neighbors = 0
        self.args.lambda_anisotropic = 0.

    @unittest.skipUnless(importlib.util.find_spec('simple_knn') is not None, '需要已有 simple-knn')
    def test_main_initialization_keywords_and_single_slice_roi(self):
        result = GaussianModel()
        result.InitializeGaussians(num_pts=8,bounds=self.args.bounds,
            roi_mask=torch.ones(1,2,2,device='cuda'),intensity_init=-.2,
            roi_layout='SHW',roi_jitter_factor=0.,initial_z_scale=1.5,
            initial_distance_floor=1e-7)
        torch.testing.assert_close(result.get_xyz[:,2],torch.zeros(8,device='cuda'))
        self.assertTrue(torch.isfinite(result.get_scaling).all())
        torch.testing.assert_close(result.get_intensity,torch.full((8,1),-.2,device='cuda'))

    @torch.no_grad()
    def test_oblique_signed_local_response_matches_cuda(self):
        model = self.model([[0.,.1,.2],[.2,-.3,0.]],[-.4,.8])
        affine = [[.08,.015,0.,-.5],[.01,.075,0.,-.5],[.025,-.02,1.,.03],[0.,0.,0.,1.]]
        covariance = activated_covariance(model._scaling,model._rotation)
        for mode,sigma in [('point',0.),('gaussian',.3)]:
            cfg = Settings(17,17,affine,response_mode=mode,psf_sigma=sigma,background=-.17)
            patches = response_patches(model.get_xyz,covariance,[cfg],10000)
            local = torch.full((17*17,),cfg.background,device='cuda',dtype=torch.float64)
            for i,patch in enumerate(patches):
                local[patch.indices] += patch.values*model.get_intensity[i,0]
            actual = package.rasterize_gaussians(model.get_xyz,model.get_intensity,cfg,
                         scales=model.get_scaling,rotations=model._rotation)[0]
            torch.testing.assert_close(local.reshape_as(actual),actual.double(),atol=2e-7,rtol=2e-6)

    def test_cuda_commit_then_forward_backward_and_adam(self):
        from Renderer import render
        model = self.model([[.8,.8,0.]],[0.])
        teacher = self.model([[.1,.2,0.]],[.7])
        with torch.no_grad():
            target = torch.stack([render(teacher,[17,17],i,2,3,**render_options(self.args))['render'] for i in range(3)])
            engine = GaussianStructuralOptimizer(model,target,render_options(self.args),ssim_options(self.args),1000)
            point = Primitive(*(getattr(teacher,name)[0].detach().clone() for name in ('_xyz','_scaling','_rotation','_intensity')))
            index = engine.register(point)
            engine.proposals = lambda:[Proposal('birth',(),(index,))]
            rng = torch.cuda.get_rng_state().clone()
            report = engine.run()
            self.assertTrue(torch.equal(rng,torch.cuda.get_rng_state()))
            self.assertEqual(report['accepted']['birth'],1)
            self.assertFalse(report['rolled_back'])
        prediction = render(model,[17,17],1,2,3,**render_options(self.args))['render']
        (prediction-target[1]).square().mean().backward()
        for group in model.optimizer.param_groups:
            gradient = group['params'][0].grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
        model.optimizer.step()
        self.assertTrue(torch.isfinite(model.get_xyz).all())

    @torch.no_grad()
    def test_cuda_proposal_cycle_with_regularizers_and_neighbors(self):
        from Renderer import render
        self.args.structural_neighbors = 2
        self.args.structural_candidates = 3
        self.args.structural_amplitude_weight = 1e-5
        self.args.lambda_anisotropic = .01
        self.args.anisotropic_max_ratio = 1.5
        model = self.model([[0.,0.,0.],[0.,0.,0.],[.7,.7,0.]],[-.2,.2,0.])
        teacher = self.model([[.2,.2,0.]],[.7])
        target = torch.stack([render(teacher,[17,17],i,2,3,**render_options(self.args))['render'] for i in range(3)])
        engine = GaussianStructuralOptimizer(model,target,render_options(self.args),ssim_options(self.args),1000)
        report = engine.run()
        self.assertFalse(report['rolled_back'])
        self.assertGreater(sum(report['accepted'].values()),0)
        self.assertLess(report['delta_j'],0.)
        self.assertTrue(torch.isfinite(model.get_intensity).all())


class MainInterfaceTests(unittest.TestCase):
    def test_default_acdc_and_fixed_volume_contract(self):
        from argparse import Namespace
        args = Namespace(dataset_paths={'acdc':'/tmp/acdc','mnms':'/tmp/mnms'},
                         train_fraction=.7,normalization_epsilon=1e-4)
        self.assertEqual(dataset_options(args)['datasets'],'acdc')
        self.assertEqual(dataset_options(args)['dim'],'3D')
        args.dataset_name='M&Ms'
        self.assertEqual(dataset_options(args)['data_path'],'/tmp/mnms')

    def test_actual_dataloader_batch_can_feed_all_main_entries(self):
        from Dataloader import dataset
        from Motion import prepare_cmr_sequence
        args = load_config(ROOT/'config_train.yaml')
        with tempfile.TemporaryDirectory() as directory:
            info = {'schema':{'array_axes':{'cmr':['T','Z','Y','X'],'labels':['phase','Z','Y','X']}},
                    'case':{'case_id':'sample','original_split':'Testing'},
                    'annotation':{'time_indices':[0,2],'phase_names':['ED','ES'],'has_labels':True,'phase_metadata_valid':True},
                    'source':{'files':{'cmr_4d':{'zooms':[1.,1.,8.,1.]}}},
                    'processing':{'output':{'cmr_shape':[3,2,4,4],'labels_shape':[2,2,4,4]}}}
            np.savez_compressed(Path(directory)/'sample.npz',cmr=np.arange(96).reshape(3,2,4,4),
                                labels=np.zeros((2,2,4,4),dtype=np.uint8),info=json.dumps(info))
            args.dataset_paths['acdc']=directory
            data = next(iter(torch.utils.data.DataLoader(dataset(mode='all',require_labels=True,**dataset_options(args)),batch_size=1)))
            self.assertEqual(data['info']['filename'][0],'sample.npz')
            self.assertEqual(data['cmr'][0,:,args.reference_phase].shape,(2,128,128))
            self.assertEqual(prepare_cmr_sequence(data['cmr'],device='cpu').shape,(3,2,128,128))
            self.assertEqual(data['label'].shape,(1,2,2,128,128))


if __name__=='__main__': unittest.main()
