"""合成 spacing/初始化回归；不读取患者像素、不进行完整训练。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if not all(importlib.util.find_spec(name) for name in ('torch','numpy','yaml','plyfile')):
    raise unittest.SkipTest('需要已有 PyTorch/NumPy/PyYAML/plyfile')
import numpy as np
import torch
from config_utils import load_config, ssim_options
from case_geometry import make_geometry, check_geometry, attach_geometry, save_model_ply, read_model_geometry
from CostFunctions import anisotropic_scale_loss
from Gaussians.GaussianModel_structural import GaussianModel, GaussianStructuralOptimizer, Primitive, Proposal
from Gaussians.initialization import IntensityOperator
from diff_gaussian_slice_rendering import GaussianRasterizationSettings as Settings
import diff_gaussian_slice_rendering as package
from test_structural_optimizer import reference_render


def reference_linear_render(model,img_size,slice_idx,axis,num_slices,**kwargs):
    view = SimpleNamespace(_scaling=model._scaling,_rotation=model._rotation,
        get_xyz=kwargs.get('xyz',model.get_xyz),get_intensity=kwargs.get('colors',model.get_intensity))
    result = reference_render(view,img_size,slice_idx,axis,num_slices,**kwargs)
    # CPU 参考测试隔离线性算子；真实可见性和重采样在 CUDA 路径检查。
    result['visibility_filter'] = torch.ones(len(model.get_xyz),dtype=torch.bool,device=model.get_xyz.device)
    return result


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.args = load_config(ROOT/'config_train.yaml')

    def test_spacing_coordinates_agree_with_renderer_including_single_slice(self):
        for slices in (1,3):
            geometry = make_geometry([1.171875,.75,10.],[slices,128,128],self.args)
            for k in range(slices):
                cfg = Settings.from_axis(128,128,num_slices=slices,slice_idx=k,bounds=geometry['bounds'])
                points = torch.tensor([[0.,0.],[127.,127.],[35.,81.]],dtype=torch.float64)
                actual = points@cfg.slice_to_world[:3,:2].T+cfg.slice_to_world[:3,3]
                expected = torch.stack(((points[:,0]+.5-64)*1.171875,
                    (points[:,1]+.5-64)*.75,torch.full((3,),(k-(slices-1)/2)*10,dtype=torch.float64)),1)/geometry['scale_mm']
                torch.testing.assert_close(actual,expected)

    def test_geometry_rejects_changed_spacing_phase_and_rendering(self):
        geometry = make_geometry([1.,1.,8.],[3,9,13],self.args)
        for field,value in [('spacing_xyz',[1.,1.,10.]),('reference_phase',1)]:
            changed = dict(geometry,**{field:value})
            with self.assertRaisesRegex(ValueError,'几何'):
                check_geometry(geometry,changed)
        self.assertEqual(make_geometry([1.,1.,8.],[3,9,13],self.args,legacy=True)['bounds'],self.args.bounds)

    def test_resume_allows_extension_but_rejects_optimizer_change(self):
        from GetInitialGaussian import check_resume_config
        saved = dict(vars(self.args))
        self.args.iters += 1
        self.args.lambda_volume += .01
        check_resume_config(saved,self.args)
        self.args.intensity_lr *= 2
        with self.assertRaisesRegex(ValueError,'优化条件'):check_resume_config(saved,self.args)

    def test_dense_flow_uses_saved_pixel_centers_and_pixel_units(self):
        from Motion import export_dense_flow
        class Translation(torch.nn.Module):
            def __init__(self):
                super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()))
                self.uses_control_points=False;self.motion_dimension='2D'
                self.control_deformation_mode='kernel';self.control_indices=torch.zeros(1,dtype=torch.long)
                self.through_plane_max_displacement=.1;self.queries=[]
            def deform_query_points(self,points,target_time_t,**kwargs):
                self.queries.append(points.clone())
                return points+points.new_tensor([dx,dy,0.])*target_time_t
        geometry=make_geometry([1.2,.8,8.],[1,3,5],self.args)
        dx=2*geometry['bounds'][0]/5;dy=2*geometry['bounds'][1]/3
        net=Translation()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'flow.npz'
            export_dense_flow(net,2,1,3,5,1,100,str(path),geometry=geometry)
            with np.load(path,allow_pickle=False) as result:
                np.testing.assert_allclose(result['flow_pixel'][1],1.,rtol=1e-6)
                check_geometry(json.loads(result['geometry_json'].item()),geometry)
            cfg=Settings.from_axis(3,5,num_slices=1,bounds=geometry['bounds'])
            y,x=torch.meshgrid(torch.arange(3),torch.arange(5),indexing='ij')
            pixels=torch.stack((x,y),-1).reshape(-1,2).double()
            expected=pixels@cfg.slice_to_world[:3,:2].T+cfg.slice_to_world[:3,3]
            torch.testing.assert_close(net.queries[0].double(),expected,atol=1e-7,rtol=1e-6)


class InitializationTests(unittest.TestCase):
    def setUp(self):
        self.args = load_config(ROOT/'config_train.yaml')
        self.args.num_pts = 48
        self.args.init_fit_steps = 8
        self.args.interpolation = False
        patch = mock.patch('Gaussians.initialization.render',side_effect=reference_linear_render)
        patch.start();self.addCleanup(patch.stop)

    def initialize(self, device='cpu', case='synthetic'):
        y,x = torch.meshgrid(torch.linspace(-1,1,9,device=device),torch.linspace(-1,1,13,device=device),indexing='ij')
        images = torch.stack([.1+.5*torch.exp(-3*((x-.1*k)**2+y*y)) for k in range(3)])
        geometry = make_geometry([1.,1.5,8.],images.shape,self.args)
        model = GaussianModel();attach_geometry(model,geometry)
        report = model.initialize_from_images(images,geometry,self.args,geometry['rendering'],case)
        return model,images,report

    def test_distribution_covariance_reproducibility_and_rng_isolation(self):
        rng = torch.get_rng_state().clone()
        first,images,report = self.initialize()
        second,_,_ = self.initialize()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        torch.testing.assert_close(first.get_xyz,second.get_xyz,rtol=0,atol=0)
        self.assertEqual(first.get_xyz.shape,(48,3))
        self.assertTrue((first.get_xyz.abs() <= first.get_xyz.new_tensor(first.bounds)).all())
        self.assertTrue((first.get_scaling.max(1).values/first.get_scaling.min(1).values <= self.args.init_max_axis_ratio+1e-5).all())
        torch.testing.assert_close(first.get_rotation,torch.tensor([[1.,0.,0.,0.]]).repeat(48,1))
        self.assertLess(report['mse_after'],report['mse_before'])
        self.assertTrue(torch.isfinite(first.get_intensity).all())
        self.assertEqual(first.geometry_reference_count,48)

    def test_linear_operator_adjoint(self):
        model,images,_ = self.initialize()
        operator = IntensityOperator(model,images,model.geometry['rendering'])
        vector = torch.linspace(-.3,.5,48,dtype=torch.float64)
        pixels = torch.linspace(-.2,.4,images.numel(),dtype=torch.float64).reshape_as(images)
        torch.testing.assert_close((operator.forward(vector)*pixels).mean(),
                                   vector@operator.adjoint(pixels),rtol=3e-6,atol=1e-8)

    def test_constant_zero_images_and_tiny_point_count(self):
        self.args.num_pts = 1
        images = torch.zeros(1,9,13)
        geometry = make_geometry([1.,1.5,8.],images.shape,self.args)
        model = GaussianModel();attach_geometry(model,geometry)
        model.initialize_from_images(images,geometry,self.args,geometry['rendering'],'constant')
        torch.testing.assert_close(model.get_intensity,torch.zeros(1,1))
        self.assertTrue(torch.isfinite(model.get_scaling).all())

    def test_ply_geometry_roundtrip_hash_and_legacy_absence(self):
        model,_,_ = self.initialize()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.ply'
            save_model_ply(model,path)
            record = read_model_geometry(path)
            check_geometry(record['geometry'],model.geometry)
            self.assertEqual(record['geometry_reference_count'],48)
            other = GaussianModel();other.load_ply(str(path),device='cpu')
            torch.testing.assert_close(other.get_xyz,model.get_xyz)
            with path.open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'不匹配'):read_model_geometry(path)
            self.assertIsNone(read_model_geometry(Path(directory)/'legacy.ply'))

    def test_zero_contribution_birth_cannot_dilute_geometry_penalty(self):
        from test_structural_optimizer import StructuralTests
        fixture = StructuralTests();fixture.setUp()
        try:
            fixture.args.lambda_anisotropic=.1
            model = fixture.model()
            with torch.no_grad():model._scaling[0]=torch.tensor([.3,.3,3.6]).log()
            engine = fixture.engine(model)
            parent = engine.primitive(1)
            index = engine.register(Primitive(torch.tensor([.7,.7,0.]),parent.scaling,parent.rotation,torch.zeros(1)))
            with torch.no_grad():
                self.assertFalse(engine.evaluate(Proposal('birth',(),(index,))))
            before = anisotropic_scale_loss(model.get_scaling,reference_count=2)
            after = anisotropic_scale_loss(torch.cat((model.get_scaling,torch.ones(1,3))),reference_count=2)
            torch.testing.assert_close(before,after)
        finally:fixture.doCleanups()


@unittest.skipUnless(torch.cuda.is_available() and package._C is not None,'需要 CUDA 与已有渲染扩展')
class CudaInitializationTests(unittest.TestCase):
    initialize = InitializationTests.initialize
    def setUp(self):
        self.args = load_config(ROOT/'config_train.yaml')
        self.args.num_pts = 24
        self.args.init_fit_steps = 8

    def test_cuda_init_adjoint_then_backward(self):
        self.args.num_pts = 48
        state = torch.cuda.get_rng_state().clone()
        model,images,report = self.initialize('cuda')
        self.assertTrue(torch.equal(state,torch.cuda.get_rng_state()))
        self.assertLess(report['mse_after'],report['mse_before'])
        operator = IntensityOperator(model,images,model.geometry['rendering'])
        vector = torch.linspace(-.3,.5,48,device='cuda',dtype=torch.float64)
        pixels = torch.linspace(-.2,.4,images.numel(),device='cuda',dtype=torch.float64).reshape_as(images)
        torch.testing.assert_close((operator.forward(vector)*pixels).mean(),vector@operator.adjoint(pixels),rtol=5e-5,atol=1e-7)
        model.training_setup(self.args)
        from Renderer import render
        prediction = render(model,[9,13],1,2,3,**model.geometry['rendering'])['render']
        (prediction-images[1]).square().mean().backward()
        for group in model.optimizer.param_groups:
            self.assertTrue(torch.isfinite(group['params'][0].grad).all())
        model.optimizer.step()

    def test_main_training_resume_repair_and_evaluation(self):
        from GetInitialGaussian import training
        from evaluate import evaluate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);data_root=root/'data';data_root.mkdir()
            y,x = np.mgrid[-1:1:12j,-1:1:16j]
            image = np.exp(-3*(x*x+y*y)).astype(np.float32)
            cmr = np.stack([np.stack([image,image*.8])*(1+.1*t) for t in range(3)])
            info = dict(schema={'array_axes':{'cmr':['T','Z','Y','X'],'labels':['phase','Z','Y','X']}},
                case={'case_id':'synthetic','original_split':'Testing'},
                annotation={'time_indices':[0,2],'phase_names':['ED','ES'],'has_labels':True,'phase_metadata_valid':True},
                source={'files':{'cmr_4d':{'zooms':[1.5,1.2,8.,1.]}}},
                processing={'output':{'cmr_shape':list(cmr.shape),'labels_shape':[2,2,12,16]}})
            np.savez_compressed(data_root/'synthetic.npz',cmr=cmr,labels=np.zeros((2,2,12,16),dtype=np.uint8),info=json.dumps(info))
            args=self.args
            args.dataset_paths['acdc']=str(data_root);args.dataset_mode='all'
            args.output_path=str(root/'training');args.iters=2;args.init_fit_steps=3
            args.checkpoint_interval=2;args.save_iterations=[1];args.reconstruct_interval=0
            args.densify_from_iter=1;args.densify_until_iter=3;args.densification_interval=1
            args.structural_candidates=2;args.structural_max_accept=1;args.structural_neighbors=1
            with mock.patch('GetInitialGaussian.update_loss_plot'):
                training(args)
                args.iters=3
                training(args)
                ply=Path(args.output_path)/'synthetic'/'point_cloud_final.ply'
                Path(str(ply)+'.geometry.json').unlink()
                # 模拟最终 PLY 记录尚未写完；完整 checkpoint 可重建产物。
                training(args)
            record=read_model_geometry(ply)
            self.assertEqual(record['iteration'],3)
            self.assertEqual(record['geometry_reference_count'],24)
            state=torch.load(ply.parent/'training_state.pth',map_location='cpu',weights_only=False)
            self.assertEqual(state['iteration'],3)
            check_geometry(state['geometry'],record['geometry'])
            test=load_config(ROOT/'config_test.yaml')
            test.dataset_paths['acdc']=str(data_root);test.dataset_mode='all'
            test.output_path=str(root/'evaluation');test.save_arrays=True;test.save_images=False
            test.stage1_ply_path=str(root/'training'/'{case}'/'point_cloud_final.ply')
            rows=evaluate(test)
            self.assertEqual(len(rows),2)
            self.assertTrue(all(np.isfinite(row['l1']) for row in rows))
            test.reference_phase=1
            with self.assertRaisesRegex(ValueError,'几何'):evaluate(test)


if __name__=='__main__':unittest.main()
