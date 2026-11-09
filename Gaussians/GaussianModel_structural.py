"""Stage1 结构优化模型：保留基元/训练接口，统一 Birth/Split/Prune/Merge/Swap。"""
from collections import OrderedDict
from dataclasses import dataclass
from itertools import zip_longest
import math
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .GaussianModel_noop import GaussianModel as BaseGaussianModel
from .utils import matrix_to_quaternion
from CostFunctions import create_window, _ssim_map
from Renderer import render
from diff_gaussian_slice_rendering import GaussianRasterizationSettings, normalize_rotations
from diff_gaussian_slice_rendering.local_response import activated_covariance, response_patches


@dataclass
class Primitive:
    xyz: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    intensity: torch.Tensor


@dataclass
class Proposal:
    kind: str
    removed: tuple = ()
    added: tuple = ()


class GaussianModel(BaseGaussianModel):
    """只替换结构操作，PLY、属性访问和学习率沿用旧模型。"""

    @property
    def get_rotation(self):
        # 与 Renderer 共用完整梯度链，绕开大点数 F.normalize 的二维行归约反向。
        return normalize_rotations(self._rotation)

    def training_setup(self, args):
        super().training_setup(args)
        self.structural_args = args
        if not hasattr(self, 'geometry_reference_count'):
            self.geometry_reference_count = len(self.get_xyz)
        for group in self.optimizer.param_groups:
            group['eps'] = args.adam_eps
            group['betas'] = tuple(args.adam_betas)

    def initialize_from_images(self, images, geometry, args, rendering, case_name):
        from .initialization import initialize_from_images
        return initialize_from_images(self,images,geometry,args,rendering,case_name)

    def InitializeGaussians(self, num_pts=50000, bounds=(1.,1.,1.), roi_mask=None,
                            intensity_init=0.6410473958869391, roi_layout='SHW',
                            roi_jitter_factor=.5, initial_z_scale=2., initial_distance_floor=1e-7):
        # 无 ROI 且使用原默认值时直接复用旧初始化，保持主训练的随机数顺序。
        if roi_mask is None and initial_z_scale == 2. and initial_distance_floor == 1e-7:
            return super().InitializeGaussians(num_pts, bounds, None, intensity_init)
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

    def _invalid_point_mask(self):
        invalid = torch.zeros(len(self._xyz),device=self._xyz.device,dtype=torch.bool)
        for value in (self._xyz,self._intensity,self._scaling,self._rotation):
            invalid |= ~torch.isfinite(value).flatten(1).all(-1)
        scales = self.get_scaling
        invalid |= ~torch.isfinite(scales).all(-1) | (scales <= 0).any(-1)
        components = tuple(component.double() for component in self._rotation.unbind(-1))
        invalid |= torch.sqrt(sum(value.square() for value in components)) < 1e-12
        return invalid

    @torch.no_grad()
    def structural_step(self, images, iteration, render_kwargs, similarity_kwargs):
        optimizer = GaussianStructuralOptimizer(self, images, render_kwargs, similarity_kwargs, iteration)
        report = optimizer.run()
        print(f'[Structure {iteration}] N={report["before"]}->{report["after"]} '
              f'candidates={report["candidates"]} accepted={report["accepted"]} '
              f'dJ={report["delta_j"]:.6g} rollback={report["rolled_back"]} '
              f'budget_rejections={report["budget_rejections"]} '
              f'time={report["seconds"]:.3f}s')
        return report

    def _commit_structure(self, engine):
        """一次 gather/concat 每组参数；旧引用保留至正式渲染校验结束。"""
        device = self.get_xyz.device
        mask = torch.ones(engine.original_count, dtype=torch.bool, device=device)
        if engine.removed:
            mask[list(engine.removed)] = False
        survivors = torch.nonzero(mask, as_tuple=False).flatten()
        added = sorted(engine.active_new)
        snapshot = ([(group,group['params'][0],self.optimizer.state.get(group['params'][0]))
                     for group in self.optimizer.param_groups],
                    {key:getattr(self,key) for key in ('xyz_gradient_accum','denom','max_radii2D')})
        positions = torch.full((engine.original_count,),-1,dtype=torch.long,device=device)
        positions[survivors] = torch.arange(len(survivors),device=device)
        names = {'xyz':'_xyz','intensity':'_intensity','scaling':'_scaling','rotation':'_rotation'}
        prepared = []
        for group, old, old_state in snapshot[0]:
            name = group['name']
            tail = [getattr(engine.new[i], name) for i in added]
            tail = torch.stack(tail) if tail else old.new_empty((0,*old.shape[1:]))
            data = torch.cat((old.detach()[survivors],tail),0)
            reset_rows = []
            if name == 'intensity':
                changed = [i for i in engine.coefficients if i < engine.original_count and i not in engine.removed]
                if changed:
                    rows = positions[changed]
                    data[rows] = torch.stack([engine.coefficients[i].reshape(1) for i in changed])
                    reset_rows.append(rows)
                if added:
                    data[len(survivors):] = torch.stack([engine.coefficient(i).reshape(1) for i in added])
            parameter = nn.Parameter(data)
            state = None
            if old_state is not None:
                state = {}
                for key,value in old_state.items():
                    if isinstance(value,torch.Tensor) and value.shape == old.shape:
                        state[key] = torch.cat((value[survivors],torch.zeros_like(tail)),0)
                        for rows in reset_rows:
                            state[key][rows] = 0
                    else:
                        # Adam step 是组级时钟；新行清零动量但沿用原组 step。
                        state[key] = value
            prepared.append((group,old,parameter,state))
        count = len(survivors)+len(added)
        statistics = (self._xyz.new_zeros((count,1)),self._xyz.new_zeros((count,1)),self._xyz.new_zeros(count))
        for group,old,parameter,state in prepared:
            group['params'][0] = parameter
            setattr(self,names[group['name']],parameter)
            self.optimizer.state.pop(old,None)
            if state is not None:
                self.optimizer.state[parameter] = state
        self.xyz_gradient_accum,self.denom,self.max_radii2D = statistics
        return snapshot

    def _rollback_structure(self, snapshot):
        names = {'xyz':'_xyz','intensity':'_intensity','scaling':'_scaling','rotation':'_rotation'}
        for group,old,state in snapshot[0]:
            self.optimizer.state.pop(group['params'][0],None)
            group['params'][0] = old
            setattr(self,names[group['name']],old)
            if state is not None:
                self.optimizer.state[old] = state
        for key,value in snapshot[1].items():
            setattr(self,key,value)


class GaussianStructuralOptimizer:
    """一个结构周期内的虚拟模型；每个提案只计算涉及基元的局部图像增量。"""
    kinds = ('birth','split','prune','merge','swap')

    def __init__(self, model, images, render_kwargs, similarity_kwargs, iteration):
        self.model, self.args = model, model.structural_args
        self.original_count = len(model.get_xyz)
        self.geometry_reference_count = model.geometry_reference_count
        self.render_kwargs, self.similarity = render_kwargs, similarity_kwargs
        self.device = model.get_xyz.device
        self.height, self.width = images.shape[-2:]
        self.area = self.height*self.width
        self.real_count = len(images)
        self.targets_real = images.double()
        self.views = self._views(images,iteration)
        self.settings = [view[0] for view in self.views]
        self.targets = torch.stack([view[1] for view in self.views]).double()
        self.l1_weights = images.new_tensor([view[2] for view in self.views]).double()
        self.ssim_weights = images.new_tensor([view[3] for view in self.views]).double()
        weights = self.l1_weights+self.ssim_weights
        self.weights = weights/weights.sum()
        self.window_size = similarity_kwargs['window_size']
        self.window = create_window(self.window_size,1,similarity_kwargs['sigma']).to(self.device,torch.float64)
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.new, self.coefficients = {}, {}
        self.removed, self.active_new = set(),set()
        self.gsum = self._geometry_penalties(model._scaling).double().sum()
        self.start_gsum = self.gsum.clone()
        self.sparsity_delta = images.new_zeros((),dtype=torch.float64)
        self.candidates = dict.fromkeys(self.kinds,0)
        self.accepted = dict.fromkeys(self.kinds,0)
        self.budget_rejections = 0

    def _views(self, images, iteration):
        args = self.args
        count = len(images)
        weights = [[args.lambda_l1/count,args.lambda_ssim/count] for _ in range(count)]
        descriptors = [(float(i),images[i]) for i in range(count)]
        # 正常训练的随机数流不变；重叠插层阶段使用固定节点近似其期望目标。
        if args.interpolation and iteration >= args.interpolate_iter and count > 1:
            probability = args.interpolation_probability
            for value in weights:
                value[0] += (1-probability)*args.lambda_interp_l1/count
                value[1] += (1-probability)*args.lambda_interp_ssim/count
            nodes = args.structural_interp_nodes
            starts = max(1,count-2)
            for left in range(starts):
                intervals = range(left,min(left+2,count-1))
                for index in intervals:
                    for node in nodes:
                        alpha = args.interpolation_alpha_min + node*(args.interpolation_alpha_max-args.interpolation_alpha_min)
                        descriptors.append((index+alpha,(1-alpha)*images[index]+alpha*images[index+1]))
                        factor = probability/(starts*len(intervals)*len(nodes))
                        weights.append([factor*args.lambda_interp_l1,factor*args.lambda_interp_ssim])
        result = []
        for (index,target),(l1,similarity) in zip(descriptors,weights):
            cfg = GaussianRasterizationSettings.from_axis(
                self.height,self.width,axis=2,slice_idx=index,num_slices=count,
                bounds=self.render_kwargs['bounds'],**self.render_kwargs['raster_options'])
            result.append((cfg,target,l1,similarity,index))
        return result

    @property
    def count(self):
        return self.original_count-len(self.removed)+len(self.active_new)

    def active(self,index):
        return (index < self.original_count and index not in self.removed) or index in self.active_new

    def primitive(self,index):
        if index >= self.original_count:
            return self.new[index]
        return Primitive(self.model._xyz[index].detach(),self.model._scaling[index].detach(),
                         self.model._rotation[index].detach(),self.model._intensity[index].detach())

    def coefficient(self,index):
        return self.coefficients.get(index,self.primitive(index).intensity).reshape(())

    def register(self,primitive):
        index = self.original_count+len(self.new)
        self.new[index] = primitive
        return index

    def _geometry_penalties(self,scaling):
        scales = scaling.exp()
        ratio = scales.max(-1).values / scales.min(-1).values.clamp_min(self.args.anisotropic_epsilon)
        return F.relu(ratio/self.args.anisotropic_max_ratio-1).square()

    def covariance(self,index):
        point = self.primitive(index)
        return activated_covariance(point.scaling[None],point.rotation[None])[0]

    def response(self,index):
        if index in self.cache:
            value = self.cache.pop(index)
            self.cache[index] = value
            return value
        point = self.primitive(index)
        value = response_patches(point.xyz[None],self.covariance(index)[None],
                                 self.settings,self.args.structural_patch_budget)[0]
        if value is not None:
            limit = self.args.structural_cache_mb*1024**2
            while self.cache and self.cache_bytes+value.nbytes > limit:
                _,previous = self.cache.popitem(last=False)
                self.cache_bytes -= previous.nbytes
            if value.nbytes <= limit:
                self.cache[index] = value
                self.cache_bytes += value.nbytes
        return value

    def matrix(self,identifiers):
        patches = [self.response(index) for index in identifiers]
        if any(patch is None for patch in patches):
            return None
        indices = torch.unique(torch.cat([patch.indices for patch in patches]))
        if len(indices) > self.args.structural_patch_budget:
            return None
        matrix = torch.zeros(len(indices),len(identifiers),device=self.device,dtype=torch.float64)
        for column,patch in enumerate(patches):
            matrix[torch.searchsorted(indices,patch.indices),column] = patch.values
        return indices,matrix

    def _pixel_weights(self,indices):
        return self.weights[indices//self.area]/self.area

    def render_all(self):
        return torch.stack([render(self.model,[self.height,self.width],view[4],2,
                                   self.real_count,**self.render_kwargs)['render']
                            for view in self.views]).double()

    def _map(self,prediction,target):
        return _ssim_map(prediction[None,None],target[None,None],self.window,self.window_size,1,
                         self.similarity['k1'],self.similarity['k2'],self.similarity['data_range'])[0,0]

    def image_loss(self,prediction):
        l1 = (prediction-self.targets).abs().mean(dim=(-1,-2))
        value = (l1*self.l1_weights).sum()
        for v in range(len(self.views)):
            if self.views[v][3] != 0:
                value += self.ssim_weights[v]*(1-self._map(prediction[v],self.targets[v]).mean())
        return value

    def image_delta(self,indices,delta):
        old = self.prediction.flatten()[indices]
        target = self.targets.flatten()[indices]
        change = ((old+delta-target).abs()-(old-target).abs())*self.l1_weights[indices//self.area]/self.area
        value = change.sum()
        if not len(indices):
            return value
        # 一次回传所有 view 的边界；每个 crop 读取 2r halo，只归约受影响窗口中心。
        view = indices//self.area
        rows,columns = (indices%self.area)//self.width,indices%self.width
        bounds = torch.empty(len(self.views),4,device=self.device,dtype=torch.long)
        bounds[:,0] = self.height; bounds[:,1] = self.width; bounds[:,2:] = -1
        bounds[:,0].scatter_reduce_(0,view,rows,reduce='amin',include_self=True)
        bounds[:,1].scatter_reduce_(0,view,columns,reduce='amin',include_self=True)
        bounds[:,2].scatter_reduce_(0,view,rows,reduce='amax',include_self=True)
        bounds[:,3].scatter_reduce_(0,view,columns,reduce='amax',include_self=True)
        radius = self.window_size//2
        for v,(y0,x0,ymax,xmax) in enumerate(bounds.cpu().tolist()):
            if ymax < 0 or self.views[v][3] == 0:
                continue
            y1,x1 = ymax+1,xmax+1
            cy0,cx0,cy1,cx1 = max(0,y0-radius),max(0,x0-radius),min(self.height,y1+radius),min(self.width,x1+radius)
            iy0,ix0,iy1,ix1 = max(0,cy0-radius),max(0,cx0-radius),min(self.height,cy1+radius),min(self.width,cx1+radius)
            before = self.prediction[v,iy0:iy1,ix0:ix1]
            after = before.clone()
            selected = view == v
            after[rows[selected]-iy0,columns[selected]-ix0] += delta[selected]
            gt = self.targets[v,iy0:iy1,ix0:ix1]
            difference = self._map(before,gt)-self._map(after,gt)
            value += self.ssim_weights[v]*difference[cy0-iy0:cy1-iy0,cx0-ix0:cx1-ix0].sum()/self.area
        return value

    def refit(self,indices,basis,target,initial):
        weights = self._pixel_weights(indices)
        norm = (basis.square().T@weights).sqrt()
        active = norm > 0
        solution = torch.zeros(basis.shape[1],device=self.device,dtype=torch.float64)
        if not bool(active.any()):
            return solution,norm
        design = basis[:,active]/norm[active]
        weighted = design*weights.sqrt()[:,None]
        rhs = target*weights.sqrt()
        # 小列数 SVD 不显式求逆，也不假定高度相关的响应列满秩。
        u,s,vh = torch.linalg.svd(weighted,full_matrices=False)
        valid = s > self.args.structural_rank_rtol*s[0]
        beta = vh.T@((u.T@rhs)/torch.where(valid,s,1)*valid)
        if self.args.structural_amplitude_weight > 0:
            gram,correlation = weighted.T@weighted,weighted.T@rhs
            # 归一化列后 L1 项为 lambda*sum(|beta|)，避免不同支撑尺度影响求解条件。
            beta = initial[active].double()*norm[active]
            for _ in range(self.args.structural_refit_steps):
                for j in range(len(beta)):
                    residual = correlation[j]-gram[j]@beta+gram[j,j]*beta[j]
                    beta[j] = residual.sign()*F.relu(residual.abs()-self.args.structural_amplitude_weight)/gram[j,j]
        solution[active] = beta/norm[active]
        return solution,norm

    def nearest(self,xyz,excluded=(),count=None):
        count = self.args.structural_neighbors if count is None else count
        if count == 0:
            return []
        distance = (self.model.get_xyz.detach()-xyz).square().sum(-1)
        blocked = self.removed|set(excluded)
        old_blocked = [index for index in blocked if index < self.original_count]
        if old_blocked:
            distance[old_blocked] = torch.inf
        # 查询数量受候选预算限制，是 K*N 搜索，不构造全局 N*N 矩阵。
        k = min(count,self.original_count-len(set(old_blocked)))
        old = torch.topk(distance,k,largest=False).indices.cpu().tolist() if k else []
        alternatives = [i for i in self.active_new if i not in blocked]
        identifiers = old+alternatives
        if not identifiers:
            return []
        positions = torch.stack([self.primitive(i).xyz for i in identifiers])
        selected = torch.topk((positions-xyz).square().sum(-1),min(count,len(identifiers)),largest=False).indices.cpu().tolist()
        return [identifiers[i] for i in selected]

    def propose_covariance(self,xyz,covariance,intensity):
        eigenvalues,rotation = torch.linalg.eigh(covariance)
        if not bool((eigenvalues > 0).all()):
            return None
        logs = .5*eigenvalues.log()
        if not bool(((logs >= self.args.log_scale_min)&(logs <= self.args.log_scale_max)).all()):
            return None
        if torch.linalg.det(rotation) < 0:
            rotation[:,0] *= -1
        quaternion = matrix_to_quaternion(rotation[None])[0].to(self.model._rotation)
        return Primitive(xyz.to(self.model._xyz),logs.to(self.model._scaling),quaternion,
                         intensity.to(self.model._intensity).reshape(1))

    def split(self,index):
        covariance = self.covariance(index)
        patch = self.response(index)
        if patch is None:
            return None
        _,axes = torch.linalg.eigh(covariance)
        direction = axes[:,-1]
        if len(patch.indices):
            v = patch.indices//self.area
            pixels = torch.stack(((patch.indices%self.area)%self.width,
                                  (patch.indices%self.area)//self.width),-1).double()
            transforms = torch.stack([cfg.slice_to_world for cfg in self.settings]).to(self.device)
            points = (transforms[v,:3,:2]@pixels[:,:,None]).squeeze(-1)+transforms[v,:3,3]
            weights = (self.targets.flatten()-self.prediction.flatten())[patch.indices].abs()*patch.values*self._pixel_weights(patch.indices)
            if bool(weights.sum() > 0):
                center = (weights[:,None]*points).sum(0)/weights.sum()
                offset = points-center
                _,axes = torch.linalg.eigh((offset.T*weights)@offset/weights.sum())
                direction = axes[:,-1]
        delta = self.args.structural_split_fraction*direction/torch.sqrt(direction@torch.linalg.solve(covariance,direction))
        child_covariance = covariance-delta[:,None]*delta[None,:]
        base = covariance+self.settings[0].covariance_jitter*torch.eye(3,device=self.device,dtype=torch.float64)
        amplitude = self.coefficient(index)/(2*torch.sqrt(1-delta@torch.linalg.solve(base,delta)))
        parent = self.primitive(index)
        children = [self.propose_covariance(parent.xyz.double()+sign*delta,child_covariance,amplitude) for sign in (-1,1)]
        if any(child is None for child in children):
            return None
        return Proposal('split',(index,),tuple(self.register(child) for child in children))

    def merge(self,first,second):
        a,b = self.coefficient(first),self.coefficient(second)
        covariance = torch.stack((self.covariance(first),self.covariance(second)))
        identity = torch.eye(3,device=self.device,dtype=torch.float64)
        mass = torch.stack((a,b)).double()*torch.linalg.det(covariance+self.settings[0].covariance_jitter*identity).sqrt()
        if not bool((mass.sum().abs()>0)&(a*b>0)):
            return None
        weights = mass/mass.sum()
        centers = torch.stack((self.primitive(first).xyz,self.primitive(second).xyz)).double()
        center = weights@centers
        offset = centers-center
        merged = (weights[:,None,None]*(covariance+offset[:,:,None]*offset[:,None,:])).sum(0)
        amplitude = mass.sum()/torch.linalg.det(merged+self.settings[0].covariance_jitter*identity).sqrt()
        point = self.propose_covariance(center,merged,amplitude)
        return Proposal('merge',(first,second),(self.register(point),)) if point is not None else None

    def birth_proposals(self):
        count = self.args.structural_candidates
        residual = (self.targets_real-self.prediction[:len(self.targets_real)]).abs()
        size = self.args.structural_peak_window
        # 每个图像块至多一个残差峰，避免所有预算集中于相邻像素。
        peaks,indices = F.max_pool2d(residual[:,None],size,stride=size,ceil_mode=True,return_indices=True)
        k = min(count,peaks.numel())
        best = peaks.flatten().topk(k).indices
        cells_per_view = peaks[0].numel()
        view = best//cells_per_view
        pixels = indices.flatten()[best]
        descriptors = torch.stack((view,pixels//self.width,pixels%self.width),-1).cpu().tolist()
        proposals = []
        for v,row,column in descriptors:
            cfg = self.settings[v]
            transform = cfg.slice_to_world.to(self.device)
            xyz = transform[:3,3]+transform[:3,:2]@transform.new_tensor([column,row])
            nearest = self.nearest(xyz.float(),count=1)[0]
            parent = self.primitive(nearest)
            # 均值落在实际观测平面；法向支撑由邻域基元约束，不从单层残差臆测深度。
            point = Primitive(xyz.to(parent.xyz),parent.scaling.clone(),parent.rotation.clone(),parent.intensity.new_zeros(1))
            proposals.append(Proposal('birth',(),(self.register(point),)))
        return proposals

    def proposals(self):
        budget = min(self.args.structural_candidates,self.original_count)
        amplitudes = self.model.get_intensity.detach().flatten()
        sizes = self.model.get_scaling.detach()
        low = torch.topk(amplitudes.abs()*sizes.prod(-1),budget,largest=False).indices.cpu().tolist()
        gradients = (self.model.xyz_gradient_accum/self.model.denom.clamp_min(1)).flatten()
        wide = torch.topk((gradients+torch.finfo(gradients.dtype).eps)*sizes.max(-1).values,budget).indices.cpu().tolist()
        groups = {kind:[] for kind in self.kinds}
        groups['prune'] = [Proposal('prune',(i,)) for i in low]
        groups['birth'] = self.birth_proposals()
        groups['split'] = [proposal for i in wide if (proposal:=self.split(i)) is not None]
        pairs = set()
        for i in dict.fromkeys(low+wide):
            for j in self.nearest(self.primitive(i).xyz,excluded=(i,)):
                pair = tuple(sorted((i,j)))
                if pair in pairs:
                    continue
                pairs.add(pair)
                data = self.matrix(pair)
                if data is None:
                    continue
                indices,basis = data
                gram = basis.T@(basis*self._pixel_weights(indices)[:,None])
                norm = (gram[0,0]*gram[1,1]).sqrt()
                if not bool(norm > 0) or not bool(gram[0,1] >= self.args.structural_merge_correlation*norm):
                    continue
                if bool(self.coefficient(i)*self.coefficient(j)>0):
                    proposal = self.merge(i,j)
                    if proposal is not None and len(groups['merge']) < budget:
                        groups['merge'].append(proposal)
                elif len(groups['prune']) < 2*budget:
                    groups['prune'].append(Proposal('prune',pair))
        for birth,old in zip(groups['birth'],low):
            point = self.primitive(birth.added[0])
            groups['swap'].append(Proposal('swap',(old,),(self.register(point),)))
        # 每轮均覆盖五种提案，避免 Prune 数量多时长期挤占 Birth/Split 预算。
        return [op for row in zip_longest(*(groups[kind] for kind in ('prune','merge','birth','split','swap')))
                for op in row if op is not None]

    def evaluate(self,proposal):
        kind,removed,added = proposal.kind,proposal.removed,proposal.added
        self.candidates[kind] += 1
        if any(not self.active(i) for i in removed) or any(self.active(i) for i in added):
            return False
        count = self.count-len(removed)+len(added)
        if not self.args.structural_min_points <= count <= self.args.max_num_pts:
            return False
        if kind == 'merge' and not bool(self.coefficient(removed[0])*self.coefficient(removed[1])>0):
            return False
        anchor = torch.stack([self.primitive(i).xyz for i in added or removed]).mean(0)
        fitted = tuple(self.nearest(anchor,excluded=removed))
        identifiers = removed+added+fitted
        data = self.matrix(identifiers)
        if data is None:
            self.budget_rejections += 1
            return False
        indices,columns = data
        removed_count = len(removed)
        old_coefficients = (torch.stack([self.coefficient(i) for i in removed+fitted]).double()
                            if removed+fitted else columns.new_empty(0))
        old_basis = torch.cat((columns[:,:removed_count],columns[:,removed_count+len(added):]),1)
        old_contribution = old_basis@old_coefficients
        basis = columns[:,removed_count:]
        rhs = (self.targets.flatten()-self.prediction.flatten())[indices]+old_contribution
        initial = torch.stack([self.coefficient(i) for i in added+fitted]) if added+fitted else columns.new_empty(0)
        if basis.shape[1]:
            coefficient,norm = self.refit(indices,basis,rhs,initial)
            coefficient = coefficient.to(self.model._intensity).double()
        else:
            coefficient = columns.new_empty(0)
            norm = columns.new_empty(0)
        if not bool(torch.isfinite(coefficient).all()):
            return False
        delta = basis@coefficient-old_contribution
        old_norm = (old_basis.square().T@self._pixel_weights(indices)).sqrt()
        amplitude_delta = (coefficient.abs()*norm).sum()-(old_coefficients.abs()*old_norm).sum()
        removed_geometry = sum((self._geometry_penalties(self.primitive(i).scaling[None])[0].double() for i in removed),columns.new_zeros(()))
        added_geometry = sum((self._geometry_penalties(self.primitive(i).scaling[None])[0].double() for i in added),columns.new_zeros(()))
        gsum = self.gsum-removed_geometry+added_geometry
        objective_delta = (self.image_delta(indices,delta)
                           + self.args.structural_count_weight*(count-self.count)
                           + self.args.structural_amplitude_weight*amplitude_delta
                           + self.args.lambda_anisotropic*(gsum-self.gsum)/self.geometry_reference_count)
        if not bool(objective_delta < -self.args.structural_min_gain):
            return False
        self.prediction.flatten()[indices] += delta
        self.removed.update(i for i in removed if i < self.original_count)
        self.active_new.difference_update(removed)
        self.active_new.update(added)
        self.coefficients.update({i:value.to(self.model._intensity) for i,value in zip(added+fitted,coefficient)})
        self.sparsity_delta += amplitude_delta
        self.gsum = gsum
        self.accepted[kind] += 1
        return True

    @torch.no_grad()
    def run(self):
        start = time.perf_counter()
        baseline = self.render_all()
        self.prediction = baseline.clone()
        start_loss = self.image_loss(baseline)
        before = self.original_count
        for proposal in self.proposals():
            self.evaluate(proposal)
            if sum(self.accepted.values()) >= self.args.structural_max_accept:
                break
        delta_j = baseline.new_zeros(())
        rolled_back = False
        if sum(self.accepted.values()):
            snapshot = self.model._commit_structure(self)
            try:
                actual = self.render_all()
                regularization = (self.args.structural_count_weight*(self.count-before)
                                  + self.args.structural_amplitude_weight*self.sparsity_delta
                                  + self.args.lambda_anisotropic*(self.gsum-self.start_gsum)/self.geometry_reference_count)
                delta_j = self.image_loss(actual)-start_loss+regularization
                consistent = torch.allclose(actual,self.prediction,
                                            atol=self.args.structural_verify_atol,rtol=self.args.structural_verify_rtol)
                if not consistent or not bool(torch.isfinite(delta_j) & (delta_j < -self.args.structural_min_gain)):
                    self.model._rollback_structure(snapshot)
                    rolled_back = True
                    self.accepted = dict.fromkeys(self.kinds,0)
                    delta_j = baseline.new_zeros(())
            except Exception:
                # 已临时替换正式参数；异常时先恢复全部引用和状态，再交给调用方处理。
                self.model._rollback_structure(snapshot)
                raise
        else:
            self.model.xyz_gradient_accum.zero_()
            self.model.denom.zero_()
            self.model.max_radii2D.zero_()
        return dict(before=before,after=len(self.model.get_xyz),candidates=self.candidates,
                    accepted=self.accepted,delta_j=float(delta_j),rolled_back=rolled_back,
                    budget_rejections=self.budget_rejections,seconds=time.perf_counter()-start)
