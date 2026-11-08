"""病例局部 spacing 坐标；PLY 几何记录与 checkpoint 使用同一契约。"""
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from config_utils import render_options


def make_geometry(spacing_xyz, shape_shw, args, legacy=False):
    spacing = np.asarray(spacing_xyz, dtype=float).reshape(3)
    slices, height, width = map(int, shape_shw)
    extent = spacing * [width, height, max(slices-1, 1)]
    scale = float(extent.max()/2)
    bounds = list(args.bounds) if legacy else (extent/(2*scale)).tolist()
    rendering = render_options(args)
    rendering['bounds'] = bounds
    return dict(version=0 if legacy else 1, frame='legacy_box' if legacy else 'spacing_stack_local',
                spacing_unit='mm_assumed_from_dataset', spacing_xyz=spacing.tolist(),
                shape_shw=[slices,height,width], scale_mm=None if legacy else scale,
                bounds=bounds, reference_phase=args.reference_phase,
                normalization_epsilon=args.normalization_epsilon, rendering=rendering)


def check_geometry(saved, current):
    """只比较坐标/观测条件，不把迭代数或输出路径当作实验几何。"""
    def same(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a, (list,tuple)) and isinstance(b, (list,tuple)):
            return len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
        if isinstance(a, (int,float)) and isinstance(b, (int,float)):
            return bool(np.isclose(a,b,rtol=1e-8,atol=1e-10))
        return a == b
    if not same(saved,current):
        raise ValueError('模型几何与当前数据/配置不一致：请核对 spacing、shape、参考相位、归一化和渲染参数')


def attach_geometry(model, geometry):
    model.geometry = copy.deepcopy(geometry)
    model.bounds = list(geometry['bounds'])


def save_model_ply(model, path, iteration=None):
    """旧 PLY 字段不变；哈希防止中断后将旧几何记录误配给新 PLY。"""
    model.save_ply(str(path)+'.tmp')
    os.replace(str(path)+'.tmp',path)
    record = dict(geometry=model.geometry,
                  geometry_reference_count=model.geometry_reference_count,
                  iteration=iteration,
                  ply_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    sidecar = str(path)+'.geometry.json'
    with open(sidecar+'.tmp','w',encoding='utf-8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2,allow_nan=False)
    os.replace(sidecar+'.tmp',sidecar)


def read_model_geometry(path):
    sidecar = Path(str(path)+'.geometry.json')
    if not sidecar.is_file():
        return None
    record = json.loads(sidecar.read_text(encoding='utf-8'))
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != record['ply_sha256']:
        raise ValueError(f'PLY 与几何记录不匹配：{sidecar}')
    return record
