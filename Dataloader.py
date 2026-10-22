import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from decimal import Decimal
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as FF
from tools import subplots, load_json,Plot_flows
import shutil
import pandas as pd
from torchvision.transforms import v2

class data_augmentation():

    def init(self,):
        self.prob = random.random()
        self.random_rotation = transforms.RandomRotation(degrees=(-180, 180))

    def RandomFlip(self, images, labels=None):
        
        if random.random() < 0.3:
            images = torch.flipud(images)
            labels = torch.flipud(labels) if labels is not None else None
        elif random.random() > 0.7:
            images = torch.fliplr(images)
            labels = torch.fliplr(labels) if labels is not None else None

        if labels is not None:
            return images, labels 
        else:
            return images
    
    def RandomRotate(self, images, labels=None):

        if self.prob < 0.8:
            images = self.random_rotation(images)
            labels = self.random_rotation(labels) if labels is not None else None

        if labels is not None:
            return images, labels 
        else:
            return images

class dataset(Dataset):

    def __init__(self, mode='train', datasets='shengyi', align = False, augmentation=False):
        

        self.augmentation = augmentation 
        self.datasets = datasets
        # self.dim = dim
        self.cmr_cases = []
        self.tmr_cases = []
        self.lge_cases = []
        self.note = {}
        self.num = 0
        if datasets == 'shengyi':
            self.data_pth = r'/data1/zhuangshuxin/datasets/shengyi_all/cropCMRnpz'
            patients = sorted(os.listdir(self.data_pth))

        elif datasets == 'acdc':
            self.data_pth = r'/data1/zhuangshuxin/datasets/ACDC/all_npz'
            patients = sorted(os.listdir(self.data_pth))
        
        length = len(patients)
        if mode == 'train':
            self.patients = patients[:int(length*0.7)]
        elif mode == 'val':
            self.patients = patients[int(length*0.7):]
        elif mode == 'all':
            self.patients = patients

        print(f'{len(self.patients)} cardiac cases.')

    def normalize_data(self, img_np, mode='cmr'):

        cmin = torch.min(img_np)
        cmax = torch.max(img_np)
        img_np = (img_np - cmin) / (cmax- cmin + 0.0001)

        if mode == 'tmr':
            if len(img_np.shape) == 3:
                meanv = torch.mean(img_np, dim=(1,2))
                meanv = meanv.view(-1,1,1)
            if len(img_np.shape) == 4:
                s, t, h, w = img_np.shape
                meanv = torch.mean(img_np, dim=(2,3))
                meanv = meanv.view(-1,1,1).reshape(s, t, 1, 1)
            img_np = img_np / (2 * meanv)
            img_np = torch.clamp(img_np, -1, 1)
        return img_np

    def split_masks(self, masks):
        unique = [x for x in torch.unique(masks) if x != 0]
        Masks = []
        for u in unique:
            mask = torch.zeros_like(masks)
            mask[masks == u] = 1
            Masks.append(mask)
        return Masks

    def __getitem__(self, idx):
        
        if self.augmentation:
            self.angle = random.uniform(-45, 45)
            self.prob = random.random()>0.5
        Data = self.get_data(idx)
        

        return Data

    def __len__(self):

        # length = []
        # length = max([len(self.cmr_cases)])
        return len(self.patients)


    def get_data(self, idx):
        Patient = self.patients[idx]
        Data = np.load(os.path.join(self.data_pth, Patient))
        # ids = Data['info']      # [ed, es]
        
        # 基础信息必定存在
        data = {'info': Patient}

        # 动态加载图像 (Images)
        if 'cmr' in Data:
            # data['cmr'] = self.normalize_data(torch.Tensor(Data['cmr']))[:,ids[0]-1:ids[1]]
            data['cmr'] = self.normalize_data(torch.Tensor(Data['cmr']))
        if 'tmr' in Data:
            data['tmr'] = self.normalize_data(torch.Tensor(Data['tmr']), mode='tmr')
        if 'lge' in Data:
            data['lge'] = self.normalize_data(torch.Tensor(Data['lge']))

        # 动态加载标签 (Labels)
        if 'cmr_lbl' in Data:
            data['cmr_lbl'] = torch.Tensor(Data['cmr_lbl'])
        if 'tmr_lbl' in Data:
            data['tmr_lbl'] = torch.Tensor(Data['tmr_lbl'])
        if 'lge_lbl' in Data:
            data['lge_lbl'] = torch.Tensor(Data['lge_lbl'])

        return data

    def get_flow_data(self,idx, dim):
        # cases = self.cases[idx]
        if dim == '2D':
            Case = self.cases[idx]
            # img, gt = self.load_data(Case, self.cmr_pth, self.cmrgt_pth)
            flows = self.load_data(Case, self.flow_pth)
            if self.augmentation:
                flows = FF.rotate(flows, angle=self.angle, interpolation=InterpolationMode.BILINEAR)

            patient = Case.split('/')[0]
            slices = Case.split('/')[1]
            data = {
                    'flows': flows,
                    'info': [patient, slices],
                    }
            
        elif dim == '3D':
            Case = self.cases[idx]
            patient = list(Case.keys())[0]
            slcs = [os.path.join(patient, s) for s in Case[patient]]
            flowsxy = self.load_data(slcs, self.flow_pth)
            flow_norm = torch.Tensor(load_json(r'/data1/zhuangshuxin/codes/MI_prediction/flow_norm.json'))
            if flowsxy.shape[0] == flow_norm.shape[0]:
                flowsxy = flowsxy / flow_norm[:, None, None, None, None]
            S, T, C, H, W = flowsxy.shape
            if self.flow_pth[0].split('/')[-1] == 'TMR_Flows':
                flowsxy = flowsxy.repeat_interleave(2, dim=0)[:-1]
                if T<14:
                    div = 14 - T
                    rep_im1 = flowsxy[:,-2:-1].repeat(1,div,1,1,1)
                    flowsxy = torch.cat([flowsxy, rep_im1], dim=1)
                elif T>14:
                    flowsxy = flowsxy[:,:14]
            elif self.flow_pth[0].split('/')[-1] == 'CMR_Flows':
                # flowsxy = flowsxy.repeat_interleave(2, dim=0)[:-1]
                if T<14:
                    div = 14 - T
                    rep_im1 = flowsxy[:,-2:-1].repeat(1,div,1,1,1)
                    flowsxy = torch.cat([flowsxy, rep_im1], dim=1)
                elif T>14:
                    flowsxy = flowsxy[:,:14]

            if self.augmentation:
                flowsx = FF.rotate(flowsxy[:,:,0], angle=self.angle, interpolation=InterpolationMode.BILINEAR)
                flowsy = FF.rotate(flowsxy[:,:,1], angle=self.angle, interpolation=InterpolationMode.BILINEAR)
                flowsxy = torch.stack([flowsx, flowsy], dim=2)
            data = {
                    'flowsxy': flowsxy,
                    'flows': torch.sqrt(flowsxy[:,:,0]**2 + flowsxy[:,:,1]**2),
                    'info': [patient, Case[patient]],
                    }
            
        return data

def fix_len(arr):
    """把 array 的第 0 维统一到 target 长度。
    不足：复制最后一帧；超出：截断。
    """
    n = arr.shape[0]
    if n == 40:
        return arr
    if n > 40:
        return arr[:40]
    # n < target: 复制最后一帧补齐
    pad = np.repeat(arr[-1:], 40 - n, axis=0)
    return np.concatenate([arr, pad], axis=0)

if __name__ == "__main__":

    Dataset = torch.utils.data.DataLoader(
        dataset(mode="val", datasets='shengyi', align=False, augmentation=False),
        batch_size=1, 
        num_workers=0, 
        drop_last=True,
        pin_memory=True,
    )
    
    for sample_idx, data in enumerate(Dataset):
            
        # if os.path.exists(os.path.join(r'output/acdc_experiment', f'sample_{sample_idx:03d}')):
        #     continue
        imgs = data['cmr'][0,:,0].cuda()    # initial time step
        lbls = data['cmr_lbl'].cuda()
        p_name = data['info'][0].split('.')[0]
    # pth1 = r'/data1/zhuangshuxin/datasets/shengyi_all/cropped/images/cmr'
    # pth2 = r'/data1/zhuangshuxin/datasets/shengyi_all/cropped/labels/cmr_Myo_mask'

    # for p in tqdm(sorted(os.listdir(pth2))):
    #     slcs = os.listdir(os.path.join(pth2,p))
    #     imgs = [nib.load(os.path.join(pth1,p,f)).get_fdata() for f in slcs]
    #     lbls = [nib.load(os.path.join(pth2,p,f)).get_fdata() for f in slcs]
    #     imgs = [fix_len(a) for a in imgs]
    #     lbls = [fix_len(a) for a in lbls]


    #     data = {'cmr':np.array(imgs),
    #             'cmr_lbl':np.array(lbls),
    #             'info': p,
    #             }
    #     save = r'/data1/zhuangshuxin/datasets/shengyi_all/crop_cmr_npz'
    #     spth = os.path.join(save, p+'.npz')
    #     np.savez(spth, **data)  

    # pth = r'/data1/zhuangshuxin/codes/MI_prediction_v/CMR_Flows/BI_PEI_YUN_041Y/-26.7484.nii.gz'
    # # load nii
    # flow = torch.Tensor(nibabel.load(pth).get_fdata())
    # rgbs = Plot_flows(flow)
    # subplots(rgbs, color=True)
    # print(flow.shape)

    # train_dataset = torch.utils.data.DataLoader(
    #         dataset(mode="train", datasets='shengyi', align=False, augmentation=False),
    #         batch_size=1, 
    #         num_workers=0, 
    #         drop_last=True,
    #         pin_memory=True,
    #     )

    
    # for data in tqdm(train_dataset):
         
    #     a = data['cmr']
    #     b = data['tmr']
    #     c = data['lge']
    #     d = data['cmr_lbl']
    #     e = data['tmr_lbl'] 
    #     f = data['lge_lbl']

    #     # subplots(f)
    #     pass

