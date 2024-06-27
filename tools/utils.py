import os
import os.path as osp
import random

import PIL

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import voc12.data
import tools.imutils as imutils

class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

class PolyOptimizer_adam(torch.optim.Adam):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


def make_path(args):

    exp_path = osp.join('./experiments', args.name)
    ckpt_path = osp.join(exp_path, 'ckpt')
    train_path = osp.join(exp_path, 'train')
    val_path = osp.join(exp_path, 'val')
    infer_path = osp.join(exp_path, 'infer')
    dict_path = osp.join(exp_path, 'dict')
    crf_path = osp.join(exp_path, 'crf')
        
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        os.makedirs(ckpt_path)
        os.makedirs(train_path)
        os.makedirs(val_path)
        os.makedirs(infer_path)
        os.makedirs(dict_path)
        os.makedirs(crf_path)
        print(exp_path + ' is built.')
    else:
        print(exp_path + ' already exsits.')

    for alpha in args.alphas:
        crf_alpha_path = osp.join(crf_path, str(alpha).zfill(2))
        if not os.path.exists(crf_alpha_path):
            os.makedirs(crf_alpha_path)

    return exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path


def make_path_with_log(args):

    exp_path = osp.join('./experiments', args.name)
    ckpt_path = osp.join(exp_path, 'ckpt')
    train_path = osp.join(exp_path, 'train')
    val_path = osp.join(exp_path, 'val')
    infer_path = osp.join(exp_path, 'infer')
    dict_path = osp.join(exp_path, 'dict')
    crf_path = osp.join(exp_path, 'crf')
    log_path = osp.join(exp_path, 'log.txt')
        
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        os.makedirs(ckpt_path)
        os.makedirs(train_path)
        os.makedirs(val_path)
        os.makedirs(infer_path)
        os.makedirs(dict_path)
        os.makedirs(crf_path)
        print(exp_path + ' is built.')
    else:
        print(exp_path + ' already exsits.')

    for alpha in args.alphas:
        crf_alpha_path = osp.join(crf_path, str(alpha).zfill(2))
        if not os.path.exists(crf_alpha_path):
            os.makedirs(crf_alpha_path)

    return exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path, log_path


def build_dataset(args, phase='train', path="voc12/train_aug.txt", root='./data/VOC2012'):

    tf_list = []
    
    crop = args.crop # 384
    resize = args.resize # 256 448
    cj = args.cj

    if phase=='train':
        tf_list.append(imutils.random_resize(resize[0], resize[1]))
        tf_list.append(transforms.RandomHorizontalFlip())
        tf_list.append(transforms.ColorJitter(brightness=cj[0], contrast=cj[1], saturation=cj[2], hue=cj[3]))

    tf_list.append(np.asarray)
    tf_list.append(imutils.normalize())
    print("ResNet38d")

    if phase=='train':
        tf_list.append(imutils.random_crop(crop))

    tf_list.append(imutils.HWC_to_CHW)

    if phase=='train':
        tf_list.append(imutils.torch.from_numpy)

    tf = transforms.Compose(tf_list)

    if phase=='train':
        dataset = voc12.data.VOC12ClsDataset(path, voc12_root=root, transform=tf)
    elif phase=='val':
        # MSF dataset augments an image to 8 images with multi-scale & flip 
        dataset = voc12.data.VOC12ClsDatasetMSF(path, voc12_root=root, scales=[0.5,1.0,1.5,2.0], inter_transform=tf)
    
    return dataset

def build_dataset_sam(args, phase='train', path="voc12/train_aug.txt", root='./data/VOC2012', use_se=False, se_path=None):

    tf_list = []
    tf_list.append(np.asarray)
    tf_list.append(imutils.normalize())
    tf_list.append(imutils.HWC_to_CHW)
    tf = transforms.Compose(tf_list)

    if phase == 'train':
        crop = args.crop
        resize = args.resize
        cj = args.cj
        dataset = voc12.data.VOC12ClsDataset_MyTF(path, voc12_root=root, crop=[crop, crop], resize=resize, cj=cj, use_se=use_se, se_path=se_path)

    elif phase == 'val':
        # MSF dataset augments an image to 8 images with multi-scale & flip
        dataset = voc12.data.VOC12ClsDatasetMSF(path, voc12_root=root, scales=[0.5, 1.0, 1.5, 2.0], inter_transform=tf, use_se=use_se, se_path=se_path)

    return dataset


def build_dataset_dl(phase='train', path="voc12/train.txt", gt_path='', root='./data/VOC2012'):
    
    if phase=='train':
        dataset = voc12.data.VOC12ImageSegDataset(gt_path, path, voc12_root=root)
        dataset.set_tf(phase)
        
    if phase=='val':
        dataset = voc12.data.VOC12ImageSegDatasetMSF(gt_path, path, voc12_root=root,scales=[0.5,0.75,1.0,1.25,1.5])#scales=[0.5, 0.75, 1, 1.25, 1.5])#
        dataset.set_tf()
    if phase=='test':
        dataset = voc12.data.VOC12ImageSegDatasetMSF_test(gt_path,path,voc12_root=root,scales=[0.5,0.75,1.0,1.25,1.5])
        dataset.set_tf()

    return dataset


def build_dataset_moco(args, phase='train', path="voc12/train_aug.txt", root='./data/VOC2012'):

    tf_list = []
    tf_list.append(np.asarray)
    tf_list.append(imutils.normalize())
    tf_list.append(imutils.HWC_to_CHW)
    tf = transforms.Compose(tf_list)

    crop = args.crop
    resize = args.resize
    cj = args.cj

    if phase == 'train':
        dataset = voc12.data.VOC12ClsDataset_SelfSup(path, voc12_root=root, crop=crop, resize=resize, cj=cj)
        # dataset = voc12.data.VOC12MocoDataset(args, path, voc12_root=root)

    elif phase == 'val':
        # MSF dataset augments an image to 8 images with multi-scale & flip
        dataset = voc12.data.VOC12ClsDatasetMSF(path, voc12_root=root, scales=[0.5, 1.0, 1.5, 2.0], inter_transform=tf)

    return dataset


def build_dataset_recon(args, phase='train', path="voc12/train_aug.txt", root='./data/VOC2012'):

    tf_list = []
    tf_list.append(np.asarray)
    tf_list.append(imutils.normalize())
    tf_list.append(imutils.HWC_to_CHW)
    tf = transforms.Compose(tf_list)

    crop = args.crop
    resize = args.resize

    if phase == 'train':
        dataset = voc12.data.VOC12ClsDatasetRecon(path, voc12_root=root, crop=crop, resize=resize)

    elif phase == 'val':
        # MSF dataset augments an image to 8 images with multi-scale & flip
        dataset = voc12.data.VOC12ClsDatasetMSF(path, voc12_root=root, scales=[0.5, 1.0, 1.5, 2.0], inter_transform=tf)

    return dataset