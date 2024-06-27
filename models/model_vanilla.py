import os, time
import os.path as osp
import argparse
import glob
import random
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools

# Image tools
import cv2
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from torchvision import transforms

import voc12.data
from tools import utils, pyutils
from tools.imutils import save_img, denorm, _crf_with_alpha, cam_on_image

# import resnet38d
from networks import resnet38d


def set_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class model_WSSS():

    def __init__(self, args, logger=None, writer=None):

        self.args = args
        self.categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                           'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        # Common things
        self.phase = 'train'
        self.dev = 'cuda'
        self.bce = nn.BCEWithLogitsLoss()
        self.bs = args.batch_size
        if logger is not None:
            self.logger = logger
        if writer is not None:
            self.writer = writer

        # Attributes
        self.C = args.C  # Number of classes - VOC : 20
        self.D = args.D  # Feature dimension - Default : 256
        self.M = args.M  # Momentum - Default : 0.997
        self.W = args.W  # Weight for each term in loss - Default : [1, 1, 1]

        # Model attributes
        self.net_names = ['net_main']
        self.base_names = ['cls']
        self.loss_names = ['loss_' + bn for bn in self.base_names]
        self.acc_names = ['acc_' + bn for bn in self.base_names]

        self.nets = []
        self.opts = []

        # Evaluation-related
        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.accs = [0] * len(self.acc_names)
        self.count = 0
        self.num_count = 0

        # Define networks
        self.net_main = resnet38d.Net_CAM(C=self.C, D=self.D)

        # Initialize networks with ImageNet pretrained weight
        self.net_main.load_state_dict(resnet38d.convert_mxnet_to_torch('./pretrained/resnet_38d.params'), strict=False)

    # Save networks
    def save_model(self, epo, ckpt_path):
        epo_str = str(epo).zfill(3)
        torch.save(self.net_main.module.state_dict(), ckpt_path + '/' + epo_str + 'net_main.pth')

    # Load networks
    def load_model(self, epo, ckpt_path):
        epo_str = str(epo).zfill(3)
        self.net_main.load_state_dict(torch.load(ckpt_path + '/' + epo_str + 'net_main.pth'), strict=True)

        self.net_main = torch.nn.DataParallel(self.net_main.to(self.dev))

    # Set networks' phase (train/eval)
    def set_phase(self, phase):

        if phase == 'train':
            self.phase = 'train'
            for name in self.net_names:
                getattr(self, name).train()
            self.logger.info('Phase : train')

        else:
            self.phase = 'eval'
            for name in self.net_names:
                getattr(self, name).eval()
            self.logger.info('Phase : eval')

    # Set optimizers and upload networks on multi-gpu
    def train_setup(self):

        args = self.args
        param_main = self.net_main.get_parameter_groups()

        self.opt_main = utils.PolyOptimizer([
            {'params': param_main[0], 'lr': 1 * args.lr, 'weight_decay': args.wt_dec},
            {'params': param_main[1], 'lr': 2 * args.lr, 'weight_decay': 0},  # non-scratch bias
            {'params': param_main[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec},  # scratch weight
            {'params': param_main[3], 'lr': 20 * args.lr, 'weight_decay': 0}  # scratch bias
        ],
            lr=args.lr, weight_decay=args.wt_dec, max_step=args.max_step)

        self.logger.info('Poly-optimizer for net_main is defined.')
        self.logger.info('* Base learning rate : ' + str(args.lr))
        self.logger.info('* non-scratch layer weight lr : ' + str(args.lr))
        self.logger.info('* non-scratch layer bias lr : ' + str(2 * args.lr))
        self.logger.info('* scratch layer weight lr : ' + str(10 * args.lr))
        self.logger.info('* scratch layer bias lr : ' + str(20 * args.lr))
        self.logger.info('* Weight decaying : ' + str(args.wt_dec) + ', max step : ' + str(args.max_step))

        self.net_main = torch.nn.DataParallel(self.net_main.to(self.dev))
        self.logger.info('Network is uploaded on multi-gpu.')

        self.nets.append(self.net_main)

    # Unpack data pack from data_loader
    def unpack(self, pack):

        if self.phase == 'train':
            self.img = pack['img'].to(self.dev)
            self.label = pack['label'].to(self.dev)
            self.name = pack['name']  # list of image names

        if self.phase == 'eval':
            self.img = pack['img']
            for i in range(8):
                self.img[i] = self.img[i].to(self.dev)
            self.label = pack['label'].to(self.dev)
            self.name = pack['name'][0]

    # Do forward/backward propagation and call optimizer to update the networks
    def update(self, epo, iter):

        # Tensor dimensions
        B = self.img.shape[0]
        H = self.img.shape[2]
        W = self.img.shape[3]
        C = self.C
        D = self.D

        self.opt_main.zero_grad()

        out_main = self.net_main(self.img)
        feat_main = out_main['feat']
        cam_main = out_main['cam']
        pred_main = out_main['pred']
        
        self.loss_cls = self.W[0] * self.bce(pred_main, self.label)
        loss = self.loss_cls

        loss.backward()
        self.opt_main.step()

        #
        ################################################### Export ###################################################
        #


        # Visualize
        if iter%300==0:
            with torch.no_grad():
                num_vis = 8
                
                cam_vis = self.scale(cam_main[:num_vis])
                label_vis = self.label[:num_vis]
                
                img_vis = torch.zeros_like(self.img[:num_vis])            
                gt_idx = torch.zeros_like(label_vis).long()

                for i in range(num_vis):
                    gt_idx[i,label_vis[i].nonzero(as_tuple=False)[:,0][0]] = 1
                    img_vis[i] = denorm(self.img[i])
                    
                cam_vis = cam_vis[gt_idx==1].unsqueeze(1)
                cam_vis = F.interpolate(cam_vis, (H,W), mode='bilinear', align_corners=False)
                
                img_cam = torch.zeros_like(img_vis)
                img_vis = img_vis.cpu().detach().numpy()
                cam_vis = cam_vis.cpu().detach().numpy()
                for i in range(num_vis):
                    img_cam[i] = torch.from_numpy(cam_on_image(img_vis[i], cam_vis[i,0]))
                    
                self.writer.add_images('0_train', img_cam/255, epo)

        for i in range(len(self.loss_names)):
            self.running_loss[i] += getattr(self, self.loss_names[i]).item()
        self.count += 1

        self.count_rw(self.label, pred_main, 0)
    
    # Initialization for msf-infer
    def infer_init(self):
        n_gpus = torch.cuda.device_count()
        self.net_main_replicas = torch.nn.parallel.replicate(self.net_main.module, list(range(n_gpus)))

    # (Multi-Thread) Infer MSF-CAM and save image/cam_dict/crf_dict
    def infer_multi(self, epo, val_path, dict_path, crf_path, vis=False, dict=False, crf=False, writer=None):

        if self.phase != 'eval':
            self.set_phase('eval')

        epo_str = str(epo).zfill(3)
        gt = self.label[0].cpu().detach().numpy()
        self.gt_cls = np.nonzero(gt)[0]

        _, _, H, W = self.img[2].shape
        n_gpus = torch.cuda.device_count()

        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i % n_gpus):
                    out = self.net_main_replicas[i % n_gpus](img.cuda())
                    cam = out['cam']
                    cam = F.interpolate(cam, (H, W), mode='bilinear', align_corners=False)[0]
                    cam = F.relu(cam)

                    cam = cam.cpu().numpy()
                    cam *= self.label.clone().cpu().view(20, 1, 1).numpy()

                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)
                    return cam

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(self.img)), batch_size=8, prefetch_size=0, processes=8)
        
        cam_list = thread_pool.pop_results()
        cam = np.sum(cam_list, axis=0)
        cam_max = np.max(cam, (1, 2), keepdims=True)
        norm_cam = cam / (cam_max + 1e-5)

        self.cam_dict = {}
        for i in range(20):
            if self.label[0, i] > 1e-5:
                self.cam_dict[i] = norm_cam[i]

        if vis:
            img_np = denorm(self.img[2][0]).cpu().detach().numpy()
            for c in self.gt_cls:
                temp = cam_on_image(img_np, norm_cam[c])
                temp_path = osp.join(val_path, epo_str + '_' + self.name + '_cam_' + self.categories[c] + '.png')
                plt.imsave(temp_path, np.transpose(temp, (1,2,0)))
                if writer is not None:
                    writer.add_image(self.name+'/'+self.categories[c], temp)

        if dict:
            np.save(osp.join(dict_path, self.name + '.npy'), self.cam_dict)

        if crf:
            for a in self.args.alphas:
                crf_dict = _crf_with_alpha(self.cam_dict, self.name, alpha=a)
                np.save(osp.join(crf_path, str(a).zfill(2), self.name + '.npy'), crf_dict)

    # Print loss/accuracy (and re-initialize them)
    def print_log(self, epo, iter, writer):

        loss_str = ''
        acc_str = ''

        for i in range(len(self.loss_names)):
            loss_str += self.loss_names[i] + ' : ' + str(round(self.running_loss[i] / self.count, 5)) + ', '
            writer.add_scalar('Loss/'+self.loss_names[i], round(self.running_loss[i] / self.count, 5), iter)

        for i in range(len(self.acc_names)):
            if self.right_count[i] != 0:
                acc = 100 * self.right_count[i] / (self.right_count[i] + self.wrong_count[i])
                acc_str += self.acc_names[i] + ' : ' + str(round(acc, 2)) + ', '
                self.accs[i] = acc
                writer.add_scalar('Acc/'+self.acc_names[i], round(acc, 2), iter)
        
        self.logger.info(loss_str[:-2])
        self.logger.info(acc_str[:-2])

        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.count = 0

    def count_rw(self, label, out, idx):
        for b in range(self.bs):
            out = out[:,:self.C] # To ignore bg prediction
            gt = label[b].cpu().detach().numpy()
            gt_cls = np.nonzero(gt)[0]
            num = len(np.nonzero(gt)[0])
            pred = out[b].cpu().detach().numpy()
            pred_cls = pred.argsort()[-num:][::-1]

            for c in gt_cls:
                if c in pred_cls:
                    self.right_count[idx] += 1
                else:
                    self.wrong_count[idx] += 1

    # Exponential moving weight with momentum (from main to sup)
    @torch.no_grad()
    def mvweight(self):
        for param_main, param_sup in zip(self.net_main.parameters(), self.net_sup.parameters()):
            param_sup.data = self.M * param_sup.data + (1 - self.M) * param_main.data

    # Scale
    def scale(self, cam):
        cam = F.relu(cam)
        cam_max = F.adaptive_max_pool2d(cam, (1,1))
        cam /= cam_max+1e-5
        return cam

    # Max_norm
    def max_norm(self, cam_cp):
        N, C, H, W = cam_cp.size()
        cam_cp = F.relu(cam_cp)
        max_v = torch.max(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        min_v = torch.min(cam_cp.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        cam_cp = F.relu(cam_cp - min_v - 1e-5) / (max_v - min_v + 1e-5)
        return cam_cp

    def cam_l1(self, cam1, cam2):
        return torch.mean(torch.abs(cam2.detach() - cam1))
