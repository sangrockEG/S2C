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
from tools.imutils import save_img, denorm, _crf_with_alpha, cam_on_image, voc_palette

from scipy import ndimage as ndi
from skimage.feature import peak_local_max

# import resnet38d
from networks import resnet38d
import torch_scatter

# import SAM
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.predictor import SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

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
        self.L1 = nn.L1Loss()
        self.bs = args.batch_size
        if logger is not None:
            self.logger = logger
        if writer is not None:
            self.writer = writer

        # Attributes
        self.C = args.C  # Number of classes - VOC : 20
        self.D = args.D  # Feature dimension - Default : 256
        self.W = args.W  # Weight for each term in loss - Default : [1, 1, 1]

        self.T = args.T
        self.th_multi = args.th_multi # Default: 0.5
        self.size_sam = 1024

        # Model attributes
        self.net_names = ['net_main']
        self.base_names = ['cls', 'ssc', 'cpm']
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
        sam_path = './pretrained/sam_vit_h.pth'
        self.net_sam = sam_model_registry['vit_h'](checkpoint=sam_path)

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
        if not self.args.debug:
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
        self.net_sam = torch.nn.DataParallel(self.net_sam.to(self.dev))
        self.logger.info('Networks are uploaded on multi-gpu.')

        self.nets.append(self.net_main)

    # Unpack data pack from data_loader
    def unpack(self, pack):

        if self.phase == 'train':
            self.img = pack['img'].to(self.dev)
            self.label = pack['label'].to(self.dev)
            self.name = pack['name']
            self.se = pack['se'].to(self.dev)

        if self.phase == 'eval':
            self.img = pack['img']
            for i in range(8):
                self.img[i] = self.img[i].to(self.dev)
            self.label = pack['label'].to(self.dev)
            self.name = pack['name'][0]
            self.se = pack['se']
            for i in range(2):
                self.se[i] = self.se[i].unsqueeze(0).to(self.dev)
            

    # Do forward/backward propagation and call optimizer to update the networks
    def update(self, epo, iter):

        # Tensor dimensions
        B = self.img.shape[0]
        H = self.img.shape[2]
        W = self.img.shape[3]
        C = self.C
        D = self.D
                
        self.net_sam.eval()
        use_cpm = epo>self.args.sstart-1

        ############################################################################################################################
        ######################################################## CPM branch ########################################################
        ############################################################################################################################
        if use_cpm:
        
            # Obtain MS-CAM
            with torch.no_grad():
                self.net_main.eval()
                img_05 = F.interpolate(self.img, scale_factor=0.5, mode='bilinear', align_corners=True)
                img_10 = self.img
                img_15 = F.interpolate(self.img, scale_factor=1.5, mode='bilinear', align_corners=True)
                img_20 = F.interpolate(self.img, scale_factor=2.0, mode='bilinear', align_corners=True)
        
                img_ms = [img_05, img_10, img_15, img_20]
                
                for k, img in enumerate(img_ms):
                    out = self.net_main(img)
                    cam_temp = F.relu(F.interpolate(out['cam'], size=(H,W), mode='bilinear', align_corners=False))
                    cam_temp *= self.label.view(B,C,1,1)
                    
                    if k==0:
                        cam_ms = cam_temp
                    else:
                        cam_ms += cam_temp
                        
                cam_max = F.adaptive_max_pool2d(cam_ms, (1, 1))
                cam_ms = cam_ms / (cam_max + 1e-5) # (B,C,H,W)
                    
            # Sample points from the MS-CAM
            with torch.no_grad():

                img_sam = F.interpolate(denorm(self.img)*255, (self.size_sam, self.size_sam), mode='bilinear', align_corners=True)
                img_sam = img_sam.to(torch.uint8)


                # For efficient inference, we get embedding first                
                features_sam = self.net_sam(run_encoder_only=True, 
                                            transformed_image=img_sam, 
                                            original_image_size=(H,W))
                del img_sam
                
                ############################################### Sample local peaks ###############################################
                points_all = {}
                for i in range(B):
                    points_img = {}
                    for ct in self.label[i].nonzero(as_tuple=False)[:,0]:
                        ct = ct.item()
                        
                        cam_target = cam_ms[i,ct]
                
                        # Global maximum                
                        cam_target_f = cam_target.view(-1)
                        argmax_indices = torch.argmax(cam_target_f)
                        coord_w = argmax_indices // W
                        coord_h = argmax_indices % W
                        peak_max = torch.cat((coord_w.view(1,1),coord_h.view(1,1)), dim=-1) # (1,2)
                        peak_max = peak_max.cpu().detach().numpy()
                
                        # Local maximums
                        cam_target_np = cam_target.cpu().detach().numpy()
                    
                        cam_filtered = ndi.maximum_filter(cam_target_np, size=3, mode='constant')
                        peaks_temp = peak_local_max(cam_filtered, min_distance=20)
                        peaks_valid = peaks_temp[cam_target_np[peaks_temp[:,0],peaks_temp[:,1]]>self.th_multi]
                        
                        # Aggregate all the peaks
                        peaks = np.concatenate((peak_max, peaks_valid[1:]),axis=0) # (NP,2)

                        points = np.flip(peaks,axis=(-1)) * self.size_sam / H
                        points = torch.from_numpy(points).cuda()
                        points_img[ct] = points
                        
                    points_all[i] = points_img                    
                
                ############################################### Get masks using SAM ###############################################
                
                sam_conf = -1e5*torch.ones_like(cam_ms)
                
                for i in range(B):
                    for k in points_all[i].keys():
                        points = points_all[i][k].unsqueeze(0)
                        points_label = torch.ones_like(points[:,:,0])
                        
                        output_sam = self.net_sam(run_decoder_only=True, 
                                          features_sam=features_sam[i].unsqueeze(0), 
                                          original_image_size=(H,W), 
                                          point_coords=points, 
                                          point_labels=points_label)

                        mask = output_sam[0] # (1,3,H,W)
                        conf = output_sam[2] # (1,3,H,W)

                        idx_max_sam = 2 # Empirically, 2 is the best.

                        target_mask = mask[0,idx_max_sam]
                        target_conf = conf[0,idx_max_sam].unsqueeze(0).unsqueeze(0)
                        target_conf = F.interpolate(target_conf, (H,W), mode='bilinear', align_corners=False)[0,0]
                        
                        # Confidence-based aggregation
                        sam_conf[i,k][target_mask] = target_conf[target_mask] * cam_ms[i,k][target_mask].mean() # scalar
                
                temp = sam_conf.max(dim=1)
                pgt_sam = temp[1]
                pgt_score = temp[0]

                pgt_sam[pgt_score<0] = 20
                pgt_score[pgt_score<0] = 0

        ############################################################################################################################
        ####################################################### Main branch ########################################################
        ############################################################################################################################   
                
        self.net_main.train()
        self.opt_main.zero_grad()
        
        loss = 0

        out_main = self.net_main(self.img)
        feat_main = out_main['feat']
        cam_main = out_main['cam']
        pred_main = out_main['pred']
        
        cam_main = F.relu(cam_main)
        cam_max = F.adaptive_max_pool2d(cam_main, (1, 1))
        cam_main = cam_main / (cam_max + 1e-5)
        cam_main = F.interpolate(cam_main, size=(H,W), mode='bilinear', align_corners=False) * self.label.view(B,C,1,1)
        
        ############################################################################################################################
        ###################################################### Loss functions ######################################################
        ############################################################################################################################

        # mere classification loss
        self.loss_cls = self.W[0] * self.bce(pred_main, self.label)
        loss += self.loss_cls
        
        # SAM-Segment Contrasting (SSC)
        feat_main = F.interpolate(feat_main, size=(H,W), mode='bilinear', align_corners=False)
        feat_main = F.normalize(feat_main, dim=1)
        feat_main_ = feat_main.view(B,D,-1) # (B,D,HW) 
        index_ = self.se.view(B,1,-1).long() # (B,1,HW)
        
        pt = torch_scatter.scatter_mean(feat_main_.detach(), index_) # (B,D,N)
        pt = F.normalize(pt, dim=1)
        index_ = index_.squeeze(1)
        pred_ssc = torch.bmm(pt.permute(0,2,1), feat_main_) # (B,N,HW)

        self.loss_ssc = F.cross_entropy(pred_ssc*self.T, index_, ignore_index=0)
        if not torch.isnan(self.loss_ssc):
            loss += self.loss_ssc
        else:
            print("loss_ssc is NaN!")
            self.loss_ssc = torch.zeros_like(self.loss_cls)
        
        # CAM-based Prompting Module (CPM)
        if use_cpm:            
            cam_bg = 1-cam_main.max(dim=1,keepdims=True)[0]
            cam_main = torch.cat((cam_main, cam_bg), dim=1)
            self.loss_cpm = F.cross_entropy(cam_main, pgt_sam, ignore_index=255)
            
            if not torch.isnan(self.loss_cpm):
                loss += self.loss_cpm
            else:
                print("loss_cpm is NaN!")
                self.loss_cpm = torch.zeros_like(self.loss_cls)
        else:
            self.loss_cpm = torch.zeros_like(self.loss_cls)

        loss.backward()
        self.opt_main.step()
    
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
                    writer.add_image(self.name+'/'+self.categories[c], temp, epo)

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

    # Relu and Scale
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
    
    def sample_target_class(self, label):
        B = label.shape[0]
        label_target = torch.zeros(B, self.C).cuda()
        for i in range(B):
            label_idx = torch.nonzero(label[i], as_tuple=False)
            rand_idx = torch.randint(0, len(label_idx), (1,))
            target = label_idx[rand_idx][0]
            label_target[i, target] = 1
        return label_target