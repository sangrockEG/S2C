from __future__ import division
from __future__ import print_function

import os
import os.path as osp
from tqdm import tqdm
import pdb
import random
import importlib
import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt
import pickle

# Custom
import tools.utils as utils
from tools.imutils import *
from evaluation import eval_in_script

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.predictor import SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import build_all_layer_point_grids, batch_iterator

import torch.nn.functional as F

import glob
import pdb
import tqdm

root_path = '/home/vilab/khj/ssd0/wsss_sam'

sam_path = root_path + '/pretrained/sam_vit_h.pth'
sam = sam_model_registry['vit_h'](checkpoint=sam_path)
# sam = sam.to('cuda')

mask_generator = SamAutomaticMaskGenerator(sam)

img_path = root_path + '/' +'/data/VOC2012/JPEGImages/'
save_path = root_path + '/se/default/'

img_list_path = root_path + '/' + '/voc12/train_aug.txt'
img_gt_name_list = open(img_list_path).read().splitlines()
img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]


for name in tqdm.tqdm(img_name_list):
    img = plt.imread(img_path+name+'.jpg')
    masks = mask_generator.generate(img)
    
    temp = np.full((img.shape[0],img.shape[1]), -1, dtype=int)
    for i, mask in enumerate(reversed(masks)):
        temp[mask['segmentation']] = i
        
    np.save(save_path+name, temp)