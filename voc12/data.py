import os.path
import random
import pdb

import numpy as np
import torch
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

import cv2
import PIL.Image
from matplotlib import pyplot as plt

##########################################################
import scipy.misc
import scipy.ndimage
import tools.imutils as imutils
##########################################################


IMG_FOLDER_NAME = "JPEGImages" ##################caution
ANNOT_FOLDER_NAME = "Annotations"

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

def save_img(x, path):
    plt.imshow(x)
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(path)
    plt.close()

def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME,img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list):

    cls_labels_dict = np.load('voc12/cls_labels.npy',allow_pickle=True).item()
    # cls_labels_dict = np.load('voc12/cls_labels_custom2.npy', allow_pickle=True).item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]

def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

    return img_name_list

class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return name, img

class VOC12ClsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])
        
        pack = {}
        pack['img'] = img
        pack['label'] = label
        pack['name'] = name

        return pack
    
class VOC12ClsDatasetMSF(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1, use_se=False, se_path=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform
        self.use_se = use_se
        self.se_path = se_path

    def __getitem__(self, idx):
        pack = super().__getitem__(idx)

        name = pack['name'] 
        img = pack['img']
        label = pack['label']

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)
                
            

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
            
        pack = {}
        pack['img'] = msf_img_list
        pack['label'] = label
        pack['name'] = name
        
        if self.use_se:
            se = np.load(self.se_path+'/'+name+'.npy')
            se = (se+1).astype(np.uint8)
            se = torch.from_numpy(se)
            se_flip = torch.flip(se, dims=(1,))
            
            pack['se'] = [se, se_flip]

        return pack
    

class VOC12ClsDataset_MyTF(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, crop, resize, cj, use_se=False, se_path=None):
        super().__init__(img_name_list_path, voc12_root)
        
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        
        self.crop = crop
        self.resize = resize
        self.cj = cj
        
        self.use_se = use_se
        self.se_path = se_path
        
        self.tf_rr = imutils.random_resize(self.resize[0], self.resize[1])
        self.tf_rc = imutils.random_crop(self.crop[0])
        
        self.tf_flip = transforms.RandomHorizontalFlip(p=0.5)
        self.tf_cj = transforms.RandomApply([transforms.ColorJitter(self.cj[0], self.cj[1], self.cj[2], self.cj[3])], p=0.8)
        self.tf_gray = transforms.RandomGrayscale(p=0.02)
        self.tf_norm = imutils.normalize()

        self.tf_list = []
        self.tf_list.append(imutils.HWC_to_CHW)
        self.tf_list.append(imutils.torch.from_numpy)
        self.tf_final = transforms.Compose(self.tf_list)

    def apply_tf(self, img):

        img = self.tf_rr(img)
        img = self.tf_flip(img)
        img = self.tf_cj(img)
        img = self.tf_gray(img)

        img = np.asarray(img)
        img = self.tf_norm(img)

        img = self.tf_rc(img)
        img = self.tf_final(img)
        
        return img
    
    def apply_tf_se(self, img):

        img = self.tf_rr(img, mode='nearest')
        img = self.tf_flip(img)
        # img = self.tf_cj(img)
        # img = self.tf_gray(img)

        img = np.asarray(img) # (H,W,3)
        # img = self.tf_norm(img)
        
        unique_values = np.unique(img)
        sorted_values = np.sort(unique_values)

        mapped_img = np.searchsorted(sorted_values, img)

        img = self.tf_rc(mapped_img)
        img = self.tf_final(img)
        
        img = img[0].unsqueeze(0)
        
        return img

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])
        
        if self.use_se:
            
            rng_state = random.getstate()
            rng_state_torch = torch.get_rng_state()
            img = self.apply_tf(img)
            
            random.setstate(rng_state)
            torch.set_rng_state(rng_state_torch)
            
            se = np.load(self.se_path+'/'+name+'.npy')

            se = (se+1).astype(np.uint8)
            se = np.expand_dims(se, axis=2)
            se = np.repeat(se, 3, axis=2)
            se = PIL.Image.fromarray(se)
            se = self.apply_tf_se(se)
            
            pack = {}
            pack['img'] = img
            pack['label'] = label
            pack['name'] = name
            pack['se'] = se
            return pack
            
        
        else:
            img = self.apply_tf(img)
        
            pack = {}
            pack['img'] = img
            pack['label'] = label
            pack['name'] = name

            return pack

    def __len__(self):
        return len(self.img_name_list)






########################################################################################################################
##################################################### SEGMENTATION #####################################################
########################################################################################################################


class VOC12ImageSegDataset(Dataset):

    def __init__(self, gt_path, img_name_list_path, voc12_root, val_flag=False):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        self.voc12_root = voc12_root
        self.gt_path = gt_path

    def set_tf(self, phase):

        self.tf_rr = imutils.random_resize(256, 768)
        self.tf_rc = imutils.random_crop(384)

        self.tf_cj = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

        self.tf_norm = imutils.normalize()

        self.tf_permute = imutils.HWC_to_CHW

        self.tf_list = []
        self.tf_list.append(imutils.HWC_to_CHW)
        
        if phase=='train':
            self.tf_list.append(imutils.torch.from_numpy)
            
        self.tf = transforms.Compose(self.tf_list)   

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx, val_flag=False):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")
        label = PIL.Image.open(self.gt_path + '/' + name + '.png')
        cls_label = torch.from_numpy(self.label_list[idx])

        # longer = max(img.size(0), img.size(1))
        # self.tf_rr = imutils.random_resize(longer*0.5, longer*2)

        if not val_flag:

            img, xy = self.tf_rr(img, get_xy=True)
            label = self.tf_rr(label, xy=xy)

            if random.random()<0.5:
                img = transforms.functional.hflip(img)
                label = transforms.functional.hflip(label)

            img = self.tf_cj(img)

            img = np.asarray(img)
            label = np.expand_dims(np.asarray(label), axis=2)

            img = self.tf_norm(img)

            img, xy = self.tf_rc(img, get_xy=True)
            label = self.tf_rc(label, xy=xy)

            img = self.tf(img)
            label = self.tf(label)

        return name, img, label, cls_label


class VOC12ImageSegDataset_test(Dataset):

    def __init__(self, gt_path, img_name_list_path, voc12_root, val_flag=False):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.gt_path = gt_path

    def set_tf(self, phase):

        self.tf_rr = imutils.random_resize(256, 768)
        self.tf_rc = imutils.random_crop(448)

        self.tf_cj = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

        self.tf_norm = imutils.normalize()

        self.tf_permute = imutils.HWC_to_CHW

        self.tf_list = []
        self.tf_list.append(imutils.HWC_to_CHW)

        if phase == 'train':
            self.tf_list.append(imutils.torch.from_numpy)

        self.tf = transforms.Compose(self.tf_list)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx, val_flag=False):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")
        # label = PIL.Image.open(self.gt_path + '/' + name + '.png')

        # longer = max(img.size(0), img.size(1))
        # self.tf_rr = imutils.random_resize(longer*0.5, longer*2)

        if not val_flag:

            img, xy = self.tf_rr(img, get_xy=True)
            # label = self.tf_rr(label, xy=xy)

            if random.random() < 0.25:
                img = transforms.functional.hflip(img)
                # label = transforms.functional.hflip(label)

            img = self.tf_cj(img)

            img = np.asarray(img)
            # label = np.expand_dims(np.asarray(label), axis=2)

            img = self.tf_norm(img)

            img, xy = self.tf_rc(img, get_xy=True)
            # label = self.tf_rc(label, xy=xy)

            img = self.tf(img)
            # label = self.tf(label)

        return name, img #, label


class VOC12ImageSegDatasetMSF(VOC12ImageSegDataset):

    def __init__(self, gt_path, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(gt_path, img_name_list_path, voc12_root, val_flag=True)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def set_tf(self):

        self.tf_norm = imutils.normalize()
        self.tf_permute = imutils.HWC_to_CHW

    def __getitem__(self, idx):
        name, img, label,_ = super().__getitem__(idx, val_flag=True)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        ms_label_list = []

        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            s_label = label.resize(target_size, resample=PIL.Image.NEAREST)
            
            ms_img_list.append(s_img)
            ms_label_list.append(s_label)

        for i in range(len(ms_img_list)):
                
            ms_img_list[i] = np.asarray(ms_img_list[i])
            ms_img_list[i] = self.tf_norm(ms_img_list[i])
            ms_img_list[i] = self.tf_permute(ms_img_list[i])
                
            ms_label_list[i] = np.expand_dims(np.asarray(ms_label_list[i]), axis=2)
            ms_label_list[i] = self.tf_permute(ms_label_list[i])
    
        msf_img_list = []
        msf_label_list = []
        
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
            msf_label_list.append(ms_label_list[i])
            msf_label_list.append(np.flip(ms_label_list[i], -1).copy())

        for i in range(len(msf_img_list)):
            msf_img_list[i] = torch.from_numpy(msf_img_list[i])
            msf_label_list[i] = torch.from_numpy(msf_label_list[i])

        return name, msf_img_list, msf_label_list


class VOC12ImageSegDatasetMSF_test(VOC12ImageSegDataset_test):

    def __init__(self, gt_path, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(gt_path, img_name_list_path, voc12_root, val_flag=True)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def set_tf(self):

        self.tf_norm = imutils.normalize()
        self.tf_permute = imutils.HWC_to_CHW

    def __getitem__(self, idx):
        name, img  = super().__getitem__(idx, val_flag=True)

        rounded_size = (
        int(round(img.size[0] / self.unit) * self.unit), int(round(img.size[1] / self.unit) * self.unit))

        ms_img_list = []
        # ms_label_list = []

        for s in self.scales:
            target_size = (round(rounded_size[0] * s),
                           round(rounded_size[1] * s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            # s_label = label.resize(target_size, resample=PIL.Image.NEAREST)

            ms_img_list.append(s_img)
            # ms_label_list.append(s_label)

        for i in range(len(ms_img_list)):
            ms_img_list[i] = np.asarray(ms_img_list[i])
            ms_img_list[i] = self.tf_norm(ms_img_list[i])
            ms_img_list[i] = self.tf_permute(ms_img_list[i])

            # ms_label_list[i] = np.expand_dims(np.asarray(ms_label_list[i]), axis=2)
            # ms_label_list[i] = self.tf_permute(ms_label_list[i])

        msf_img_list = []
        msf_label_list = []

        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())
            # msf_label_list.append(ms_label_list[i])
            # msf_label_list.append(np.flip(ms_label_list[i], -1).copy())

        for i in range(len(msf_img_list)):
            msf_img_list[i] = torch.from_numpy(msf_img_list[i])
            # msf_label_list[i] = torch.from_numpy(msf_label_list[i])

        return name, msf_img_list






















########################################################################################################################
####################################################### AFFINITY #######################################################
########################################################################################################################


class ExtractAffinityLabelInRadius():

    def __init__(self, cropsize, radius=5):
        self.radius = radius

        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))

        for y in range(1, radius):
            for x in range(-radius+1, radius):
                if x*x + y*y < radius*radius:
                    self.search_dist.append((y, x))

        self.radius_floor = radius-1

        self.crop_height = cropsize - self.radius_floor
        self.crop_width = cropsize - 2 * self.radius_floor
        return

    def __call__(self, label):

        labels_from = label[:-self.radius_floor, self.radius_floor:-self.radius_floor]
        labels_from = np.reshape(labels_from, [-1])

        labels_to_list = []
        valid_pair_list = []

        for dy, dx in self.search_dist:
            labels_to = label[dy:dy+self.crop_height, self.radius_floor+dx:self.radius_floor+dx+self.crop_width]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, 255), np.less(labels_from, 255))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)

        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)), concat_valid_pair).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), torch.from_numpy(neg_affinity_label)


class VOC12AffDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_la_dir, label_ha_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_la_dir = label_la_dir
        self.label_ha_dir = label_ha_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_la_path = os.path.join(self.label_la_dir, name + '.npy')

        label_ha_path = os.path.join(self.label_ha_dir, name + '.npy')

        label_la = np.load(label_la_path,allow_pickle=True).item()
        label_ha = np.load(label_ha_path,allow_pickle=True).item()

        label = np.array(list(label_la.values()) + list(label_ha.values()))
        label = np.transpose(label, (1, 2, 0)) # H x W x (2 x num_cls)

        for joint_transform, img_transform, label_transform \
                in zip(self.joint_transform_list, self.img_transform_list, self.label_transform_list):

            if joint_transform:
                img_label = np.concatenate((img, label), axis=-1)
                img_label = joint_transform(img_label)
                img = img_label[..., :3]
                label = img_label[..., 3:]

            if img_transform:
                img = img_transform(img)
            if label_transform:
                label = label_transform(label)

        no_score_region = np.max(label, -1) < 1e-5
        label_la, label_ha = np.array_split(label, 2, axis=-1)
        label_la = np.argmax(label_la, axis=-1).astype(np.uint8)
        label_ha = np.argmax(label_ha, axis=-1).astype(np.uint8)
        label = label_la.copy()
        label[label_la == 0] = 255
        label[label_ha == 0] = 0
        label[no_score_region] = 255 # mostly outer of cropped region
        label = self.extract_aff_lab_func(label)

        return img, label


class VOC12AffGtDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_dir = label_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_path = os.path.join(self.label_dir, name + '.png')

        label = scipy.misc.imread(label_path)

        for joint_transform, img_transform, label_transform \
                in zip(self.joint_transform_list, self.img_transform_list, self.label_transform_list):

            if joint_transform:
                img_label = np.concatenate((img, label), axis=-1)
                img_label = joint_transform(img_label)
                img = img_label[..., :3]
                label = img_label[..., 3:]

            if img_transform:
                img = img_transform(img)
            if label_transform:
                label = label_transform(label)

        label = self.extract_aff_lab_func(label)

        return img, label
