import os
import os.path as osp
import random

import PIL
from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import voc12
import cv2

def cam_on_image(img, cam):
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap.transpose((2,0,1)) + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def save_img(path, img, cam=None):

    plt.imshow(img)
    if cam is not None:
        plt.imshow(cam, cmap='jet', alpha=0.6)
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(path)
    plt.close()

def cam_on_image(img, cam):
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap.transpose((2,0,1)) + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def cam_on_image2(img, cam_pos, cam_neg):
    
    heatmap_pos = cv2.applyColorMap(np.uint8(255 * cam_pos), cv2.COLORMAP_HOT)
    heatmap_neg = cv2.applyColorMap(np.uint8(255 * cam_neg), cv2.COLORMAP_COOL)
    heatmap_pos = cv2.cvtColor(heatmap_pos, cv2.COLOR_BGR2RGB)
    heatmap_neg = cv2.cvtColor(heatmap_neg, cv2.COLOR_BGR2RGB)
    heatmap_pos = np.float32(heatmap_pos) / 255
    heatmap_neg = np.float32(heatmap_neg) / 255

    cam = heatmap_pos.transpose((2,0,1))+ img #+heatmap_neg.transpose((2,0,1)) 
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class random_resize():

    def __init__(self, min_long, max_long):
        self.min_long = min_long
        self.max_long = max_long

    def __call__(self, img, sal=None, get_xy=False, xy=None, mode=None):

        target_long = random.randint(self.min_long, self.max_long)
        w, h = img.size

        if xy:
            target_long = xy

        if w < h:
            target_shape = (int(round(w * target_long / h)), target_long)
        else:
            target_shape = (target_long, int(round(h * target_long / w)))

        if xy:
            img = img.resize(target_shape, resample=PIL.Image.NEAREST)
        else:
            if mode is None:
                img = img.resize(target_shape, resample=PIL.Image.CUBIC)
            else:
                img = img.resize(target_shape, resample=PIL.Image.NEAREST)
            
        if sal:
           sal = sal.resize(target_shape, resample=PIL.Image.CUBIC)
           return img, sal

        if get_xy:
            return img, target_long
        else:
            return img

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


class random_crop():

    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, imgarr, sal=None, get_xy=False, xy=None, seg=True):

        h, w, c = imgarr.shape

        ch = min(self.cropsize, h)
        cw = min(self.cropsize, w)
        
        w_space = w - self.cropsize
        h_space = h - self.cropsize

        if w_space > 0:
            cont_left = 0
            img_left = random.randrange(w_space+1)
        else:
            cont_left = random.randrange(-w_space+1)
            img_left = 0



        if h_space > 0:
            cont_top = 0
            img_top = random.randrange(h_space+1)
        else:
            cont_top = random.randrange(-h_space+1)
            img_top = 0

        if xy:
            cont_left, img_left, cont_top, img_top = xy

        if not xy or not seg:
            container = np.zeros((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)
        else:
            container = np.ones((self.cropsize, self.cropsize, imgarr.shape[-1]), np.float32)*255
        container[cont_top:cont_top+ch, cont_left:cont_left+cw] = \
            imgarr[img_top:img_top+ch, img_left:img_left+cw]
        if sal is not None:
            container_sal = np.zeros((self.cropsize, self.cropsize,1), np.float32)
            container_sal[cont_top:cont_top+ch, cont_left:cont_left+cw,0] = \
                sal[img_top:img_top+ch, img_left:img_left+cw]
            return container, container_sal

        if get_xy:
            xy = cont_left, img_left, cont_top, img_top
            return container, xy
        else:
            return container

class normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

class normalize_res2vgg():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class normalize_vgg():
    def __init__(self, mean = (122.675, 116.669, 104.008)):

        self.mean = mean

    def __call__(self, img):
        imgarr = np.asarray(img) #RGB
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 2] - self.mean[2])
        proc_img[..., 1] = (imgarr[..., 1] - self.mean[1])
        proc_img[..., 2] = (imgarr[..., 0] - self.mean[0])

        return proc_img #BGR

def invTrans(img):
    inv_trans= transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    return inv_trans(img)

def denorm(img):
    # ImageNet statistics
    mean_img = [0.485, 0.456, 0.406]
    std_img = [0.229, 0.224, 0.225]

    tf_denorm = transforms.Normalize(mean = [-mean_img[0] / std_img[0], -mean_img[1] / std_img[1], -mean_img[2] / std_img[2]],
                                     std = [1 / std_img[0], 1 / std_img[1], 1 / std_img[2]])

    return tf_denorm(img)

def denorm(img):
    # ImageNet statistics
    mean_img = [0.485, 0.456, 0.406]
    std_img = [0.229, 0.224, 0.225]

    tf_denorm = transforms.Normalize(mean = [-mean_img[0] / std_img[0], -mean_img[1] / std_img[1], -mean_img[2] / std_img[2]],
                                     std = [1 / std_img[0], 1 / std_img[1], 1 / std_img[2]])

    return tf_denorm(img)

def norm(img):
    # ImageNet statistics
    mean_img = [0.485, 0.456, 0.406]
    std_img = [0.229, 0.224, 0.225]

    tf_norm = transforms.Normalize(mean = [mean_img[0], mean_img[1], mean_img[2]],std = [std_img[0],std_img[1], std_img[2]])

    return tf_norm(img)

def norm_vgg(img):
    # ImageNet statistics
    mean_img = [122.675, 116.669, 104.008]
    # std_img = [0.229, 0.224, 0.225]

    tf_norm = transforms.Normalize(mean = [mean_img[0], mean_img[1], mean_img[2]],std = [1,1,1])

    return tf_norm(img)

def norm_gray(img):
    # ImageNet statistics
    mean_img = [0.485, 0.456, 0.406]
    std_img = [0.229, 0.224, 0.225]

    tf_norm = transforms.Normalize(mean = [mean_img[0]],std = [std_img[0]])

    return tf_norm(img)



class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


def HWC_to_CHW(tensor, sal=False):
    if sal:
        tensor = np.expand_dims(tensor, axis=0)
    else:
        tensor = np.transpose(tensor, (2, 0, 1))
    return tensor


def voc_palette(label):
	m = label.astype(np.uint8)
	r,c = m.shape
	cmap = np.zeros((r,c,3), dtype=np.uint8)
	cmap[:,:,0] = (m&1)<<7 | (m&8)<<3
	cmap[:,:,1] = (m&2)<<6 | (m&16)<<2
	cmap[:,:,2] = (m&4)<<5
	cmap[m==255] = [255,255,255]
	return cmap


def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def _crf_with_alpha(cam_dict, img, alpha=10):
        orig_img = np.ascontiguousarray(np.uint8(Image.open(os.path.join('./data/VOC2012/JPEGImages', img + '.jpg'))))
        mean_img = [0.485, 0.456, 0.406]
        std_img = [0.229, 0.224, 0.225]

        # tf_denorm = transforms.Normalize(
        #     mean=[-mean_img[0] / std_img[0], -mean_img[1] / std_img[1], -mean_img[2] / std_img[2]],
        #     std=[1 / std_img[0], 1 / std_img[1], 1 / std_img[2]])

        # img = tf_denorm(img).permute(1,2,0).detach().cpu().numpy()
        # orig_img = np.ascontiguousarray(np.uint8(img*255))

        v = np.array(list(cam_dict.values()))
        bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
        bgcam_score = np.concatenate((bg_score, v), axis=0)
        crf_score = crf_inference_ysh(orig_img, bgcam_score, labels=bgcam_score.shape[0])

        n_crf_al = dict()

        n_crf_al[0] = crf_score[0]
        for i, key in enumerate(cam_dict.keys()):
            n_crf_al[key + 1] = crf_score[i + 1]

        return n_crf_al

def _crf_with_alpha_rrm(ori_img, cam_dict, alpha):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference(np.ascontiguousarray(ori_img), bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = np.zeros([21, bg_score.shape[1], bg_score.shape[2]])
    n_crf_al[0, :, :] = crf_score[0, :, :]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al


def _crf_with_alpha_nodict(cam_dict, img, alpha=10):
    # orig_img = np.ascontiguousarray(np.uint8(Image.open(os.path.join('./data/VOC2012/JPEGImages', img + '.jpg'))))
    mean_img = [0.485, 0.456, 0.406]
    std_img = [0.229, 0.224, 0.225]

    tf_denorm = transforms.Normalize(
        mean=[-mean_img[0] / std_img[0], -mean_img[1] / std_img[1], -mean_img[2] / std_img[2]],
        std=[1 / std_img[0], 1 / std_img[1], 1 / std_img[2]])

    # img = tf_denorm(img).permute(1,2,0).detach().cpu().numpy()
    # orig_img = np.ascontiguousarray(np.uint8(img*255))
    img = img.permute(1,2,0).detach().cpu().numpy()
    orig_img = np.ascontiguousarray(np.uint8(img*255))


    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference_ysh(orig_img, bgcam_score, labels=bgcam_score.shape[0])

    # return crf_score
    n_crf_al = dict()

    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al



import PIL.Image
import random
import numpy as np


def crf_inference_ysh(img, probs, t=10, scale_factor=1, labels=21): #t=10 best
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    # d.addPairwiseGaussian(sxy=1, compat=3)
    # d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=np.copy(img), compat=4)
    # d.addPairwiseGaussian(sxy=3, compat=3)
    # d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=np.copy(img), compat=4)

    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))



def crf_dl(name, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax



    image_path = os.path.join('./data/VOC2012/JPEGImages', name + '.jpg')
    img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
    # img = img - np.array([104.008,116.669,122.675]) #Not used
    img = np.ascontiguousarray(np.uint8(img))

    # img_temp = cv2.imread(image_path)
    # img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2RGB).astype(np.float)
    # img = img_temp.astype(np.uint8)

    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)

    # d.addPairwiseGaussian(sxy=1, compat=3)  # 64.85
    # d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=np.copy(img), compat=4)  # 64.85

    d.addPairwiseGaussian(sxy=1, compat=3)  # 64.86/no normal
    d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=np.copy(img), compat=4)  #64.86/no normal

    # d.addPairwiseGaussian(sxy=3, compat=3)  #RRM
    # d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=np.copy(img), compat=10)  #RRM t=10

    # d.addPairwiseGaussian(sxy=1, compat=3) #good
    # d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=np.copy(img), compat=4) #good

    # d.addPairwiseGaussian(sxy=2, compat=3,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
    # d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=np.copy(img),kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC,compat=4) #good

    # d.addPairwiseGaussian(sxy=2, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    # d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=np.copy(img), kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC, compat=4)  # good


    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


