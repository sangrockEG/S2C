from cProfile import label
import os
import sys
import time
import visdom
import numpy as np
from tools import  utils
import pdb

class Visualizer():

    def __init__(self, port, legends_loss, legends_acc=None):

        self.vis = visdom.Visdom(port=port)
        self.data_loss = {'X': [], 'Y': [], 'legend': legends_loss}
        if legends_acc is not None:
            self.data_acc = {'X': [], 'Y': [], 'legend': legends_acc}


    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def plot_loss(self, epoch, losses):

        self.data_loss['X'].append(epoch)
        self.data_loss['Y'].append([losses[i] for i in range(len(losses))])

        self.vis.line(
            X=np.stack([np.array(self.data_loss['X'])] * len(self.data_loss['legend']), 1),
            Y=np.array(self.data_loss['Y']),
            opts={
                'title': 'loss over time',
                'legend': self.data_loss['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=0)
    
    def plot_acc(self, epoch, accs):

        self.data_acc['X'].append(epoch)
        self.data_acc['Y'].append([accs[i] for i in range(len(accs))])

        self.vis.line(
            X=np.stack([np.array(self.data_acc['X'])] * len(self.data_acc['legend']), 1),
            Y=np.array(self.data_acc['Y']),
            opts={
                'title': 'acc over time',
                'legend': self.data_acc['legend'],
                'xlabel': 'epoch',
                'ylabel': 'acc'},
            win=1)

    def plot_mask(self, imgs, imgs_remain, masks, mask_remain, labels, all_preds, remain_preds, reset=True):
        if reset:
            self.vis.close(env="mask")
        # self.vis.text(labels, env="mask")
        self.vis.images(imgs, opts=dict(title='image', caption=all_preds), env="mask")
        self.vis.images(masks, opts=dict(title='mask', caption=labels), env="mask")
        self.vis.images(imgs_remain, opts=dict(title='image remain', caption=remain_preds), env="mask")
        self.vis.images(mask_remain, opts=dict(title='mask remain', caption=labels), env="mask")

    def plot_mask_gpp(self, imgs, cams, I_P, t_H, t_L, labels, all_preds, remain_preds, reset=True):
        if reset:
            self.vis.close(env="mask")
        # self.vis.text(labels, env="mask")
        self.vis.images(imgs, opts=dict(title='image', caption=all_preds), env="mask")
        self.vis.images(cams, opts=dict(title='cam', caption=labels), env="mask")
        self.vis.images(I_P, opts=dict(title='gpp', caption=labels), env="mask")
        self.vis.images(t_H, opts=dict(title='image remain', caption=remain_preds), env="mask")
        self.vis.images(t_L, opts=dict(title='cam remain', caption=labels), env="mask")
    
    def plot_mask_5cams(self, imgs, cam, cam0, cam1, cam2, cam3, labels, all_preds, remain_preds, reset=True):
        if reset:
            self.vis.close(env="mask")
        # self.vis.text(labels, env="mask")
        self.vis.images(imgs, opts=dict(title='image', caption=all_preds), env="mask")
        self.vis.images(cam, opts=dict(title='cam', caption=labels), env="mask")
        self.vis.images(cam0, opts=dict(title='cam0', caption=labels), env="mask")
        self.vis.images(cam1, opts=dict(title='cam1', caption=labels), env="mask")
        self.vis.images(cam2, opts=dict(title='cam2', caption=labels), env="mask")
        self.vis.images(cam3, opts=dict(title='cam3', caption=labels), env="mask")
    
    def plot_mask_recon(self, imgs, cam, imgs_recon_masked, cam_recon, imgs_recon_fg, cam_recon_fg,imgs_recon_bg, cam_recon_bg, labels, original_preds, fg_preds, bg_pred,reset=True):
        if reset:
            self.vis.close(env="mask")
        self.vis.images(imgs, opts=dict(title='image', caption=original_preds), env="mask")
        self.vis.images(cam, opts=dict(title='cam', caption=labels), env="mask")
        self.vis.images(imgs_recon_masked, opts=dict(title='image', caption=original_preds), env="mask")
        self.vis.images(cam_recon, opts=dict(title='cam', caption=labels), env="mask")
        self.vis.images(imgs_recon_fg, opts=dict(title='image', caption=fg_preds), env="mask")
        self.vis.images(cam_recon_fg, opts=dict(title='cam', caption=labels), env="mask")
        self.vis.images(imgs_recon_bg, opts=dict(title='image', caption=bg_pred), env="mask")
        self.vis.images(cam_recon_bg, opts=dict(title='cam', caption=labels), env="mask")