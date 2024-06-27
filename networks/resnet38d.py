import torch
from torch import nn
import numpy as np
import pdb
# from torch.nn.common_types import T

import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, first_dilation=None, dilation=1):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        if first_dilation == None: first_dilation = dilation

        self.bn_branch2a = nn.BatchNorm2d(in_channels)

        self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride,
                                       padding=first_dilation, dilation=first_dilation, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(mid_channels)

        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False, start_imd = False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.conv_branch2b1(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)

class ResBlock_bot(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.):
        super(ResBlock_bot, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        self.bn_branch2a = nn.BatchNorm2d(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, out_channels//4, 1, stride, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(out_channels//4)
        self.dropout_2b1 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b1 = nn.Conv2d(out_channels//4, out_channels//2, 3, padding=dilation, dilation=dilation, bias=False)

        self.bn_branch2b2 = nn.BatchNorm2d(out_channels//2)
        self.dropout_2b2 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b2 = nn.Conv2d(out_channels//2, out_channels, 1, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False,start_imd=False):
        if not start_imd:
            branch2 = self.bn_branch2a(x)
            branch2 = F.relu(branch2)
        else:
            branch2 = x
        x_bn_relu = branch2

        branch1 = self.conv_branch1(branch2)

        branch2 = self.conv_branch2a(branch2)

        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        branch2 = self.bn_branch2b2(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False, start_imd=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu,start_imd=start_imd)

class Normalize():
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

def convert_mxnet_to_torch(filename):
    import mxnet

    save_dict = mxnet.nd.load(filename)

    renamed_dict = dict()

    bn_param_mx_pt = {'beta': 'bias', 'gamma': 'weight', 'mean': 'running_mean', 'var': 'running_var'}

    for k, v in save_dict.items():

        v = torch.from_numpy(v.asnumpy())
        toks = k.split('_')

        if 'conv1a' in toks[0]:
            renamed_dict['conv1a.weight'] = v

        elif 'linear1000' in toks[0]:
            pass

        elif 'branch' in toks[1]:

            pt_name = []

            if toks[0][-1] != 'a':
                pt_name.append('b' + toks[0][-3] + '_' + toks[0][-1])
            else:
                pt_name.append('b' + toks[0][-2])

            if 'res' in toks[0]:
                layer_type = 'conv'
                last_name = 'weight'

            else:  # 'bn' in toks[0]:
                layer_type = 'bn'
                last_name = bn_param_mx_pt[toks[-1]]

            pt_name.append(layer_type + '_' + toks[1])

            pt_name.append(last_name)

            torch_name = '.'.join(pt_name)
            renamed_dict[torch_name] = v

        else:
            last_name = bn_param_mx_pt[toks[-1]]
            renamed_dict['bn7.' + last_name] = v

    return renamed_dict

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


class Net_CAM(nn.Module):
    def __init__(self, D=256, C=20):
        super(Net_CAM, self).__init__()
        
        self.D = D
        self.C = C

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.2)
        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.2)
        self.bn7 = nn.BatchNorm2d(4096)

        self.fc8 = nn.Conv2d(4096, D, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(D)
        self.fc9 = nn.Conv2d(D, C, 1, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.3)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.fc9.weight)

        self.from_scratch_layers = []
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

    def forward(self, x):

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x = self.b4(x)  # B x 512 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)

        x = self.b4_4(x)
        x = self.b4_5(x)

        x = self.b5(x)  # B x 1024 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x = self.b6(x)
        x = self.b7(x)

        x = F.relu(self.bn7(x))

        feat = F.relu(self.fc8(x))  # B x D x 56 x 56
        cam = self.fc9(feat)
        pred = self.avgpool(cam).squeeze(3).squeeze(2)
        
        out = {}
        out['feat'] = feat
        out['cam'] = cam
        out['pred'] = pred        

        return out


    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    if not isinstance(c, torch.nn.Dropout2d):
                        c.weight.requires_grad = False
                        if c.bias is not None:
                            c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups




class Net_CAM_seg(nn.Module):
    def __init__(self, D=256, C=20):
        super(Net_CAM_seg, self).__init__()
        
        self.D = D
        self.C = C

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.2)
        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.2)
        self.bn7 = nn.BatchNorm2d(4096)

        self.fc8 = nn.Conv2d(4096, D, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(D)
        self.fc9 = nn.Conv2d(D, C, 1, bias=False)

        self.fc8_seg = nn.Conv2d(4096, 512, 3, dilation=12, padding=12, bias=True)
        self.fc9_seg = nn.Conv2d(512, 21, 3, dilation=12, padding=12, bias=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.3)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.fc9.weight)

        tt = torch.get_rng_state()

        torch.nn.init.xavier_uniform_(self.fc8_seg.weight)
        torch.nn.init.xavier_uniform_(self.fc9_seg.weight)

        torch.set_rng_state(tt)

        self.from_scratch_layers = []
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

    def forward(self, x):

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x = self.b4(x)  # B x 512 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)

        x = self.b4_4(x)
        x = self.b4_5(x)

        x = self.b5(x)  # B x 1024 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x = self.b6(x)
        x = self.b7(x)

        x = F.relu(self.bn7(x))

        feat = F.relu(self.fc8(x))  # B x D x 56 x 56
        cam = self.fc9(feat)
        pred = self.avgpool(cam).squeeze(3).squeeze(2)

        # seg branch
        x_seg = F.relu(self.fc8_seg(x))#originally no inplace
        x_seg = self.fc9_seg(x_seg)#originally no inplace

        
        out = {}
        out['feat'] = feat
        out['cam'] = cam
        out['pred'] = pred
        out['seg'] = x_seg        

        return out


    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    if not isinstance(c, torch.nn.Dropout2d):
                        c.weight.requires_grad = False
                        if c.bias is not None:
                            c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups






class Net_gpp(nn.Module):
    def __init__(self, D=256, C=20):
        super(Net_gpp, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.2)
        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.2)
        self.bn7 = nn.BatchNorm2d(4096)

        self.fc8 = nn.Conv2d(4096, D, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(D)
        self.fc9 = nn.Conv2d(D, C, 1, bias=False)
        
        self.gconv1 = gconv(20,20)
        self.gconv2 = gconv(20,20)
        self.gconv3 = gconv(20,20)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.3)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.fc9.weight)

        self.from_scratch_layers = []
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

    def forward(self, x):

        B,_,H,W = x.size()
        
        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x = self.b4(x)  # B x 512 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)

        x = self.b4_4(x)
        x = self.b4_5(x)

        x = self.b5(x)  # B x 1024 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)


        x,b5 = self.b6(x,get_x_bn_relu=True)
        x = self.b7(x)

        x = F.relu(self.bn7(x))

        feat = F.relu(self.fc8(x))  # B x D x 56 x 56 #기존
        cam = self.fc9(feat)
        
        gap_2 = F.interpolate(F.adaptive_avg_pool2d(cam,(2,2)),size=cam.size()[2:],mode='bilinear',align_corners=False)
        gap_4 = F.interpolate(F.adaptive_avg_pool2d(cam,(4,4)),size=cam.size()[2:],mode='bilinear',align_corners=False)
        gap_8 = F.interpolate(F.adaptive_avg_pool2d(cam,(8,8)),size=cam.size()[2:],mode='bilinear',align_corners=False)
        gap_16 = F.interpolate(F.adaptive_avg_pool2d(cam,(16,16)),size=cam.size()[2:],mode='bilinear',align_corners=False)
        
        att1 = self.gconv1(gap_2,gap_4)
        att2 = self.gconv2(att1,gap_8)
        gpp = self.gconv3(att2,gap_16)
        
        outs = self.avgpool(F.relu(cam)*F.relu(gpp)).squeeze(3).squeeze(2)
        outs -= self.avgpool(F.relu(-cam)*F.relu(-gpp)).squeeze(3).squeeze(2)

        out = {}
        out['feat'] = feat
        out['cam'] = cam
        out['pred'] = outs        

        return out

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


class gconv(nn.Module):
    def __init__(self,in_ch1,in_ch2):
        super(gconv,self).__init__()
        
        self.att_p = nn.Sequential(
                    nn.Conv2d(in_ch1*2, 2, 3,1,1,bias=False),
                    nn.Sigmoid(),
                )
        self.att_n = nn.Sequential(
                    nn.Conv2d(in_ch1*2, 2, 3,1,1,bias=False),
                    nn.Sigmoid(),
                )
    
    def forward(self,x,y):
        n,c,h,w = y.size()
        xy_p = torch.cat([F.relu(x),F.relu(y)],dim=1)
        xy_n = torch.cat([F.relu(-x),F.relu(-y)],dim=1)
        att_p = self.att_p(xy_p)
        att_n = self.att_n(xy_n)
        final = (F.relu(x)*att_p[:,0].view(n,1,h,w)+F.relu(y)*att_p[:,1].view(n,1,h,w))/2
        final -=(F.relu(-x)*att_n[:,0].view(n,1,h,w)+F.relu(-y)*att_n[:,1].view(n,1,h,w))/2

        return final




class Net_reCAM(nn.Module):
    def __init__(self, D=256, C=20):
        super(Net_reCAM, self).__init__()
        
        self.D = D
        self.C = C

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.2)
        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.2)
        self.bn7 = nn.BatchNorm2d(4096)

        self.fc8 = nn.Conv2d(4096, D, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(D)
        self.fc9 = nn.Conv2d(D, C, 1, bias=False)
        self.fc9_re = nn.Conv2d(D, C, 1, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.3)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.fc9.weight)

        self.from_scratch_layers = []
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

    def forward(self, x):

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x = self.b4(x)  # B x 512 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)

        x = self.b4_4(x)
        x = self.b4_5(x)

        x = self.b5(x)  # B x 1024 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x = self.b6(x)
        x = self.b7(x)

        x = F.relu(self.bn7(x))

        feat = F.relu(self.fc8(x))  # (B,D,H,W)
        cam = self.fc9(feat) # (B,C,H,W)
        pred = self.avgpool(cam).squeeze(3).squeeze(2)

        recam_feat = cam.unsqueeze(2)*feat.unsqueeze(1) # (B,C,D,H,W)

        B,C,D,H,W = recam_feat.shape
        recam_feat = recam_feat.view(B*C,D,H,W)

        recam = self.fc9_re(recam_feat) # (B*C,C,H,W)
        recam_pred = self.avgpool(recam).squeeze(3).squeeze(2) # (B*C,C)
        recam_pred = recam_pred.view(B,C,C) # (B,C,C)
        
        
        out = {}
        out['feat'] = feat
        out['cam'] = cam
        out['pred'] = pred
        out['recam_pred'] = recam_pred

        return out


    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    if not isinstance(c, torch.nn.Dropout2d):
                        c.weight.requires_grad = False
                        if c.bias is not None:
                            c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


class Net_CAM_sep(nn.Module):
    def __init__(self, D=256, C=20):
        super(Net_CAM_sep, self).__init__()
        
        self.D = D
        self.C = C

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.2)
        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.2)
        self.bn7 = nn.BatchNorm2d(4096)

        self.fc8 = nn.Conv2d(4096, D, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(D)
        self.fc9 = nn.Conv2d(D, C, 1, bias=False)

        self.fc_sep = nn.Conv2d(D, D//2, 1, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.3)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.fc9.weight)
        torch.nn.init.xavier_uniform_(self.fc_sep.weight)

        self.from_scratch_layers = []
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

    def forward(self, x):

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x = self.b4(x)  # B x 512 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)

        x = self.b4_4(x)
        x = self.b4_5(x)

        x = self.b5(x)  # B x 1024 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x = self.b6(x)
        x = self.b7(x)

        x = F.relu(self.bn7(x))

        feat = F.relu(self.fc8(x))  # B x D x 56 x 56
        cam = self.fc9(feat)
        pred = self.avgpool(cam).squeeze(3).squeeze(2)

        feat = self.fc_sep(feat)
        
        out = {}
        out['feat'] = feat
        out['cam'] = cam
        out['pred'] = pred        

        return out


    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    if not isinstance(c, torch.nn.Dropout2d):
                        c.weight.requires_grad = False
                        if c.bias is not None:
                            c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups




class Net_CAM_SAM(nn.Module):
    def __init__(self, D=256, C=20):
        super(Net_CAM_SAM, self).__init__()
        
        self.D = D
        self.C = C

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.2)
        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.2)
        self.bn7 = nn.BatchNorm2d(4096)

        self.fc8 = nn.Conv2d(4096, D, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(D)
        self.fc9 = nn.Conv2d(D, C, 1, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.3)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.fc9.weight)

        self.from_scratch_layers = []
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        
    
    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    if not isinstance(c, torch.nn.Dropout2d):
                        c.weight.requires_grad = False
                        if c.bias is not None:
                            c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups

    def forward(self, x, se):

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x = self.b4(x)  # B x 512 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)

        x = self.b4_4(x)
        x = self.b4_5(x)

        x = self.b5(x)  # B x 1024 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x = self.b6(x)
        x = self.b7(x)

        x = F.relu(self.bn7(x))

        feat = F.relu(self.fc8(x))  # B x D x 56 x 56
        # feat = F.normalize(feat, dim=1)        
        
        cam = self.fc9(feat)
        pred = self.avgpool(cam).squeeze(3).squeeze(2)
        
        feat = F.interpolate(feat, (se.shape[2], se.shape[3]), mode='bilinear', align_corners=False)
        feat_se = torch.zeros_like(feat)
        for idx in range(se.shape[0]):
            for jdx, sdx in enumerate(se[idx,0].unique()):
                if sdx==0 or jdx<2:
                    continue
                else:
                    mask = se[idx,0]==sdx
                    feat_temp = feat[idx,:,mask]
                    feat_se[idx,:,mask] = feat_temp.mean(dim=-1,keepdim=True)

        cam_se = self.fc9(feat_se)
        pred_se = self.avgpool(cam_se).squeeze(3).squeeze(2)
        
        out = {}
        out['feat'] = feat
        out['feat_se'] = feat_se
        out['cam'] = cam
        out['cam_se'] = cam_se
        out['pred'] = pred        
        out['pred_se'] = pred_se

        return out






class Net_eps(nn.Module):
    def __init__(self, D=256, C=20):
        super(Net_eps, self).__init__()
        
        self.D = D
        self.C = C

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.2)
        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.2)
        self.bn7 = nn.BatchNorm2d(4096)

        self.fc8 = nn.Conv2d(4096, D, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(D)
        self.fc9 = nn.Conv2d(D, C, 1, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.3)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.fc9.weight)

        self.from_scratch_layers = []
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

    def reset_cls_head(self):
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2, self.b3, self.b3_1, self.b3_2,
                             self.b4, self.b4_1, self.b4_2, self.b4_3, self.b4_4, self.b4_5,
                             self.b5, self.b5_1, self.b5_2, self.b6, self.b7, self.bn7, self.fc8, self.bn8]
        self.fc9 = nn.Conv2d(self.D, self.C, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc9.weight)

    def forward(self, x):

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x = self.b4(x)  # B x 512 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)

        x = self.b4_4(x)
        x = self.b4_5(x)

        x = self.b5(x)  # B x 1024 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x = self.b6(x)
        x = self.b7(x)

        x = F.relu(self.bn7(x))

        feat = F.relu(self.fc8(x))  # B x D x 56 x 56
        cam = self.fc9(feat)
        pred = self.avgpool(cam).squeeze(3).squeeze(2)

        return feat, cam, pred


    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    if not isinstance(c, torch.nn.Dropout2d):
                        c.weight.requires_grad = False
                        if c.bias is not None:
                            c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups



class Net_dl(nn.Module):
    def __init__(self):
        super(Net_dl, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5)
        self.bn7 = nn.BatchNorm2d(4096)

        self.dropout = nn.Dropout2d(0.5)

        self.b8 = nn.Conv2d(4096, 512, 3, dilation=12, padding=12, bias=True)

        self.b9 = nn.Conv2d(512, 21, 3, dilation=12, padding=12, bias=True)

        self.not_training = []  # self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.b8, self.b9] #original version on

        torch.nn.init.xavier_uniform_(self.b8.weight)
        torch.nn.init.xavier_uniform_(self.b9.weight)

        self.normalize = Normalize()

    def forward(self, x):

        _, _, H, W = x.shape

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        # x = self.b4(x)
        x, conv3 = self.b4(x, get_x_bn_relu=True)  # B x 256 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x, get_x_bn_relu=True)  # B x 512 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)  # B x 1024 x 56 x 56

        x = self.b7(x)
        x = F.relu(self.bn7(x))#originally no inplace
        # x = self.dropout(x)

        x = F.relu(self.b8(x))#originally no inplace

        x = self.b9(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        return x

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5)

        self.bn7 = nn.BatchNorm2d(4096)

        self.not_training = [self.conv1a]

        self.normalize = Normalize()

        return

    def forward(self, x):
        return self.forward_as_dict(x)['conv6']

    def forward_as_dict(self, x):

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        #x = self.b4(x)
        x, conv3 = self.b4(x, get_x_bn_relu=True)  # B x 256 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x, get_x_bn_relu=True)  # B x 512 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)  # B x 1024 x 56 x 56

        x = self.b7(x)
        conv6 = F.relu(self.bn7(x)) # B x 4096 x 56 x 56

        return dict({'conv3': conv3, 'conv4': conv4, 'conv5': conv5, 'conv6': conv6})

    def forward_conv5(self, x):

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        #x = self.b4(x)
        x = self.b4(x)  # B x 256 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x = self.b5(x)  # B x 512 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)  # B x 1024 x 56 x 56

        return x, conv5

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return