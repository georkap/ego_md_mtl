# -*- coding: utf-8 -*-
"""
Created on 18-9-19 by georkap
This is an extension of the Multi Fiber network for multiple arbitrary outputs

Original Author: Yunpeng Chen
https://github.com/cypw/PyTorch-MFNet/blob/master/network/mfnet_3d.py

mo stands for multiple output

@author: Γιώργος
"""
from collections import OrderedDict
import torch.nn as nn

from src.models.layers import CoordRegressionLayer, MultitaskClassifiers, ObjectPresenceLayer, MF_UNIT, get_norm_layers
from src.utils.initializer import xavier
from torch.functional import F

class MFNET_3D_MO_COMB(nn.Module):
    def __init__(self, num_classes, dropout=None, **kwargs):
        super(MFNET_3D_MO_COMB, self).__init__()
        # support for arbitrary number of output layers, but it is the user's job to make sure they make sense
        # (e.g. actions->actions and not actions->verbs,nouns etc.)
        self.num_classes = num_classes[0:4] # to remove egtea VN tasks
        self.num_coords = kwargs.get('num_coords', 0) - 2 # to remove one of the hand tasks
        self.num_objects = kwargs.get('num_objects', None)
        self.num_obj_cat = kwargs.get('num_obj_cat', None)
        self.one_object_layer = kwargs.get('one_object_layer', False)
        self.norm = kwargs.get('norm', 'BN') # else 'GN' or 'IN'
        self.ensemble_eval = kwargs.get('ensemble_eval', False)
        self.t_dim_in = kwargs.get('num_frames', 16)
        self.s_dim_in = kwargs.get('spatial_size', 224)
        self.interpolate_coords = kwargs.get('interpolate_coordinates', 1)
        in_ch = kwargs.get('input_channels', 3)
        groups = 16
        # k_sec = {2: 3, 3: 4, 4: 6, 5: 3}
        k_sec = kwargs.get('k_sec', {2: 3, 3: 4, 4: 6, 5: 3})

        c1_out = 16
        c2_out = 96
        c3_out = 2 * c2_out
        c4_out = 2 * c3_out
        c5_out = 2 * c4_out

        conv1normlayer, tailnorm = get_norm_layers(self.norm, c1_out, c5_out)

        # conv1 - x224 (x16)
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(in_ch, c1_out, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2), bias=False)),
            conv1normlayer,
            ('relu', nn.ReLU(inplace=True))
        ]))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # conv2 - x56 (x8)
        num_mid = 96

        self.conv2 = nn.Sequential(OrderedDict([
            ("B%02d" % i, MF_UNIT(num_in=c1_out if i == 1 else c2_out, num_mid=num_mid, num_out=c2_out,
                                  stride=(2, 1, 1) if i == 1 else (1, 1, 1), g=groups, first_block=(i == 1),
                                  norm=self.norm)) for i in range(1, k_sec[2]+1)
        ]))

        # conv3 - x28 (x8)
        num_mid *= 2
        self.conv3 = nn.Sequential(OrderedDict([
            ("B%02d" % i, MF_UNIT(num_in=c2_out if i == 1 else c3_out, num_mid=num_mid, num_out=c3_out,
                                  stride=(1, 2, 2) if i == 1 else (1, 1, 1), g=groups, first_block=(i == 1),
                                  norm=self.norm)) for i in range(1, k_sec[3]+1)
        ]))

        # conv4 - x14 (x8)
        num_mid *= 2
        self.conv4 = nn.Sequential(OrderedDict([
            ("B%02d" % i, MF_UNIT(num_in=c3_out if i == 1 else c4_out, num_mid=num_mid, num_out=c4_out,
                                  stride=(1, 2, 2) if i == 1 else (1, 1, 1), g=groups, first_block=(i == 1),
                                  norm=self.norm)) for i in range(1, k_sec[4]+1)
        ]))

        # conv5 - x7 (x8)
        num_mid *= 2
        self.conv5 = nn.Sequential(OrderedDict([
            ("B%02d" % i, MF_UNIT(num_in=c4_out if i == 1 else c5_out, num_mid=num_mid, num_out=c5_out,
                                  stride=(1, 2, 2) if i == 1 else (1, 1, 1), g=groups, first_block=(i == 1),
                                  norm=self.norm)) for i in range(1, k_sec[5]+1)
        ]))

        # create heatmaps
        if self.num_coords > 0:
            self.coord_layers = CoordRegressionLayer(c5_out, self.num_coords, self.interpolate_coords)

        # final
        self.tail = nn.Sequential(OrderedDict([tailnorm, ('relu', nn.ReLU(inplace=True))]))

        self.globalpool = nn.Sequential()
        self.globalpool.add_module('avg', nn.AvgPool3d(kernel_size=(self.t_dim_in // 2, self.s_dim_in // 32,
                                                                    self.s_dim_in // 32), stride=(1, 1, 1)))

        if dropout:
            self.globalpool.add_module('dropout', nn.Dropout(p=dropout))

        self.classifier_list = MultitaskClassifiers(c5_out, self.num_classes)

        if self.num_objects:
            for ii, no in enumerate(self.num_objects): # if there are more than one object presence layers, e.g. one per dataset
                object_presence_layer = ObjectPresenceLayer(c5_out, no, one_layer=self.one_object_layer)
                self.add_module('object_presence_layer_{}'.format(ii), object_presence_layer)
        if self.num_obj_cat:
            for ii, no in enumerate(self.num_obj_cat):
                object_presence_layer = ObjectPresenceLayer(c5_out, no, one_layer=self.one_object_layer)
                self.add_module('objcat_presence_layer_{}'.format(ii), object_presence_layer)
        #############
        # Initialization
        xavier(net=self)

    def forward(self, x):
        assert x.shape[2] == 16

        h = self.conv1(x)   # x224 -> x112
        # print(h.shape)
        h = self.maxpool(h)  # x112 ->  x56
        # print(h.shape)
        h = self.conv2(h)  # x56 ->  x56
        # print(h.shape)
        h = self.conv3(h)  # x56 ->  x28
        # print(h.shape)
        h = self.conv4(h)  # x28 ->  x14
        # print(h.shape)
        h = self.conv5(h)  # x14 ->   x7
        # print(h.shape)

        h = self.tail(h)
        coords, heatmaps, probabilities = None, None, None
        if self.num_coords > 0:
            coords, heatmaps, probabilities = self.coord_layers(h)

        if not self.training and self.ensemble_eval: # not fully supported yet
            h_ens = F.avg_pool3d(h, (1, self.s_dim_in//32, self.s_dim_in//32), (1, 1, 1))
            h_ens = h_ens.view(h_ens.shape[0], h_ens.shape[1], -1)
            h_ens = [self.classifier_list(h_ens[:, :, ii]) for ii in range(h_ens.shape[2])]

        h = self.globalpool(h)
        h = h.view(h.shape[0], -1)
        h_out = self.classifier_list(h)

        objects = None
        if self.num_objects:
            objects = [self.__getattr__('object_presence_layer_{}'.format(ii))(h) for ii in range(len(self.num_objects))]
        cat_obj = None
        if self.num_obj_cat:
            cat_obj = [self.__getattr__('objcat_presence_layer_{}'.format(ii))(h) for ii in range(len(self.num_obj_cat))]
        if not self.training and self.ensemble_eval:
            h_out = [h_out, h_ens]

        return h_out, coords, heatmaps, probabilities, objects, cat_obj

if __name__ == "__main__":
    import torch, time
    # ---------
    kwargs = {'num_coords': 5, 'num_objects': None, 'num_obj_cat': None, 'one_object_layer': True,
              'ensemble_eval': False}
    net = MFNET_3D_MO_COMB(num_classes=[2513, 125, 352, 106, 19, 53], dropout=0.5, **kwargs)
    data = torch.randn(1, 3, 16, 224, 224, requires_grad=True)
    net.cuda()
    data = data.cuda()
    net.eval()
    # loss = torch.tensor([10]).cuda()
    output = net(data)
    t0 = time.time()
    for i in range(10):
        output = net(data)
    t1 = time.time()
    print('forward time:', t1 - t0)
    # h, htail = net.forward_shared_block(data)
    # coords, heatmaps, probabilities = net.forward_coord_layers(htail)
    # output = net.forward_cls_layers(h)
   # torch.save({'state_dict': net.state_dict()}, './tmp.pth')
   #  print(len(output))
