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
import torch
import torch.nn as nn

from src.models.layers import CoordRegressionLayer, MultitaskDFBClassifiers, MultitaskClassifiers, ObjectPresenceLayer, MF_UNIT, get_norm_layers
from src.utils.initializer import xavier
from torch.functional import F

class MFNET_3D_DFB(nn.Module):
    def __init__(self, num_classes, dropout=None, **kwargs):
        super(MFNET_3D_DFB, self).__init__()
        # support for arbitrary number of output layers, but it is the user's job to make sure they make sense
        # (e.g. actions->actions and not actions->verbs,nouns etc.)
        self.num_classes = num_classes
        self.num_coords = kwargs.get('num_coords', 0)
        self.num_objects = kwargs.get('num_objects', None)
        self.num_obj_cat = kwargs.get('num_obj_cat', None)
        self.one_object_layer = kwargs.get('one_object_layer', False)
        self.norm = kwargs.get('norm', 'BN') # else 'GN' or 'IN'
        self.ensemble_eval = kwargs.get('ensemble_eval', False)
        self.t_dim_in = kwargs.get('num_frames', 16)
        self.s_dim_in = kwargs.get('spatial_size', 224)
        input_channels = kwargs.get('input_channels', 3)
        groups = 16
        k_sec = kwargs.get('k_sec', {2: 3, 3: 4, 4: 6, 5: 3})

        conv1_num_out = 16
        conv2_num_out = 96
        conv3_num_out = 2 * conv2_num_out
        conv4_num_out = 2 * conv3_num_out
        conv5_num_out = 2 * conv4_num_out

        conv1normlayer, tailnorm = get_norm_layers(self.norm, conv1_num_out, conv5_num_out)

        # conv1 - x224 (x16)
        self.conv1 = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv3d(input_channels, conv1_num_out, kernel_size=(3, 5, 5), padding=(1, 2, 2),
                                       stride=(1, 2, 2), bias=False)),
                    conv1normlayer,
                    ('relu', nn.ReLU(inplace=True))
                    ]))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # conv2 - x56 (x8)
        num_mid = 96

        self.conv2 = nn.Sequential(
            OrderedDict([("B%02d" % i, MF_UNIT(num_in=conv1_num_out if i == 1 else conv2_num_out,
                                               num_mid=num_mid,
                                               num_out=conv2_num_out,
                                               stride=(2, 1, 1) if i == 1 else (1, 1, 1),
                                               g=groups,
                                               first_block=(i == 1),
                                               norm=self.norm)) for i in range(1, k_sec[2]+1)])
        )

        # conv3 - x28 (x8)
        num_mid *= 2
        self.conv3 = nn.Sequential(
            OrderedDict([("B%02d" % i, MF_UNIT(num_in=conv2_num_out if i == 1 else conv3_num_out,
                                               num_mid=num_mid,
                                               num_out=conv3_num_out,
                                               stride=(1, 2, 2) if i == 1 else (1, 1, 1),
                                               g=groups,
                                               first_block=(i == 1),
                                               norm=self.norm)) for i in range(1, k_sec[3]+1)])
        )

        # conv4 - x14 (x8)
        num_mid *= 2
        self.conv4 = nn.Sequential(
            OrderedDict([("B%02d" % i, MF_UNIT(num_in=conv3_num_out if i == 1 else conv4_num_out,
                                               num_mid=num_mid,
                                               num_out=conv4_num_out,
                                               stride=(1, 2, 2) if i == 1 else (1, 1, 1),
                                               g=groups,
                                               first_block=(i == 1),
                                               norm=self.norm)) for i in range(1, k_sec[4]+1)])
        )

        # conv5 - x7 (x8)
        num_mid *= 2
        self.conv5 = nn.Sequential(
            OrderedDict([("B%02d" % i, MF_UNIT(num_in=conv4_num_out if i == 1 else conv5_num_out,
                                               num_mid=num_mid,
                                               num_out=conv5_num_out,
                                               stride=(1, 2, 2) if i == 1 else (1, 1, 1),
                                               g=groups,
                                               first_block=(i == 1),
                                               norm=self.norm)) for i in range(1, k_sec[5]+1)])
        )
        self.conv5_2 = nn.Sequential(
            OrderedDict([("B%02d" % i, MF_UNIT(num_in=conv4_num_out if i == 1 else conv5_num_out,
                                               num_mid=num_mid,
                                               num_out=conv5_num_out,
                                               stride=(1, 2, 2) if i == 1 else (1, 1, 1),
                                               g=groups,
                                               first_block=(i == 1),
                                               norm=self.norm)) for i in range(1, k_sec[5] + 1)])
        )

        # create heatmaps
        if self.num_coords > 0:
            self.coord_layers = CoordRegressionLayer(conv5_num_out, self.num_coords)

        # final
        self.tail = nn.Sequential(OrderedDict([tailnorm, ('relu', nn.ReLU(inplace=True))]))

        pooling_kernel_avg = (self.t_dim_in // 2, self.s_dim_in // 32, self.s_dim_in // 32)
        pooling_kernel_max = (self.t_dim_in // 2, self.s_dim_in // 16, self.s_dim_in // 16)
        self.globalpool = nn.Sequential()
        self.globalpool.add_module('avg', nn.AvgPool3d(kernel_size=pooling_kernel_avg, stride=(1, 1, 1)))

        if dropout:
            self.globalpool.add_module('dropout', nn.Dropout(p=dropout))

        self.classifier_list = MultitaskClassifiers(conv5_num_out, num_classes)

        self.dfb_classifier_list = MultitaskDFBClassifiers(conv5_num_out+conv4_num_out, num_classes, pooling_kernel_max,
                                                           dropout)

        # if self.num_objects:
        #     for ii, no in enumerate(self.num_objects): # if there are more than one object presence layers, e.g. one per dataset
        #         object_presence_layer = ObjectPresenceLayer(conv5_num_out, no, one_layer=self.one_object_layer)
        #         self.add_module('object_presence_layer_{}'.format(ii), object_presence_layer)
        # if self.num_obj_cat:
        #     for ii, no in enumerate(self.num_obj_cat):
        #         object_presence_layer = ObjectPresenceLayer(conv5_num_out, no, one_layer=self.one_object_layer)
        #         self.add_module('objcat_presence_layer_{}'.format(ii), object_presence_layer)
        #############
        # Initialization
        xavier(net=self)

    def forward(self, x):
        h = self.conv1(x)   # x224 -> x112
        h = self.maxpool(h)  # x112 ->  x56
        h = self.conv2(h)  # x56 ->  x56
        h = self.conv3(h)  # x56 ->  x28
        h = self.conv4(h)  # x28 ->  x14

        # local branch
        h2 = self.conv5_2(h)
        h2 = F.interpolate(h2, scale_factor=(1, 2, 2), mode='trilinear')
        h2 = torch.cat([h, h2], dim=1)

        h = self.conv5(h)  # x14 ->   x7
        h = self.tail(h)
        coords, heatmaps, probabilities = None, None, None
        # if self.num_coords > 0:
        #     coords, heatmaps, probabilities = self.coord_layers(h)

        # if not self.training and self.ensemble_eval: # not fully supported yet
        #     h_ens = F.avg_pool3d(h, (1, self.s_dim_in//32, self.s_dim_in//32), (1, 1, 1))
        #     h_ens = h_ens.view(h_ens.shape[0], h_ens.shape[1], -1)
        #     h_ens = [self.classifier_list(h_ens[:, :, ii]) for ii in range(h_ens.shape[2])]

        h_ch, h_max = self.dfb_classifier_list(h2)
        # h_ch = self.dfb_classifier_list(h)

        h = self.globalpool(h)
        h = h.view(h.shape[0], -1)
        h_out = self.classifier_list(h)

        objects = None
        # if self.num_objects:
        #     objects = [self.__getattr__('object_presence_layer_{}'.format(ii))(h) for ii in range(len(self.num_objects))]
        cat_obj = None
        # if self.num_obj_cat:
        #     cat_obj = [self.__getattr__('objcat_presence_layer_{}'.format(ii))(h) for ii in range(len(self.num_obj_cat))]
        # if not self.training and self.ensemble_eval:
        #     return h_out, h_ens, coords, heatmaps, probabilities, objects, cat_obj
        h_out = [h_out, h_ch, h_max]
        # h_out = h_ch
        # h_out = [out + ch + hmax for out, ch, hmax in zip(h_out, h_ch, h_max)]
        return h_out, coords, heatmaps, probabilities, objects, cat_obj


if __name__ == "__main__":
    import torch, time
    # ---------
    kwargs = {'num_coords': 0, 'num_objects': None, 'num_obj_cat': None, 'one_object_layer': True,
              'ensemble_eval': False}
    net = MFNET_3D_DFB(num_classes=[106, 19, 53], dropout=0.5, **kwargs)
    data = torch.randn(1, 3, 16, 224, 224, requires_grad=True)
    net.cuda()
    data = data.cuda()
    net.eval()
    # loss = torch.tensor([10]).cuda()
    t0 = time.time()
    # for i in range(10):
    output = net(data)
    t1 = time.time()
    print('forward time:', t1-t0)
    # h, htail = net.forward_shared_block(data)
    # coords, heatmaps, probabilities = net.forward_coord_layers(htail)
    # output = net.forward_cls_layers(h)
   # torch.save({'state_dict': net.state_dict()}, './tmp.pth')
   #  print(len(output))
