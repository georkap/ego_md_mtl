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

from src.models.custom_layers import CoordRegressionLayer
from src.utils.initializer import xavier


class MultitaskClassifiers(nn.Module):
    def __init__(self, last_conv_size, num_classes):
        super(MultitaskClassifiers, self).__init__()
        self.num_classes = [num_cls for num_cls in num_classes if num_cls > 0]
        self.classifier_list = nn.ModuleList(
            [nn.Linear(last_conv_size, num_cls) for num_cls in self.num_classes])

    def forward(self, h):
        h_out = []
        for i, cl in enumerate(self.classifier_list):
            h_out.append(cl(h))

        return h_out


class BN_AC_CONV3D(nn.Module):
    def __init__(self, num_in, num_filter,
                 kernel=(1, 1, 1), pad=(0, 0, 0), stride=(1, 1, 1), g=1, bias=False):
        super(BN_AC_CONV3D, self).__init__()
        self.bn = nn.BatchNorm3d(num_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_filter, kernel_size=kernel, padding=pad, stride=stride, groups=g, bias=bias)

    def forward(self, x):
        h = self.relu(self.bn(x))
        h = self.conv(h)
        return h


class MF_UNIT(nn.Module):
    def __init__(self, num_in, num_mid, num_out, g=1, stride=(1, 1, 1), first_block=False, use_3d=True):
        super(MF_UNIT, self).__init__()
        num_ix = int(num_mid/4)
        kt, pt = (3, 1) if use_3d else (1, 0)
        # prepare input
        self.conv_i1 = BN_AC_CONV3D(num_in=num_in,  num_filter=num_ix,  kernel=(1, 1, 1), pad=(0, 0, 0))
        self.conv_i2 = BN_AC_CONV3D(num_in=num_ix,  num_filter=num_in,  kernel=(1, 1, 1), pad=(0, 0, 0))
        # main part
        self.conv_m1 = BN_AC_CONV3D(num_in=num_in,  num_filter=num_mid, kernel=(kt, 3, 3), pad=(pt, 1, 1),
                                    stride=stride, g=g)
        if first_block:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1, 1, 1), pad=(0, 0, 0))
        else:
            self.conv_m2 = BN_AC_CONV3D(num_in=num_mid, num_filter=num_out, kernel=(1, 3, 3), pad=(0, 1, 1), g=g)
        # adapter
        if first_block:
            self.conv_w1 = BN_AC_CONV3D(num_in=num_in,  num_filter=num_out, kernel=(1, 1, 1), pad=(0, 0, 0),
                                        stride=stride)

    def forward(self, x):
        h = self.conv_i1(x)
        x_in = x + self.conv_i2(h)

        h = self.conv_m1(x_in)
        h = self.conv_m2(h)

        if hasattr(self, 'conv_w1'):
            x = self.conv_w1(x)

        return h + x


class MFNET_3D(nn.Module):

    def __init__(self, num_classes, dropout=None, **kwargs):
        super(MFNET_3D, self).__init__()
        # support for arbitrary number of output layers, but it is the user's job to make sure they make sense
        # (e.g. actions->actions and not actions->verbs,nouns etc.)
        self.num_classes = num_classes
        self.num_coords = kwargs.get('num_coords', 0)

        groups = 16
        k_sec = {2: 3, 3: 4, 4: 6, 5: 3}

        # conv1 - x224 (x16)
        conv1_num_out = 16
        self.conv1 = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv3d(3, conv1_num_out, kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=(1, 2, 2),
                                       bias=False)),
                    ('bn', nn.BatchNorm3d(conv1_num_out)),
                    ('relu', nn.ReLU(inplace=True))
                    ]))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # conv2 - x56 (x8)
        num_mid = 96
        conv2_num_out = 96
        self.conv2 = nn.Sequential(
            OrderedDict([("B%02d" % i, MF_UNIT(num_in=conv1_num_out if i == 1 else conv2_num_out,
                                               num_mid=num_mid,
                                               num_out=conv2_num_out,
                                               stride=(2, 1, 1) if i == 1 else (1, 1, 1),
                                               g=groups,
                                               first_block=(i == 1))) for i in range(1, k_sec[2]+1)])
        )

        # conv3 - x28 (x8)
        num_mid *= 2
        conv3_num_out = 2 * conv2_num_out
        self.conv3 = nn.Sequential(
            OrderedDict([("B%02d" % i, MF_UNIT(num_in=conv2_num_out if i == 1 else conv3_num_out,
                                               num_mid=num_mid,
                                               num_out=conv3_num_out,
                                               stride=(1, 2, 2) if i == 1 else (1, 1, 1),
                                               g=groups,
                                               first_block=(i == 1))) for i in range(1, k_sec[3]+1)])
        )

        # conv4 - x14 (x8)
        num_mid *= 2
        conv4_num_out = 2 * conv3_num_out
        self.conv4 = nn.Sequential(
            OrderedDict([("B%02d" % i, MF_UNIT(num_in=conv3_num_out if i == 1 else conv4_num_out,
                                               num_mid=num_mid,
                                               num_out=conv4_num_out,
                                               stride=(1, 2, 2) if i == 1 else (1, 1, 1),
                                               g=groups,
                                               first_block=(i == 1))) for i in range(1, k_sec[4]+1)])
        )

        # conv5 - x7 (x8)
        num_mid *= 2
        conv5_num_out = 2 * conv4_num_out
        self.conv5 = nn.Sequential(
            OrderedDict([("B%02d" % i, MF_UNIT(num_in=conv4_num_out if i == 1 else conv5_num_out,
                                               num_mid=num_mid,
                                               num_out=conv5_num_out,
                                               stride=(1, 2, 2) if i == 1 else (1, 1, 1),
                                               g=groups,
                                               first_block=(i == 1))) for i in range(1, k_sec[5]+1)])
        )

        # create heatmaps
        if self.num_coords > 0:
            self.coord_layers = CoordRegressionLayer(conv5_num_out, self.num_coords)

        # final
        self.tail = nn.Sequential(OrderedDict([('bn', nn.BatchNorm3d(conv5_num_out)), ('relu', nn.ReLU(inplace=True))]))

        if dropout:
            self.globalpool = nn.Sequential(
                OrderedDict([('avg', nn.AvgPool3d(kernel_size=(8, 7, 7), stride=(1, 1, 1))),
                             ('dropout', nn.Dropout(p=dropout))]))
        else:
            self.globalpool = nn.Sequential(
                OrderedDict([('avg', nn.AvgPool3d(kernel_size=(8, 7, 7),  stride=(1, 1, 1)))]))

        self.classifier_list = MultitaskClassifiers(conv5_num_out, num_classes)

        #############
        # Initialization
        xavier(net=self)

    def forward(self, x, upto=None):
        if upto is None:
            assert x.shape[2] == 16

            h = self.conv1(x)   # x224 -> x112
            h = self.maxpool(h)  # x112 ->  x56

            h = self.conv2(h)  # x56 ->  x56
            h = self.conv3(h)  # x56 ->  x28
            h = self.conv4(h)  # x28 ->  x14
            h = self.conv5(h)  # x14 ->   x7

            h = self.tail(h)
            coords, heatmaps = None, None
            if self.num_coords > 0:
                coords, heatmaps = self.coord_layers(h)

            h = self.globalpool(h)

            h = h.view(h.shape[0], -1)

            h_out = self.classifier_list(h)

            return h_out, coords, heatmaps
        elif upto == 'shared':
            return self.forward_shared_block(x)
        elif upto == 'cls':
            return self.forward_cls_layers(x)
        elif upto == 'coord':
            return self.forward_coord_layers(x)
    
    def forward_shared_block(self, x):
        assert x.shape[2] == 16

        h = self.conv1(x)   # x224 -> x112
        h = self.maxpool(h)  # x112 ->  x56

        h = self.conv2(h)  # x56 ->  x56
        h = self.conv3(h)  # x56 ->  x28
        h = self.conv4(h)  # x28 ->  x14
        h = self.conv5(h)  # x14 ->   x7

        h_tail = self.tail(h)

        h = self.globalpool(h_tail)

        h = h.view(h.shape[0], -1)

        return h, h_tail

    def forward_coord_layers(self, h_tail):
        coords, heatmaps = None, None
        if self.num_coords > 0:
            coords, heatmaps = self.coord_layers(h_tail)
        return coords, heatmaps

    def forward_cls_layers(self, h):
        h_out = self.classifier_list(h)
        return h_out


if __name__ == "__main__":
    import torch
    # ---------
    kwargs = {'num_coords': 3}
    net = MFNET_3D(num_classes=[2, 3], pretrained=False, **kwargs)
    data = torch.randn(1, 3, 16, 224, 224, requires_grad=True)
    output = net(data)
    h, htail = net.forward_shared_block(data)
    coords, heatmaps = net.forward_coord_layers(htail)
    output = net.forward_cls_layers(h)
#    torch.save({'state_dict': net.state_dict()}, './tmp.pth')
    print(len(output))
