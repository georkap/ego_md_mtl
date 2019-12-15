import dsntnn
import torch
import torch.nn as nn
from torch import nn as nn


class CoordRegressionLayer(nn.Module):
    def __init__(self, input_filters, n_locations):
        super(CoordRegressionLayer, self).__init__()
        self.hm_conv = nn.Conv3d(input_filters, n_locations, kernel_size=1, bias=False)
        self.probability = nn.Linear(input_filters, 1, bias=False)

    def forward(self, h):
        probabilities = torch.zeros(0, device=h.device) # torch.nn.ReLU(self.probability(torch.squeeze(h)))
        # 1. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(h)
        # 2. Transpose the heatmap volume to keep the temporal dimension in the volume
        unnormalized_heatmaps.transpose_(2, 1).transpose_(1, 0)
        # 3. Normalize the heatmaps
        heatmaps = [dsntnn.flat_softmax(uhm) for uhm in unnormalized_heatmaps]
        # 4. Calculate the coordinates
        coords = [dsntnn.dsnt(hm) for hm in heatmaps]
        heatmaps = torch.stack(heatmaps, 1)
        coords = torch.stack(coords, 1)

        return coords, heatmaps, probabilities

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

class ObjectPresenceLayer(nn.Module):
    def __init__(self, input_shape, num_objects, one_layer=False):
        super(ObjectPresenceLayer, self).__init__()
        if one_layer:
            self.object_classifier = nn.Linear(input_shape, num_objects)
        else:
            self.object_classifier = nn.ModuleList([nn.Linear(input_shape, 1) for _ in range(num_objects)])

        self.one_layer = one_layer
        self.num_objects = num_objects

    def forward(self, h):
        h_out = []
        if self.one_layer:
            h_out = self.object_classifier(h)
            if not self.training:
                h_out = torch.sigmoid(h_out)
        else:
            for i, cl in enumerate(self.object_classifier):
                h_out.append(torch.sigmoid(cl(h)) if not self.training else cl(h))
        return h_out

class IN_AC_CONV3D(nn.Module):
    def __init__(self, num_in, num_filter,
                 kernel=(1, 1, 1), pad=(0, 0, 0), stride=(1, 1, 1), g=1, bias=False):
        super(IN_AC_CONV3D, self).__init__()
        self.gn = nn.InstanceNorm3d(num_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_filter, kernel_size=kernel, padding=pad, stride=stride, groups=g, bias=bias)

    def forward(self, x):
        h = self.relu(self.gn(x))
        h = self.conv(h)
        return h


class GN_AC_CONV3D(nn.Module):
    def __init__(self, num_in, num_filter,
                 kernel=(1, 1, 1), pad=(0, 0, 0), stride=(1, 1, 1), g=1, bias=False):
        super(GN_AC_CONV3D, self).__init__()
        self.gn = nn.GroupNorm(num_in, num_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_in, num_filter, kernel_size=kernel, padding=pad, stride=stride, groups=g, bias=bias)

    def forward(self, x):
        h = self.relu(self.gn(x))
        h = self.conv(h)
        return h


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

def get_norm_layers(norm_type, conv1_num_out, conv5_num_out):
    if norm_type == 'GN':
        conv1normlayer = ('gn', nn.GroupNorm(num_groups=1, num_channels=conv1_num_out))
        tailnorm = ('gn', nn.GroupNorm(num_groups=1, num_channels=conv5_num_out))
    elif norm_type == 'IN':
        conv1normlayer = ('in', nn.InstanceNorm3d(conv1_num_out))
        tailnorm = ('in', nn.InstanceNorm3d(conv5_num_out))
    else:
        conv1normlayer = ('bn', nn.BatchNorm3d(conv1_num_out))
        tailnorm = ('bn', nn.BatchNorm3d(conv5_num_out))
    return conv1normlayer, tailnorm

class MF_UNIT(nn.Module):
    def __init__(self, num_in, num_mid, num_out, g=1, stride=(1, 1, 1), first_block=False, use_3d=True, norm='BN'):
        super(MF_UNIT, self).__init__()
        num_ix = int(num_mid/4)
        kt, pt = (3, 1) if use_3d else (1, 0)
        if norm == 'GN':
            layer = GN_AC_CONV3D
        elif norm == 'IN':
            layer = IN_AC_CONV3D
        else: # if norm == 'BN':
            layer = BN_AC_CONV3D
        # prepare input
        self.conv_i1 = layer(num_in=num_in,  num_filter=num_ix,  kernel=(1, 1, 1), pad=(0, 0, 0))
        self.conv_i2 = layer(num_in=num_ix,  num_filter=num_in,  kernel=(1, 1, 1), pad=(0, 0, 0))
        # main part
        self.conv_m1 = layer(num_in=num_in,  num_filter=num_mid, kernel=(kt, 3, 3), pad=(pt, 1, 1), stride=stride, g=g)
        if first_block:
            self.conv_m2 = layer(num_in=num_mid, num_filter=num_out, kernel=(1, 1, 1), pad=(0, 0, 0))
        else:
            self.conv_m2 = layer(num_in=num_mid, num_filter=num_out, kernel=(1, 3, 3), pad=(0, 1, 1), g=g)
        # adapter
        if first_block:
            self.conv_w1 = layer(num_in=num_in,  num_filter=num_out, kernel=(1, 1, 1), pad=(0, 0, 0), stride=stride)

    def forward(self, x):
        h = self.conv_i1(x)
        x_in = x + self.conv_i2(h)

        h = self.conv_m1(x_in)
        h = self.conv_m2(h)

        if hasattr(self, 'conv_w1'):
            x = self.conv_w1(x)

        return h + x

class DiscriminativeFilterBankClassifier(nn.Module):
    def __init__(self, num_in, num_out_classes, max_kernel, dropout, num_dc=5): # input should be Bx768x8x7x7
        super(DiscriminativeFilterBankClassifier, self).__init__()
        self.dfb = nn.Conv3d(in_channels=num_in, out_channels=num_out_classes*num_dc, kernel_size=(1,1,1), bias=False)
        self.maxpool = nn.MaxPool3d(kernel_size=max_kernel, stride=(num_dc, 1, 1))
        self.meanpool = nn.AvgPool1d(kernel_size=num_dc, stride=num_dc) # xchannel classifier output
        self.classifier_max = nn.Linear(num_out_classes*num_dc, num_out_classes) # max classifier output
        self.num_out_classes = num_out_classes
        if dropout:
            self.add_module('dropout', nn.Dropout(p=dropout))

    def forward(self, x):
        x = self.dfb(x)
        x = self.maxpool(x)

        # if exists apply dropout before output layers
        if hasattr(self, 'dropout'):
            x = self.dropout(x)

        x_max = self.classifier_max(x.view(x.shape[0], -1)) # max classifier applied

        x = self.meanpool(x.view(x.shape[0], self.num_out_classes, -1))
        x_ch = x.view(x.shape[0], -1) # xchannel classifier applied

        return x_ch, x_max
        # return x_ch

class MultitaskDFBClassifiers(nn.Module):
    def __init__(self, last_conv_size, num_classes, max_kernel, dropout, num_dc=5):
        super(MultitaskDFBClassifiers, self).__init__()
        self.num_classes = [num_cls for num_cls in num_classes if num_cls > 0]
        self.dfb_classifier_list = nn.ModuleList([
            DiscriminativeFilterBankClassifier(last_conv_size, num_cls, max_kernel, dropout, num_dc)
            for num_cls in self.num_classes])

    def forward(self, h):
        h_ch, h_max = [], []
        # h_ch = []
        for i, cl in enumerate(self.dfb_classifier_list):
            x_ch, x_max = cl(h)
            # x_ch = cl(h)
            h_ch.append(x_ch)
            h_max.append(x_max)
        return h_ch, h_max
        # return h_ch

