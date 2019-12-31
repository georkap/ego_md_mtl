import dsntnn
import torch
from torch import nn as nn
from torch.functional import F


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

class MultitaskLSTMClassifiers(nn.Module):
    def __init__(self, last_conv_size, num_classes, dropout, num_lstm_layers=1, hidden_multiplier=1):
        super(MultitaskLSTMClassifiers, self).__init__()
        self.num_classes = [num_cls for num_cls in num_classes if num_cls > 0]
        self.num_lstm_layers = num_lstm_layers
        self.hidden_multiplier = hidden_multiplier
        for i, cls_task_size in enumerate(self.num_classes):
            lstm = nn.LSTM(last_conv_size, int(last_conv_size*hidden_multiplier), num_lstm_layers, bias=True,
                           batch_first=False, dropout=0)
            fc = nn.Linear(last_conv_size * hidden_multiplier * len(self.num_classes), cls_task_size)
            self.add_module('lstm_{}'.format(i), lstm)
            self.add_module('classifier_{}'.format(i), fc)
        if dropout:
            self.add_module('dropout', nn.Dropout(p=dropout))

    def forward(self, h): # h is a volume of Bx768x8x7x7 -> 8xBx768
        batch_size = h.size(0)
        feat_size = h.size(1)

        h = F.avg_pool3d(h, (1, h.size(3), h.size(4)), (1, 1, 1)) # Bx768x8x1x1
        h = h.view(h.shape[0], h.shape[1], -1) # Bx768x8
        h = h.transpose(1, 2).transpose(0, 1) # 8xBx768

        # seq_size = h.size(0)

        h_temp = []
        for i, cls_task_size in enumerate(self.num_classes):
            lstm_for_task = getattr(self, 'lstm_{}'.format(i))
            h0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_multiplier * feat_size, device=h.device)
            c0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_multiplier * feat_size, device=h.device)
            lstm_out, (ht, ct) = lstm_for_task(h, (h0, c0))
            h_temp.append(lstm_out[-1])

        h_temp = torch.cat(h_temp, dim=1)
        if hasattr(self, 'dropout'):
            h_temp = self.dropout(h_temp)

        h_out = []
        for i in range(len(self.num_classes)):
            fc = getattr(self, 'classifier_{}'.format(i))
            h_out.append(fc(h_temp))

        return h_out

class MultitaskLSTMAttnClassifiers(nn.Module):
    def __init__(self, last_conv_size, num_classes, dropout, max_seq_len, num_lstm_layers=1, hidden_multiplier=1):
        super(MultitaskLSTMAttnClassifiers, self).__init__()
        self.num_classes = [num_cls for num_cls in num_classes if num_cls > 0]
        self.num_lstm_layers = num_lstm_layers
        self.hidden_multiplier = hidden_multiplier
        for i, cls_task_size in enumerate(self.num_classes):
            hidden_size = int(last_conv_size * hidden_multiplier)
            lstm = nn.LSTM(last_conv_size, hidden_size, num_lstm_layers, bias=True,
                           batch_first=False, dropout=0)
            attn_decoder = AttnDecoderLSTM(last_conv_size, hidden_size, max_seq_len)
            fc = nn.Linear(hidden_size * len(self.num_classes), cls_task_size)
            self.add_module('encoder_{}'.format(i), lstm)
            self.add_module('attn_decoder_{}'.format(i), attn_decoder)
            self.add_module('classifier_{}'.format(i), fc)
        if dropout:
            self.add_module('dropout', nn.Dropout(p=dropout))

    def forward(self, h):# h is a volume of Bx768x8x7x7 -> 8xBx768
        batch_size = h.size(0)
        feat_size = h.size(1)

        h = F.avg_pool3d(h, (1, h.size(3), h.size(4)), (1, 1, 1)) # Bx768x8x1x1
        h = h.view(h.shape[0], h.shape[1], -1) # Bx768x8
        h = h.transpose(1, 2).transpose(0, 1) # 8xBx768

        h_temp = []
        for i, cls_task_size in enumerate(self.num_classes):
            encoder = getattr(self, 'encoder_{}'.format(i))
            h0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_multiplier * feat_size, device=h.device)
            c0 = torch.zeros(self.num_lstm_layers, batch_size, self.hidden_multiplier * feat_size, device=h.device)
            encoder_out, (ht, ct) = encoder(h, (h0, c0))
            attn_decoder = getattr(self, 'attn_decoder_{}'.format(i))
            decoder_out = attn_decoder(h, encoder_out)
            h_temp.append(decoder_out[-1])

        h_temp = torch.cat(h_temp, dim=1)
        if hasattr(self, 'dropout'):
            h_temp = self.dropout(h_temp)

        h_out = []
        for i in range(len(self.num_classes)):
            fc = getattr(self, 'classifier_{}'.format(i))
            h_out.append(fc(h_temp))

        return h_out


class AttnDecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, max_seq_len):
        super(AttnDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # attn layer will calculate which step's weights to pay attention to.
        self.attn = nn.Linear(input_size + hidden_size, max_seq_len)
        self.attn_combine = nn.Linear(input_size + hidden_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, 1, bias=True, batch_first=False, dropout=0)

    def forward(self, h, encoder_out):
        attended_encoder_out = torch.zeros_like(encoder_out)
        for seq_index in range(self.max_seq_len):
            cat_for_attn = torch.cat((h[seq_index], encoder_out[seq_index]), 1)
            attn = self.attn(cat_for_attn)
            attn = F.softmax(attn, dim=1)
            attn_applied = torch.bmm(attn.unsqueeze(1), torch.transpose(encoder_out, 0, 1))
            temp_encoder_out = torch.cat((h[seq_index], attn_applied[:, 0, :]), 1)
            attended_encoder_out[seq_index] = self.attn_combine(temp_encoder_out)

        batch_size = h.size(1)
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=h.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=h.device)
        decoder_out, (ht, ct) = self.decoder(h, (h0, c0))

        return decoder_out

class TemporalAttention(nn.Module):
    def __init__(self, num_in_feat, t_dim, num_cls_tasks):
        super(TemporalAttention, self).__init__()
        self.num_cls_tasks = num_cls_tasks
        self.num_in_feat = num_in_feat
        self.t_dim = t_dim
        for task_id in range(num_cls_tasks):
            attention = nn.ModuleList([nn.Linear(num_in_feat, 1) for _ in range(t_dim)])
            self.add_module('attn_{}'.format(task_id), attention)

    def forward(self, h_ens): # Bx768x8
        h_ens_out = torch.zeros((h_ens.shape[0], self.t_dim, self.num_cls_tasks), device=h_ens.device)
        for task_id in range(self.num_cls_tasks):
            attention = getattr(self, "attn_{}".format(task_id))
            for ens_id, ens_layer in enumerate(attention):
                prob = torch.sigmoid(ens_layer(h_ens[:, :, ens_id]))
                h_ens_out[:, ens_id, task_id] = prob.squeeze(1)
        return h_ens_out # B x 8 x num_cls_tasks


