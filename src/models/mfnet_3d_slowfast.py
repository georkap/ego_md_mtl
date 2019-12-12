from collections import OrderedDict
import torch
import torch.nn as nn
from src.models.layers import MultitaskClassifiers, BN_AC_CONV3D
from src.utils.initializer import xavier

class MF_UNIT_SF(nn.Module):
    def __init__(self, num_in, num_mid, num_out, g=1, stride=(1, 1, 1), first_block=False, temporal_kernel=1):
        super(MF_UNIT_SF, self).__init__()
        num_ix = int(num_mid/4)
        temporal_pad = 1 if temporal_kernel == 3 else 0
        # prepare input
        self.conv_i1 = BN_AC_CONV3D(num_in=num_in, num_filter=num_ix,  kernel=(1, 1, 1), pad=(0, 0, 0))
        self.conv_i2 = BN_AC_CONV3D(num_in=num_ix, num_filter=num_in,  kernel=(1, 1, 1), pad=(0, 0, 0))
        # main part
        self.conv_m1 = BN_AC_CONV3D(num_in=num_in, num_filter=num_mid, kernel=(temporal_kernel, 3, 3),
                                    pad=(temporal_pad, 1, 1), stride=stride, g=g)
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
        x_in = x + self.conv_i2(h)[:,:]

        h = self.conv_m1(x_in)
        h = self.conv_m2(h)

        if hasattr(self, 'conv_w1'):
            x = self.conv_w1(x)

        return h + x

class FuseFastToSlow(nn.Module):
    def __init__(self, num_in, fusion_conv_channel_ratio, fusion_kernel, alpha, eps=1e-5, bn_mmt=0.1,
                 inplace_relu=True):
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            num_in,
            num_in * fusion_conv_channel_ratio, kernel_size=[fusion_kernel, 1, 1], stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False
        )
        self.bn = nn.BatchNorm3d(num_in * fusion_conv_channel_ratio, eps=eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


class MFNET_3D_SF(nn.Module):
    """
    MFNET adaptation with SlowFast routes
    following:
    https://github.com/cypw/PyTorch-MFNet/blob/master/network/mfnet_3d.py for mfnet
    and
    https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/video_model_builder.py
    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "Slowfast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    for slowfast
    """

    def __init__(self, num_classes, dropout=None, **kwargs):
        super(MFNET_3D_SF, self).__init__()

        self.num_classes = num_classes
        self.num_coords = kwargs.get('num_coords', 0)
        self.num_objects = kwargs.get('num_objects', 0)
        self.sf_a = kwargs.get('sf_a', 6) # 6 temporal dim
        self.sf_b = kwargs.get('sf_b', 6) # 6 channel dim
        temporal_dim_slow = kwargs.get('num_frames', 4) # could fit even 8
        temporal_dim_fast = temporal_dim_slow * self.sf_a
        spatial_dim = kwargs.get('spatial_size', 224)
        groups = kwargs.get('groups', 16)
        k_sec = kwargs.get('k_sec', {2: 3, 3: 4, 4: 6, 5: 3})
        num_mid = [96, 192, 384, 768]
        conv_num_out_slow = [16, 96, 192, 384, 768]
        conv_num_out_fast = [4, 16, 32, 64, 128]
        fusion_conv_channel_ratio = 4
        # for fast out channels are conv_num_out/b (=6) [2.67, 16, 32, 64, 128]->[3,16,32,64,128]

        # Slow intro conv
        self.slow_conv1 = nn.Sequential(OrderedDict([
                          ('conv', nn.Conv3d(3, conv_num_out_slow[0], kernel_size=(1, 5, 5), padding=(0, 2, 2),
                                             stride=(1, 2, 2), bias=False)),
                          ('bn', nn.BatchNorm3d(conv_num_out_slow[0])),
                          ('relu', nn.ReLU(inplace=True))]))
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.fast_conv1 = nn.Sequential(OrderedDict([
                          ('conv', nn.Conv3d(3, conv_num_out_fast[0], kernel_size=(5, 5, 5), padding=(2, 2, 2),
                                             stride=(1, 2, 2), bias=False)),
                          ('bn', nn.BatchNorm3d(conv_num_out_fast[0])),
                          ('relu', nn.ReLU(inplace=True))]))
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.s1_fuse = FuseFastToSlow(conv_num_out_fast[0], fusion_conv_channel_ratio=fusion_conv_channel_ratio,
                                      fusion_kernel=5, alpha=self.sf_a)

        for block_id, (key, value) in enumerate(k_sec.items()):
            slow_temporal_kernel = 1 if key in [2, 3] else 3
            block_slow = nn.Sequential(OrderedDict([
                ("B%02d" % i,
                 MF_UNIT_SF(num_in=conv_num_out_slow[block_id]+fusion_conv_channel_ratio*conv_num_out_fast[block_id] if i == 1 else conv_num_out_slow[block_id+1],
                            num_mid=num_mid[block_id], num_out=conv_num_out_slow[block_id+1],
                            stride=(1, 2, 2) if i == 1 and block_id != 0 else (1, 1, 1),
                            g=groups, first_block=(i == 1), temporal_kernel=slow_temporal_kernel))
                for i in range(1, value+1)]))
            block_fast = nn.Sequential(OrderedDict([
                ("B%02d" % i,
                 MF_UNIT_SF(num_in=conv_num_out_fast[block_id] if i == 1 else conv_num_out_fast[block_id + 1],
                            num_mid=num_mid[block_id]//self.sf_b, num_out=conv_num_out_fast[block_id + 1],
                            stride=(1, 2, 2) if i == 1 and block_id != 0 else (1, 1, 1),
                            g=1 if block_id == 0 else groups, first_block=(i == 1), temporal_kernel=3))
                for i in range(1, value+1)]))

            fuse = FuseFastToSlow(conv_num_out_fast[block_id+1], fusion_conv_channel_ratio=fusion_conv_channel_ratio,
                                  fusion_kernel=5, alpha=self.sf_a)

            self.add_module("slow_conv{}".format(key), block_slow)
            self.add_module("fast_conv{}".format(key), block_fast)
            self.add_module("s{}_fuse".format(key), fuse)

        # final
        self.tail_slow = nn.Sequential(OrderedDict([('bn', nn.BatchNorm3d(conv_num_out_slow[-1] + fusion_conv_channel_ratio * conv_num_out_fast[-1])),
                                                    ('relu', nn.ReLU(inplace=True))]))
        self.tail_fast = nn.Sequential(OrderedDict([('bn', nn.BatchNorm3d(conv_num_out_fast[-1])), ('relu', nn.ReLU(inplace=True))]))

        self.globalpool_slow = nn.AvgPool3d(kernel_size=(temporal_dim_slow, spatial_dim//32, spatial_dim//32), stride=(1, 1, 1))
        self.globalpool_fast = nn.AvgPool3d(kernel_size=(temporal_dim_fast, spatial_dim//32, spatial_dim//32), stride=(1, 1, 1))
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

        self.classifier_list = MultitaskClassifiers(conv_num_out_slow[-1] + (fusion_conv_channel_ratio+1)*conv_num_out_fast[-1], num_classes)

        #############
        # Initialization
        xavier(net=self)

    # def forward(self, x_s, x_f):
    def forward(self, x_f):
        x_s = x_f[:, :, ::6]
        x_s = self.slow_conv1(x_s)
        x_f = self.fast_conv1(x_f)
        x_s = self.slow_maxpool(x_s)
        x_f = self.fast_maxpool(x_f)

        x_s, x_f = self.s1_fuse([x_s, x_f])

        for block_id in [2, 3, 4, 5]:
            slow_block = getattr(self, "slow_conv{}".format(block_id))
            fast_block = getattr(self, "fast_conv{}".format(block_id))
            fuse = getattr(self, "s{}_fuse".format(block_id))
            x_s = slow_block(x_s)
            x_f = fast_block(x_f)
            x_s, x_f = fuse([x_s, x_f])
            # print("forwarded block {}".format(block_id))

        x_s = self.tail_slow(x_s)
        x_f = self.tail_fast(x_f)
        x_s = self.globalpool_slow(x_s)
        x_f = self.globalpool_fast(x_f)

        x = torch.cat([x_s, x_f], 1)
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)

        h_out = self.classifier_list(x)

        coords, heatmaps, probabilities, objects = None, None, None, None
        return h_out, coords, heatmaps, probabilities, objects


if __name__ == "__main__":
    import torch
    # ---------
    kwargs = {'num_coords': 3}
    net = MFNET_3D_SF(num_classes=[3], dropout=0.5, **kwargs)
    net = net.cuda()
    # data = [torch.randn(1, 3, 4, 224, 224, requires_grad=True),
    #         torch.randn(1, 1, 24, 224, 224, requires_grad=True)]
    # output = net(data[0], data[1])

    data = torch.randn(1, 3, 24, 224, 224, requires_grad=True).cuda()
    output = net(data)

    # loss = torch.nn.CrossEntropyLoss()(output[0][0],torch.tensor([0]).long())
    # print(loss)
    # loss.backward()
    # h, htail = net.forward_shared_block(data)
    # coords, heatmaps, probabilities = net.forward_coord_layers(htail)
    # output = net.forward_cls_layers(h)
#    torch.save({'state_dict': net.state_dict()}, './tmp.pth')

#     print(len(output))
