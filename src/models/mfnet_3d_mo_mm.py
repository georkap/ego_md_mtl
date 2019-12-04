from collections import OrderedDict
import torch
import torch.nn as nn
from src.models.mfnet_3d_mo import BN_AC_CONV3D, MF_UNIT
from src.models.custom_layers import MultitaskClassifiers
from src.utils.initializer import xavier

class Modality_Block(nn.Module):
    def __init__(self, mod_name, input_channels, g, k_sec, conv_num_mid, conv_num_out):
        super(Modality_Block, self).__init__()
        stride = MFNET_3D_MO_MM.STRD['down_spat']
        conv1 = nn.Sequential(OrderedDict([  # RGB: 16x224x224 -> 16x112x112, Flow: 8x224x224->8x112x112
            ('conv', nn.Conv3d(input_channels, conv_num_out[0], kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=stride,
                               bias=False)),
            ('bn', nn.BatchNorm3d(conv_num_out[0])),
            ('relu', nn.ReLU(inplace=True))]))
        self.add_module('conv1', conv1)
        # RGB: 16x112x112 -> 16x56x56, Flow:8x112x112->8x56x56
        maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1))
        self.add_module('maxpool', maxpool)
        # block io sizes
        # [RGB: 16x56x56->8x56x56->8x28x28->8x14x14->8x7x7, Flow: 8x56x56->8x56x56->8x28x28->8x14x14->8x7x7]
        for block_id, (key, value) in enumerate(k_sec.items()):
            block_parts = OrderedDict()
            for i in range(1, value + 1):
                if i == 1:  # temporal or spatial subsampling
                    if block_id == 0:
                        if mod_name == 'Flow':
                            stride = MFNET_3D_MO_MM.STRD['stable']
                        else:
                            stride = MFNET_3D_MO_MM.STRD['down_temp']
                    else:
                        stride = MFNET_3D_MO_MM.STRD['down_spat']
                    num_in = conv_num_out[block_id]
                    fb = True  # first mf unit
                else:  # no subsampling
                    stride = MFNET_3D_MO_MM.STRD['stable']
                    num_in = conv_num_out[block_id + 1]
                    fb = False
                num_mid = conv_num_mid[block_id]
                num_out = conv_num_out[block_id + 1]
                mfunit = MF_UNIT(num_in=num_in, num_mid=num_mid, num_out=num_out, stride=stride, g=g, first_block=fb)
                block_parts["B{:02d}".format(i)] = mfunit
            self.add_module('conv{}'.format(key), nn.Sequential(block_parts))

    def forward(self, x):
        h = self.conv1(x)   # x224 -> x112
        h1 = self.maxpool(h)  # x112 ->  x56

        h2 = self.conv2(h1)  # x56 ->  x56
        h3 = self.conv3(h2)  # x56 ->  x28
        h4 = self.conv4(h3)  # x28 ->  x14
        h5 = self.conv5(h4)  # x14 ->   x7

        return [h1, h2, h3, h4, h5]

class Fusion_Block(nn.Module):
    def __init__(self, modalities, g, k_sec, conv_num_mid, conv_num_out):
        self.num_modalities = len(modalities)
        self.num_blocks = len(k_sec) + 1
        super(Fusion_Block, self).__init__()
        self.num_fusion_blocks_per_layer = self.num_modalities - 1
        for i in range(1, self.num_blocks + 1): # for len(k_sec)=4 this counts from 1 to 5 (incl.)
            fusion_layer = OrderedDict()
            num_in = conv_num_out[i - 1]
            num_mid = conv_num_mid[i - 1] // 2 if i == 1 else conv_num_mid[i - 2]
            num_out = conv_num_out[i - 1]
            for j, mod_name in enumerate(modalities):
                if i == 1 and mod_name == 'RGB':
                    stride = MFNET_3D_MO_MM.STRD['down_temp']
                else:
                    stride = MFNET_3D_MO_MM.STRD['stable']
                reducer = BN_AC_CONV3D(num_in, num_in//2, stride=stride)
                fusion_layer['reducer_{}'.format(j)] = reducer
            for j in range(self.num_fusion_blocks_per_layer):
                c_num_in = num_in + (conv_num_out[i-2] if i > 1 else 0)
                if i == 1 or i == self.num_blocks: # don't downsample for the first and last layers
                    stride = MFNET_3D_MO_MM.STRD['stable']
                else:
                    stride = MFNET_3D_MO_MM.STRD['down_spat']
                combiner = MF_UNIT(num_in=c_num_in, num_mid=num_mid, num_out=num_out, stride=stride, g=g, first_block=True)
                fusion_layer['combiner_{}'.format(j)] = combiner
            self.add_module('fusion{}'.format(i), nn.Sequential(fusion_layer))

    def forward(self, x):
        assert len(x) == self.num_modalities
        prev_combined_output = None
        for i in range(1, self.num_blocks + 1):
            reduced_output = []
            fusion_layer = getattr(self, 'fusion{}'.format(i))
            for j in range(self.num_modalities):
                reducer = getattr(fusion_layer, 'reducer_{}'.format(j))
                reduced_output.append(reducer(x[j][i-1]))
            for j in range(self.num_fusion_blocks_per_layer):
                combiner = getattr(fusion_layer, 'combiner_{}'.format(j))
                combined_input = torch.cat(reduced_output, dim=1)
                permutation = list(range(0, combined_input.shape[1], 2)) + list(
                    range(1, combined_input.shape[1] + 1, 2))
                combined_input = combined_input[:, permutation]
                if prev_combined_output is not None:  # append channels from previous layer
                    combined_input = torch.cat((combined_input, prev_combined_output), dim=1)
                prev_combined_output = combiner(combined_input)

        return prev_combined_output

class MFNET_3D_MO_MM(nn.Module):
    """
    Multi-fiber network 3d, multi-output, multi-modal
    Like the MFNET_3D_MO but supports plenty of parallel modalities with a separate network for each,
    including internal connections between the blocks.
    """
    STRD = {'down_temp': (2, 1, 1), 'down_spat': (1, 2, 2), 'down_spatemp': (2, 2, 2), 'stable': (1, 1, 1)}

    def __init__(self, num_classes, dropout=None, **kwargs):
        super(MFNET_3D_MO_MM, self).__init__()

        self.num_classes = num_classes
        self.dropout_val = dropout
        # dict contains name of modality and num input channels, for flow this would be {'Flow':2}
        self.modalities = kwargs.get('modalities', {'RGB': 3})
        self.num_coords = kwargs.get('num_coords', 0)
        self.num_objects = kwargs.get('num_objects', 0)
        self.num_modalities = len(self.modalities)
        self.fusion_nets = self.num_modalities - 1

        k_sec = kwargs.get('k_sec', {2: 3, 3: 4, 4: 6, 5: 3})
        groups = kwargs.get('groups', 16)
        num_out = [16, 96, 192, 384, 768]
        num_mid = [96, 192, 384, 768]

        for mod_name, input_channels in self.modalities.items():
            # modality_block = self.instantiate_modality_block(mod_name, input_channels, groups, k_sec, num_mid, num_out)
            modality_block = Modality_Block(mod_name, input_channels, groups, k_sec, num_mid, num_out)
            self.add_module(mod_name, modality_block)
            tail = nn.Sequential(OrderedDict([('bn', nn.BatchNorm3d(num_out[-1])), ('relu', nn.ReLU(inplace=True))]))
            self.add_module("{}_tail".format(mod_name), tail)

            gap = nn.AvgPool3d(kernel_size=(8, 7, 7), stride=MFNET_3D_MO_MM.STRD['stable'])
            self.add_module("{}_gap".format(mod_name), gap)

        fusion_block = Fusion_Block(self.modalities.keys(), groups, k_sec, num_mid, num_out)
        self.add_module('Fusion', fusion_block)
        tail = nn.Sequential(OrderedDict([('bn', nn.BatchNorm3d(num_out[-1])), ('relu', nn.ReLU(inplace=True))]))
        self.add_module('Fusion_tail', tail)
        gap = nn.AvgPool3d(kernel_size=(8, 7, 7), stride=MFNET_3D_MO_MM.STRD['stable'])
        self.add_module('Fusion_gap', gap)
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

        self.classifier_list = MultitaskClassifiers((self.num_modalities + self.fusion_nets)*num_out[-1],
                                                    num_classes)

        #############
        # Initialization
        xavier(net=self)

    def instantiate_modality_block(self, mod_name, input_channels, g, k_sec, conv_num_mid, conv_num_out):
        modality_block = nn.Sequential()
        stride = MFNET_3D_MO_MM.STRD['down_spat']
        conv1 = nn.Sequential(OrderedDict([ # RGB: 16x224x224 -> 16x112x112, Flow: 8x224x224->8x112x112
            ('conv', nn.Conv3d(input_channels, conv_num_out[0], kernel_size=(3, 5, 5), padding=(1, 2, 2), stride=stride, bias=False)),
            ('bn', nn.BatchNorm3d(conv_num_out[0])),
            ('relu', nn.ReLU(inplace=True))]))
        modality_block.add_module('conv1', conv1)
        # RGB: 16x112x112 -> 16x56x56, Flow:8x112x112->8x56x56
        maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1))
        modality_block.add_module('maxpool', maxpool)
        # block io sizes
        # [RGB: 16x56x56->8x56x56->8x28x28->8x14x14->8x7x7, Flow: 8x56x56->8x56x56->8x28x28->8x14x14->8x7x7]
        for block_id, (key, value) in enumerate(k_sec.items()):
            block_parts = OrderedDict()
            for i in range(1, value + 1):
                if i == 1: # temporal or spatial subsampling
                    if block_id == 0:
                        if mod_name == 'Flow':
                            stride = MFNET_3D_MO_MM.STRD['stable']
                        else:
                            stride = MFNET_3D_MO_MM.STRD['down_temp']
                    else:
                        stride = MFNET_3D_MO_MM.STRD['down_spat']
                    num_in = conv_num_out[block_id]
                    fb = True # first mf unit
                else: # no subsampling
                    stride = MFNET_3D_MO_MM.STRD['stable']
                    num_in = conv_num_out[block_id+1]
                    fb = False
                num_mid = conv_num_mid[block_id]
                num_out = conv_num_out[block_id+1]
                mfunit = MF_UNIT(num_in=num_in, num_mid=num_mid, num_out=num_out, stride=stride, g=g, first_block=fb)
                block_parts["B{:02d}".format(i)] = mfunit
            modality_block.add_module('conv{}'.format(key), nn.Sequential(block_parts))
        return modality_block

    def instantiate_fusion_blocks(self, num_modalities, g, k_sec, conv_num_mid, conv_num_out):
        num_fusion_blocks_per_layer = num_modalities - 1
        fusion_block = nn.Sequential()
        for i in range(1, len(k_sec) + 2): # for len(k_sec)=4 this counts from 1 to 5 (incl.)
            fusion_layer = OrderedDict()
            num_in = conv_num_out[i - 1]
            num_mid = conv_num_mid[i - 1] // 2 if i == 1 else conv_num_mid[i - 1]
            num_out = conv_num_out[i]
            for j in num_modalities:
                reducer = BN_AC_CONV3D(num_in, num_in//2)
                fusion_layer['reducer_{}_{}'.format(i, j)] = reducer
            for j in num_fusion_blocks_per_layer:
                combiner = MF_UNIT(num_in=num_in, num_mid=num_mid, num_out=num_out, stride=MFNET_3D_MO_MM.STRD['stable'], g=g, first_block=True)
                fusion_layer['combiner_{}_{}'.format(i, j)] = combiner
            fusion_block.add_module('fusion{}'.format(i), fusion_layer)
        return fusion_block

    def forward(self, data):
        rgb, flow = data
        block_rgb = getattr(self, 'RGB')
        x_rgb = block_rgb(rgb)
        block_flow = getattr(self, 'Flow')
        x_flow = block_flow(flow)
        block_fusion = getattr(self, 'Fusion')
        x_fusion = block_fusion([x_rgb, x_flow])

        rgb_tail = getattr(self, 'RGB_tail')
        flow_tail = getattr(self, 'Flow_tail')
        fusion_tail = getattr(self, 'Fusion_tail')
        x_rgb = rgb_tail(x_rgb[-1])
        x_flow = flow_tail(x_flow[-1])
        x_fusion = fusion_tail(x_fusion)

        rgb_gap = getattr(self, 'RGB_gap')
        flow_gap = getattr(self, 'Flow_gap')
        fusion_gap = getattr(self, 'Fusion_gap')
        x_rgb = rgb_gap(x_rgb)
        x_flow = flow_gap(x_flow)
        x_fusion = fusion_gap(x_fusion)

        x = torch.cat([x_rgb, x_flow, x_fusion], 1)
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)

        h_out = self.classifier_list(x)

        coords, heatmaps, probabilities, objects = None, None, None, None
        return h_out, coords, heatmaps, probabilities, objects


if __name__ == '__main__':
    import torch
    num_classes = [10]
    kwargs = {}
    kwargs['modalities'] = {'RGB':3, 'Flow':2}

    net = MFNET_3D_MO_MM(num_classes, **kwargs)

    _rgb = torch.randn((1, 3, 16, 224, 224), requires_grad=True)
    _flow = torch.randn((1, 2, 8, 224, 224), requires_grad=True)

    output = net([_rgb, _flow])

    print('')


