from collections import OrderedDict
import torch.nn as nn
from src.models.mfnet_3d_mo import MF_UNIT, MultitaskClassifiers
from src.utils.initializer import xavier

class MFNET_3D_MO_MM(nn.Module):
    """
    Multi-fiber network 3d, multi-output, multi-modal
    Like the MFNET_3D_MO but supports plenty of parallel modalities with a separate network for each,
    including internal connections between the blocks.
    """
    STRD = {'down_temp': (2, 1, 1), 'down_spat': (1, 2, 2), 'stable': (1, 1, 1)}

    def __init__(self, num_classes, dropout=None, **kwargs):
        super(MFNET_3D_MO_MM, self).__init__()

        self.num_classes = num_classes
        self.dropout = dropout
        # dict contains name of modality and num input channels, for flow this would be {'Flow':2}
        self.modalities = kwargs.get('modalities', {'RGB': 3})
        self.num_coords = kwargs.get('num_coords', 0)
        self.num_objects = kwargs.get('num_objects', 0)

        k_sec = kwargs.get('k_sec', {2: 3, 3: 4, 4: 6, 5: 3})
        groups = kwargs.get('groups', 16)
        num_out = [16, 96, 192, 384, 768]
        num_mid = [96, 192, 384, 768]

        for mod_name, input_channels in self.modalities.items():
            modality_block = self.instantiate_modality_block(mod_name, input_channels, groups, k_sec, num_mid, num_out)
            self.add_module(mod_name, modality_block)

        num_modalities = len(self.modalities.keys())

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

    def instantiate_fusion_blocks(self, num_modalities, conv_num_out):
        pass

    def forward(self, rgb, flow):
        block_rgb = getattr(self, 'RGB')
        x_rgb = block_rgb(rgb)
        block_flow = getattr(self, 'Flow')
        x_flow = block_flow(flow)

        return x_rgb, x_flow


if __name__ == '__main__':
    import torch
    num_classes = [10]
    kwargs = {}
    kwargs['modalities'] = {'RGB':3, 'Flow':2}

    net = MFNET_3D_MO_MM(num_classes, **kwargs)

    rgb = torch.randn((1, 3, 16, 224, 224), requires_grad=True)
    flow = torch.randn((1, 2, 8, 224, 224), requires_grad=True)

    out_rgb, out_flow = net(rgb, flow)

    print('')


