import logging
import torch.nn as nn


def xavier(net):
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and hasattr(m, 'weight'):
            nn.init.xavier_uniform_(m.weight.data, gain=1.)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight.data, gain=1.)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname in ['Sequential', 'AvgPool3d', 'MaxPool3d', 'Dropout', 'ReLU', 'Softmax', 'BnActConv3d'] or \
                'Block' in classname:
            pass
        else:
            if classname != classname.upper():
                logging.warning("Initializer:: '{}' is uninitialized.".format(classname))
    net.apply(weights_init)
