import dsntnn
import torch
import torch.nn as nn

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



