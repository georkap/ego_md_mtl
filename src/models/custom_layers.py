import dsntnn
import torch
import torch.nn as nn


class CoordRegressionLayer(nn.Module):
    def __init__(self, input_filters, n_locations):
        super(CoordRegressionLayer, self).__init__()
        self.hm_conv = nn.Conv3d(input_filters, n_locations, kernel_size=1, bias=False)

    def forward(self, h):
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

        return coords, heatmaps
