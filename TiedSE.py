import torch
import torch.nn as nn
import torch.nn.functional as F

class TiedSELayer(nn.Module):
    '''Tied Block Squeeze and Excitation Layer'''
    def __init__(self, channel, B=1, reduction=16):
        super(TiedSELayer, self).__init__()
        assert channel % B == 0
        self.B = B
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        channel = channel//B
        self.fc = nn.Sequential(
                nn.Linear(channel, max(2, channel//reduction)),
                nn.ReLU(inplace=True),
                nn.Linear(max(2, channel//reduction), channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b*self.B, c//self.B)
        y = self.fc(y).view(b, c, 1, 1)
        return x*y