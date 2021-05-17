import torch
import torch.nn as nn
import torch.nn.functional as F

class TFC(nn.Module):
    '''Tied Block Fully Connected Layer'''
    def __init__(self, in_planes, num_classes=100, B=1):
        super(TFC, self).__init__()
        assert in_planes % B == 0
        self.num_classes = num_classes
        self.B = B
        self.linear = nn.Linear(in_planes//self.B, num_classes//self.B)

    def forward(self, x):
        n, c = x.size()
        x = x.view(n*self.B, c//self.B)
        x = self.linear(x).view(n, self.num_classes)
        return x