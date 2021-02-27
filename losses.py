
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedDiceLoss(nn.Module):
    def __init__(self, eps=1.0, gamma=5.0):
        super().__init__()
        self.eps = eps
        self.gamma = gamma

    def forward(self, input, target):
        weight = 1 + self.gamma*torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)

        intersection = ((input*target)*weight).sum()
        dice = (2*intersection + self.eps)/(((input*weight).sum() + (target*weight).sum()) + self.eps)

        loss = 1 - dice

        return loss