
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelSpatialAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.ca_module = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1), nn.BatchNorm2d(in_channels // ratio), nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels), 
            nn.Sigmoid()
        )
        self.sa_module = nn.Sequential(
            nn.Conv2d(1, in_channels // ratio, kernel_size=3, padding=1), nn.BatchNorm2d(in_channels // ratio), nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels // ratio, 1, kernel_size=3, padding=1), nn.BatchNorm2d(1), 
            nn.Sigmoid()
        )

    def forward(self, x_res):
        x_res = self.ca_module(torch.mean(x_res, (2, 3)).unsqueeze(2).unsqueeze(3))*x_res
        x_res = self.sa_module(torch.mean(x_res, 1).unsqueeze(1))*x_res

        return x_res