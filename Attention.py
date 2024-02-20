import matplotlib.pyplot as plt
from torch import nn, einsum
import torch.nn.functional as F
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random
import torch.nn as nn
import torch
import cv2
import torch.nn as nn
import numpy as np


class ChannelAttention(nn.Module):
    def __init__(self, in_planes=128, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=(5 - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(in_planes)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=3)
        self.channelG =nn.Parameter(torch.tensor(0.5,requires_grad=True))
    def forward(self, x):
        # print(self.channelG.shape,x.shape)
        # x = self.channelG*x
        # return x
        # dx = self.conv1(x)
        if (x.size(3) % 2 == 0):
            patch_size = x.size(3)//2  # pixels
            patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) c s1 s2 ', s1=patch_size, s2=patch_size)  #(h s1) (w s2) ##滑动窗口  (h w)滑动窗口的区域
            x1 = patches[:, 0]
            x2 = patches[:, 1]
            x3 = patches[:, 2]
            x4 = patches[:, 3]

            avg_out1=self.avg_pool(x1)
            max_out1=self.max_pool(x1)

            avgout = self.conv(avg_out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            maxout = self.conv(max_out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            out = avgout + maxout
            out = self.sigmoid(out)*self.channelG

            x1 =out*x1

            avg_out1 = self.avg_pool(x2)
            max_out1 = self.max_pool(x2)
            avgout = self.conv(avg_out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            maxout = self.conv(max_out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            out = avgout + maxout
            out = self.sigmoid(out)*self.channelG
            x2 = out * x2

            avg_out1 = self.avg_pool(x3)
            max_out1 = self.max_pool(x3)
            avgout = self.conv(avg_out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            maxout = self.conv(max_out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            out = avgout + maxout
            out = self.sigmoid(out)*self.channelG
            x3 = out * x3

            avg_out1 = self.avg_pool(x4)
            max_out1 = self.max_pool(x4)
            avgout = self.conv(avg_out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            maxout = self.conv(max_out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            out = avgout + maxout
            out = self.sigmoid(out)*self.channelG
            x4 = out * x4

            newx1 = torch.cat([x1, x2], dim=3)
            newx2 = torch.cat([x3, x4], dim=3)
            newx = torch.cat([newx1, newx2], dim=2)

            avg_out1 = self.avg_pool(x)
            max_out1 = self.max_pool(x)

            avgout = self.conv(avg_out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            maxout = self.conv(max_out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            out = avgout + maxout
            out = self.sigmoid(out)
            x = out * x

            return newx + x

        else:
            avg_out1 = self.avg_pool(x)
            max_out1 = self.max_pool(x)

            avgout = self.conv(avg_out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            maxout = self.conv(max_out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            out = avgout + maxout
            out = self.sigmoid(out)
            x = out * x
            return x;


class SpatialAttention(nn.Module):
    def __init__(self,in_channel=256):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 =self.conv1(x)
        avg_pool1 = torch.mean(x1, dim=1, keepdim=True)
        max_pool1, _ = torch.max(x1, dim=1, keepdim=True)

        out = torch.cat([avg_pool1, max_pool1], dim=1)
        out = self.conv2(out)
        # out =out*out1

        attention = self.sigmoid(out)
        return attention * x