import torch
import torch.nn as nn
import torch.nn.functional as F

class ERPConv2d(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(padding,0), dilation=dilation, groups=groups, bias=bias, padding_mode='zeros')
        

    def forward(self, x):
        x = F.pad(x,(self.padding,self.padding,0,0),mode='circular')
        x = self.conv(x)
        return x

class ERPConv3d(nn.Module):
    #depth is first dimension
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=(padding,padding,0), dilation=dilation, groups=groups, bias=bias, padding_mode="zeros")

    def forward(self, x):
        x = F.pad(x,(self.padding,self.padding,0,0,0,0))
        x = self.conv(x)
        return x


   