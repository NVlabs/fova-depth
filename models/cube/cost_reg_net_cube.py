import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cube.cube_conv import SparseIdxCubePadImproved

class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=(1,0,0) ,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu


        self.padder = SparseIdxCubePadImproved(1)


    def forward(self, args):
        #(k,c,d,w,w) (k) 
        #return (k,c',d',w',w')
        x, to_process, batch_size = args
        k,c,d,w,_ = x.shape
        
        x = self.padder(x.flatten(1,2),to_process, batch_size).unflatten(1,(c,d))
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x, to_process, batch_size

class CostRegNetCube(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2,mode='nearest')

        self.conv0 = Conv3d(in_channels, base_channels)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8)

        self.conv7 = Conv3d(base_channels * 8, base_channels * 4, 3, stride=1)

        self.conv9 = Conv3d(base_channels * 4, base_channels * 2, stride=1)

        self.conv11 = Conv3d(base_channels * 2, base_channels * 1, stride=1)

        self.prob = Conv3d(base_channels, 1, 3, stride=1, relu=False,bn=False)

    def forward(self, x):
        #(x, to_process, batch_size)
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x, to_process, batch_size = self.conv6(self.conv5(conv4))
       
        x = conv4[0] + self.conv7((self.upsample(x),to_process,batch_size))[0]
        x = conv2[0] + self.conv9((self.upsample(x),to_process,batch_size))[0]
        x = conv0[0] + self.conv11((self.upsample(x),to_process,batch_size))[0]
        x = self.prob((x,to_process,batch_size))[0]
        return x