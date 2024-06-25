# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import torch.nn as nn
from models.erp.resnet_erp import erp_resnet_34


class ERPFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        resnet = erp_resnet_34(pretrained=pretrained)
        layer_out_channels = [64, 128, 256, 512]

        self.resnet_conv1 = resnet.conv1
        self.resnet_bn1 = resnet.bn1
        self.resnet_relu = resnet.relu
        self.resnet_maxpool = resnet.maxpool

        self.resnet_layers = nn.ModuleList([resnet.layer1, resnet.layer2, resnet.layer3])
        self.reduction_convs = nn.ModuleList(
            [
                nn.ConvTranspose2d(layer_out_channels[0], 32, 1),
                nn.ConvTranspose2d(layer_out_channels[1], 32, 2, stride=2),
                nn.ConvTranspose2d(layer_out_channels[2], 64, 4, stride=4),
            ]
        )

        self.out_channels = 32 + 32 + 64

    def forward(self, image):
        # (b,M,c,h,w)
        # return (b,M,c,h/4,w/4)
        b, M = image.shape[0:2]
        image = image.flatten(0, 1)

        x = self.resnet_conv1(image)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)

        features = []
        for resnet_layer, red_conv in zip(self.resnet_layers, self.reduction_convs):
            x = resnet_layer(x)
            f = red_conv(x)
            features.append(f)

        feature = torch.cat(features, dim=1)
        feature = feature.unflatten(0, (b, M))

        return feature
