# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import torch.nn as nn
from models.cube.resnet_sparse_cube import resnet34
from models.cube.cube_conv import sparse_to_dense, dense_to_sparse


class SparsePadFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, add_mixing=False):
        super().__init__()
        self.add_mixing = add_mixing
        resnet = resnet34(pretrained=pretrained)
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

        if self.add_mixing:
            self.mixer = nn.Conv2d(self.out_channels, self.out_channels, 1)

    def forward(self, cube_maps, idx_to_process):
        # (b,M,6,c,w,w) (b,M,6)
        # return (b,M,6,c,w/4,w/4)
        device = cube_maps.device
        b, M = cube_maps.shape[0:2]
        batch_size = b * M
        cube_maps = cube_maps.flatten(0, 1)
        idx_flat = idx_to_process.flatten(0)

        to_process = torch.arange(idx_flat.size(0), device=device)[idx_flat]  # (k)
        sparse_cube_maps = dense_to_sparse(cube_maps, to_process)  # (k,c,w,w)

        x = self.resnet_conv1((sparse_cube_maps, to_process, batch_size))[0]
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool((x, to_process, batch_size))

        features = []
        for resnet_layer, red_conv in zip(self.resnet_layers, self.reduction_convs):
            x = resnet_layer(x)
            f = red_conv(x[0])
            features.append(f)

        feature = torch.cat(features, dim=1)
        if self.add_mixing:
            feature = self.mixer(feature)
        feature = sparse_to_dense(feature, to_process, b * M)
        feature = feature.unflatten(0, (b, M))

        return feature
