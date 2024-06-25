# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cube.cost_reg_net_cube import CostRegNetCube
from models.cube.cube_conv import sparse_to_dense
from models.shared.cost_volume import cost_volume, BasicWeightModule


class DepthModuleDotMulti(nn.Module):
    def __init__(self, groups=8, base_channels=8, greater_than_two=False):
        super().__init__()
        self.groups = groups
        self.greater_than_two = greater_than_two
        self.cost_reg = CostRegNetCube(in_channels=groups, base_channels=base_channels)

        if self.greater_than_two:
            self.weight_module = BasicWeightModule(groups)

    def forward(self, features, cameras, to_worlds, hypo_distances, idx_to_process):
        # (b,M,6,f,w,w) (b,6) (b,M,4,4) (b,d,6,w,w) (b,d) (b,6)
        # return (b,6,1,w,w)

        device = features.device
        b, M, _, _, w, _ = features.shape
        idx_flat = idx_to_process.flatten(0)
        # (k)
        to_process = torch.arange(idx_flat.size(0), device=device)[idx_flat]
        k = to_process.size(0)

        features = features.transpose(2, 3).flatten(3, 4)  # (b,M,f,6w,w)

        ref_vol, src_vols = cost_volume(
            features, cameras, to_worlds, hypo_distances.unsqueeze(-1).unsqueeze(-1)
        )  # (b,1,f,d,6w,w), (b,M-1,f,d,6w,w)
        ref_vol = ref_vol.unflatten(-2, (6, -1)).squeeze(1)  # (b,f,d,6,w,w)
        ref_vol = ref_vol.permute(0, 3, 1, 2, 4, 5)  # (b,6,f,d,w,w)

        src_vols = src_vols.unflatten(-2, (6, -1))  # (b,M-1,f,d,6,w,w)
        src_vols = src_vols.permute(0, 4, 1, 2, 3, 5, 6)  # (b,6,M-1,f,d,w,w)

        ref_vol_sparse = ref_vol.flatten(0, 1)[idx_flat].unsqueeze(1)  # (k,1,f,d,w,w)
        src_vols_sparse = src_vols.flatten(0, 1)[idx_flat]  # (k,M-1,f,d,w,w)

        ref_vol_sparse = ref_vol_sparse.unflatten(2, (self.groups, -1))  # (k,1,g,f/g,d,w,w)
        src_vols_sparse = src_vols_sparse.unflatten(2, (self.groups, -1))  # (k,M-1,g,f/g,d,w,w)

        similarity = torch.mean(ref_vol_sparse * src_vols_sparse, dim=3)  # (k,M-1,g,d,w,w)

        if self.greater_than_two:
            weight = self.weight_module(similarity)
            combined = torch.sum(similarity * weight, dim=1)  # (k,g,d,w,w)
        else:
            combined = similarity.squeeze(1)  # (k,g,d,w,w)

        cost_reg = self.cost_reg((combined, to_process, b))  # (k,1,d,w,w)
        prob_volume = F.softmax(cost_reg, dim=2)

        hypos_sparse = hypo_distances.unsqueeze(1).repeat(1, 6, 1).flatten(0, 1)[idx_flat]  # (k,d)
        hypos_sparse = hypos_sparse.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # (k,1,d,1,1)
        distance_sparse = torch.sum(prob_volume * hypos_sparse, dim=2)  # (k,1,w,w)

        distance = sparse_to_dense(distance_sparse, to_process, b)  # (b,6,1,w,w)

        out = {"distance": distance}

        return out
