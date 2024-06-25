# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.erp.cost_reg_net_erp import CostRegNetERP
from models.shared.cost_volume import cost_volume, BasicWeightModule


class DepthModuleDotMulti(nn.Module):
    def __init__(self, groups=8, base_channels=8, greater_than_two=False):
        super().__init__()
        self.groups = groups
        self.greater_than_two = greater_than_two
        self.cost_reg = CostRegNetERP(in_channels=groups, base_channels=base_channels)

        if self.greater_than_two:
            self.weight_module = BasicWeightModule(groups)

    def forward(self, features, cameras, to_worlds, hypo_distances):
        # (b,M,f,h,w) (b,M) (b,M,4,4) (b,d)

        hypo_distances = hypo_distances.unsqueeze(-1).unsqueeze(-1)  # (b,d,1,1)

        # (b,1,f,d,h,w), (b,M-1,f,d,h,w)
        ref_vol, src_vols = cost_volume(
            features, cameras, to_worlds, hypo_distances, sphere_sweep=True
        )

        ref_vol = ref_vol.unflatten(2, (self.groups, -1))  # (b,1,g,f/g,d,h,w)

        src_vols = src_vols.unflatten(2, (self.groups, -1))  # (b,M-1,g,f/g,d,h,w)

        similarity = torch.mean(ref_vol * src_vols, dim=3)  # (b,M-1,g,d,h,w)

        if self.greater_than_two:
            # (b,M-1,1,d,h,w)
            weight = self.weight_module(similarity)
            # (b,g,d,h,w)
            combined = torch.sum(similarity * weight, dim=1)
        else:
            combined = similarity.squeeze(1)

        cost_reg = self.cost_reg(combined)  # (b,1,d,h,w)
        prob_volume = F.softmax(cost_reg, dim=2)

        hypo_distances = hypo_distances.unsqueeze(1)  # (b,1,d,1,1)
        distance = torch.sum(prob_volume * hypo_distances, dim=2)  # (b,1,h,s)

        out = {"distance": distance}

        return out
