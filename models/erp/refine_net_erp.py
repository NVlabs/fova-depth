# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.erp.midas_erp import ERPMidasNet


class RefineNetMono(nn.Module):
    def __init__(self, model="resnet34", pretrained=True, scale_factor=1, pred_residual=True):
        super().__init__()

        self.network = ERPMidasNet(
            input_ch=4, model=model, pretrained=pretrained, use_new_output=True
        )

        self.scale_factor = scale_factor
        self.pred_residual = pred_residual

    def forward(self, image, initial):
        # (b,3,h,w) (b,1,h,w)
        with torch.no_grad():
            input = torch.cat([initial, image], dim=1)

        residual = self.network(input)
        if self.pred_residual:
            refined = self.scale_factor * residual + input[:, 0:1, ...]
        else:
            refined = residual

        out = {"refined": refined}
        return out
