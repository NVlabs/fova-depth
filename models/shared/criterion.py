# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import torch.nn as nn
import torch.nn.functional as F


def nan_L1_loss(gt, x):
    # gt may have nans
    # (b,k)
    # return (b)
    mask = torch.isfinite(gt)
    gt[~mask] = 0
    x = x * mask
    total = torch.sum(torch.abs(gt - x), dim=1)
    count = torch.sum(mask, dim=1).clamp(min=1)
    return total / count


class Criterion(nn.Module):
    def __init__(
        self,
        min_depth=0.1,
        distance_keys=["pred_distance", "pred_refined_distance"],
        distance_weights=[1, 1],
    ):
        super().__init__()
        self.min_depth = min_depth
        self.distance_keys = distance_keys
        self.distance_weights = distance_weights

    def get_depth_loss(self, gt_depth, pred_depth):
        # (b,1,h,w) (b,1,h',w')
        # return (b,) (b,) (b,)
        if gt_depth.shape[-2:] != pred_depth.shape[-2:]:
            # OK for cubemap because of nearest neighbor interpolation
            gt_depth = F.interpolate(gt_depth, size=pred_depth.shape[-2:], mode="nearest")
        gt_depth = gt_depth.flatten(1)
        pred_depth = pred_depth.flatten(1)
        gt_depth_log = torch.log(gt_depth.clamp(min=self.min_depth))
        pred_depth_log = torch.log(pred_depth.clamp(min=self.min_depth))
        depth_loss_per_item = nan_L1_loss(gt_depth_log, pred_depth_log)

        abs_rel_per_item = torch.nanmean(
            (torch.abs(pred_depth - gt_depth) / gt_depth.clamp(min=self.min_depth)).squeeze(-1),
            dim=1,
        ).detach()

        return depth_loss_per_item, abs_rel_per_item

    def forward(self, net_out, sample):
        gt_distance = sample["canon_distance"][:, 0]

        total_loss_per_item = 0
        loss_dict = {}
        for key, weight in zip(self.distance_keys, self.distance_weights):
            depth_loss, abs_rel = self.get_depth_loss(gt_distance, net_out[key])
            total_loss_per_item = total_loss_per_item + weight * depth_loss

            loss_dict[key + "_loss"] = depth_loss.detach()
            loss_dict[key + "_abs_rel"] = abs_rel.detach()

        loss = torch.mean(total_loss_per_item)

        return loss, loss_dict
