# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import torch.nn as nn
import torch.nn.functional as F
from nvtorchcam.utils import samples_from_image, samples_from_cubemap, apply_affine
from nvtorchcam.cameras import CubeCamera


class BasicWeightModule(nn.Module):
    def __init__(self, groups, hidden=16):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(groups, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, groups),
            nn.ReLU(True),
            nn.Linear(groups, 1),
        )

    def forward(self, vol):
        # (b,M-1,g,d,h,w)
        # return #(b,M-1,1,d,h,w) torch.sum(out,dim=1) == 1
        vol = vol.permute(0, 1, 3, 4, 5, 2)
        x = self.net(vol)
        x = torch.softmax(x, dim=1)
        x = x.permute(0, 1, 5, 2, 3, 4)
        return x


def cost_volume(image, cameras, to_world, hypo_depths, sphere_sweep=True):
    # (b,M,c,h,w) Cam[b,M] (b,M,4,4) (b,d,h,w)|(b,d,1,1)
    # return (b,1,f,d,h,w), (b,M-1,c,d,h,w)
    src_to_worlds = to_world[:, 1:, :, :]
    src_cams = cameras[:, 1:]
    src_images = image[:, 1:]

    ref_to_world = to_world[:, 0, :, :]
    ref_cam = cameras[:, 0:1]
    ref_image = image[:, 0:1]

    origin, dir, valid = ref_cam.get_camera_rays(ref_image.shape[3:5], sphere_sweep)
    pc_ref = origin + dir * hypo_depths.unsqueeze(-1)  # (b,d,h,w,3)
    ref_to_src = torch.einsum(
        "bmij,bjk->bmik", torch.inverse(src_to_worlds.float()), ref_to_world
    )  # (b,M-1,4,4)
    pc_src = apply_affine(
        ref_to_src, pc_ref.unsqueeze(1).expand(-1, ref_to_src.size(1), -1, -1, -1, -1)
    )  # (b,M-1,d,h,w,3)
    pc_src_proj, _, valid2 = src_cams.project_to_pixel(
        pc_src, depth_is_along_ray=sphere_sweep
    )  # (b,M-1,d,h,w,2)
    if isinstance(cameras, CubeCamera):
        src_vols = samples_from_cubemap(src_images, pc_src_proj)  # (b,M-1,d,c,h,w)
    else:
        src_vols = samples_from_image(src_images, pc_src_proj)  # (b,M-1,d,c,h,w)
    ref_vol = ref_image.unsqueeze(3).expand(-1, -1, -1, src_vols.size(3), -1, -1)
    return ref_vol, src_vols
