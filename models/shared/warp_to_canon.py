# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import torch.nn as nn
from nvtorchcam.cameras import EquirectangularCamera, CubeCamera
from nvtorchcam.warpings import resample_by_intrinsics
from kornia.geometry.liegroup import So3


class WarpToCanon(nn.Module):
    def __init__(
        self, h_w, canonical_model="erp", normalize_baseline=True, aug_intrinsic_rotation="none"
    ):
        super().__init__()
        assert aug_intrinsic_rotation in ["full", "none"]
        assert canonical_model in ["erp", "cube"]
        self.h_w = h_w

        self.normalize_baseline = normalize_baseline
        self.aug_intrinsic_rotation = aug_intrinsic_rotation
        self.canonical_model = canonical_model

    def forward(self, sample):

        to_world = sample["to_world"].clone()
        distance = sample["distance"]  # (b,M,1,h,w)
        device = to_world.device
        b, M = to_world.shape[:2]

        if self.canonical_model == "erp":
            canonical_camera = EquirectangularCamera.make(batch_shape=(b, M))
        elif self.canonical_model == "cube":
            canonical_camera = CubeCamera.make((b, M))

        canonical_camera = canonical_camera.to(to_world.device)
        sample["canon_camera"] = canonical_camera
        # transform depth and extrinsics some camera are at locations (0,0,0), (0,0,1)
        t0 = to_world[:, 0, :3, 3]
        t1 = to_world[:, 1, :3, 3]

        delta_t = t1 - t0  # (b,3)

        baseline_distance = torch.norm(delta_t, dim=1).clamp(min=1e-6)  # (b)
        if self.normalize_baseline:
            to_world[:, :, :3, 3] = to_world[:, :, :3, 3] / baseline_distance.reshape(-1, 1, 1)
            distance = distance / baseline_distance.reshape(-1, 1, 1, 1, 1)
            sample["baseline_distance"] = baseline_distance
        else:
            sample["baseline_distance"] = torch.ones_like(baseline_distance)

        # Augmentation
        # get new to world
        if self.training and (self.aug_intrinsic_rotation == "full"):
            R_new_to_world = (
                So3.random(batch_size=b * M, device=device).matrix().reshape(b, M, 3, 3)
            )
        else:
            R_new_to_world = to_world[:, :, :3, :3]

        R_old_to_world_inv = to_world[:, :, :3, :3].transpose(-1, -2)
        R = torch.bmm(R_old_to_world_inv.flatten(0, 1), R_new_to_world.flatten(0, 1)).unflatten(
            0, (b, M)
        )

        canon_image_and_distance, _ = resample_by_intrinsics(
            [sample["image"], distance],
            sample["camera"],
            canonical_camera,
            self.h_w,
            rotation_trg_to_src=R,
            interp_mode=["bilinear", "nearest"],
            padding_mode="border",
            depth_is_along_ray=True,
            set_invalid_pix_to_nan=True,
        )

        canon_image = canon_image_and_distance[0]  # (b,M,3,h,w)
        canon_distance = canon_image_and_distance[1]  # (b,M,1,h,w)

        canonical_to_world = to_world.clone()
        canonical_to_world[:, :, :3, :3] = R_new_to_world
        sample["canon_rotation"] = R
        sample["canon_to_world"] = canonical_to_world
        sample["canon_image"] = canon_image
        sample["canon_distance"] = canon_distance

        return sample
