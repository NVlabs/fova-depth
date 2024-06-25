# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import glob
import os
import shutil

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from dgp.datasets import SynchronizedSceneDataset, SynchronizedScene


def get_sparse_depth(depth):
    valid = depth > 1e-6
    values = depth[valid]
    grid = np.indices(depth.shape)
    xy = grid[:, valid]
    xy = np.stack((xy[1], xy[0]), axis=1)
    return values, xy


datum_names = [
    "CAMERA_01",
    "CAMERA_05",
    "CAMERA_06",
    "CAMERA_07",
    "CAMERA_08",
    "CAMERA_09",
    "lidar",
]
camera_names = datum_names[:-1]


def export_scene(original_scene_path, new_scene_path, new_im_size=(968, 608)):
    scene_json = glob.glob(os.path.join(original_scene_path, "scene_*.json"))[0]
    dataset = SynchronizedScene(
        scene_json,
        datum_names=datum_names,
        generate_depth_from_datum="lidar",
    )

    shutil.copy2(scene_json, new_scene_path)
    shutil.copytree(
        os.path.join(original_scene_path, "calibration"),
        os.path.join(new_scene_path, "calibration"),
    )

    rgb_dir = os.path.join(new_scene_path, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    for cam_name in camera_names:
        os.makedirs(os.path.join(rgb_dir, cam_name), exist_ok=True)

    depth_dir = os.path.join(new_scene_path, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    for cam_name in camera_names:
        os.makedirs(os.path.join(depth_dir, cam_name), exist_ok=True)

    for data in dataset:
        for i in range(0, 6):

            m = data[0][i]
            cam_name = m["datum_name"]
            t = str(m["timestamp"])
            rgb = np.array(m["rgb"])
            rgb = cv2.resize(rgb, new_im_size, interpolation=cv2.INTER_LINEAR)
            imageio.imwrite(os.path.join(rgb_dir, cam_name, t + ".jpg"), rgb)

            depth = m["depth"]
            values, xy = get_sparse_depth(depth)
            xy = xy.astype(np.uint16)
            sparse_depth = {"values": values, "xy": xy}
            np.save(os.path.join(depth_dir, cam_name, t + ".npy"), sparse_depth)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ddad_path", type=str, default="ddad_train_val")
    parser.add_argument("--resized_ddad_path", type=str, default="ddad_resize")

    args = parser.parse_args()
    os.makedirs(args.resized_ddad_path, exist_ok=True)
    original_scene_paths = sorted(glob.glob(os.path.join(args.ddad_path, "000*")))

    for original_sp in tqdm(original_scene_paths):
        new_scene_path = os.path.join(args.resized_ddad_path, os.path.basename(original_sp))
        os.makedirs(new_scene_path, exist_ok=True)
        export_scene(original_sp, new_scene_path, new_im_size=(968, 608))
