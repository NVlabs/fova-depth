# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

from typing import Optional, Tuple
import glob
import json
import os
import pickle
from collections import defaultdict

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

from datasets.temporal_dataset_base import TemporalDatasetBase
from nvtorchcam.cameras import PinholeCamera
from nvtorchcam.utils import normalized_intrinsics_from_pixel_intrinsics


class DDADTemporalDataset(TemporalDatasetBase):
    def __init__(
        self,
        dataset_path: str = "/mnt/disk1/Downloads/DDAD/ddad_resize",
        cameras: list = [
            "CAMERA_01",
            "CAMERA_05",
            "CAMERA_06",
            "CAMERA_07",
            "CAMERA_08",
            "CAMERA_09",
        ],
        init_resize: Optional[Tuple[int, int]] = None,
        scene_nums: range = None,
        num_forward_context: int = 1,
        num_backward_context: int = 1,
        mode: str = "index_context",
        mask_ego_vehical_depth: bool = False,
        forward_index_context_step: Optional[int] = None,
        backward_index_context_step: Optional[int] = None,
        nominal_forward_context_distance: Optional[float] = None,
        forward_look_ahead: Optional[int] = None,
        nominal_backward_context_distance: Optional[float] = None,
        backward_look_ahead: Optional[int] = None,
        normalize_trans: bool = True,
        start_ref_frames_remove: int = 0,
        end_ref_frames_remove: int = 0,
        distance_filter_threshold: Optional[float] = None,
        gpu_transforms: Optional[nn.Module] = None,
    ):
        super().__init__(
            num_forward_context,
            num_backward_context,
            mode,
            forward_index_context_step,
            backward_index_context_step,
            nominal_forward_context_distance,
            forward_look_ahead,
            nominal_backward_context_distance,
            backward_look_ahead,
            normalize_trans,
            start_ref_frames_remove=start_ref_frames_remove,
            end_ref_frames_remove=end_ref_frames_remove,
            distance_filter_threshold=distance_filter_threshold,
            gpu_transforms=gpu_transforms,
        )

        self.init_resize = init_resize
        self.dataset_path = dataset_path
        self.cameras = cameras
        self.mask_ego_vehical_depth = mask_ego_vehical_depth

        cache_path = "datasets/ddad_temporal_cache/ddad_pair_cache.pkl"
        if os.path.exists(cache_path):
            print("loading ddad cache")
            scene_dicts = pickle.load(open(cache_path, "rb"))
        else:
            print("crawling ddad")
            # os.makedirs('datasets/ddad_temporal_cache')
            scene_dicts = get_ddad_scene_dict(dataset_path, range(0, 200))
            remove_missing_in_scene_dict(dataset_path, scene_dicts)
            # pickle.dump(scene_dicts,open(cache_path,'wb'))
            print("done crawling ddad")

        sequence_dict = {}
        for scene_num in scene_nums:
            for cam in self.cameras:
                scene_name = "%06d" % scene_num
                sequence_dict[scene_name + "_" + cam] = scene_dicts[scene_name][cam]

        self.sequence_dict = sequence_dict
        self.setup()

    def load_frame(self, sequence_key, frame_idx, index_in_sample):
        seq = self.sequence_dict[sequence_key]
        frame = seq["frame_list"][frame_idx]

        im_path = os.path.join(self.dataset_path, sequence_key[0:6], frame["im_path"])
        image = imageio.imread(im_path).astype(np.float32) / 255
        if self.init_resize is not None:
            image = cv2.resize(
                image, (self.init_resize[1], self.init_resize[0]), interpolation=cv2.INTER_LINEAR
            )
        image = torch.from_numpy(image).permute(2, 0, 1)

        base_name = os.path.splitext(os.path.basename(im_path))[0]
        sparse_depth_path = os.path.join(
            self.dataset_path, sequence_key[0:6], "depth", sequence_key[7:], base_name + ".npy"
        )
        sparse_depth_dict = np.load(sparse_depth_path, allow_pickle=True)[()]
        xy, val = sparse_depth_dict["xy"], sparse_depth_dict["values"]
        u = xy[:, 1]
        v = xy[:, 0]
        if self.init_resize is None:
            init_resize = (608, 968)
        else:
            init_resize = self.init_resize

        u = ((u + 0.5) * (init_resize[0] / 1216)) - 0.5
        v = ((v + 0.5) * (init_resize[1] / 1936)) - 0.5
        u = np.round(u)
        v = np.round(v)
        u = np.clip(u, 0, init_resize[0] - 1).astype(np.uint)
        v = np.clip(v, 0, init_resize[1] - 1).astype(np.uint)

        depth = np.zeros(image.shape[1:3]) * np.nan
        depth[u, v] = val

        if self.mask_ego_vehical_depth:
            mask_path = os.path.join(
                self.dataset_path, "masks", sequence_key[7:], sequence_key[:6], "mask.png"
            )
            mask = imageio.imread(mask_path)[:, :, 0]
            mask = cv2.resize(
                mask, (init_resize[1], init_resize[0]), interpolation=cv2.INTER_NEAREST
            )
            depth[mask > 1] = np.nan

        depth = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0)

        to_world = frame["to_world"]
        to_world = torch.from_numpy(to_world).float()

        intrinsics = torch.from_numpy(seq["intrinsics"])
        intrinsics = normalized_intrinsics_from_pixel_intrinsics(intrinsics.float(), (1216, 1936))
        camera = PinholeCamera.make(intrinsics)

        sample = {"image": image, "depth": depth, "camera": camera, "to_world": to_world}

        return sample


def ddad_pose_to_matrix(pose):
    R = Rotation.from_quat(
        [
            pose["rotation"]["qx"],
            pose["rotation"]["qy"],
            pose["rotation"]["qz"],
            pose["rotation"]["qw"],
        ]
    ).as_matrix()
    trans = np.array([pose["translation"]["x"], pose["translation"]["y"], pose["translation"]["z"]])
    to_world = np.eye(4)
    to_world[:3, :3] = R
    to_world[:3, 3] = trans

    return to_world


def make_frame_list(pose_data):
    out_dict = defaultdict(list)

    for pd in pose_data:
        if pd["id"]["name"] == "LIDAR":
            continue

        t = pd["id"]["timestamp"]
        pose = pd["datum"]["image"]["pose"]
        im_path = pd["datum"]["image"]["filename"]
        im_path = im_path[:-4] + ".jpg"

        to_world = ddad_pose_to_matrix(pose)

        out_dict[pd["id"]["name"]].append(
            {"to_world": to_world, "im_path": im_path, "time_stamp": t}
        )

    for k in out_dict.keys():
        out_dict[k] = sorted(out_dict[k], key=lambda x: x["time_stamp"])

    out_dict2 = {}
    for k, v in out_dict.items():
        out_dict2[k] = {"frame_list": v}
    return out_dict2


def get_ddad_scene_dict(dataset_path, scene_nums):
    scene_dicts = {}
    for scene_num in scene_nums:
        scene_name = "%06d" % scene_num
        cam_file = glob.glob(os.path.join(dataset_path, scene_name, "calibration", "*.json"))[0]
        with open(cam_file, "r") as f:
            cam_data = json.load(f)

        pose_file = glob.glob(os.path.join(dataset_path, scene_name, "*.json"))[0]
        with open(pose_file, "r") as f:
            pose_data = json.load(f)

        scene_dict = make_frame_list(pose_data["data"])
        for cam_name, extrinsics, intrinsics in zip(
            cam_data["names"], cam_data["extrinsics"], cam_data["intrinsics"]
        ):
            if cam_name == "LIDAR":
                continue

            K = np.eye(3)
            K[0, 0] = intrinsics["fx"]
            K[1, 1] = intrinsics["fy"]
            K[0, 2] = intrinsics["cx"]
            K[1, 2] = intrinsics["cy"]

            scene_dict[cam_name]["intrinsics"] = K
            scene_dict[cam_name]["extrinsics"] = ddad_pose_to_matrix(extrinsics)

        scene_dicts[scene_name] = scene_dict

    return scene_dicts


def test_frame_exists(dataset_path, scene_name, cam_name, frame):
    im_path = os.path.join(dataset_path, scene_name, frame["im_path"])
    im_exists = os.path.exists(im_path)
    base_name = os.path.splitext(os.path.basename(im_path))[0]
    depth_path = os.path.join(dataset_path, scene_name, "depth", cam_name, base_name + ".npy")
    depth_exists = os.path.exists(depth_path)
    ret_val = depth_exists & im_exists
    return ret_val


def remove_missing_in_scene_dict(dataset_path, scene_dicts):
    for scene_name, scene in scene_dicts.items():
        for cam_name, cam in scene.items():
            cam["frame_list"] = [
                f
                for f in cam["frame_list"]
                if test_frame_exists(dataset_path, scene_name, cam_name, f)
            ]
