import os
import pickle
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import PIL.Image
import torch
import torch.utils.data as data
from imageio import imread
from jsonargparse import lazy_instance

import nvtorchcam.utils as utils
from nvtorchcam.cameras import PinholeCamera


def load_as_float(path):
    return imread(path).astype(np.float32)


def load_png(path):
    return np.array(PIL.Image.open(path).convert("RGB")).astype(np.float32)


def load_h5(path):
    return np.transpose(np.array(h5py.File(path, "r")["result"]), (1, 2, 0))


def read_pose(path, fid):
    with open(path) as f:
        line = f.readline()
        while line:
            if "Frame" in line:
                if fid == int(line[6:]):
                    line = f.readline()
                    pose_tgt = np.array(line[2:])
                    line = f.readline()
                    pose_src = np.array(line[2:])
                    return pose_tgt, pose_src
            line = f.readline()
        return None


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
    root/scene_1/0000000.jpg
    root/scene_1/0000001.jpg
    ..
    root/scene_1/cam.txt
    root/scene_2/0000000.jpg
    .

    transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, ttype="train.txt"):

        self.root = root
        self.ttype = ttype

        print("crawling scannet")
        self.crawl_folders()
        print("done crawling scannet")

    def crawl_folders(self):
        cache_path = "datasets/scannet_cache/scan_net_" + self.ttype[:-4] + "_dump.pkl"
        if os.path.exists(cache_path):
            sequence_set = pickle.load(open(cache_path, "rb"))
            print("skipping crawl")
        else:
            sequence_set = []
            imgs = np.genfromtxt(
                os.path.join(
                    self.root,
                    "new_orders/" + self.ttype[:-4] + "/" + self.ttype[:-4] + "new_orders_v.list",
                ),
                delimiter=" ",
                dtype="unicode",
            )
            imgs = imgs[imgs[:, 0].argsort()]

            for i in range(len(imgs)):
                if self.ttype[:-4] == "train":
                    my_type = "train"
                elif self.ttype[:-4] == "test":
                    my_type = "val"

                scene = os.path.join(my_type, imgs[i, 0][2:-9])

                img = imgs[i, 0][-9:][:-5] + ".jpg"
                n_img = imgs[i, 1] + ".jpg"
                intrinsics = (
                    np.genfromtxt(os.path.join(self.root, scene, "intrinsic/intrinsic_depth.txt"))
                    .astype(np.float32)
                    .reshape((4, 4))[:3, :3]
                )
                gt_nmap = os.path.join(self.root, "normals/", scene, img[:-4] + "_normal.npy")
                depth = os.path.join(self.root, scene, "depth/", (img[:-4] + ".npy"))
                pose_tgt = os.path.join(self.root, scene, "pose/", (img[:-4] + ".txt"))
                n_index = [n_img]
                sample = {
                    "intrinsics": intrinsics,
                    "tgt": os.path.join(self.root, scene, "color/" + img),
                    "tgt_depth": depth,
                    "ref_imgs": [],
                    "pose_tgt": pose_tgt,
                    "pose_src": [],
                    "gt_nmap": gt_nmap,
                    "ref_depths": [],
                }
                for j in n_index:
                    sample["ref_imgs"].append(os.path.join(self.root, scene, "color/" + j))
                    sample["ref_depths"].append(
                        os.path.join(self.root, scene, "depth/", (j[:-4] + ".npy"))
                    )
                    sample["pose_src"].append(
                        os.path.join(self.root, scene, "pose/", (j[:-4] + ".txt"))
                    )

                    # make relative to root directory
                for k in ["tgt", "tgt_depth", "pose_tgt", "gt_nmap"]:
                    sample[k] = os.path.relpath(sample[k], self.root)
                for k in ["ref_imgs", "ref_depths", "pose_src"]:
                    sample[k] = [os.path.relpath(x, self.root) for x in sample[k]]

                sequence_set.append(sample)

            os.makedirs(os.path.split(cache_path)[0], exist_ok=True)
            pickle.dump(sequence_set, open(cache_path, "wb"))

        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]

        ref_img = load_as_float(os.path.join(self.root, sample["tgt"])) / 255
        ref_depth = np.load(os.path.join(self.root, sample["tgt_depth"]))
        ref_pose = (
            np.genfromtxt(os.path.join(self.root, sample["pose_tgt"]))
            .astype(np.float32)
            .reshape((4, 4))
        )

        src_poses = []
        for p in sample["pose_src"]:
            p_src = np.genfromtxt(os.path.join(self.root, p)).astype(np.float32).reshape((4, 4))
            src_poses.append(p_src)

        src_depths = [
            np.load(os.path.join(self.root, ref_depth)) for ref_depth in sample["ref_depths"]
        ]
        src_imgs = [
            load_as_float(os.path.join(self.root, ref_img)) / 255 for ref_img in sample["ref_imgs"]
        ]

        images = [ref_img] + src_imgs
        depths = [ref_depth] + src_depths
        to_worlds = [ref_pose] + src_poses

        intrinsics = sample["intrinsics"]
        return images, depths, to_worlds, intrinsics

    def __len__(self):
        return len(self.samples)


class ScanNetDataset(data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        train_val: str,
        init_resize: Optional[Tuple[int, int]] = None,
        gpu_transforms: torch.nn.Module = lazy_instance(torch.nn.Identity),
    ):
        if train_val == "train":
            ttype = "train.txt"
        elif train_val == "val":
            ttype = "test.txt"
        else:
            raise RuntimeError("train_val must either be either train or val")

        self.wrapped_dataset = SequenceFolder(dataset_path, ttype=ttype)
        self.gpu_transforms = gpu_transforms
        self.init_resize = init_resize

    def __len__(self):
        return self.wrapped_dataset.__len__()

    def __getitem__(self, idx):
        images, depths, to_worlds, intrinsics = self.wrapped_dataset.__getitem__(idx)
        depths = [(x / 1000.0).astype(np.float32) for x in depths]

        if self.init_resize is not None:
            images = [cv2.resize(x, (self.init_resize[1], self.init_resize[0])) for x in images]
            depths = [
                cv2.resize(
                    x, (self.init_resize[1], self.init_resize[0]), interpolation=cv2.INTER_NEAREST
                )
                for x in depths
            ]

        intrinsics = np.copy(intrinsics)
        n_intrinsics = utils.normalized_intrinsics_from_pixel_intrinsics(
            torch.from_numpy(intrinsics), (480, 640)
        )

        n_intrinsics = n_intrinsics.expand(len(images), -1, -1)
        cameras = PinholeCamera.make(intrinsics=n_intrinsics)

        images = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2)
        depths = torch.from_numpy(np.stack(depths, axis=0)).unsqueeze(1)

        sample = {}
        sample["idx"] = idx
        sample["image"] = images
        sample["depth"] = depths
        sample["camera"] = cameras
        sample["to_world"] = torch.from_numpy(np.stack(to_worlds, axis=0))

        return sample


if __name__ == "__main__":

    dataset_path = "/media/daniel/drive3/Datasets/scannet"
    dataset = ScanNetDataset(dataset_path, "val", init_resize=None)
    print(len(dataset))
    batch_size = 2
    device = "cuda"
    sample = dataset[0]
    print(sample.keys())
    for k, v in sample.items():
        if k != "idx":
            print(k, v.shape)
