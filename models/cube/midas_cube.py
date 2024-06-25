# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cube.cube_conv import (
    dense_to_sparse,
    sparse_to_dense,
    SparseCubeConv2d,
    SparseBilinearUpsample,
)
from models.cube.resnet_sparse_cube import resnet34


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = SparseCubeConv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = SparseCubeConv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, args):
        x, to_process, batch_size = args
        out = self.relu(x)
        out = self.conv1((out, to_process, batch_size))[0]
        out = self.relu(out)
        out = self.conv2((out, to_process, batch_size))[0]

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.upsampler = SparseBilinearUpsample(2)
        self.resConfUnit = ResidualConvUnit(features)

    def forward(self, x, to_process, batch_size, res=None):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = x

        if res is not None:
            output = output + self.resConfUnit((res, to_process, batch_size))

        output = self.resConfUnit((output, to_process, batch_size))

        output, _, _ = self.upsampler((output, to_process, batch_size))
        # output = nn.functional.interpolate(output, scale_factor=2, mode="nearest")
        return output


class MidasSparsePadFeatureExtractor(nn.Module):
    def __init__(self, resnet_model=resnet34, pretrained=False, input_ch=3):
        super().__init__()
        resnet = resnet_model(pretrained=pretrained)

        if input_ch == 3:
            self.resnet_conv1 = resnet.conv1
        else:
            self.resnet_conv1 = SparseCubeConv2d(input_ch, 64, 7, stride=2, bias=False, padding=3)

        self.resnet_bn1 = resnet.bn1
        self.resnet_relu = resnet.relu
        self.resnet_maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, sparse_cube_maps, to_process, batch_size):

        x = self.resnet_conv1((sparse_cube_maps, to_process, batch_size))[0]
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool((x, to_process, batch_size))

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        return layer1, layer2, layer3, layer4


class OutputConv(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = SparseCubeConv2d(features, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = SparseCubeConv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.upsampler = SparseBilinearUpsample(2)

    def forward(self, x, to_process, batch_size):
        x = self.conv1((x, to_process, batch_size))[0]
        x = self.conv2((x, to_process, batch_size))[0]
        x, _, _ = self.upsampler((x, to_process, batch_size))
        return x


class OutputConvNew(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = SparseCubeConv2d(features, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = SparseCubeConv2d(128, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = SparseCubeConv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.upsampler = SparseBilinearUpsample(2)

    def forward(self, x, to_process, batch_size):
        x = self.conv1((x, to_process, batch_size))[0]
        x = nn.functional.relu(x)
        x = self.conv2((x, to_process, batch_size))[0]
        x = nn.functional.relu(x)
        x, _, _ = self.upsampler((x, to_process, batch_size))
        x = self.conv3((x, to_process, batch_size))[0]
        return x


class CubeMidasNet(nn.Module):

    def __init__(
        self,
        features=256,
        out_ch=1,
        pretrained=True,
        input_ch=3,
        model="resnet34",
        use_new_output=False,
    ):
        super().__init__()

        if model == "resnet34":
            resnet_model = resnet34
            layer_out_channels = [64, 128, 256, 512]
        elif model == "resnet50":
            resnet_model = resnet50
            layer_out_channels = [256, 512, 1024, 2048]
        else:
            raise RuntimeError("Cube Midas feature extractor resnet name not recognized", model)

        self.pretrained = MidasSparsePadFeatureExtractor(
            pretrained=pretrained, input_ch=input_ch, resnet_model=resnet_model
        )

        self.refinenet4 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet1 = FeatureFusionBlock(features)

        self.layer1_rn = SparseCubeConv2d(
            layer_out_channels[0], features, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer2_rn = SparseCubeConv2d(
            layer_out_channels[1], features, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer3_rn = SparseCubeConv2d(
            layer_out_channels[2], features, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer4_rn = SparseCubeConv2d(
            layer_out_channels[3], features, kernel_size=3, stride=1, padding=1, bias=False
        )

        if use_new_output:
            self.output_conv = OutputConvNew(features)
        else:
            self.output_conv = OutputConv(features)

    def forward(self, sparse_cube_maps, to_process, batch_size):

        layer_1, layer_2, layer_3, layer_4 = self.pretrained(
            sparse_cube_maps, to_process, batch_size
        )

        layer_1_rn = self.layer1_rn(layer_1)[0]
        layer_2_rn = self.layer2_rn(layer_2)[0]
        layer_3_rn = self.layer3_rn(layer_3)[0]
        layer_4_rn = self.layer4_rn(layer_4)[0]

        path_4 = self.refinenet4(layer_4_rn, to_process, batch_size)
        path_3 = self.refinenet3(path_4, to_process, batch_size, res=layer_3_rn)
        path_2 = self.refinenet2(path_3, to_process, batch_size, res=layer_2_rn)
        path_1 = self.refinenet1(path_2, to_process, batch_size, layer_1_rn)

        out = self.output_conv(path_1, to_process, batch_size)
        return out


def load_pretrained(net, cp_path="/home/dlichy/Desktop/Project/ComparisonCode/MiDaS/model.pt"):
    sd_original = torch.load(cp_path)
    sd_cube = net.state_dict()

    new_sd = {}
    for k, v in sd_cube.items():
        k_list = k.split(".")
        if k_list[0] == "pretrained":
            if k_list[-2] == "conv":
                k_list.pop(-2)
            if k_list[-3] == "downsample" and k_list[-2] == "2":
                k_list[-2] = "1"

            if k_list[1][:-1] == "layer" and k_list[1] != "layer1":
                old_key = ".".join(k_list)
                new_sd[k] = sd_original[old_key]
            elif k_list[1][:-1] == "layer" and k_list[1] == "layer1":
                k_list.insert(2, "4")
                old_key = ".".join(k_list)
                new_sd[k] = sd_original[old_key]
            else:
                if k_list[1] == "resnet_conv1":
                    old_key = "pretrained.layer1.0.weight"
                else:
                    k_list[1] = "layer1.1"
                    old_key = ".".join(k_list)
                new_sd[k] = sd_original[old_key]

        elif k_list[0] == "output_conv":
            if k_list[-2] == "conv":
                k_list.pop(-2)
            if k_list[1] == "conv1":
                k_list[1] = "0"
            if k_list[1] == "conv2":
                k_list[1] = "1"

            k_list.insert(0, "scratch")
            old_key = ".".join(k_list)
            new_sd[k] = sd_original[old_key]
        else:
            if k_list[-2] == "conv":
                k_list.pop(-2)
            k_list.insert(0, "scratch")
            old_key = ".".join(k_list)
            new_sd[k] = sd_original[old_key]

    # for k,v in sd_cube.items():
    #    if k not in new_sd:
    #        print('not yet in: ', k)
    #    else:
    #        if new_sd[k].shape != v.shape:
    #            print('wrong shape: ', k)

    # count = 0
    # for k,v in sd_cube.items():
    #    count += v.numel()

    # count2 = 0
    # for k,v in sd_original.items():
    #    count2 += v.numel()
    # print(count,count2)

    net.load_state_dict(new_sd)


def basic_single_image_test():
    device = "cuda"
    net = CubeMidasNet(input_ch=3, model="resnet50").to(device)
    load_pretrained(net)

    image0 = imageio.imread(
        "/home/dlichy/Desktop/Project/ComparisonCode/Normal-Assisted-Stereo/scannet/val/scene0011_00/color/0000.jpg"
    )
    image = image0.astype(np.float32) / 255
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    image = image[:, :, :, 80:-80]
    print(image.shape)
    batch_size = 1
    to_process = torch.tensor([0], device=device)

    depth = net(image, to_process, batch_size)

    out_dir = "test_cube_midas_basic"
    os.makedirs(out_dir, exist_ok=True)
    imageio.imwrite(os.path.join(out_dir, "image.jpg"), image.squeeze(0).permute(1, 2, 0).cpu())
    imageio.imwrite(os.path.join(out_dir, "depth.png"), depth.detach().squeeze().cpu())


def test_on_cube_map():
    device = "cuda"
    dataset = Matterport360NNDataset(
        "/home/dlichy/Downloads/matterport360/data",
        "datasets/matterport_split/scenes_train.txt",
        init_resize=(2 * 128, 2 * 256),
    )
    wrapped_dataset = AnyDatasetWrapperModule(dataset, batch_size=2, num_workers=1, shuffle=True)
    data_loader = wrapped_dataset.val_dataloader()
    batch_transforms = wrapped_dataset.get_val_gpu_transforms()

    it = iter(data_loader)
    sample = it.__next__()

    sample = sample_to_device(sample)

    sample = batch_transforms(sample)

    data_prep = CubeDataPrepare(512, aug_intrinsic_rotation="none").to(device)

    sample_cube = data_prep(sample)
    batch_size = sample_cube["gt_cube_image"].size(0)

    # (b,6,3,w,w)
    image = sample_cube["gt_cube_image"][:, 0]
    idx_flat = sample_cube["idx_to_process"][:, 0].flatten(0, 1)
    idx_flat[:] = False
    idx_flat[2] = True
    idx_flat[4] = True
    # (k)
    to_process = torch.arange(idx_flat.size(0), device=device)[idx_flat]
    print(to_process)

    # (k,c,w,w)
    sparse_input = dense_to_sparse(image, to_process)

    net = CubeMidasNet(input_ch=3, model="resnet50").to(device)
    load_pretrained(net)

    sparse_output = net(sparse_input, to_process, batch_size)

    dense_output = sparse_to_dense(sparse_output, to_process, batch_size)

    cube_image_flat = flatten_cubemap(image)
    cube_depth_flat = flatten_cubemap(dense_output)

    save_batch = 0
    out_dir = "test_cube_midas"
    os.makedirs(out_dir, exist_ok=True)
    imageio.imwrite(
        os.path.join(out_dir, "image.jpg"), cube_image_flat[save_batch].permute(1, 2, 0).cpu()
    )
    imageio.imwrite(
        os.path.join(out_dir, "depth.png"), cube_depth_flat[save_batch].detach().squeeze().cpu()
    )


if __name__ == "__main__":
    import time
    import imageio
    import os
    import numpy as np
    import sys

    sys.path.append(".")
    from datasets.any_dataset_wrapper_module import AnyDatasetWrapperModule
    from datasets.matterport360_nn_dataset import Matterport360NNDataset
    from datasets.test_loader_warp import sample_to_device
    from models.sparse_cube_mvs_consolidated.my_modules import CubeDataPrepare
    from data_processing.cube_map_diff_rast import flatten_cubemap

    net = CubeMidasNet()
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    exit()

    basic_single_image_test()
    test_on_cube_map()
