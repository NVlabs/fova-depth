import os

import imageio
import numpy as np
import torch
import torch.nn as nn

from nvtorchcam.utils import (
    samples_from_cubemap,
    get_normalized_grid_cubemap,
    flatten_cubemap_for_visual,
)


def sparse_to_dense(sparse_cube, idx, batch_size):
    # (k,c,w,w) (k) int
    # return (b,6,c,w,w)
    dense_cube = torch.zeros(
        batch_size, 6, *sparse_cube.shape[1:], dtype=sparse_cube.dtype, device=sparse_cube.device
    )
    dense_cube.flatten(0, 1)[idx, ...] = sparse_cube
    return dense_cube


def dense_to_sparse(dense_cube, idx):
    # (b,6,c,w,w) (k)
    # return (k,c,w,w)
    return dense_cube.flatten(0, 1)[idx, ...]


def get_padding_indices(channels, width, padding):
    # return (6,c,w+2p,w+2p) (6,w+2p,w+2p)
    device = "cuda"
    pixel_cube = (
        torch.arange(channels * width * width, device=device)
        .unsqueeze(0)
        .expand(6, -1)
        .reshape(6, channels, width, width)
    )
    # (6,width+2*padding,width+2*padding,3)
    rays = get_normalized_grid_cubemap(width, device, pad=padding).unflatten(0, (6, -1))
    pixel_idx = samples_from_cubemap(
        pixel_cube.transpose(0, 1).flatten(1, 2), rays, mode="nearest"
    ).transpose(0, 1)
    pixel_idx = pixel_idx.clamp(
        min=0, max=channels * width * width - 1
    )  # why this is necessary needs more investigation
    pixel_idx = pixel_idx.long()
    # get face each index is pointing to
    face_cube = (
        torch.arange(0, 6).reshape(-1, 1, 1).expand(6, width, width).unsqueeze(0).flatten(1, 2)
    )
    face_idx = samples_from_cubemap(face_cube.to(device), rays, mode="nearest").long().squeeze(0)
    return pixel_idx, face_idx


def get_padding_indices_pad_only(channels, width, padding):
    # return (6,c,2p,w+2p) (6,c,w,2p) (6,2p,w+2p) (6,w,2p)

    pixel_idx, face_idx = get_padding_indices(
        channels, width, padding
    )  # (6,c,w+2p,w+2p) (6,w+2p,w+2p)

    pixel_idx_top = pixel_idx[:, :, :padding, :]  # (6,c,p,w+2p)
    pixel_idx_bottom = pixel_idx[:, :, -padding:, :]  # (6,c,p,w+2p)

    pixel_idx_tb = torch.cat((pixel_idx_top, pixel_idx_bottom), dim=2)  # (6,c,2p,w+wp)

    pixel_idx_left = pixel_idx[:, :, padding:-padding, :padding]  # (6,c,w,p)
    pixel_idx_right = pixel_idx[:, :, padding:-padding, -padding:]  # (6,c,w,p)

    pixel_idx_lr = torch.cat((pixel_idx_left, pixel_idx_right), dim=3)  # (6,c,w,2p)

    face_idx_top = face_idx[:, :padding, :]  # (6,p,w+2p)
    face_idx_bottom = face_idx[:, -padding:, :]  # (6,p,w+2p)

    face_idx_tb = torch.cat((face_idx_top, face_idx_bottom), dim=1)  # (6,2p,w+2p)

    face_idx_left = face_idx[:, padding:-padding, :padding]  # (6,w,p)
    face_idx_right = face_idx[:, padding:-padding, -padding:]  # (6,w,p)

    face_idx_lr = torch.cat((face_idx_left, face_idx_right), dim=2)  # (6,w,2p)

    return pixel_idx_tb, pixel_idx_lr, face_idx_tb, face_idx_lr


class SparseIdxCubePadImproved(nn.Module):
    def __init__(self, pad_width):
        super().__init__()
        self.pad_width = pad_width
        self.input_width = -1
        self.channels = -1

    # (k,c,w,w) (k)
    # return (k,c,w,w)
    def forward(self, cube, to_process, batch_size):
        # print('running sparse cube padding improved')
        k, c, w, _ = cube.shape
        device = cube.device
        if self.input_width != w or self.channels != c:

            # (6,c,2p,w+2p) (6,c,w,2p) (6,2p,w+2p) (6,w,2p)
            pixel_idx_tb, pixel_idx_lr, face_idx_tb, face_idx_lr = get_padding_indices_pad_only(
                c, w, self.pad_width
            )
            self.pixel_idx_tb = pixel_idx_tb.to(device)
            self.pixel_idx_lr = pixel_idx_lr.to(device)

            self.face_idx_tb = face_idx_tb.to(device)
            self.face_idx_lr = face_idx_lr.to(device)

            self.input_width = w
            self.channels = c

        with torch.no_grad():

            inv_to_process = -torch.ones(6 * batch_size, device=device, dtype=torch.long)
            inv_to_process[to_process] = torch.arange(k, device=device)
            # ------------calculate to bottom pad------------
            # (b,6,2p,w+2p)
            face_idx_tb = self.face_idx_tb.unsqueeze(0) + 6 * torch.arange(
                batch_size, device=device
            ).reshape(-1, 1, 1, 1)
            # (k,2p,w+2p)
            face_idx_tb = face_idx_tb.flatten(0, 1)[to_process]
            face_idx_tb = inv_to_process[face_idx_tb]
            # (k,c,2p,w+2p)
            pixel_idx_tb = self.pixel_idx_tb[to_process % 6, :, :, :]
            idx_tb = face_idx_tb.unsqueeze(1) * (c * w * w) + pixel_idx_tb
            mask_tb = idx_tb >= 0
            idx_tb = idx_tb.clamp(min=0)

            # ------------calculate to left right pad------------
            # (b,6,w,2p)
            face_idx_lr = self.face_idx_lr.unsqueeze(0) + 6 * torch.arange(
                batch_size, device=device
            ).reshape(-1, 1, 1, 1)
            # (k,w,2p)
            face_idx_lr = face_idx_lr.flatten(0, 1)[to_process]
            face_idx_lr = inv_to_process[face_idx_lr]
            # (k,c,w,2p)
            pixel_idx_lr = self.pixel_idx_lr[to_process % 6, :, :, :]
            idx_lr = face_idx_lr.unsqueeze(1) * (c * w * w) + pixel_idx_lr
            mask_lr = idx_lr >= 0
            idx_lr = idx_lr.clamp(min=0)

            flat_cube = cube.flatten(0)

        # pad_tb = flat_cube[idx_tb]*mask_tb
        # pad_lr =flat_cube[idx_lr]*mask_lr

        pad_tb = torch.index_select(flat_cube, 0, idx_tb.flatten(0)).reshape(idx_tb.shape) * mask_tb
        pad_lr = torch.index_select(flat_cube, 0, idx_lr.flatten(0)).reshape(idx_lr.shape) * mask_lr

        padded_cube = torch.cat(
            [pad_lr[:, :, :, : self.pad_width], cube, pad_lr[:, :, :, -self.pad_width :]], dim=3
        )
        padded_cube = torch.cat(
            [pad_tb[:, :, : self.pad_width, :], padded_cube, pad_tb[:, :, -self.pad_width :, :]],
            dim=2,
        )

        return padded_cube


class SparseCubeConv2d(nn.Module):
    def __init__(
        self, f_in, f_out, kernel_size=3, stride=1, padding=0, bias=True, groups=1, dilation=1
    ):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(
            f_in, f_out, kernel_size, stride, padding=0, bias=bias, groups=groups, dilation=dilation
        )
        if padding != 0:
            self.padder = SparseIdxCubePadImproved(padding)
        else:
            self.padder = None

    # (k,c,w,w) (k)
    # return (k,c,w,w)
    def forward(self, args):
        x, to_process, batch_size = args
        b = x.size(0)
        if self.padder is not None:
            x = self.padder(x, to_process, batch_size)
        x = self.conv(x)
        return x, to_process, batch_size


class SparseCubeMaxPool2d(nn.Module):

    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        if padding != 0:
            self.padder = SparseIdxCubePadImproved(padding)
        else:
            self.padder = None

    # (k,c,w,w) (k)
    # return (k,c,w,w)
    def forward(self, args):
        x, to_process, batch_size = args
        b = x.size(0)
        if self.padder is not None:
            x = self.padder(x, to_process, batch_size)
        x = self.pool(x)
        return x, to_process, batch_size


class TakeFirstLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0]


class SparseBilinearUpsample(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.padder = SparseIdxCubePadImproved(1)
        self.scale_factor = scale_factor

    # (k,c,w,w) (k)
    # return (k,c,w,w)
    def forward(self, args):
        x, to_process, batch_size = args
        x = self.padder(x, to_process, batch_size)
        x = nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        x = x[:, :, self.scale_factor : -self.scale_factor, self.scale_factor : -self.scale_factor]
        return x, to_process, batch_size


def get_cube_map_for_testing_each_side_diff(batch_size, width=64):
    # return (batch_size,6,3,width, width)
    cube_map = torch.zeros(batch_size, 6, width, width, 3)
    for i in range(1, 7):
        color = torch.tensor([int(i / 4) % 2, int(i / 2) % 2, i % 2])
        cube_map[:, i - 1, :, :, :] = color.reshape(1, 1, 3)

    cube_map = cube_map.permute(0, 1, 4, 2, 3)
    return cube_map


def test_sparse_padding(cube_map, idx_to_process=None, pad_width=16):
    batch_size = cube_map.size(0)
    if idx_to_process is None:
        idx_to_process = torch.arange(batch_size * 6)

    sparse_cube_map = dense_to_sparse(cube_map, idx_to_process)

    padder = SparseIdxCubePadImproved(pad_width).to(sparse_cube_map.device)

    padded_sparse_cube_map = padder(sparse_cube_map, idx_to_process, batch_size)

    cube_map = sparse_to_dense(sparse_cube_map, idx_to_process, batch_size)  # (b,6,3,w,w)
    padded_cube_map = sparse_to_dense(padded_sparse_cube_map, idx_to_process, batch_size)

    flat_cubemap = flatten_cubemap_for_visual(cube_map.transpose(1, 2).flatten(2, 3)).cpu()
    padded_flat_cubemap = flatten_cubemap_for_visual(
        padded_cube_map.transpose(1, 2).flatten(2, 3)
    ).cpu()

    save_dir = "test_sparse_cube_padding"
    os.makedirs(save_dir, exist_ok=True)
    for i in range(batch_size):
        imageio.imwrite(
            os.path.join(save_dir, "original_%d.png" % i),
            (flat_cubemap[i] * 255).permute(1, 2, 0).numpy().astype(np.uint8),
        )
        imageio.imwrite(
            os.path.join(save_dir, "padded_%d.png" % i),
            (padded_flat_cubemap[i] * 255).permute(1, 2, 0).numpy().astype(np.uint8),
        )


if __name__ == "__main__":
    cube_map = get_cube_map_for_testing_each_side_diff(2)
    test_sparse_padding(cube_map, idx_to_process=torch.tensor([0, 1, 2, 5, 6, 7, 8]))
