from typing import List

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

from nvtorchcam.warpings import affine_transform_image


class SequentialFromList(nn.Sequential):
    def __init__(self, module_list: List[nn.Module]):
        super().__init__(*module_list)


class RandomCropFlip(nn.Module):
    def __init__(
        self,
        crop_scale_range=None,
        flip_probability=0,
        keys=["image", "depth"],
        interp_modes=["bilinear", "nearest"],
        share_crop_across_frames=False,
    ):
        super().__init__()
        assert len(interp_modes) == len(keys)
        self.crop_scale_range = crop_scale_range
        self.flip_probability = flip_probability
        self.keys = keys
        self.interp_modes = interp_modes
        self.share_crop_across_frames = share_crop_across_frames

    def get_crop_matrix(self, N, device):
        new_half_width = (self.crop_scale_range[1] - self.crop_scale_range[0]) * torch.rand(
            N, device=device
        ) + self.crop_scale_range[0]
        # (N,2)
        new_center = (1 - new_half_width.unsqueeze(1)) * torch.rand(N, 2, device=device)

        crop = torch.zeros(N, 3, 3, device=device)
        crop[:, 0, 0] = 1 / new_half_width
        crop[:, 1, 1] = 1 / new_half_width
        crop[:, 2, 2] = 1
        crop[:, :2, 2] = new_center
        return crop

    def forward(self, sample):
        camera = sample["camera"]  # (b,M)
        to_world = sample["to_world"].clone()  # (b,M)
        device = camera.device
        b, M = camera.shape

        new_camera = camera
        if self.flip_probability > 0:
            flip = 2 * (torch.rand(b, device=device) > self.flip_probability).float() - 1

            flip3 = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(b, M, 1, 1)
            flip3[:, :, 0, 0] = flip.reshape(-1, 1)

            # conjugate by flip3
            new_camera = new_camera.affine_transform(flip3)
            new_camera = new_camera.affine_transform(flip3, multiply_on_right=True)

            # conjugate two world by flip4
            to_world[:, :, 0, :] *= flip.reshape(-1, 1, 1)
            to_world[:, :, :, 0] *= flip.reshape(-1, 1, 1)
        else:
            flip3 = torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(b, M, 1, 1)

        if self.crop_scale_range is not None:
            if self.share_crop_across_frames:
                crop_matrix = self.get_crop_matrix(b, device).unsqueeze(1).expand(b, M, 3, 3)
            else:
                crop_matrix = self.get_crop_matrix(b * M, device).reshape(b, M, 3, 3)
            new_camera = new_camera.affine_transform(crop_matrix)
        else:
            crop_matrix = torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(b, M, 1, 1)
        crop_matrix = torch.bmm(crop_matrix.flatten(0, 1), flip3.flatten(0, 1)).unflatten(0, (b, M))

        to_crop_list = [sample[k] for k in self.keys]
        cropped_list, _ = affine_transform_image(
            to_crop_list, crop_matrix, interp_mode=self.interp_modes
        )
        for k, cropped in zip(self.keys, cropped_list):
            sample[k] = cropped

        sample["to_world"] = to_world
        sample["camera"] = new_camera
        return sample


class ColorJitterBatch(nn.Module):
    # individually=False applys the same augmentation to each view.
    def __init__(
        self,
        brightness=(0.8, 1.2),
        contrast=(0.8, 1.2),
        saturation=(0.8, 1.2),
        hue=(-0.1, 0.1),
        individually=False,
    ):
        super().__init__()
        self.color_aug = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.individually = individually

    # all images in a batch get the same augmentation
    def forward(self, sample):
        for b in range(len(sample["image"])):
            params = self.color_aug.get_params(
                brightness=self.color_aug.brightness,
                contrast=self.color_aug.contrast,
                saturation=self.color_aug.saturation,
                hue=self.color_aug.hue,
            )
            for i in range(len(sample["image"][b])):
                sample["image"][b][i] = color_jitter_specify_params(sample["image"][b][i], *params)
                if self.individually:
                    params = self.color_aug.get_params(
                        brightness=self.color_aug.brightness,
                        contrast=self.color_aug.contrast,
                        saturation=self.color_aug.saturation,
                        hue=self.color_aug.hue,
                    )
        return sample


def color_jitter_specify_params(
    img, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor
):
    for fn_id in fn_idx:
        if fn_id == 0 and brightness_factor is not None:
            img = TF.adjust_brightness(img, brightness_factor)
        elif fn_id == 1 and contrast_factor is not None:
            img = TF.adjust_contrast(img, contrast_factor)
        elif fn_id == 2 and saturation_factor is not None:
            img = TF.adjust_saturation(img, saturation_factor)
        elif fn_id == 3 and hue_factor is not None:
            img = TF.adjust_hue(img, hue_factor)

    return img


class AddDistanceFromDepth(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample):
        camera = sample["camera"]  # (b,n)
        assert camera.is_central()
        depth = sample["depth"]  # (b,n,1,h,w)
        _, dirs, valid = camera.get_camera_rays(depth.shape[-2:], False)
        distance = depth * torch.norm(dirs, dim=-1).unsqueeze(2)
        distance[~valid.unsqueeze(2)] = torch.nan
        sample["distance"] = distance
        return sample
