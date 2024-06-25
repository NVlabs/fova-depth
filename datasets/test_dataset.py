import os

import imageio
import numpy as np
import torch
import lightning as pl
from lightning.pytorch.cli import LightningCLI

import nvtorchcam.warpings as warpings
from nvtorchcam.utils import flatten_cubemap_for_visual
from data_processing.write_ply import write_ply_standard
from models.shared.warp_to_canon import WarpToCanon
from models.cube.cube_conv import test_sparse_padding


class DummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--type_to_test", default="train")  # train test val
        parser.add_argument("--sample_number", type=int, default=0)
        parser.add_argument("--canon_type", type=str, default="erp")
        parser.add_argument("--normalize_baseline", action="store_true")
        parser.add_argument("--test_cube_padding", action="store_true")


def cli_main():
    out_dir = "test_dataset_output"
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda"
    cli = MyLightningCLI(DummyModel, run=False)

    if cli.config.type_to_test == "train":
        dataset = cli.datamodule.train_dataset
    elif cli.config.type_to_test == "test":
        dataset = cli.datamodule.test_datasets[0]
    elif cli.config.type_to_test == "val":
        dataset = cli.datamodule.val_datasets[0]
    else:
        raise RuntimeError(
            "type_to_test must be train, test, or val. Got: %s" % cli.config.type_to_test
        )

    sample = dataset[cli.config.sample_number]
    for k, v in sample.items():
        if k != "idx":
            sample[k] = sample[k].unsqueeze(0).to(device)

    if cli.config.canon_type == "erp":
        warp_to_canon = WarpToCanon(
            (512, 1024),
            canonical_model="erp",
            normalize_baseline=cli.config.normalize_baseline,
            aug_intrinsic_rotation="full",
        )
    elif cli.config.canon_type == "cube":
        warp_to_canon = WarpToCanon(
            (384 * 6, 384),
            canonical_model="cube",
            normalize_baseline=cli.config.normalize_baseline,
            aug_intrinsic_rotation="full",
        )
    else:
        raise RuntimeError("canon_type mush be cube or erp, Got: %s" % cli.config.canon_type)

    if dataset.gpu_transforms is not None:
        sample = dataset.gpu_transforms(sample)

    sample = warp_to_canon(sample)

    distance = sample["distance"].squeeze(0)
    to_world = sample["to_world"].squeeze(0)
    image = sample["image"].squeeze(0)
    cam = sample["camera"].squeeze(0)
    point_cloud, _ = cam.unproject_depth(distance, to_world=to_world, depth_is_along_ray=True)
    for im_num in range(image.size(0)):
        im = image[im_num].permute(1, 2, 0)
        imageio.imwrite(
            os.path.join(out_dir, "image_%d.jpg" % im_num),
            (im * 255).cpu().numpy().astype(np.uint8),
        )
        pc = point_cloud[im_num].reshape(-1, 3)
        pc_color = im.reshape(-1, 3)
        mask = torch.all(torch.isfinite(pc), dim=1)
        pc = pc[mask, :]
        pc_color = pc_color[mask, :]
        write_ply_standard(os.path.join(out_dir, "pc_%d.ply" % im_num), pc, colors=pc_color)

    for src_num in range(1, image.size(0)):
        im_rec, _, _ = warpings.stereo_rectify(
            image[[0, src_num]], cam[[0, src_num]], to_world[[0, src_num]], (512, 1024)
        )
        im_rec = im_rec.permute(0, 2, 3, 1).flatten(0, 1)
        baseline = torch.norm(to_world[0, :3, 3] - to_world[src_num, :3, 3])
        print("baseline: ", baseline)
        imageio.imwrite(
            os.path.join(out_dir, "rect_ref_src%d.jpg" % src_num),
            (im_rec * 255).cpu().numpy().astype(np.uint8),
        )

    distance = sample["canon_distance"].squeeze(0)
    to_world = sample["canon_to_world"].squeeze(0)
    image = sample["canon_image"].squeeze(0)
    if cli.config.canon_type == "cube":
        vis_image = flatten_cubemap_for_visual(image)
    else:
        vis_image = image
    cam = sample["canon_camera"].squeeze(0)
    point_cloud, _ = cam.unproject_depth(distance, to_world=to_world, depth_is_along_ray=True)
    for im_num in range(image.size(0)):
        im = image[im_num].permute(1, 2, 0)
        im_vis = vis_image[im_num].permute(1, 2, 0)
        imageio.imwrite(
            os.path.join(out_dir, "canon_image_%d.jpg" % im_num),
            (im_vis * 255).cpu().numpy().astype(np.uint8),
        )
        pc = point_cloud[im_num].reshape(-1, 3)
        pc_color = im.reshape(-1, 3)
        mask = torch.all(torch.isfinite(pc), dim=1)
        pc = pc[mask, :]
        pc_color = pc_color[mask, :]
        write_ply_standard(os.path.join(out_dir, "canon_pc_%d.ply" % im_num), pc, colors=pc_color)

    if cli.config.test_cube_padding and cli.config.canon_type == "cube":
        cubemap = sample["canon_image"]
        cubemap = cubemap.squeeze(0).unflatten(2, (6, -1)).transpose(1, 2)
        test_sparse_padding(cubemap, pad_width=128)


if __name__ == "__main__":
    with torch.no_grad():
        cli_main()
