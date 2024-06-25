import torch
import torch.nn as nn
import torch.nn.functional as F
from models.shared.warp_to_canon import WarpToCanon
from models.cube.feature_extractor_cube import SparsePadFeatureExtractor
from models.cube.depth_net_cube import DepthModuleDotMulti
from models.cube.refine_net_cube import RefineNetMono
from nvtorchcam.warpings import resample_by_intrinsics


class FOVADepthCube(nn.Module):
    def __init__(
        self,
        greater_than_two_images: bool = False,
        pretrained: bool = True,
        cube_side: int = 256,
        num_depth_hypos: int = 48,
        refine_net_backbone: str = "resnet34",
        aug_intrinsic_rotation: str = "full",
        normalize_baseline: bool = True,
        groups: int = 8,
        refine_scale_factor: float = 0.01,
        refine_pred_residual: bool = True,
        warp_to_original_cam: bool = False,
        add_feature_mixing: bool = False,
    ):
        super().__init__()

        self.num_depth_hypos = num_depth_hypos

        self.prepare_data = WarpToCanon(
            (6 * cube_side, cube_side),
            canonical_model="cube",
            normalize_baseline=normalize_baseline,
            aug_intrinsic_rotation=aug_intrinsic_rotation,
        )

        self.feature = SparsePadFeatureExtractor(
            pretrained=pretrained, add_mixing=add_feature_mixing
        )
        print("feature out channels", self.feature.out_channels)

        self.depth_net = DepthModuleDotMulti(
            groups=groups, greater_than_two=greater_than_two_images
        )
        self.refine_network = RefineNetMono(
            model=refine_net_backbone,
            pretrained=pretrained,
            scale_factor=refine_scale_factor,
            pred_residual=refine_pred_residual,
        )

        self.warp_to_original_cam = warp_to_original_cam

    def forward(self, sample):
        with torch.no_grad():
            sample = self.prepare_data(sample)
        canon_image = sample["canon_image"]
        b, M = canon_image.shape[:2]
        device = canon_image.device
        canon_to_world = sample["canon_to_world"]
        canon_camera = sample["canon_camera"]

        # move to internal cubemap rep
        canon_image = canon_image.unflatten(-2, (6, -1)).transpose(2, 3)  # (b,M,6,3,w,w)
        idx_to_process = torch.any(torch.isfinite(canon_image.flatten(3)), dim=-1)  # (b,M,6)
        canon_image = torch.nan_to_num(canon_image)

        # (b,M,f_s,h,w)
        features = self.feature(canon_image, idx_to_process)

        with torch.no_grad():
            # f(x) = (2/pi) ( 1/tan(pi/2 x) )
            pi_over2 = 3.14159 / 2
            min_depth = 0.1
            max_depth = 1000
            upper = (1 / pi_over2) * torch.atan((1 / pi_over2) / torch.tensor(min_depth))
            lower = (1 / pi_over2) * torch.atan((1 / pi_over2) / torch.tensor(max_depth))
            x = torch.linspace(lower, upper, self.num_depth_hypos, device=device)
            depth_hypos = (1 / pi_over2) * (1 / torch.tan(x * pi_over2))
            depth_hypos = depth_hypos.reshape(1, -1).expand(b, -1)

        out = self.depth_net(
            features, canon_camera, canon_to_world, depth_hypos, idx_to_process[:, 0]
        )
        pred_dist = out["distance"]  # (b,6,1,cube_side/4,cube_side/4)
        pred_inv_dist = 1 / pred_dist.detach().clamp(min=min_depth)

        # Refine stage
        up_pred_inv_dist = F.interpolate(
            pred_inv_dist.flatten(0, 1), scale_factor=4, mode="nearest"
        ).unflatten(0, (-1, 6))

        refine_out = self.refine_network(canon_image[:, 0], up_pred_inv_dist, idx_to_process[:, 0])
        pred_refined_inv_dist = refine_out["refined"]
        pred_refined_dist = 1 / pred_refined_inv_dist.clamp(min=(1 / max_depth))

        # back to nvtorchcam cube format
        pred_dist = pred_dist.squeeze(2).unsqueeze(1).flatten(2, 3)
        pred_refined_dist = pred_refined_dist.squeeze(2).unsqueeze(1).flatten(2, 3)

        net_out = {"pred_distance": pred_dist, "pred_refined_distance": pred_refined_dist}

        if self.warp_to_original_cam:
            RT = sample["canon_rotation"][:, 0, :, :].transpose(-1, -2)

            recon_depth, _ = resample_by_intrinsics(
                net_out["pred_distance"],
                canon_camera[:, 0],
                sample["camera"][:, 0],
                sample["image"].shape[3:5],
                rotation_trg_to_src=RT,
                interp_mode="bilinear",
                depth_is_along_ray=True,
            )
            net_out["pred_distance_original"] = recon_depth * sample["baseline_distance"].reshape(
                -1, 1, 1, 1
            )

            recon_depth, _ = resample_by_intrinsics(
                net_out["pred_refined_distance"],
                canon_camera[:, 0],
                sample["camera"][:, 0],
                sample["image"].shape[3:5],
                rotation_trg_to_src=RT,
                interp_mode="bilinear",
                depth_is_along_ray=True,
            )
            net_out["pred_refined_distance_original"] = recon_depth * sample[
                "baseline_distance"
            ].reshape(-1, 1, 1, 1)

        return net_out
