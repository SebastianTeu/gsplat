import math
import argparse
import os

from gsplat.cuda._wrapper import (
    ood_filter,
)

import tqdm
import torch
import torch.nn.functional as F

from gsplat.distributed import cli

def compute_gaussian_instability(
    means,
    quats,
    scales,
    opacities,
    viewmat,
    K,
    W,
    H,
    xg_thresh=0.02,
    nx=5,
    ny=5,
    near_plane=0.01,
):
    means = means.contiguous()
    quats = quats.contiguous()
    scales = scales.contiguous()
    opacities = opacities.contiguous()
    K = K.contiguous()
    viewmat = viewmat.contiguous()

    return ood_filter(
        means,
        quats,
        scales,
        opacities,
        viewmat,
        K,
        W,
        H,
        xg_thresh,
        nx,
        ny,
        near_plane,
    )

def look_at(device, cam_pos, target, up=torch.tensor([0., 0., 1.])):
    up = up.to(device)
    forward = F.normalize(target - cam_pos, dim=0)
    right   = F.normalize(torch.cross(forward, up, dim=0), dim=0)
    up      = torch.cross(right, forward, dim=0)
    # W2C viewmat
    R = torch.stack([right, up, -forward], dim=0)  # [3,3]
    t = -(R @ cam_pos.unsqueeze(-1)).squeeze(-1)   # [3]
    viewmat = torch.eye(4, device=device)
    viewmat[:3, :3] = R
    viewmat[:3,  3] = t
    return viewmat

def compute_instability_mask(
    device,
    means,
    quats,
    scales,
    opacities,
    xg_thresh=0.02,
    ratio_thresh=0.01,
    nx=5,
    ny=5,
):
    # Compute scene center and radius from Gaussian means
    with torch.no_grad():
        scene_center = means.mean(dim=0)  # [3]
        scene_radius = (means - scene_center).norm(dim=-1).quantile(0.9) # 90th percentile distance to not be skewed by outliers

    # Synthetic intrinsics
    fx = fy = 400.0
    W, H = 1600, 1200
    K_synth = torch.tensor([
        [fx,  0., W/2],
        [0.,  fy, H/2],
        [0.,  0., 1. ]
    ], dtype=torch.float32, device=device)

    N = len(means)
    reject_counts = torch.zeros(N, dtype=torch.int32, device=device)
    total_counts  = torch.zeros(N, dtype=torch.int32, device=device)

    elevations = [-70, 0, 5, 20, 45, 70, 85]
    azimuths   = [0, 45, 90, 135, 180, 225, 270, 315]
    distances  = [1.5, 2.5, 4.0, 7.0]

    outward_cameras = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]  # add outward-facing cameras from scene center at multiple angles

    total_cams = (
        len(distances) * len(elevations) * len(azimuths)
        + len(outward_cameras)
    )

    pbar = tqdm.tqdm(total=total_cams, desc="Evaluating OOD cameras")

    for dist in distances:
        for elev_deg in elevations:
            for azim_deg in azimuths:
                elev = math.radians(elev_deg)
                azim = math.radians(azim_deg)
                cam_pos = scene_center + scene_radius * dist * torch.tensor([
                    math.cos(elev) * math.cos(azim),
                    math.cos(elev) * math.sin(azim),
                    math.sin(elev),
                ], dtype=torch.float32, device=device)

                viewmat = look_at(device, cam_pos, scene_center)

                r, t = compute_gaussian_instability(
                    means, quats, scales, opacities,
                    viewmat=viewmat,
                    K=K_synth,
                    W=W, H=H,
                    xg_thresh=xg_thresh,
                    nx=nx,
                    ny=ny,
                    near_plane=scene_radius * 0.05,
                )
                reject_counts += r
                total_counts  += t

                pbar.update(1)

    # Also add cameras looking outward from scene center
    for azim_deg in outward_cameras:
        azim = math.radians(azim_deg)
        # Camera inside the scene looking outward
        cam_pos = scene_center + scene_radius * 0.1 * torch.tensor([
            math.cos(azim), math.sin(azim), 0.
        ], dtype=torch.float32, device=device)
        # Look outward instead of at scene center
        look_target = scene_center + scene_radius * 5.0 * torch.tensor([
            math.cos(azim), math.sin(azim), 0.
        ], dtype=torch.float32, device=device)

        viewmat = look_at(device, cam_pos, look_target)
        r, t = compute_gaussian_instability(
            means, quats, scales, opacities,
            viewmat=viewmat,
            K=K_synth,
            W=W, H=H,
            xg_thresh=xg_thresh,
            nx=nx, ny=ny,
            near_plane=scene_radius * 0.05
        )
        reject_counts += r
        total_counts  += t

        pbar.update(1)

    pbar.close()

    ratio = reject_counts.float() / total_counts.float().clamp(min=1)
    unstable_mask = ratio > ratio_thresh

    print(f"Pruning {unstable_mask.sum()} unstable gaussians")
    return unstable_mask

def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []

    # Quats, scales, and opacities are stored in the checkpoint in their unnormalized/log space form, 
    # so we need to keep track of the original values to save the filtered checkpoint correctly
    original_quats, original_scales, original_opacities = [], [], []

    ckpt = torch.load(args.ckpt, map_location=device)["splats"]
    means.append(ckpt["means"])
    quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
    scales.append(torch.exp(ckpt["scales"]))
    opacities.append(torch.sigmoid(ckpt["opacities"]))
    sh0.append(ckpt["sh0"])
    shN.append(ckpt["shN"])

    original_quats.append(ckpt["quats"])
    original_scales.append(ckpt["scales"])
    original_opacities.append(ckpt["opacities"])
    
    means = torch.cat(means, dim=0)
    quats = torch.cat(quats, dim=0)
    scales = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    sh0 = torch.cat(sh0, dim=0)
    shN = torch.cat(shN, dim=0)

    original_quats = torch.cat(original_quats, dim=0)
    original_scales = torch.cat(original_scales, dim=0)
    original_opacities = torch.cat(original_opacities, dim=0)

    print("Number of Gaussians before filter:", len(means))

    unstable_mask = compute_instability_mask(
        device=device,
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        xg_thresh=0.000000000001,
        ratio_thresh=0.01,
        nx=100,
        ny=100,
    )
    filtered_means       = means[~unstable_mask]
    filtered_quats       = original_quats[~unstable_mask]
    filtered_scales      = original_scales[~unstable_mask]
    filtered_opacities   = original_opacities[~unstable_mask]
    filtered_sh0         = sh0[~unstable_mask, ...]
    filtered_shN         = shN[~unstable_mask, ...]
    print("Number of Gaussians after filter:", len(filtered_means))

    # Saving the filtered Gaussians back to a new checkpoint
    output_ckpt_path = args.ckpt.replace(".pt", "_filtered.pt")
    torch.save({
        "splats": {
            "means": filtered_means,
            "quats": filtered_quats,
            "scales": filtered_scales,
            "opacities": filtered_opacities,
            "sh0": filtered_sh0,
            "shN": filtered_shN,
        }
    }, output_ckpt_path)
    print(f"Filtered checkpoint saved to: {output_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True, default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--xg_thresh", type=float, default=0.000000000001, help="threshold for Gaussian instability"
    )
    parser.add_argument(
        "--ratio_thresh", type=float, default=0.01, help="threshold for rejecting Gaussians based on instability ratio"
    )
    parser.add_argument(
        "--nx", type=int, default=100, help="number of pixels to sample along x-axis for instability evaluation"
    )
    parser.add_argument(
        "--ny", type=int, default=100, help="number of pixels to sample along y-axis for instability evaluation"
    )
    args = parser.parse_args()
    cli(main, args)