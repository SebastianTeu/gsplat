import math
import argparse

from gsplat.cuda._wrapper import (
    ood_filter,
)

import tqdm
import torch
from torch import Tensor
import torch.nn.functional as F 
from typing import Tuple, Optional

from gsplat.distributed import cli

def compute_gaussian_instability(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    viewmat: Tensor,
    K: Tensor,
    W: int,
    H: int,
    xg_thresh: float = 1e-13,
    nx: int = 100,
    ny: int = 100,
    near_plane: float = 0.01,
) -> Tuple[Tensor, Tensor]:
    means = means.contiguous()
    quats = quats.contiguous()
    scales = scales.contiguous()
    opacities = opacities.contiguous()
    K = K.contiguous()
    viewmat = viewmat.contiguous()

    # CUDA kernel call
    return ood_filter(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        viewmat=viewmat,
        K=K,
        W=W,
        H=H,
        xg_thresh=xg_thresh,
        nx=nx,
        ny=ny,
        near_plane=near_plane,
    )

# Create a view matrix given camera position, target, and up vector
def look_at(device: torch.device, cam_pos: Tensor, target: Tensor, up: Tensor = torch.tensor([0., 0., 1.])
) -> Tensor:
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

def compute_pca_ellipsoid(
    means: Tensor,
    percentile: float = 0.9,
) -> Tuple[Tensor, Tensor, Tensor]:
    
    center = means.mean(dim=0)
    centered = means - center

    # PCA with SVD
    _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
    axes = Vt # each row is a principal axis

    # Project points onto each axis
    projections = centered @ axes.T
    # Compute distance along each axis by the given percentile
    radii = torch.quantile(projections.abs(), percentile, dim=0)

    return center, axes, radii


def sample_ellipsoid_cameras(
    center: Tensor,         # [3]
    axes: Tensor,           # [3, 3], row = PCA axis
    radii: Tensor,          # [3]
    num_slices: int = 7,
    num_cameras_per_slice: int = 8,
) -> list[Tensor]:
    
    device = center.device

    # From PCA: axis[0] = major, axis[1] = middle, axis[2] = minor
    # We slice along the minor axis
    up_axis   = axes[2]  # slicing direction
    ax0       = axes[0]  # equatorial axis
    ax1       = axes[1]  # equatorial axis
    r0, r1, r2 = radii[0], radii[1], radii[2]

    cam_positions = []

    # Slice along the up axis
    # num_slices + 2 since we drop the poles
    slice_positions = torch.linspace(-1.0, 1.0, num_slices + 2, device=device)[1:-1]

    for slice_h in slice_positions:
        h = slice_h * r2 # Height along the up axis

        # Radius of the ellipse slice at this height
        slice_scale = (1.0 - slice_h*slice_h).clamp(min=0.0).sqrt()

        for i in range(num_cameras_per_slice):
            angle = 2.0 * math.pi * i / num_cameras_per_slice
            # Point on the ellipse in the slice plane
            cam_pos = (
                center
                + h * up_axis
                + slice_scale * r0 * math.cos(angle) * ax0
                + slice_scale * r1 * math.sin(angle) * ax1
            )
            cam_positions.append(cam_pos)

    return cam_positions


def compute_instability_mask(
    device: torch.device,
    means: Tensor,                          # [N, 3]
    quats: Tensor,                          # [N, 4]
    scales: Tensor,                         # [N, 3]
    opacities: Tensor,                      # [N]
    xg_thresh: float = 1e-13,
    ratio_thresh: float = 0.01,
    nx: int = 100,
    ny: int = 100,
    near_plane: Optional[float] = None,
    percentile: float = 0.9,
    ellipsoid_scalars: list[float] = [2.0, 4.0],
    num_slices: int = 7,
    num_cameras_per_slice: int = 8,
) -> Tensor:
    
    assert means.shape[0] == quats.shape[0] == scales.shape[0] == opacities.shape[0], "All Gaussian parameter tensors must have the same number of Gaussians"
    assert means.ndim == 2 and means.shape[1] == 3
    assert quats.ndim == 2 and quats.shape[1] == 4
    assert scales.ndim == 2 and scales.shape[1] == 3
    assert opacities.ndim == 1

    N = len(means)
    reject_counts = torch.zeros(N, dtype=torch.int32, device=device)
    total_counts  = torch.zeros(N, dtype=torch.int32, device=device)

    # Get PCA ellipsoid parameters from Gaussian means
    with torch.no_grad():
        center, axes, base_radii  = compute_pca_ellipsoid(means, percentile=percentile)

    if near_plane is None:
        near_plane = base_radii.min().item() * 0.05

    # Synthetic intrinsics. These don't matter much just need to be reasonable
    fx = fy = 400.0
    W, H = 1600, 1200
    K_synth = torch.tensor([
        [fx,  0., W/2],
        [0.,  fy, H/2],
        [0.,  0., 1. ]
    ], dtype=torch.float32, device=device)

    # Get camera positions on ellipsoid surface
    exterior_positions = []

    for s in ellipsoid_scalars:
        radii = base_radii * s

        positions = sample_ellipsoid_cameras(
            center, axes, radii,
            num_slices, num_cameras_per_slice
        )

        exterior_positions.extend(positions)

    pbar = tqdm.tqdm(total=len(exterior_positions), desc="Evaluating OOD cameras")

    # For each camera, compute instability of all Gaussians and accumulate reject/total counts
    for cam_pos in exterior_positions:
        viewmat = look_at(device, cam_pos, center) # Looking at the scene center
        r, t = compute_gaussian_instability(
            means=means, quats=quats, scales=scales, opacities=opacities,
            viewmat=viewmat, K=K_synth, W=W, H=H,
            xg_thresh=xg_thresh, nx=nx, ny=ny, near_plane=near_plane,
        )
        reject_counts += r
        total_counts  += t
        pbar.update(1)

    pbar.close()

    ratio = reject_counts.float() / total_counts.float().clamp(min=1)
    unstable_mask = ratio > ratio_thresh

    print(f"Pruning {unstable_mask.sum()} unstable gaussians")
    return unstable_mask

def main(local_rank: int, world_rank: int, world_size: int, args):
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
        xg_thresh=args.xg_thresh,
        ratio_thresh=args.ratio_thresh,
        nx=args.nx,
        ny=args.ny,
        near_plane=args.near_plane,
        percentile=args.pca_percentile,
        ellipsoid_scalars=args.ellipsoid_scalars,
        num_slices=args.num_slices,
        num_cameras_per_slice=args.num_cameras_per_slice,
    )

    # Filter out unstable Gaussians based on the computed mask
    filtered_means       = means[~unstable_mask]
    filtered_quats       = original_quats[~unstable_mask]
    filtered_scales      = original_scales[~unstable_mask]
    filtered_opacities   = original_opacities[~unstable_mask]
    filtered_sh0         = sh0[~unstable_mask, ...]
    filtered_shN         = shN[~unstable_mask, ...]
    print("Number of Gaussians after filter:", len(filtered_means))

    # Saving the filtered Gaussians back to a new checkpoint
    output_ckpt_path = args.ckpt.replace(
        ".pt",
        f"_ood_r{args.ratio_thresh:g}"
        f"_p{args.pca_percentile:g}"
        f"_s{args.num_slices}x{args.num_cameras_per_slice}"
        f"_e{'-'.join(map(str, args.ellipsoid_scalars))}"
        f"_n{len(filtered_means)}.pt"
    )

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
        "--xg_thresh", type=float, default=1e-13, help="threshold for Gaussian instability"
    )
    parser.add_argument(
        "--ratio_thresh", type=float, default=0.01, help="threshold for rejecting Gaussians based on instability ratio"
    )
    parser.add_argument(
        "--nx", type=int, default=100, help="number of rays to sample along x-axis for instability evaluation"
    )
    parser.add_argument(
        "--ny", type=int, default=100, help="number of rays to sample along y-axis for instability evaluation"
    )
    parser.add_argument(
        "--near_plane", type=float, default=None, help="near plane for instability evaluation"
    )
    parser.add_argument(
        "--ellipsoid_scalars", type=float, default=[2.0, 4.0], nargs="+", help="List of scalar multipliers applied to the given percentile PCA radii (e.g. 1.5 2.0, 4.0)"
    )
    parser.add_argument(
        "--num_slices", type=int, default=7, help="number of horizontal ellipsoid slices to sample cameras from"
    )
    parser.add_argument(
        "--num_cameras_per_slice", type=int, default=8, help="number of cameras equally spaced around each slice"
    )
    parser.add_argument(
        "--pca_percentile", type=float, default=0.9, help="percentile of PCA radii to use for camera sampling"
    )
    args = parser.parse_args()
    cli(main, args)