#pragma once
#include <torch/extension.h>

namespace gsplat {

void launch_ood_filter_counts(
    const at::Tensor means,
    const at::Tensor quats,
    const at::Tensor scales,
    const at::Tensor opacities,
    const at::Tensor viewmat,
    const at::Tensor K,
    const int W,
    const int H,
    const float xg_thresh,
    const int nx,
    const int ny,
    const float near_plane,
    at::Tensor reject_counts,
    at::Tensor total_counts
);

} // namespace gsplat