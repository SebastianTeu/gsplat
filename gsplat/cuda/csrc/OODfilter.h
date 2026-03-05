#pragma once
#include <torch/extension.h>

namespace gsplat {

void launch_ood_filter_counts(
    const at::Tensor means,     // [N, 3]
    const at::Tensor quats,     // [N, 4]
    const at::Tensor scales,    // [N, 3]
    const at::Tensor opacities, // [N]
    const at::Tensor viewmat,   // [4, 4]
    const at::Tensor K,         // [3, 3]
    const int W,                // width
    const int H,                // height
    const float xg_thresh,      // threshold for instability metric
    const int nx,               // number of rays in x direction for sampling
    const int ny,               // number of rays in y direction for sampling
    const float near_plane,     
    at::Tensor reject_counts,   // [N]
    at::Tensor total_counts     // [N]
);

} // namespace gsplat