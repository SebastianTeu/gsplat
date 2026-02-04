#pragma once
#include <torch/extension.h>

namespace gsplat {

void launch_ood_filter_kernel(
    const at::Tensor means,        // [..., N, 3]
    const at::Tensor quats,        // [..., N, 4]
    const at::Tensor visibility,   // [N]
    float threshold,
    at::Tensor render_mask         // [N]
);

} // namespace gsplat