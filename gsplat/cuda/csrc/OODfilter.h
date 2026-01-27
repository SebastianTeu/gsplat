#pragma once

#include <cstdint>

namespace at {
class Tensor;
}

namespace gsplat {

void launch_ood_filter_kernel(
    // inputs
    const at::Tensor means,
    const at::Tensor quats,
    const at::Tensor scales,
    const at::Tensor viewmats,
    float threshold,

    // outputs
    at::Tensor used_count,
    at::Tensor reject_count
);

} // namespace gsplat