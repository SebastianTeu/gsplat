#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD

#include "Common.h"
#include "OODfilter.h"

namespace gsplat {

void ood_filter_radii(
    const at::Tensor means,         // [..., N, 3]
    const at::Tensor quats,         // [..., N, 4]
    at::Tensor radii,               // [..., N]
    const at::Tensor visibility,    // [N] bool
    float threshold
) 
{
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(radii);
    CHECK_INPUT(visibility);

    TORCH_CHECK(visibility.dim() == 1);
    TORCH_CHECK(visibility.size(0) == means.size(0));
    TORCH_CHECK(visibility.scalar_type() == torch::kBool);

    auto render_mask = visibility.clone();

    launch_ood_filter_kernel(
        means,
        quats,
        visibility,
        threshold,
        render_mask
    );

    auto mask = render_mask.view({1, -1, 1}).expand_as(radii);
    radii.masked_fill_(~mask, 0);
}

} // namespace gsplat