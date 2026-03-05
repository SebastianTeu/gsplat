#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD

#include "Common.h"
#include "OODfilter.h"

namespace gsplat {

std::tuple<at::Tensor, at::Tensor> ood_filter(
    const at::Tensor means,         // [N, 3]
    const at::Tensor quats,         // [N, 4]
    const at::Tensor scales,        // [N, 3]
    const at::Tensor opacities,     // [N]
    const at::Tensor viewmat,       // [4, 4]
    const at::Tensor K,             // [3, 3]
    const int W,                    // width
    const int H,                    // height
    const float xg_thresh,          // threshold for instability metric
    const int nx,                   // number of rays in x direction for sampling
    const int ny,                   // number of rays in y direction for sampling
    const float near_plane
) 
{
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(opacities);
    CHECK_INPUT(K);

    auto reject_counts = at::zeros(opacities.sizes(), at::dtype(torch::kInt32).device(means.device()));
    auto total_counts = at::zeros(opacities.sizes(), at::dtype(torch::kInt32).device(means.device()));
    launch_ood_filter_counts(
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
        reject_counts,
        total_counts
    );
    return std::make_tuple(reject_counts, total_counts);
}

} // namespace gsplat