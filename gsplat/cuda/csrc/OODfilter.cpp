#include "OODfilter.h"
#include "Common.h"

namespace gsplat {

void ood_filter(
    // TODO: Add support for batching to improve performance

    // inputs
    const at::Tensor means,     // [N, 3]
    const at::Tensor quats,     // [N, 4]
    const at::Tensor scales,    // [N, 3]
    const at::Tensor viewmats,  // [C, 4, 4]
    float threshold,

    // outputs
    at::Tensor used_count,      // [N]
    at::Tensor reject_count     // [N]
) 
{
    DEVICE_GUARD(means); 
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(used_count);
    CHECK_INPUT(reject_count);

    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3, "means must be of shape [N, 3]");
    TORCH_CHECK(quats.dim() == 2 && quats.size(1) == 4, "quats must be of shape [N, 4]");
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3, "scales must be of shape [N, 3]");
    TORCH_CHECK(viewmats.dim() == 3 && viewmats.size(1) == 4 && viewmats.size(2) == 4, "viewmats must be of shape [C, 4, 4]");
    TORCH_CHECK(used_count.dim() == 1 && used_count.size(0) == means.size(0), "used_count must be of shape [N]");
    TORCH_CHECK(reject_count.dim() == 1 && reject_count.size(0) == means.size(0), "reject_count must be of shape [N]");
    TORCH_CHECK(used_count.scalar_type() == at::kInt, "used_count must be int32");
    TORCH_CHECK(reject_count.scalar_type() == at::kInt, "reject_count must be int32");
    const int N = means.size(0);

    launch_ood_filter_kernel(
        means,
        quats,
        scales,
        viewmats,
        threshold,
        used_count,
        reject_count
    );
}

} // namespace gsplat