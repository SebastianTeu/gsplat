#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "Common.h"
#include "OODfilter.h"

namespace gsplat {

template <typename scalar_t>
__global__ void ood_filter_kernel(
    // inputs
    const int N,
    const scalar_t *__restrict__ means,     // [N, 3]
    const scalar_t *__restrict__ quats,     // [N, 4]
    const scalar_t *__restrict__ scales,    // [N, 3]
    const scalar_t *__restrict__ viewmats,  // [C, 4, 4]
    const int C,
    const float threshold,
    // outputs
    int *__restrict__ used_count,
    int *__restrict__ reject_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    scalar_t mx = means[idx * 3 + 0];
    scalar_t my = means[idx * 3 + 1];
    scalar_t mz = means[idx * 3 + 2];

    scalar_t qx = quats[idx * 4 + 0];
    scalar_t qy = quats[idx * 4 + 1];
    scalar_t qz = quats[idx * 4 + 2];
    scalar_t qw = quats[idx * 4 + 3];

    scalar_t sx = scales[idx * 3 + 0];
    scalar_t sy = scales[idx * 3 + 1];  
    scalar_t sz = scales[idx * 3 + 2];

    // TODO: Implement math to measure unstableness of each gaussian
    

    scalar_t score = 0; // placeholder

    atomicAdd(&used_count[idx], 1);

    if(score < threshold) {
        atomicAdd(&reject_count[idx], 1);
    }
}

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
) {
    const int N = means.size(0);
    const int C = viewmats.size(0);

    int64_t n_elements = N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0;

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(), "ood_filter_kernel", ([&] {
            ood_filter_kernel<scalar_t>
            <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                N,
                means.data_ptr<scalar_t>(),
                quats.data_ptr<scalar_t>(),
                scales.data_ptr<scalar_t>(),
                viewmats.data_ptr<scalar_t>(),
                C,
                threshold,
                used_count.data_ptr<int>(),
                reject_count.data_ptr<int>());
        }));
}

} // namespace gsplat