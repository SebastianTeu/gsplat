#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "Common.h"
#include "OODfilter.h"

namespace gsplat {

template <typename scalar_t>
__device__ inline void quat_rotate(
    const scalar_t* q,
    scalar_t& x, scalar_t& y, scalar_t& z
) {
    // q assumed normalized
    scalar_t qx = q[0], qy = q[1], qz = q[2], qw = q[3];

    scalar_t tx = 2 * (qy * z - qz * y);
    scalar_t ty = 2 * (qz * x - qx * z);
    scalar_t tz = 2 * (qx * y - qy * x);

    x += qw * tx + (qy * tz - qz * ty);
    y += qw * ty + (qz * tx - qx * tz);
    z += qw * tz + (qx * ty - qy * tx);
}

template <typename scalar_t>
__global__ void ood_filter_kernel(
    const int N,
    const scalar_t* __restrict__ means,   // [N, 3]
    const scalar_t* __restrict__ quats,   // [N, 4]
    const bool* __restrict__ visibility,  // [N]
    const float threshold,
    bool* __restrict__ render_mask        // [N]
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    // If gaussian already isn't visible, skip
    if (!visibility[k]) {
        render_mask[k] = false;
        return;
    }

    // TODO: This math is not getting good results and needs to be redone closer to EV3DGS paper.
    // It is mostly just a placeholder as the code is compiling as of now.

    // Vector to gaussian center
    scalar_t vx = means[3*k + 0];
    scalar_t vy = means[3*k + 1];
    scalar_t vz = means[3*k + 2];

    // Normalize vector
    scalar_t inv_norm = rsqrt(vx*vx + vy*vy + vz*vz + 1e-6);
    vx *= inv_norm;
    vy *= inv_norm;
    vz *= inv_norm;

    // Put vector to center into gaussians local coordinate space
    quat_rotate(quats + 4*k, vx, vy, vz);

    // Calculate how aligned the vector is to gaussian's principle axis
    scalar_t px = vx * vx;
    scalar_t py = vy * vy;
    scalar_t pz = vz * vz;

    // Score how strongly aligned direction is to gaussian axis
    scalar_t score =
        fabs(px - scalar_t(1.0/3.0)) +
        fabs(py - scalar_t(1.0/3.0)) +
        fabs(pz - scalar_t(1.0/3.0));

    // // Print first 10 scores for debugging
    // if (k < 10) {
    //     printf("Thread %d: score = %f\n", k, static_cast<float>(score));
    // }

    // If gaussian is being looked at edge on, don't render it
    render_mask[k] = (score <= threshold);
}

void launch_ood_filter_kernel(
    const at::Tensor means,
    const at::Tensor quats,
    const at::Tensor visibility,
    float threshold,
    at::Tensor render_mask
) {
    const int N = means.size(0);

    dim3 threads(256);
    dim3 grid((N + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(means.scalar_type(), "ood_filter_kernel", [&]() {
        ood_filter_kernel<scalar_t>
        <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            N,
            means.data_ptr<scalar_t>(),
            quats.data_ptr<scalar_t>(),
            visibility.data_ptr<bool>(),
            threshold,
            render_mask.data_ptr<bool>()
        );
    });
}

} // namespace gsplat