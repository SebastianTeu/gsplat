#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

#include "Common.h"
#include "OODfilter.h"

namespace gsplat {

#define MAX_HITS 2048 // Max amount of gaussians that can contribute to one ray

// Device helpers

__device__ inline float dot3(const float a[3], const float b[3]) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__device__ inline void quat_to_rotmat(const float q[4], float R[9]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = 1 - 2*(y*y + z*z);  R[1] = 2*(x*y - w*z);      R[2] = 2*(x*z + w*y);
    R[3] = 2*(x*y + w*z);      R[4] = 1 - 2*(x*x + z*z);  R[5] = 2*(y*z - w*x);
    R[6] = 2*(x*z - w*y);      R[7] = 2*(y*z + w*x);      R[8] = 1 - 2*(x*x + y*y);
}

// Computes cam_local = S^{-1} R^T (cam_pos - mean)
//          ray_local = S^{-1} R^T ray_d
// then t = -dot(ray_local, cam_local) / dot(ray_local, ray_local)
// Returns false if t <= 0.
__device__ inline bool compute_t_local(
    const float cam_pos[3],
    const float ray_d[3],
    const float mean[3],
    const float scale[3],
    const float R[9],
    float& t_out,
    float cam_local[3],
    float ray_local[3]
) {
    float offset[3] = {
        cam_pos[0] - mean[0],
        cam_pos[1] - mean[1],
        cam_pos[2] - mean[2]
    };

    for (int r = 0; r < 3; r++) {
        float ro = 0.f, rr = 0.f;
        for (int c = 0; c < 3; c++) {
            ro += R[c * 3 + r] * offset[c];   // R^T @ offset
            rr += R[c * 3 + r] * ray_d[c];    // R^T @ ray_d
        }
        // Ignoring scaling as per the EV3DGS paper
        cam_local[r] = ro;
        ray_local[r] = rr;
    }

    float myA = dot3(ray_local, ray_local);
    float myB = 2.f * dot3(ray_local, cam_local);
    float t   = -myB / (2.f * myA);

    if (t <= 0.f) return false;
    t_out = t;
    return true;
}

__device__ inline void compute_xg(
    const float cam_local[3], const float ray_local[3], float t, float x_g[3]
) {
    x_g[0] = cam_local[0] + t * ray_local[0];
    x_g[1] = cam_local[1] + t * ray_local[1];
    x_g[2] = cam_local[2] + t * ray_local[2];
}

__device__ inline float compute_alpha(const float x_g[3], float opacity) {
    float myPow = fminf(-0.5f * dot3(x_g, x_g), 0.f);
    return fminf(0.99f, opacity * expf(myPow));
}

__device__ inline float compute_gs_xg(
    const float x_g[3], float myAlpha, float myT, const float sum_xg[3]
) {
    float d[3] = {
        myAlpha * myT * (sum_xg[0] - x_g[0]),
        myAlpha * myT * (sum_xg[1] - x_g[1]),
        myAlpha * myT * (sum_xg[2] - x_g[2])
    };
    return dot3(d, d);
}

__device__ inline void update_sum_xg(
    float sum_xg[3], const float x_g[3], float myAlpha
) {
    float w = myAlpha / (1.f - myAlpha);
    sum_xg[0] += w * x_g[0];
    sum_xg[1] += w * x_g[1];
    sum_xg[2] += w * x_g[2];
}

__device__ inline void insertion_sort(float* ts, int* ids, int n) {
    for (int i = 1; i < n; i++) {
        float kt = ts[i];
        int   ki = ids[i];
        int j = i - 1;
        while (j >= 0 && ts[j] > kt) {
            ts[j+1] = ts[j];
            ids[j+1] = ids[j];
            j--;
        }
        ts[j+1] = kt;
        ids[j+1] = ki;
    }
}

// One thread per ray
template <typename scalar_t>
__global__ void ood_filter_kernel(
    const int N,
    const int num_rays,                         // nx * ny
    const int nx,
    const int ny,
    const scalar_t* __restrict__ means,         // [N, 3]
    const scalar_t* __restrict__ quats,         // [N, 4]
    const scalar_t* __restrict__ scales,        // [N, 3]
    const scalar_t* __restrict__ opacities,     // [N]
    // Camera pose (world-space)
    const float cam_pos_x, const float cam_pos_y, const float cam_pos_z,
    // cam_R: row-major 3x3 rotation (C2W rotation, columns = camera axes in world)
    const float r00, const float r01, const float r02,
    const float r10, const float r11, const float r12,
    const float r20, const float r21, const float r22,
    // Intrinsics
    const float fx, const float fy,
    const float px, const float py,
    const int W, const int H,
    // Thresholds
    const float xg_thresh,
    const float near_plane,
    // Output counters
    int* __restrict__ reject_counts,   // [N]
    int* __restrict__ total_counts     // [N]
) {
    int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= num_rays) return;

    // Decode ray_id -> pixel grid indices
    int pi = ray_id / ny;   // x index in grid
    int pj = ray_id % ny;   // y index in grid

    // Map grid indices to pixel coordinates
    float u = (nx > 1) ? (float)pi * (float)(W - 1) / (float)(nx - 1) : (float)(W / 2);
    float v = (ny > 1) ? (float)pj * (float)(H - 1) / (float)(ny - 1) : (float)(H / 2);

    // Camera position
    float cam_pos[3] = {cam_pos_x, cam_pos_y, cam_pos_z};

    // Ray direction in camera space, then rotate to world space
    float ray_cam[3] = {(u - px) / fx, (v - py) / fy, 1.f};
    float ray_len = sqrtf(ray_cam[0]*ray_cam[0] + ray_cam[1]*ray_cam[1] + ray_cam[2]*ray_cam[2]);
    ray_cam[0] /= ray_len; ray_cam[1] /= ray_len; ray_cam[2] /= ray_len;

    // cam_R @ ray_cam  (cam_R stored as individual floats to avoid array arg overhead)
    float ray_d[3] = {
        r00*ray_cam[0] + r01*ray_cam[1] + r02*ray_cam[2],
        r10*ray_cam[0] + r11*ray_cam[1] + r12*ray_cam[2],
        r20*ray_cam[0] + r21*ray_cam[1] + r22*ray_cam[2],
    };

    // Collect candidates and their depths
    float hit_t[MAX_HITS];
    int   hit_id[MAX_HITS];
    int   num_hits = 0;

    for (int i = 0; i < N; i++) {
        float mean[3]  = {(float)means[i*3],   (float)means[i*3+1],   (float)means[i*3+2]};
        float scale[3] = {(float)scales[i*3],  (float)scales[i*3+1],  (float)scales[i*3+2]};
        float quat[4]  = {(float)quats[i*4],   (float)quats[i*4+1],   (float)quats[i*4+2],  (float)quats[i*4+3]};

        float R[9];
        quat_to_rotmat(quat, R);

        float t, cam_local[3], ray_local[3];
        if (!compute_t_local(cam_pos, ray_d, mean, scale, R, t, cam_local, ray_local))
            continue;
        if (t <= near_plane)
            continue;

        if (num_hits < MAX_HITS) {
            hit_t[num_hits]  = t;
            hit_id[num_hits] = i;
            num_hits++;
        }
    }

    insertion_sort(hit_t, hit_id, num_hits);

    // Score each (ray, gaussian) pair
    float sum_xg[3] = {0.f, 0.f, 0.f};
    float myT = 1.f;

    for (int h = 0; h < num_hits; h++) {
        int   i = hit_id[h];
        float t = hit_t[h];

        float mean[3]  = {(float)means[i*3],   (float)means[i*3+1],   (float)means[i*3+2]};
        float scale[3] = {(float)scales[i*3],  (float)scales[i*3+1],  (float)scales[i*3+2]};
        float quat[4]  = {(float)quats[i*4],   (float)quats[i*4+1],   (float)quats[i*4+2],  (float)quats[i*4+3]};
        float opacity  = (float)opacities[i];

        float R[9];
        quat_to_rotmat(quat, R);

        float cam_local[3], ray_local[3], t_dummy;
        compute_t_local(cam_pos, ray_d, mean, scale, R, t_dummy, cam_local, ray_local);

        float x_g[3];
        compute_xg(cam_local, ray_local, t, x_g);

        float myAlpha = compute_alpha(x_g, opacity);
        if (myAlpha < 1.f / 255.f)
            continue;

        float gs_xg = compute_gs_xg(x_g, myAlpha, myT, sum_xg);

        atomicAdd(&total_counts[i], 1);
        if (xg_thresh > 0.f && gs_xg > xg_thresh)
            atomicAdd(&reject_counts[i], 1);
            continue; // Seeing if this will improve wall and ceiling removal

        update_sum_xg(sum_xg, x_g, myAlpha);
        myT *= (1.f - myAlpha);

        if (myT < 0.0001f)
            break;
    }
}

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
) {
    const int N = means.size(0);

    auto vm = viewmat.cpu().to(at::kFloat);
    float R00 = vm[0][0].item<float>(), R01 = vm[0][1].item<float>(), R02 = vm[0][2].item<float>(), t0 = vm[0][3].item<float>();
    float R10 = vm[1][0].item<float>(), R11 = vm[1][1].item<float>(), R12 = vm[1][2].item<float>(), t1 = vm[1][3].item<float>();
    float R20 = vm[2][0].item<float>(), R21 = vm[2][1].item<float>(), R22 = vm[2][2].item<float>(), t2 = vm[2][3].item<float>();

    float c2w_r00 = R00, c2w_r01 = R10, c2w_r02 = R20;
    float c2w_r10 = R01, c2w_r11 = R11, c2w_r12 = R21;
    float c2w_r20 = R02, c2w_r21 = R12, c2w_r22 = R22;

    float cam_pos_x = -(R00*t0 + R10*t1 + R20*t2);
    float cam_pos_y = -(R01*t0 + R11*t1 + R21*t2);
    float cam_pos_z = -(R02*t0 + R12*t1 + R22*t2);

    auto Kc = K.cpu().to(at::kFloat);
    float fx  = Kc[0][0].item<float>();
    float fy  = Kc[1][1].item<float>();
    float ppx = Kc[0][2].item<float>();
    float ppy = Kc[1][2].item<float>();

    const int num_rays = nx * ny;
    dim3 threads(256);
    dim3 grid((num_rays + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(means.scalar_type(), "ood_filter_kernel", [&]() {
        ood_filter_kernel<scalar_t><<<grid, threads, 0,
            at::cuda::getCurrentCUDAStream()>>>(
            N, num_rays, nx, ny,
            means.data_ptr<scalar_t>(),
            quats.data_ptr<scalar_t>(),
            scales.data_ptr<scalar_t>(),
            opacities.data_ptr<scalar_t>(),
            cam_pos_x, cam_pos_y, cam_pos_z,
            c2w_r00, c2w_r01, c2w_r02,
            c2w_r10, c2w_r11, c2w_r12,
            c2w_r20, c2w_r21, c2w_r22,
            fx, fy, ppx, ppy,
            W, H,
            xg_thresh, near_plane,
            reject_counts.data_ptr<int>(),
            total_counts.data_ptr<int>()
        );
    });
}

} // namespace gsplat
