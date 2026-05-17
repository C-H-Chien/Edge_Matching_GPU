#ifndef APPLY_PRECOMPUTED_SIFT_FILTER_CU
#define APPLY_PRECOMPUTED_SIFT_FILTER_CU

#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#include "definitions.h"
#include "gpu_settings.h"
#include "gpu_kernels.h"

// ---------------------------------------------------------------------------
// apply_precomputed_sift_filter_kernel
//
// Replaces the on-the-fly SIFT descriptor computation with a lookup into
// pre-computed OpenCV SIFT descriptor arrays, exactly matching the CPU
// apply_SIFT_filtering_quads which calls cv::norm(a.first, b.first, L2) etc.
//
// Each descriptor pair is stored as two contiguous 128-float half-descriptors:
//   d_kf_sift[mate_idx * 256 + 0..127]   = first keypoint descriptor
//   d_kf_sift[mate_idx * 256 + 128..255] = second keypoint descriptor
//
// NaN sentinel (stored for empty cv::Mat descriptors) triggers rejection.
//
// min_dist = min over all 4 combinations (kf_a/b × cf_a/b) of L2 distance.
// ---------------------------------------------------------------------------
__global__ void apply_precomputed_sift_filter_kernel(
    const Match_by_Edge_Index* d_candidates,
    const float*               d_kf_sift,   //> [N_kf × 256] floats
    const float*               d_cf_sift,   //> [N_cf × 256] floats
    Match_by_Edge_Index*       d_out,
    int*                       d_out_count,
    float*                     d_out_scores,
    int                        num_candidates,
    float                      threshold)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_candidates) return;

    const Match_by_Edge_Index m = d_candidates[tid];

    const float* kf_a = d_kf_sift + static_cast<size_t>(m.left_edge_idx)  * 256;
    const float* kf_b = kf_a + 128;
    const float* cf_a = d_cf_sift + static_cast<size_t>(m.right_edge_idx) * 256;
    const float* cf_b = cf_a + 128;

    //> Reject if any half-descriptor is the NaN sentinel (empty cv::Mat).
    if (isnan(kf_a[0]) || isnan(kf_b[0]) || isnan(cf_a[0]) || isnan(cf_b[0])) return;

    //> Compute L2 for all 4 half-descriptor combinations.
    float d_aa = 0.0f, d_ab = 0.0f, d_ba = 0.0f, d_bb = 0.0f;
    for (int k = 0; k < 128; ++k) {
        float aa = kf_a[k] - cf_a[k]; d_aa += aa * aa;
        float ab = kf_a[k] - cf_b[k]; d_ab += ab * ab;
        float ba = kf_b[k] - cf_a[k]; d_ba += ba * ba;
        float bb = kf_b[k] - cf_b[k]; d_bb += bb * bb;
    }
    const float min_dist = sqrtf(fminf(fminf(d_aa, d_ab), fminf(d_ba, d_bb)));

    if (min_dist < threshold) {
        const int slot = atomicAdd(d_out_count, 1);
        d_out[slot]        = m;
        d_out_scores[slot] = min_dist;
    }
}

// ---------------------------------------------------------------------------
// apply_precomputed_SIFT_filter_pipeline
//
// Lazily allocates output buffers on first call.  Returns elapsed GPU ms.
// ---------------------------------------------------------------------------
float apply_precomputed_SIFT_filter_pipeline(
    int                         device_id,
    int&                        h_count_out,
    const Match_by_Edge_Index*  d_candidates,
    const float*                d_kf_sift,
    const float*                d_cf_sift,
    Match_by_Edge_Index*&       d_out_matches,
    int*&                       d_out_count,
    float*&                     d_out_scores,
    int                         num_candidates,
    cudaEvent_t                 start,
    cudaEvent_t                 stop,
    float                       sift_threshold)
{
    cudacheck(cudaSetDevice(device_id));

    const size_t match_bytes = static_cast<size_t>(num_candidates > 0 ? num_candidates : 1) * sizeof(Match_by_Edge_Index);
    const size_t score_bytes = static_cast<size_t>(num_candidates > 0 ? num_candidates : 1) * sizeof(float);

    if (d_out_matches == nullptr) cudacheck(cudaMalloc(&d_out_matches, match_bytes));
    if (d_out_count   == nullptr) cudacheck(cudaMalloc(&d_out_count,   sizeof(int)));
    if (d_out_scores  == nullptr) cudacheck(cudaMalloc(&d_out_scores,  score_bytes));

    cudacheck(cudaMemset(d_out_count, 0, sizeof(int)));
    cudacheck(cudaEventRecord(start));

    if (num_candidates > 0) {
        const int threads = 256;
        const int blocks  = (num_candidates + threads - 1) / threads;
        apply_precomputed_sift_filter_kernel<<<blocks, threads>>>(
            d_candidates, d_kf_sift, d_cf_sift,
            d_out_matches, d_out_count, d_out_scores,
            num_candidates, sift_threshold);
        cudacheck(cudaDeviceSynchronize());
    }

    float elapsed = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&elapsed, start, stop));

    h_count_out = 0;
    cudacheck(cudaMemcpy(&h_count_out, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));

    return elapsed;
}

#endif // APPLY_PRECOMPUTED_SIFT_FILTER_CU
