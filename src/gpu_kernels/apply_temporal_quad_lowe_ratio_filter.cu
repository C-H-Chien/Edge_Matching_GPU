#ifndef APPLY_TEMPORAL_QUAD_LOWE_RATIO_FILTER_CU
#define APPLY_TEMPORAL_QUAD_LOWE_RATIO_FILTER_CU

#include <cuda_runtime.h>
#include <cuda/std/functional>
#include <cfloat>
#include <cstddef>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "definitions.h"
#include "gpu_kernels.h"
#include "gpu_settings.h"

struct sort_temporal_quad_by_kf {
    __device__ bool operator()(const Temporal_Refined_Quad_Match_GPU& a,
                               const Temporal_Refined_Quad_Match_GPU& b) const
    {
        if (a.kf_mate_idx != b.kf_mate_idx) return a.kf_mate_idx < b.kf_mate_idx;
        return a.photometric_rms < b.photometric_rms;
    }
};

__global__ void mark_temporal_quad_segment_head_kernel(
    const Temporal_Refined_Quad_Match_GPU* __restrict__ sorted,
    int n,
    int* __restrict__ flags)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    flags[i] = (i == 0 || sorted[i].kf_mate_idx != sorted[i - 1].kf_mate_idx) ? 1 : 0;
}

//> One thread per kf_mate_idx segment. Unambiguous segments (k==1) pass through.
//> Ambiguous segments (k>1): keep the lowest-RMS quad only when
//> best_rms / second_best_rms < lowe_ratio_threshold (SIFT-style Lowe's ratio).
__global__ void temporal_quad_lowe_ratio_filter_kernel(
    const Temporal_Refined_Quad_Match_GPU* __restrict__ sorted,
    const int* __restrict__ seg_starts_ext,
    int n_segments,
    Temporal_Refined_Quad_Match_GPU* __restrict__ d_out,
    int* __restrict__ d_out_count,
    float lowe_ratio_threshold)
{
    const int seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= n_segments) return;

    const int seg_begin = seg_starts_ext[seg];
    const int seg_end   = seg_starts_ext[seg + 1];
    const int k         = seg_end - seg_begin;
    if (k <= 0) return;

    if (k == 1) {
        const int oid = atomicAdd(d_out_count, 1);
        d_out[oid] = sorted[seg_begin];
        return;
    }

    const float best_rms   = sorted[seg_begin].photometric_rms;
    const float second_rms = sorted[seg_begin + 1].photometric_rms;
    if (!isfinite(best_rms) || !isfinite(second_rms)) {
        return;
    }

    const float ratio = best_rms / fmaxf(second_rms, 1e-6f);
    if (ratio >= lowe_ratio_threshold) {
        return;
    }

    const int oid = atomicAdd(d_out_count, 1);
    d_out[oid] = sorted[seg_begin];
}

float apply_temporal_quad_lowe_ratio_filter_pipeline(
    int device_id,
    Temporal_Refined_Quad_Match_GPU* d_clustered_matches,
    int& h_clustered_count_inout,
    int*& d_clustered_count,
    float lowe_ratio_threshold,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    cudacheck(cudaSetDevice(device_id));
    cudacheck(cudaEventRecord(start));

    const int n_in = h_clustered_count_inout;
    if (n_in <= 0 || d_clustered_matches == nullptr) {
        h_clustered_count_inout = 0;
        cudacheck(cudaEventRecord(stop));
        cudacheck(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        cudacheck(cudaEventElapsedTime(&elapsed, start, stop));
        return elapsed;
    }

    thrust::device_ptr<Temporal_Refined_Quad_Match_GPU> d_ptr(d_clustered_matches);
    thrust::sort(thrust::device, d_ptr, d_ptr + n_in, sort_temporal_quad_by_kf());

    thrust::device_vector<int> d_flags(static_cast<size_t>(n_in));
    {
        const int threads = 256;
        const int blocks  = (n_in + threads - 1) / threads;
        mark_temporal_quad_segment_head_kernel<<<blocks, threads>>>(
            d_clustered_matches, n_in, thrust::raw_pointer_cast(d_flags.data()));
        cudacheck(cudaDeviceSynchronize());
    }

    thrust::device_vector<int> d_seg_starts(static_cast<size_t>(n_in));
    thrust::exclusive_scan(thrust::device, d_flags.begin(), d_flags.end(),
                           d_seg_starts.begin());

    int n_segs = 0;
    {
        int last_flag = 0, last_seg = 0;
        cudacheck(cudaMemcpy(&last_flag, thrust::raw_pointer_cast(d_flags.data()) + (n_in - 1),
                             sizeof(int), cudaMemcpyDeviceToHost));
        cudacheck(cudaMemcpy(&last_seg, thrust::raw_pointer_cast(d_seg_starts.data()) + (n_in - 1),
                             sizeof(int), cudaMemcpyDeviceToHost));
        n_segs = last_seg + last_flag;
    }

    if (n_segs <= 0) {
        h_clustered_count_inout = 0;
        cudacheck(cudaEventRecord(stop));
        cudacheck(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        cudacheck(cudaEventElapsedTime(&elapsed, start, stop));
        return elapsed;
    }

    thrust::device_vector<int> d_seg_starts_ext(static_cast<size_t>(n_segs + 1));
    {
        thrust::device_vector<int> d_iota(static_cast<size_t>(n_in));
        thrust::sequence(thrust::device, d_iota.begin(), d_iota.end(), 0);
        thrust::copy_if(thrust::device, d_iota.begin(), d_iota.end(),
                        d_flags.begin(), d_seg_starts_ext.begin(),
                        cuda::std::identity{});
        cudacheck(cudaMemcpy(thrust::raw_pointer_cast(d_seg_starts_ext.data()) + n_segs,
                             &n_in, sizeof(int), cudaMemcpyHostToDevice));
    }

    Temporal_Refined_Quad_Match_GPU* d_filtered = nullptr;
    cudacheck(cudaMalloc(&d_filtered, static_cast<size_t>(n_in) * sizeof(Temporal_Refined_Quad_Match_GPU)));
    if (d_clustered_count == nullptr) {
        cudacheck(cudaMalloc(&d_clustered_count, sizeof(int)));
    }
    cudacheck(cudaMemset(d_clustered_count, 0, sizeof(int)));

    {
        const int threads = 128;
        const int blocks  = (n_segs + threads - 1) / threads;
        temporal_quad_lowe_ratio_filter_kernel<<<blocks, threads>>>(
            d_clustered_matches,
            thrust::raw_pointer_cast(d_seg_starts_ext.data()),
            n_segs,
            d_filtered,
            d_clustered_count,
            lowe_ratio_threshold);
        cudacheck(cudaDeviceSynchronize());
    }

    int h_filtered_count = 0;
    cudacheck(cudaMemcpy(&h_filtered_count, d_clustered_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_filtered_count > n_in) {
        h_filtered_count = n_in;
    }

    if (h_filtered_count > 0) {
        cudacheck(cudaMemcpy(d_clustered_matches, d_filtered,
                             static_cast<size_t>(h_filtered_count) * sizeof(Temporal_Refined_Quad_Match_GPU),
                             cudaMemcpyDeviceToDevice));
    }
    cudacheck(cudaFree(d_filtered));

    h_clustered_count_inout = h_filtered_count;

    float elapsed = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&elapsed, start, stop));
    return elapsed;
}

#endif // APPLY_TEMPORAL_QUAD_LOWE_RATIO_FILTER_CU
