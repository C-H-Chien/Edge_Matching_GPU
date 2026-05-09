#ifndef APPLY_BEST_SECOND_NCC_PER_LEFT_EDGE_CU
#define APPLY_BEST_SECOND_NCC_PER_LEFT_EDGE_CU

#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "definitions.h"
#include "gpu_settings.h"
#include "gpu_kernels.h"

struct SortRow {
    int left_idx;
    float neg_ncc;
    int candidate_match_idx;
};

struct SortRowLess {
    __host__ __device__ bool operator()(const SortRow& a, const SortRow& b) const
    {
        if (a.left_idx != b.left_idx)
            return a.left_idx < b.left_idx;
        if (a.neg_ncc != b.neg_ncc)
            return a.neg_ncc < b.neg_ncc;
        return a.candidate_match_idx < b.candidate_match_idx;
    }
};

__global__ void fill_sort_rows_for_best_second_ncc_kernel(
    const Merged_Refined_Stereo_Match_GPU* __restrict__ d_matches,
    const float* __restrict__ d_scores,
    int num_candidates,
    SortRow* __restrict__ d_rows)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_candidates)
        return;
    d_rows[tid].left_idx = d_matches[tid].left_edge_idx;
    d_rows[tid].neg_ncc = -d_scores[tid];
    d_rows[tid].candidate_match_idx = tid;
}

__global__ void compact_sorted_best_per_left_kernel(
    const SortRow* __restrict__ d_sorted,
    const Merged_Refined_Stereo_Match_GPU* __restrict__ d_src_matches,
    int num_candidates,
    Merged_Refined_Stereo_Match_GPU* __restrict__ d_dst_matches,
    int* __restrict__ d_out_count)
{
    //> Each thread is responsible for one candidate match of the sorted array
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_candidates)
        return;

    //> The first row of the sorted array is always a group start
    //> For the rest of the rows, if the left edge index is different from the previous row, 
    //  it is also a group start (the start of the new group/segment of the left edge)
    const bool head = (tid == 0) || (d_sorted[tid].left_idx != d_sorted[tid - 1].left_idx);
    if (!head)
        return;
    
    const int out_idx = atomicAdd(d_out_count, 1);
    d_dst_matches[out_idx] = d_src_matches[d_sorted[tid].candidate_match_idx];
}

float apply_best_second_ncc_per_left_edge_pipeline(
    int device_id,
    int& h_match_count_out,
    Merged_Refined_Stereo_Match_GPU*& d_merged_matches,
    int*& d_merged_count,
    float*& d_second_ncc_scores,
    int num_candidates,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    cudacheck(cudaSetDevice(device_id));

    if (num_candidates <= 0) {
        LOG_ERROR("apply_best_second_ncc_per_left_edge_pipeline: num_candidates <= 0");
        h_match_count_out = 0;
        return 0.0f;
    }
    if (d_merged_matches == nullptr || d_merged_count == nullptr || d_second_ncc_scores == nullptr) {
        LOG_ERROR("apply_best_second_ncc_per_left_edge_pipeline: null merged matches, count, or scores");
        h_match_count_out = 0;
        return 0.0f;
    }

    //> Each sort row of d_sort_rows is a candidate match, represented as [left_edge_index, -NCC_score, candidate_match_index]
    SortRow* d_sort_rows = nullptr;
    cudacheck(cudaMalloc(&d_sort_rows, static_cast<size_t>(num_candidates) * sizeof(SortRow)));

    Merged_Refined_Stereo_Match_GPU* d_compacted = nullptr;
    cudacheck(cudaMalloc(&d_compacted, static_cast<size_t>(num_candidates) * sizeof(Merged_Refined_Stereo_Match_GPU)));

    int* d_compact_count = nullptr;
    cudacheck(cudaMalloc(&d_compact_count, sizeof(int)));
    cudacheck(cudaMemset(d_compact_count, 0, sizeof(int)));

    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));
    const int threads_per_block = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block)
                                      ? NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING
                                      : max_threads_per_block;
    const int num_blocks = (num_candidates + threads_per_block - 1) / threads_per_block;

    //> Start the CUDA event timer
    cudacheck(cudaEventRecord(start));

    //> The first kernel is to fill the sort rows (d_sort_rows) for the best second-NCC per left edge
    //> This assigns one thread "per candidate match"
    fill_sort_rows_for_best_second_ncc_kernel<<<num_blocks, threads_per_block>>>( d_merged_matches, d_second_ncc_scores, num_candidates, d_sort_rows );
    cudacheck(cudaPeekAtLastError());

    //> Now sort the sort rows by left_edge_index, then by -NCC_score, then by candidate_match_idx
    thrust::sort(thrust::device, d_sort_rows, d_sort_rows + num_candidates, SortRowLess());

    //> The second kernel is to compact the sorted sort rows into the compacted matches array
    //> This assigns one thread "per candidate match" as well
    compact_sorted_best_per_left_kernel<<<num_blocks, threads_per_block>>>( d_sort_rows, d_merged_matches, num_candidates, d_compacted, d_compact_count );
    cudacheck(cudaPeekAtLastError());
    cudacheck(cudaDeviceSynchronize());

    float elapsed_ms = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&elapsed_ms, start, stop));

    LOG_INFO("Time for best second-NCC per left edge (one-to-one): " + std::to_string(elapsed_ms) + " milliseconds");

    h_match_count_out = 0;
    cudacheck(cudaMemcpy(&h_match_count_out, d_compact_count, sizeof(int), cudaMemcpyDeviceToHost));

    cudacheck(cudaFree(d_sort_rows));
    d_sort_rows = nullptr;

    cudacheck(cudaFree(d_merged_matches));
    d_merged_matches = nullptr;
    cudacheck(cudaFree(d_merged_count));
    d_merged_count = nullptr;
    cudacheck(cudaFree(d_second_ncc_scores));
    d_second_ncc_scores = nullptr;

    d_merged_matches = d_compacted;
    d_merged_count = d_compact_count;

    return elapsed_ms;
}

#endif
