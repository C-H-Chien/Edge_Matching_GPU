#ifndef TEMPORAL_CANDIDATE_GENERATION_CU
#define TEMPORAL_CANDIDATE_GENERATION_CU

#include <cuda_runtime.h>
#include <math.h>
#include <cstddef>
#include <iostream>

#include "definitions.h"
#include "gpu_settings.h"
#include "gpu_kernels.h"

static constexpr float TC_PI = 3.14159265358979323846f;

//> Absolute angular difference in degrees, folded into [0, 180].
__device__ __forceinline__
float orientation_diff_deg(float a, float b)
{
    float d = fabsf(a - b) * (180.0f / TC_PI);
    if (d > 180.0f) d = 360.0f - d;
    return d;
}

//> True if the angular difference is within thresh, considering 180-deg ambiguity.
__device__ __forceinline__
bool passes_orientation_check(float diff_deg, float thresh_deg)
{
    return (diff_deg < thresh_deg) || (fabsf(diff_deg - 180.0f) < thresh_deg);
}

//> One thread per KF stereo mate.
//> Iterates the CF-left spatial grid cells within the circular search neighbourhood of the KF-left
//> edge position.  For every CF mate index found in those cells we directly look up the CF-right
//> edge (same index) and check that it is also within the search radius of the KF-right edge.
//> This simultaneously implements the "left-grid query ∩ right-grid query" intersection used in
//> the CPU pipeline (apply_spatial_grid_filtering_quads), orientation filtering on both sides, and
//> quad emission – all in one fused kernel.
__global__ void temporal_candidate_generation_kernel(
    const Edge_GPU*          d_kf_left_edges,
    const Edge_GPU*          d_kf_right_edges,
    int                      num_kf_mates,
    const Edge_GPU*          d_cf_left_edges,
    const Edge_GPU*          d_cf_right_edges,
    const int*               d_cf_left_edge_indices,   //> sorted CF-left edge indices (by cell)
    const int*               d_cf_left_cell_start_idx, //> start offset in d_cf_left_edge_indices per cell
    const int*               d_cf_left_num_in_cell,    //> edge count per grid cell
    int                      grid_W,
    int                      grid_H,
    float                    search_radius,
    float                    orient_thresh_deg,
    Match_by_Edge_Index*     d_candidates,
    int*                     d_candidate_count,
    int                      max_candidates)
{
    const int kf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (kf_idx >= num_kf_mates) return;

    const float kl_x = d_kf_left_edges[kf_idx].location_x;
    const float kl_y = d_kf_left_edges[kf_idx].location_y;
    const float kl_o = d_kf_left_edges[kf_idx].orientation;
    const float kr_x = d_kf_right_edges[kf_idx].location_x;
    const float kr_y = d_kf_right_edges[kf_idx].location_y;
    const float kr_o = d_kf_right_edges[kf_idx].orientation;

    const float sr2 = search_radius * search_radius;

    //> Grid cell range covering the search circle for the KF-left edge position
    const int cx0 = max(0,          static_cast<int>((kl_x - search_radius) / GRID_CELL_SIZE));
    const int cx1 = min(grid_W - 1, static_cast<int>((kl_x + search_radius) / GRID_CELL_SIZE));
    const int cy0 = max(0,          static_cast<int>((kl_y - search_radius) / GRID_CELL_SIZE));
    const int cy1 = min(grid_H - 1, static_cast<int>((kl_y + search_radius) / GRID_CELL_SIZE));

    for (int cy = cy0; cy <= cy1; ++cy) {
        for (int cx = cx0; cx <= cx1; ++cx) {
            const int cell  = cy * grid_W + cx;
            const int start = d_cf_left_cell_start_idx[cell];
            const int count = d_cf_left_num_in_cell[cell];

            for (int j = 0; j < count; ++j) {
                const int cf_idx = d_cf_left_edge_indices[start + j];

                //> --- Left check ---
                const float dl_x = d_cf_left_edges[cf_idx].location_x - kl_x;
                const float dl_y = d_cf_left_edges[cf_idx].location_y - kl_y;
                if (dl_x * dl_x + dl_y * dl_y > sr2) continue;

                const float diff_left = orientation_diff_deg(kl_o, d_cf_left_edges[cf_idx].orientation);
                if (!passes_orientation_check(diff_left, orient_thresh_deg)) continue;

                //> --- Right check (implicit intersection with the CF-right grid) ---
                const float dr_x = d_cf_right_edges[cf_idx].location_x - kr_x;
                const float dr_y = d_cf_right_edges[cf_idx].location_y - kr_y;
                if (dr_x * dr_x + dr_y * dr_y > sr2) continue;

                const float diff_right = orientation_diff_deg(kr_o, d_cf_right_edges[cf_idx].orientation);
                if (!passes_orientation_check(diff_right, orient_thresh_deg)) continue;

                //> --- Emit quad candidate ---
                //> left_edge_idx  == kf_mate_idx
                //> right_edge_idx == cf_mate_idx
                const int slot = atomicAdd(d_candidate_count, 1);
                if (slot < max_candidates) {
                    d_candidates[slot].left_edge_idx  = kf_idx;
                    d_candidates[slot].right_edge_idx = cf_idx;
                }
            }
        }
    }
}

void temporal_candidate_generation_kernel_launcher(
    int                  device_id,
    const Edge_GPU*      d_kf_left_edges,
    const Edge_GPU*      d_kf_right_edges,
    int                  num_kf_mates,
    const Edge_GPU*      d_cf_left_edges,
    const Edge_GPU*      d_cf_right_edges,
    const int*           d_cf_left_edge_indices,
    const int*           d_cf_left_cell_start_idx,
    const int*           d_cf_left_num_in_cell,
    int                  grid_W,
    int                  grid_H,
    float                search_radius,
    float                orient_thresh_deg,
    Match_by_Edge_Index* d_candidates,
    int*                 d_candidate_count,
    int                  max_candidates)
{
    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));

    const int threads = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block)
                        ? NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING
                        : max_threads_per_block;
    const int blocks  = (num_kf_mates + threads - 1) / threads;

    void* args[] = {
        (void*)&d_kf_left_edges,
        (void*)&d_kf_right_edges,
        (void*)&num_kf_mates,
        (void*)&d_cf_left_edges,
        (void*)&d_cf_right_edges,
        (void*)&d_cf_left_edge_indices,
        (void*)&d_cf_left_cell_start_idx,
        (void*)&d_cf_left_num_in_cell,
        (void*)&grid_W,
        (void*)&grid_H,
        (void*)&search_radius,
        (void*)&orient_thresh_deg,
        (void*)&d_candidates,
        (void*)&d_candidate_count,
        (void*)&max_candidates
    };

    cudacheck(cudaLaunchKernel(
        reinterpret_cast<const void*>(temporal_candidate_generation_kernel),
        dim3(blocks), dim3(threads), args, 0, nullptr));
}

float temporal_candidate_generation_pipeline(
    int                  device_id,
    const Edge_GPU*      d_kf_left_edges,
    const Edge_GPU*      d_kf_right_edges,
    int                  num_kf_mates,
    const Edge_GPU*      d_cf_left_edges,
    const Edge_GPU*      d_cf_right_edges,
    const int*           d_cf_left_edge_indices,
    const int*           d_cf_left_cell_start_idx,
    const int*           d_cf_left_num_in_cell,
    int                  grid_W,
    int                  grid_H,
    float                search_radius,
    float                orient_thresh_deg,
    Match_by_Edge_Index* d_candidates,
    int*                 d_candidate_count,
    int&                 h_candidate_count_out,
    int                  max_candidates,
    cudaEvent_t          start,
    cudaEvent_t          stop)
{
    cudacheck(cudaSetDevice(device_id));
    cudacheck(cudaMemset(d_candidate_count, 0, sizeof(int)));
    cudacheck(cudaEventRecord(start));

    if (num_kf_mates > 0) {
        temporal_candidate_generation_kernel_launcher(
            device_id,
            d_kf_left_edges, d_kf_right_edges, num_kf_mates,
            d_cf_left_edges, d_cf_right_edges,
            d_cf_left_edge_indices, d_cf_left_cell_start_idx, d_cf_left_num_in_cell,
            grid_W, grid_H, search_radius, orient_thresh_deg,
            d_candidates, d_candidate_count, max_candidates);
        cudacheck(cudaDeviceSynchronize());
    }

    float elapsed = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&elapsed, start, stop));

    h_candidate_count_out = 0;
    cudacheck(cudaMemcpy(&h_candidate_count_out, d_candidate_count, sizeof(int), cudaMemcpyDeviceToHost));

    return elapsed;
}

#endif // TEMPORAL_CANDIDATE_GENERATION_CU
