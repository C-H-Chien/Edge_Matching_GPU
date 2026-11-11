#ifndef TEMPORAL_EDGE_TRACKING_MONO_CU
#define TEMPORAL_EDGE_TRACKING_MONO_CU

#include <cuda_runtime.h>
#include <math.h>
#include <cfloat>
#include <cstddef>
#include <cstdio>

#include "definitions.h"
#include "gpu_settings.h"
#include "gpu_kernels.h"
#include "gpu_helpers/eval_edges_in_grid_cells.cuh"
#include "gpu_helpers/compute_NCC.cuh"
#include "gpu_helpers/extract_gpu_sift_descriptor.cuh"
#include "gpu_helpers/get_interp_photometry_in_texture.cuh"
#include "gpu_helpers/patch_photometry_helpers.cuh"
#include "gpu_helpers/GN_refinement_2D_free.cuh"

//> CH Notes: Left-only counterpart of the temporal quad kernels. Edges are raw third-order edges of the
//  KF and CF left images (no stereo mates), so every stage only evaluates the KF-left <-> CF-left pair.

namespace {

int mono_threads_per_block(int device_id)
{
    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));
    return (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block)
           ? NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING : max_threads_per_block;
}

float mono_elapsed_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float elapsed_ms = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&elapsed_ms, start, stop));
    return elapsed_ms;
}

}  // namespace

// ==================================================================================================
// Candidate generation
// ==================================================================================================
__global__ void mono_temporal_candidate_generation_kernel(
    const Edge_GPU* __restrict__ d_kf_edges,
    const Edge_GPU* __restrict__ d_cf_edges,
    const int* __restrict__      d_cf_edge_indices,
    const int* __restrict__      d_cf_cell_start_idx,
    const int* __restrict__      d_cf_num_in_cell,
    int                          grid_W,
    int                          grid_H,
    int                          num_kf_edges,
    Match_by_Edge_Index*         d_candidates_out,
    int*                         d_candidate_count,
    int                          max_candidates)
{
    const int kf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (kf_idx >= num_kf_edges) return;

    const Edge_GPU kf_edge = d_kf_edges[kf_idx];

    //> Same square-box cell range as the quad candidate generation
    const int cx0 = max(0,          static_cast<int>((kf_edge.location_x - TEMPORAL_SEARCH_RADIUS_PIXELS) / GRID_CELL_SIZE));
    const int cx1 = min(grid_W - 1, static_cast<int>((kf_edge.location_x + TEMPORAL_SEARCH_RADIUS_PIXELS) / GRID_CELL_SIZE));
    const int cy0 = max(0,          static_cast<int>((kf_edge.location_y - TEMPORAL_SEARCH_RADIUS_PIXELS) / GRID_CELL_SIZE));
    const int cy1 = min(grid_H - 1, static_cast<int>((kf_edge.location_y + TEMPORAL_SEARCH_RADIUS_PIXELS) / GRID_CELL_SIZE));

    for (int cy = cy0; cy <= cy1; ++cy) {
        for (int cx = cx0; cx <= cx1; ++cx) {
            const int cell  = cy * grid_W + cx;
            const int start = d_cf_cell_start_idx[cell];
            const int count = d_cf_num_in_cell[cell];

            for (int j = 0; j < count; ++j) {
                const int cf_idx = d_cf_edge_indices[start + j];
                if (!edge_pair_passes_orientation_prefilter_deg(kf_edge.orientation, d_cf_edges[cf_idx].orientation,
                                                                static_cast<float>(EDGE_PAIR_ORIENTATION_PREFILTER_DEG)))
                    continue;

                const int slot = atomicAdd(d_candidate_count, 1);
                if (slot < max_candidates) {
                    d_candidates_out[slot].left_edge_idx  = kf_idx;
                    d_candidates_out[slot].right_edge_idx = cf_idx;
                }
            }
        }
    }
}

float mono_temporal_candidate_generation_pipeline(
    int                         device_id,
    const Edge_GPU*             d_kf_edges,
    const Edge_GPU*             d_cf_edges,
    const int*                  d_cf_edge_indices,
    const int*                  d_cf_cell_start_idx,
    const int*                  d_cf_num_in_cell,
    int                         grid_W,
    int                         grid_H,
    int                         num_kf_edges,
    Match_by_Edge_Index*        d_candidates,
    int*                        d_candidate_count,
    int&                        h_candidate_count_out,
    int                         max_candidates,
    cudaEvent_t                 start,
    cudaEvent_t                 stop)
{
    cudacheck(cudaSetDevice(device_id));
    cudacheck(cudaMemset(d_candidate_count, 0, sizeof(int)));
    cudacheck(cudaEventRecord(start));

    if (num_kf_edges > 0) {
        const int threads = mono_threads_per_block(device_id);
        const int blocks  = (num_kf_edges + threads - 1) / threads;
        mono_temporal_candidate_generation_kernel<<<blocks, threads>>>(
            d_kf_edges, d_cf_edges,
            d_cf_edge_indices, d_cf_cell_start_idx, d_cf_num_in_cell,
            grid_W, grid_H, num_kf_edges,
            d_candidates, d_candidate_count, max_candidates);
        cudacheck(cudaGetLastError());
        cudacheck(cudaDeviceSynchronize());
    }

    const float elapsed_ms = mono_elapsed_ms(start, stop);

    int device_count = 0;
    cudacheck(cudaMemcpy(&device_count, d_candidate_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (device_count > max_candidates) {
        printf("[Temporal Mono GPU] WARNING: candidate buffer overflow (%d > %d); truncating to buffer size\n",
               device_count, max_candidates);
        h_candidate_count_out = max_candidates;
    } else {
        h_candidate_count_out = device_count;
    }
    return elapsed_ms;
}

// ==================================================================================================
// NCC filter (left-left)
// ==================================================================================================
__device__ __forceinline__
static float mono_max_patch_pair_ncc(const Precomputed_Edge_Patches_Photometry& a,
                                     const Precomputed_Edge_Patches_Photometry& b)
{
    const float sim_pp = compute_NCC(a.pL_plus,  a.mL_plus,  a.vL_plus,  b.pL_plus,  b.mL_plus,  b.vL_plus);
    const float sim_pn = compute_NCC(a.pL_plus,  a.mL_plus,  a.vL_plus,  b.pL_minus, b.mL_minus, b.vL_minus);
    const float sim_np = compute_NCC(a.pL_minus, a.mL_minus, a.vL_minus, b.pL_plus,  b.mL_plus,  b.vL_plus);
    const float sim_nn = compute_NCC(a.pL_minus, a.mL_minus, a.vL_minus, b.pL_minus, b.mL_minus, b.vL_minus);
    return fmaxf(fmaxf(sim_pp, sim_pn), fmaxf(sim_np, sim_nn));
}

__global__ void mono_temporal_ncc_filter_kernel(
    const Match_by_Edge_Index* __restrict__                 d_candidates,
    const Precomputed_Edge_Patches_Photometry* __restrict__ d_kf_patches,
    const Precomputed_Edge_Patches_Photometry* __restrict__ d_cf_patches,
    Match_by_Edge_Index*                                    d_final_matches_out,
    int*                                                    d_final_count,
    float*                                                  d_ncc_scores_out,
    int                                                     num_candidates,
    int                                                     num_kf_edges,
    int                                                     num_cf_edges)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_candidates) return;

    const Match_by_Edge_Index candidate = d_candidates[tid];
    const int kf_idx = candidate.left_edge_idx;
    const int cf_idx = candidate.right_edge_idx;
    if (kf_idx < 0 || cf_idx < 0 || kf_idx >= num_kf_edges || cf_idx >= num_cf_edges) return;

    const float ncc = mono_max_patch_pair_ncc(d_kf_patches[kf_idx], d_cf_patches[cf_idx]);
    if (ncc <= TEMPORAL_NCC_THRESHOLD) return;

    const int out_idx = atomicAdd(d_final_count, 1);
    if (out_idx >= num_candidates) return;
    d_final_matches_out[out_idx] = candidate;
    d_ncc_scores_out[out_idx]    = ncc;
}

float mono_temporal_ncc_filter_pipeline(
    int                         device_id,
    int                         &h_match_count_out,
    const Match_by_Edge_Index*  d_candidates,
    const Precomputed_Edge_Patches_Photometry* d_kf_patches,
    const Precomputed_Edge_Patches_Photometry* d_cf_patches,
    Match_by_Edge_Index*&       d_final_matches_out,
    int*&                       d_final_count_out,
    float*&                     d_ncc_scores_out,
    int                         num_candidates,
    int                         num_kf_edges,
    int                         num_cf_edges,
    cudaEvent_t                 start,
    cudaEvent_t                 stop)
{
    cudacheck(cudaSetDevice(device_id));
    cudacheck(cudaEventRecord(start));

    const size_t n = static_cast<size_t>(num_candidates > 0 ? num_candidates : 1);
    cudacheck(cudaMalloc(&d_final_matches_out, n * sizeof(Match_by_Edge_Index)));
    cudacheck(cudaMalloc(&d_ncc_scores_out,    n * sizeof(float)));
    cudacheck(cudaMalloc(&d_final_count_out,   sizeof(int)));
    cudacheck(cudaMemset(d_final_count_out, 0, sizeof(int)));

    if (num_candidates > 0 && d_kf_patches != nullptr && d_cf_patches != nullptr) {
        const int threads = mono_threads_per_block(device_id);
        const int blocks  = (num_candidates + threads - 1) / threads;
        mono_temporal_ncc_filter_kernel<<<blocks, threads>>>(
            d_candidates, d_kf_patches, d_cf_patches,
            d_final_matches_out, d_final_count_out, d_ncc_scores_out,
            num_candidates, num_kf_edges, num_cf_edges);
        cudacheck(cudaGetLastError());
        cudacheck(cudaDeviceSynchronize());
    }

    const float elapsed_ms = mono_elapsed_ms(start, stop);

    h_match_count_out = 0;
    cudacheck(cudaMemcpy(&h_match_count_out, d_final_count_out, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_match_count_out > num_candidates) h_match_count_out = num_candidates;
    return elapsed_ms;
}

// ==================================================================================================
// SIFT filter (left-left)
// ==================================================================================================
__device__ __forceinline__
static float mono_min_sift_pair_distance(const Precomputed_Edge_SIFT_Descriptor_GPU& a,
                                         const Precomputed_Edge_SIFT_Descriptor_GPU& b)
{
    const float d_pp = sift_l2_distance(a.pL_plus,  b.pL_plus);
    const float d_pn = sift_l2_distance(a.pL_plus,  b.pL_minus);
    const float d_np = sift_l2_distance(a.pL_minus, b.pL_plus);
    const float d_nn = sift_l2_distance(a.pL_minus, b.pL_minus);
    return fminf(fminf(d_pp, d_pn), fminf(d_np, d_nn));
}

__global__ void mono_temporal_sift_filter_kernel(
    const Match_by_Edge_Index* __restrict__                  d_candidates,
    const Precomputed_Edge_SIFT_Descriptor_GPU* __restrict__ d_kf_sift,
    const Precomputed_Edge_SIFT_Descriptor_GPU* __restrict__ d_cf_sift,
    Match_by_Edge_Index*                                     d_final_matches_out,
    int*                                                     d_final_count,
    float*                                                   d_sift_scores_out,
    const float*                                             d_ncc_scores_in,
    float*                                                   d_ncc_scores_out,
    int                                                      num_candidates,
    int                                                      num_kf_edges,
    int                                                      num_cf_edges)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_candidates) return;

    const Match_by_Edge_Index candidate = d_candidates[tid];
    const int kf_idx = candidate.left_edge_idx;
    const int cf_idx = candidate.right_edge_idx;
    if (kf_idx < 0 || cf_idx < 0 || kf_idx >= num_kf_edges || cf_idx >= num_cf_edges) return;

    const float sift = mono_min_sift_pair_distance(d_kf_sift[kf_idx], d_cf_sift[cf_idx]);
    if (sift >= TEMPORAL_SIFT_THRESHOLD) return;

    const int out_idx = atomicAdd(d_final_count, 1);
    if (out_idx >= num_candidates) return;
    d_final_matches_out[out_idx] = candidate;
    d_sift_scores_out[out_idx]   = sift;
    if (d_ncc_scores_in != nullptr && d_ncc_scores_out != nullptr)
        d_ncc_scores_out[out_idx] = d_ncc_scores_in[tid];
}

float mono_temporal_sift_filter_pipeline(
    int                         device_id,
    int                         &h_match_count_out,
    const Match_by_Edge_Index*  d_candidates,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_kf_sift,
    const Precomputed_Edge_SIFT_Descriptor_GPU* d_cf_sift,
    Match_by_Edge_Index*&       d_final_matches_out,
    int*&                       d_final_count_out,
    float*&                     d_sift_scores_out,
    const float*                d_ncc_scores_in,
    float*&                     d_ncc_scores_out,
    int                         num_candidates,
    int                         num_kf_edges,
    int                         num_cf_edges,
    cudaEvent_t                 start,
    cudaEvent_t                 stop)
{
    cudacheck(cudaSetDevice(device_id));
    cudacheck(cudaEventRecord(start));

    const size_t n = static_cast<size_t>(num_candidates > 0 ? num_candidates : 1);
    cudacheck(cudaMalloc(&d_final_matches_out, n * sizeof(Match_by_Edge_Index)));
    cudacheck(cudaMalloc(&d_sift_scores_out,   n * sizeof(float)));
    cudacheck(cudaMalloc(&d_final_count_out,   sizeof(int)));
    cudacheck(cudaMemset(d_final_count_out, 0, sizeof(int)));
    d_ncc_scores_out = nullptr;
    if (d_ncc_scores_in != nullptr)
        cudacheck(cudaMalloc(&d_ncc_scores_out, n * sizeof(float)));

    if (num_candidates > 0 && d_kf_sift != nullptr && d_cf_sift != nullptr) {
        const int threads = mono_threads_per_block(device_id);
        const int blocks  = (num_candidates + threads - 1) / threads;
        mono_temporal_sift_filter_kernel<<<blocks, threads>>>(
            d_candidates, d_kf_sift, d_cf_sift,
            d_final_matches_out, d_final_count_out, d_sift_scores_out,
            d_ncc_scores_in, d_ncc_scores_out,
            num_candidates, num_kf_edges, num_cf_edges);
        cudacheck(cudaGetLastError());
        cudacheck(cudaDeviceSynchronize());
    }

    const float elapsed_ms = mono_elapsed_ms(start, stop);

    h_match_count_out = 0;
    cudacheck(cudaMemcpy(&h_match_count_out, d_final_count_out, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_match_count_out > num_candidates) h_match_count_out = num_candidates;
    return elapsed_ms;
}

// ==================================================================================================
// Best-nearly-best per KF edge
// ==================================================================================================
//> Order-preserving float <-> int mapping so atomicMax/atomicMin on int give float max/min.
__device__ __forceinline__ static int mono_float_to_ordered_int(float f)
{
    const int i = __float_as_int(f);
    return (i >= 0) ? i : (i ^ 0x7FFFFFFF);
}

__device__ __forceinline__ static float mono_ordered_int_to_float(int i)
{
    return __int_as_float((i >= 0) ? i : (i ^ 0x7FFFFFFF));
}

__global__ void mono_init_best_scores_kernel(int* d_best, int num_kf_edges, bool higher_is_better)
{
    const int kf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (kf_idx >= num_kf_edges) return;
    d_best[kf_idx] = mono_float_to_ordered_int(higher_is_better ? -FLT_MAX : FLT_MAX);
}

__global__ void mono_reduce_best_scores_kernel(
    const Match_by_Edge_Index* __restrict__ d_candidates,
    const float* __restrict__               d_rank_scores,
    int*                                    d_best,
    int                                     num_candidates,
    int                                     num_kf_edges,
    bool                                    higher_is_better)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_candidates) return;
    const int kf_idx = d_candidates[tid].left_edge_idx;
    const float score = d_rank_scores[tid];
    if (kf_idx < 0 || kf_idx >= num_kf_edges || !isfinite(score)) return;

    if (higher_is_better) atomicMax(&d_best[kf_idx], mono_float_to_ordered_int(score));
    else                  atomicMin(&d_best[kf_idx], mono_float_to_ordered_int(score));
}

__global__ void mono_apply_bnb_kernel(
    const Match_by_Edge_Index* __restrict__ d_candidates,
    const float* __restrict__               d_rank_scores,
    const float* __restrict__               d_carry_scores,
    const int* __restrict__                 d_best,
    Match_by_Edge_Index*                    d_out_matches,
    float*                                  d_out_carry_scores,
    int*                                    d_out_count,
    int                                     num_candidates,
    int                                     num_kf_edges,
    float                                   ratio_threshold,
    bool                                    higher_is_better)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_candidates) return;

    const Match_by_Edge_Index candidate = d_candidates[tid];
    const int kf_idx = candidate.left_edge_idx;
    if (kf_idx < 0 || kf_idx >= num_kf_edges) return;

    const float best  = mono_ordered_int_to_float(d_best[kf_idx]);
    const float score = d_rank_scores[tid];
    if (!isfinite(best) || !isfinite(score)) return;

    bool keep = false;
    if (higher_is_better) {
        keep = (score >= ratio_threshold * best);
    } else {
        const float denom = fmaxf(ratio_threshold, 1e-6f);
        keep = (score <= best / denom);
    }
    if (!keep) return;

    const int out_idx = atomicAdd(d_out_count, 1);
    if (out_idx >= num_candidates) return;
    d_out_matches[out_idx] = candidate;
    if (d_out_carry_scores != nullptr)
        d_out_carry_scores[out_idx] = (d_carry_scores != nullptr) ? d_carry_scores[tid] : score;
}

float mono_temporal_bnb_filter_pipeline(
    int                         device_id,
    int                         &h_match_count_out,
    const Match_by_Edge_Index*  d_candidates,
    const float*                d_rank_scores,
    const float*                d_carry_scores,
    Match_by_Edge_Index*&       d_out_matches,
    float*&                     d_out_carry_scores,
    int*&                       d_out_count,
    int                         num_candidates,
    int                         num_kf_edges,
    float                       ratio_threshold,
    bool                        higher_is_better,
    cudaEvent_t                 start,
    cudaEvent_t                 stop)
{
    cudacheck(cudaSetDevice(device_id));
    cudacheck(cudaEventRecord(start));

    const size_t n = static_cast<size_t>(num_candidates > 0 ? num_candidates : 1);
    cudacheck(cudaMalloc(&d_out_matches, n * sizeof(Match_by_Edge_Index)));
    cudacheck(cudaMalloc(&d_out_count, sizeof(int)));
    cudacheck(cudaMemset(d_out_count, 0, sizeof(int)));
    d_out_carry_scores = nullptr;
    if (d_carry_scores != nullptr)
        cudacheck(cudaMalloc(&d_out_carry_scores, n * sizeof(float)));

    int* d_best = nullptr;
    cudacheck(cudaMalloc(&d_best, static_cast<size_t>(num_kf_edges > 0 ? num_kf_edges : 1) * sizeof(int)));

    if (num_candidates > 0 && num_kf_edges > 0) {
        const int threads = 256;
        mono_init_best_scores_kernel<<<(num_kf_edges + threads - 1) / threads, threads>>>(
            d_best, num_kf_edges, higher_is_better);
        cudacheck(cudaGetLastError());

        const int cand_blocks = (num_candidates + threads - 1) / threads;
        mono_reduce_best_scores_kernel<<<cand_blocks, threads>>>(
            d_candidates, d_rank_scores, d_best, num_candidates, num_kf_edges, higher_is_better);
        cudacheck(cudaGetLastError());

        mono_apply_bnb_kernel<<<cand_blocks, threads>>>(
            d_candidates, d_rank_scores, d_carry_scores, d_best,
            d_out_matches, d_out_carry_scores, d_out_count,
            num_candidates, num_kf_edges, ratio_threshold, higher_is_better);
        cudacheck(cudaGetLastError());
        cudacheck(cudaDeviceSynchronize());
    }
    cudacheck(cudaFree(d_best));

    const float elapsed_ms = mono_elapsed_ms(start, stop);

    h_match_count_out = 0;
    cudacheck(cudaMemcpy(&h_match_count_out, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_match_count_out > num_candidates) h_match_count_out = num_candidates;
    return elapsed_ms;
}

// ==================================================================================================
// Photometric refinement (left only) + clustering
// ==================================================================================================
__global__ void mono_temporal_photometric_refine_kernel(
    const Match_by_Edge_Index* __restrict__                 d_matches,
    const Precomputed_Edge_Patches_Photometry* __restrict__ d_kf_patches,
    const Edge_GPU* __restrict__                            d_kf_edges,
    const Edge_GPU* __restrict__                            d_cf_edges,
    cudaTextureObject_t                                     cf_tex,
    Temporal_Refined_Match_GPU* __restrict__                d_out,
    int* __restrict__                                       d_out_count,
    int                                                     num_matches,
    int                                                     cf_img_w,
    int                                                     cf_img_h)
{
    const int mi = blockIdx.x * blockDim.x + threadIdx.x;
    if (mi >= num_matches) return;

    const Match_by_Edge_Index m = d_matches[mi];
    const int kf_idx = m.left_edge_idx;
    const int cf_idx = m.right_edge_idx;

    const Edge_GPU kf_edge = d_kf_edges[kf_idx];
    const Edge_GPU cf_edge = d_cf_edges[cf_idx];
    const Precomputed_Edge_Patches_Photometry kfp = d_kf_patches[kf_idx];

    float Lc_plus[TOTAL_NUM_OF_PATCH_PIXELS];
    float Lc_minus[TOTAL_NUM_OF_PATCH_PIXELS];
    for (int k = 0; k < TOTAL_NUM_OF_PATCH_PIXELS; ++k) {
        Lc_plus[k]  = kfp.pL_plus[k]  - kfp.mL_plus;
        Lc_minus[k] = kfp.pL_minus[k] - kfp.mL_minus;
    }

    float dx = 0.0f, dy = 0.0f, rms = 1e6f;
    bool  valid = false;
    GN_refinement_2D_free(Lc_plus, Lc_minus,
                          kf_edge.location_x, kf_edge.location_y, kf_edge.orientation,
                          cf_edge.location_x, cf_edge.location_y,
                          cf_tex, cf_img_w, cf_img_h,
                          dx, dy, rms, valid);
    if (!valid) return;

    const int slot = atomicAdd(d_out_count, 1);
    Temporal_Refined_Match_GPU& out = d_out[slot];
    out.kf_mate_idx          = kf_idx;
    out.cf_mate_idx          = cf_idx;
    out.cf_left_x            = kf_edge.location_x - dx;
    out.cf_left_y            = kf_edge.location_y - dy;
    out.cf_left_orientation  = cf_edge.orientation;
    out.cf_right_x           = 0.0f;
    out.cf_right_y           = 0.0f;
    out.cf_right_orientation = 0.0f;
    out.rms                  = rms;
}

float mono_temporal_photometric_refine_and_cluster_pipeline(
    int                                         device_id,
    const Match_by_Edge_Index*                  d_matches,
    int                                         num_matches,
    const Precomputed_Edge_Patches_Photometry*  d_kf_patches,
    const Edge_GPU*                             d_kf_edges,
    const Edge_GPU*                             d_cf_edges,
    cudaTextureObject_t                         cf_tex,
    int                                         cf_img_w,
    int                                         cf_img_h,
    Temporal_Refined_Match_GPU*&                d_refined_matches,
    int*&                                       d_refined_count,
    int&                                        h_refined_count_out,
    Temporal_Refined_Quad_Match_GPU*&           d_clustered_matches,
    int*&                                       d_clustered_count,
    int&                                        h_clustered_count_out,
    cudaEvent_t                                 start,
    cudaEvent_t                                 stop)
{
    cudacheck(cudaSetDevice(device_id));
    cudacheck(cudaEventRecord(start));

    h_refined_count_out   = 0;
    h_clustered_count_out = 0;

    if (num_matches > 0) {
        if (d_refined_matches == nullptr)
            cudacheck(cudaMalloc(&d_refined_matches, static_cast<size_t>(num_matches) * sizeof(Temporal_Refined_Match_GPU)));
        if (d_refined_count == nullptr)
            cudacheck(cudaMalloc(&d_refined_count, sizeof(int)));
        cudacheck(cudaMemset(d_refined_count, 0, sizeof(int)));

        const int threads = 128;  //> lower occupancy to give GN iterations more registers
        const int blocks  = (num_matches + threads - 1) / threads;
        mono_temporal_photometric_refine_kernel<<<blocks, threads>>>(
            d_matches, d_kf_patches, d_kf_edges, d_cf_edges, cf_tex,
            d_refined_matches, d_refined_count,
            num_matches, cf_img_w, cf_img_h);
        cudacheck(cudaGetLastError());
        cudacheck(cudaDeviceSynchronize());

        cudacheck(cudaMemcpy(&h_refined_count_out, d_refined_count, sizeof(int), cudaMemcpyDeviceToHost));

        cluster_temporal_refined_matches(d_refined_matches, h_refined_count_out,
                                         d_clustered_matches, d_clustered_count, h_clustered_count_out);
    }

    return mono_elapsed_ms(start, stop);
}

#endif // TEMPORAL_EDGE_TRACKING_MONO_CU
