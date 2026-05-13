#ifndef APPLY_SIFT_DESCRIPTOR_FILTER_CU
#define APPLY_SIFT_DESCRIPTOR_FILTER_CU

#include <cuda_runtime.h>
#include <math.h>
#include <cstddef>
#include <iostream>

#include "definitions.h"
#include "gpu_settings.h"
#include "gpu_kernels.h"

static constexpr float PI_F = 3.14159265358979323846f;
static constexpr float TWO_PI_F = 2.0f * PI_F;

__device__ __forceinline__
float wrap_angle_0_2pi(float angle)
{
    while (angle < 0.0f)
        angle += TWO_PI_F;
    while (angle >= TWO_PI_F)
        angle -= TWO_PI_F;
    return angle;
}

__device__ __forceinline__
float tex_clamped(cudaTextureObject_t tex, float x, float y, int width, int height)
{
    x = fminf(fmaxf(x, 0.0f), static_cast<float>(width - 1));
    y = fminf(fmaxf(y, 0.0f), static_cast<float>(height - 1));
    return tex2D<float>(tex, x, y);
}

__device__ __forceinline__
void extract_gpu_sift_descriptor(
    cudaTextureObject_t tex,
    int width,
    int height,
    float cx,
    float cy,
    float orientation,
    float* desc)
{
    #pragma unroll
    for (int i = 0; i < SIFT_DESCRIPTOR_DIM; ++i)
        desc[i] = 0.0f;

    const int grid = SIFT_DESCRIPTOR_GRID_SIZE;
    const int bins = SIFT_DESCRIPTOR_ORIENTATION_BINS;
    const int sample_width = SIFT_DESCRIPTOR_SAMPLE_WIDTH;
    const int half_width = sample_width / 2;
    const float cell_width = static_cast<float>(sample_width) / static_cast<float>(grid);
    const float cos_t = cosf(orientation);
    const float sin_t = sinf(orientation);
    const float sigma = 0.5f * static_cast<float>(sample_width);
    const float inv_two_sigma2 = 1.0f / (2.0f * sigma * sigma);

    for (int yy = -half_width; yy < half_width; ++yy) {
        for (int xx = -half_width; xx < half_width; ++xx) {
            const float local_x = static_cast<float>(xx) + 0.5f;
            const float local_y = static_cast<float>(yy) + 0.5f;

            const int cell_x = static_cast<int>((local_x + static_cast<float>(half_width)) / cell_width);
            const int cell_y = static_cast<int>((local_y + static_cast<float>(half_width)) / cell_width);
            if (cell_x < 0 || cell_x >= grid || cell_y < 0 || cell_y >= grid)
                continue;

            const float img_x = cx + local_x * cos_t - local_y * sin_t;
            const float img_y = cy + local_x * sin_t + local_y * cos_t;

            const float gx = tex_clamped(tex, img_x + 1.0f, img_y, width, height) -
                             tex_clamped(tex, img_x - 1.0f, img_y, width, height);
            const float gy = tex_clamped(tex, img_x, img_y + 1.0f, width, height) -
                             tex_clamped(tex, img_x, img_y - 1.0f, width, height);
            const float mag = sqrtf(gx * gx + gy * gy);
            if (mag <= 1e-6f)
                continue;

            const float rel_angle = wrap_angle_0_2pi(atan2f(gy, gx) - orientation);
            int bin = static_cast<int>(floorf(rel_angle * static_cast<float>(bins) / TWO_PI_F));
            if (bin >= bins)
                bin = bins - 1;

            const float gaussian = expf(-(local_x * local_x + local_y * local_y) * inv_two_sigma2);
            const int idx = (cell_y * grid + cell_x) * bins + bin;
            desc[idx] += mag * gaussian;
        }
    }

    float norm_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < SIFT_DESCRIPTOR_DIM; ++i)
        norm_sq += desc[i] * desc[i];
    float inv_norm = rsqrtf(fmaxf(norm_sq, 1e-12f));

    //> Lowe-style clipping followed by renormalization.
    norm_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < SIFT_DESCRIPTOR_DIM; ++i) {
        desc[i] = fminf(desc[i] * inv_norm, 0.2f);
        norm_sq += desc[i] * desc[i];
    }
    inv_norm = rsqrtf(fmaxf(norm_sq, 1e-12f));
    #pragma unroll
    for (int i = 0; i < SIFT_DESCRIPTOR_DIM; ++i)
        desc[i] = desc[i] * inv_norm * static_cast<float>(SIFT_DESCRIPTOR_SCALE);
}

__device__ __forceinline__
float l2_distance(const float* a, const float* b)
{
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < SIFT_DESCRIPTOR_DIM; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum);
}

__global__
void apply_SIFT_descriptor_filter_kernel(
    const Match_by_Edge_Index* d_ncc_matches,
    const Edge_GPU* d_left_edges,
    const Edge_GPU* d_right_edges,
    Match_by_Edge_Index* d_final_matches,
    int* d_final_count,
    float* d_final_scores,
    cudaTextureObject_t left_tex,
    cudaTextureObject_t right_tex,
    int left_width,
    int left_height,
    int right_width,
    int right_height,
    int num_ncc_matches,
    float sift_threshold)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_ncc_matches)
        return;

    //> Get the candidate match from the NCC matches
    const Match_by_Edge_Index candidate = d_ncc_matches[tid];
    if (candidate.left_edge_idx < 0 || candidate.right_edge_idx < 0)
        return;

    const Edge_GPU left_edge = d_left_edges[candidate.left_edge_idx];
    const Edge_GPU right_edge = d_right_edges[candidate.right_edge_idx];

    const float cos_l = cosf(left_edge.orientation);
    const float sin_l = sinf(left_edge.orientation);
    const float lnx = sin_l * static_cast<float>(SIFT_ORTHOGONAL_SHIFT);
    const float lny = -cos_l * static_cast<float>(SIFT_ORTHOGONAL_SHIFT);

    const float cos_r = cosf(right_edge.orientation);
    const float sin_r = sinf(right_edge.orientation);
    const float rnx = sin_r * static_cast<float>(SIFT_ORTHOGONAL_SHIFT);
    const float rny = -cos_r * static_cast<float>(SIFT_ORTHOGONAL_SHIFT);

    float l_plus[SIFT_DESCRIPTOR_DIM];
    float l_minus[SIFT_DESCRIPTOR_DIM];
    float r_plus[SIFT_DESCRIPTOR_DIM];
    float r_minus[SIFT_DESCRIPTOR_DIM];

    extract_gpu_sift_descriptor(left_tex, left_width, left_height, left_edge.location_x + lnx, left_edge.location_y + lny, left_edge.orientation, l_plus); 
    extract_gpu_sift_descriptor(left_tex, left_width, left_height, left_edge.location_x - lnx, left_edge.location_y - lny, left_edge.orientation, l_minus);
    extract_gpu_sift_descriptor(right_tex, right_width, right_height, right_edge.location_x + rnx, right_edge.location_y + rny, right_edge.orientation, r_plus);
    extract_gpu_sift_descriptor(right_tex, right_width, right_height, right_edge.location_x - rnx, right_edge.location_y - rny, right_edge.orientation, r_minus);

    const float d_pp = l2_distance(l_plus,  r_plus);
    const float d_pn = l2_distance(l_plus,  r_minus);
    const float d_np = l2_distance(l_minus, r_plus);
    const float d_nn = l2_distance(l_minus, r_minus);
    const float min_dist = fminf(fminf(d_pp, d_pn), fminf(d_np, d_nn));

    if (min_dist < sift_threshold) {
        const int out_idx = atomicAdd(d_final_count, 1);
        d_final_matches[out_idx] = candidate;
        d_final_scores[out_idx] = min_dist;
    }
}

void apply_SIFT_descriptor_filter_kernel_launcher(
    int device_id,
    const Match_by_Edge_Index* d_ncc_matches,
    const Edge_GPU* d_left_edges,
    const Edge_GPU* d_right_edges,
    Match_by_Edge_Index* d_final_matches,
    int* d_final_count,
    float* d_final_scores,
    cudaTextureObject_t left_tex,
    cudaTextureObject_t right_tex,
    int left_width,
    int left_height,
    int right_width,
    int right_height,
    int num_ncc_matches,
    float sift_threshold)
{
    int max_threads_per_block = 1024;
    cudacheck(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id));

    const int threads_per_block = (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < max_threads_per_block) 
                                  ? NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING : max_threads_per_block;
    const int num_blocks = (num_ncc_matches + threads_per_block - 1) / threads_per_block;

    dim3 block_dim(threads_per_block, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);

    void *kernel_args[] = {
        reinterpret_cast<void*>(&d_ncc_matches),
        reinterpret_cast<void*>(&d_left_edges),
        reinterpret_cast<void*>(&d_right_edges),
        reinterpret_cast<void*>(&d_final_matches),
        reinterpret_cast<void*>(&d_final_count),
        reinterpret_cast<void*>(&d_final_scores),
        reinterpret_cast<void*>(&left_tex),
        reinterpret_cast<void*>(&right_tex),
        reinterpret_cast<void*>(&left_width),
        reinterpret_cast<void*>(&left_height),
        reinterpret_cast<void*>(&right_width),
        reinterpret_cast<void*>(&right_height),
        reinterpret_cast<void*>(&num_ncc_matches),
        reinterpret_cast<void*>(&sift_threshold)
    };

    cudacheck(cudaLaunchKernel(reinterpret_cast<const void*>(apply_SIFT_descriptor_filter_kernel), grid_dim, block_dim, kernel_args, 0, nullptr));
}

float apply_SIFT_descriptor_filter_pipeline(
    int                         device_id,
    /* Data live in the host memory */
    int                         &h_SIFT_match_count_out,
    /* Data live in the device memory */
    cudaTextureObject_t         left_tex,
    cudaTextureObject_t         right_tex,
    const Match_by_Edge_Index*  d_ncc_matches,
    const Edge_GPU*             d_left_edges,
    const Edge_GPU*             d_right_edges,
    Match_by_Edge_Index*        &d_final_matches_passing_SIFT_filter,
    int*                        &d_final_count_passing_SIFT_filter,
    float*                      &d_sift_scores_out,
    /* Others */
    int                         left_img_width,
    int                         left_img_height,
    int                         right_img_width,
    int                         right_img_height,
    int                         num_ncc_matches,
    cudaEvent_t                 start,
    cudaEvent_t                 stop,
    float                       sift_threshold)
{
    cudacheck(cudaSetDevice(device_id));

    //> Start the CUDA event timer
    cudacheck(cudaEventRecord(start));

    //> Allocate memory for the final matches and number of matches passing the SIFT filter
    cudacheck(cudaMalloc(&d_final_matches_passing_SIFT_filter, static_cast<size_t>(num_ncc_matches) * sizeof(Match_by_Edge_Index)));
    cudacheck(cudaMalloc(&d_final_count_passing_SIFT_filter, sizeof(int)));
    cudacheck(cudaMalloc(&d_sift_scores_out, static_cast<size_t>(num_ncc_matches) * sizeof(float)));

    //> Initialize the number of matches passing the SIFT filter to 0
    cudacheck(cudaMemset(d_final_count_passing_SIFT_filter, 0, sizeof(int)));

    //> Launch the SIFT descriptor filter kernel
    apply_SIFT_descriptor_filter_kernel_launcher( device_id, d_ncc_matches, d_left_edges, d_right_edges, d_final_matches_passing_SIFT_filter,
                                                  d_final_count_passing_SIFT_filter, d_sift_scores_out, left_tex, right_tex, left_img_width, left_img_height,
                                                  right_img_width, right_img_height, num_ncc_matches, sift_threshold );
    cudacheck(cudaDeviceSynchronize());

    //> End the CUDA event timer for SIFT descriptor filter
    float apply_SIFT_filter_time = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&apply_SIFT_filter_time, start, stop));

    h_SIFT_match_count_out = 0;
    cudacheck(cudaMemcpy(&h_SIFT_match_count_out, d_final_count_passing_SIFT_filter, sizeof(int), cudaMemcpyDeviceToHost));

    return apply_SIFT_filter_time;
}

#endif // APPLY_SIFT_DESCRIPTOR_FILTER_CU
