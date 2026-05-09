#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#include <cuda_runtime.h>
#include <cstddef>
#include <stdio.h>
#include <string>
#include "definitions.h"
#include "gpu_settings.h"

//> Helper Edge structure
struct Edge_GPU { 
    float location_x;
    float location_y; 
    float orientation;
};

struct Point2D_GPU {
    float x;
    float y;
};

struct Match_by_Edge_Index { 
    int left_edge_idx;
    int right_edge_idx; 
};

//> CUDA texture wrapper holds the GPU texture object and its underlying array
struct CUDA_Texture_Wrapper {
    cudaArray_t array;
    cudaTextureObject_t texObj;
};

//> This struct is per edge pair hypothesis used in the edge location refinement step
struct Refined_Edge_Hypothesis_Match_GPU {
    int left_edge_idx;
    int right_edge_idx;
    float refined_right_x;
    float refined_right_y;
    float photometric_rms;
    float source_right_orientation;
};

//> One consolidated match per distance-cluster after merge (no TOED right index — geometry is fused).
struct Merged_Refined_Stereo_Match_GPU {
    int left_edge_idx;
    float merged_right_x;
    float merged_right_y;
    float merged_right_orientation;
};

//> The left edge patches are pre-computed and stored in global memory
struct Precomputed_Edge_Patches_Photometry {
    float pL_plus[TOTAL_NUM_OF_PATCH_PIXELS];
    float mL_plus; //> mean
    float vL_plus; //> standard deviation
    
    float pL_minus[TOTAL_NUM_OF_PATCH_PIXELS];
    float mL_minus; //> mean
    float vL_minus; //> standard deviation
};

inline CUDA_Texture_Wrapper create_CUDA_texture_object_for_img(float* h_img, int img_width, int img_height) 
{
    CUDA_Texture_Wrapper wrapper;

    //> Define the channel format (1 channel, 32-bit float)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    //> CUDA array allocated in the memory where the swizzled 2D memory lives
    cudacheck( cudaMallocArray(&wrapper.array, &channelDesc, img_width, img_height) );

    //> Copy linear host data to the swizzled CUDA array
    size_t spitch = img_width * sizeof(float);
    cudacheck( cudaMemcpy2DToArray(wrapper.array, 0, 0, h_img, spitch, img_width * sizeof(float), img_height, cudaMemcpyHostToDevice) );
    
    //> Specify the resource descriptor
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = wrapper.array;

    //> Specify the texture descriptor, namely, how the hardware reads it
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp; // Clamp X to edge
    texDesc.addressMode[1] = cudaAddressModeClamp; // Clamp Y to edge
    texDesc.filterMode     = cudaFilterModeLinear; // Enable Hardware Bilinear Interpolation
    texDesc.readMode       = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;               //> Absolute pixel coordinates are used rather than normalized coordinates

    //> Create the texture object
    cudacheck( cudaCreateTextureObject(&wrapper.texObj, &resDesc, &texDesc, nullptr) );

    return wrapper;
}

//> Thrust + GPU kernel + sort pipeline for spatial grid mapping (defined in spatial_grid_mapping.cu)
float edge_spatial_map_thrust_pipeline(
    int             device_id,
    /* Data live in the host memory */
    const Edge_GPU* h_right_edges,
    int*            &h_cell_ids_out,
    /* Data live in the device memory */
    Edge_GPU*       d_right_edges,
    int*            d_cell_ids,
    int*            d_edge_indices,
    int*            d_cell_start_idx,
    int*            d_num_of_edges_in_cell,
    /* Others */
    std::size_t     num_right_edges,
    int             total_num_of_grid_cells,
    int             num_of_grid_cells_in_width,
    int             num_of_grid_cells_in_height,
    cudaEvent_t     start,
    cudaEvent_t     stop
);

//> GPU kernel pipeline for epipolar rasterization (defined in epipolar_rasterization.cu)
float epipolar_rasterization_thick_band_pipeline(
    int                     device_id,
    /* Data live in the host memory, used for results retrieval */
    const float*            h_F,
    const Edge_GPU*         h_left_edges,
    int                     &h_match_count_out,
    Match_by_Edge_Index*    &h_matches_out,
    /* Data live in the device memory */
    Edge_GPU*               d_left_edges,
    const Edge_GPU*         d_right_edges, 
    const int*              d_edge_indices,
    const int*              d_cell_start_idx, 
    const int*              d_num_of_edges_in_cell, 
    float*                  d_F,
    Match_by_Edge_Index*    d_matches, 
    int*                    d_match_count, 
    /* Others */
    int                     num_of_left_edges, 
    int                     img_right_W, 
    int                     img_right_H, 
    int                     num_of_grid_cells_in_width, 
    int                     num_of_grid_cells_in_height, 
    int                     max_matches,
    cudaEvent_t             start,
    cudaEvent_t             stop
);

//> GPU kernel pipeline for precompute left edge patches photometry (defined in precompute_left_edge_patches_photometry.cu)
float precompute_left_edge_patches_photometry_pipeline(
    int             device_id,
    /* Data live in the CUDA texture memory */
    cudaTextureObject_t left_tex,
    /* Data live in the device memory */
    Precomputed_Edge_Patches_Photometry* &d_left_patches,
    Edge_GPU*       d_left_edges,
    /* Others */
    std::size_t     num_of_left_edges,
    cudaEvent_t     start,
    cudaEvent_t     stop
);

//> GPU-only SIFT-style descriptor filter (defined in apply_SIFT_descriptor_filter.cu).
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
    cudaEvent_t                 stop
);

//> GPU kernel pipeline for apply NCC filter (defined in apply_NCC_on_the_fly.cu)
float apply_NCC_filter_pipeline(
    int                         device_id,
    int                         &h_NCC_match_count_out,
    /* Data live in the CUDA texture memory */
    cudaTextureObject_t         right_tex,
    /* Data live in the device memory */
    const Match_by_Edge_Index*  d_candidate_matches,
    const Edge_GPU*             d_left_edges,
    const Edge_GPU*             d_right_edges,
    Match_by_Edge_Index*        &d_final_matches_passing_NCC_filter,
    int*                        &d_match_count_passing_NCC_filter,
    float*                      &d_ncc_scores_out,
    Precomputed_Edge_Patches_Photometry* d_left_patches,
    /* Others */
    int                         num_candidates,
    cudaEvent_t                 start,
    cudaEvent_t                 stop
);

//> GPU kernel pipeline for Best-Nearly-Best filtering (defined in apply_best_nearly_best_filter.cu)
float apply_best_nearly_best_filter_pipeline(
    int                         device_id,
    int                         &h_bnb_match_count_out,
    const Match_by_Edge_Index*  d_candidate_matches,
    const float*                d_candidate_scores,
    Match_by_Edge_Index*        &d_final_matches_passing_bnb_filter,
    float*                      &d_final_scores_passing_bnb_filter,
    int*                        &d_final_count_passing_bnb_filter,
    int                         num_candidates,
    int                         num_left_edges,
    float                       ratio_threshold,
    bool                        higher_is_better,
    cudaEvent_t                 start,
    cudaEvent_t                 stop
);

//> Shift right candidate edges onto the epipolar line, GN refine along the epipolar line based on Huber photometric residual,
//> then merge hypotheses per left edge using distance-only clustering (defined in combined_edge_epipolar_shift_and_refine_and_merge_pipeline.cu)
float combined_edge_epipolar_shift_and_refine_and_merge_pipeline(
    int                                         device_id,
    /* Data live in the CUDA texture memory */
    cudaTextureObject_t                         right_tex,
    /* Data live in the device memory */
    const float*                                d_F,
    const Edge_GPU*                             d_left_edges,
    const Edge_GPU*                             d_right_edges,
    const Precomputed_Edge_Patches_Photometry*  d_left_patches,
    const Match_by_Edge_Index*                  d_matches_in,
    Merged_Refined_Stereo_Match_GPU*&           d_merged_matches_out,
    int*&                                       d_merged_count_out,
    /* Others */
    int                                         num_matches_in,
    int                                         img_right_width,
    int                                         img_right_height,    
    cudaEvent_t                                 start,
    cudaEvent_t                                 stop
);

#endif // GPU_KERNELS_H
