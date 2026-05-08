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
    /* Data live in the host memory */
    float*          h_left_img,
    /* Data live in the device memory */
    Precomputed_Edge_Patches_Photometry* &d_left_patches,
    Edge_GPU*       d_left_edges,
    /* Others */
    int             img_width,
    int             img_height,
    std::size_t     num_of_left_edges,
    cudaEvent_t     start,
    cudaEvent_t     stop
);

//> GPU kernel pipeline for apply NCC filter (defined in apply_NCC_on_the_fly.cu)
float apply_NCC_filter_pipeline(
    int                         device_id,
    /* Data live in the host memory */
    float*                      h_right_img,
    int                         &h_NCC_match_count_out,
    /* Data live in the device memory */
    const Match_by_Edge_Index*  d_candidate_matches,
    const Edge_GPU*             d_left_edges,
    const Edge_GPU*             d_right_edges,
    Match_by_Edge_Index*        &d_final_matches_passing_NCC_filter,
    int*                        &d_match_count_passing_NCC_filter,
    Precomputed_Edge_Patches_Photometry* d_left_patches,
    /* Others */
    int                         right_img_width,
    int                         right_img_height,
    int                         num_candidates,
    cudaEvent_t                 start,
    cudaEvent_t                 stop
);

#endif // GPU_KERNELS_H