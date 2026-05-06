#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#include <cuda_runtime.h>
#include <cstddef>
#include <stdio.h>
#include <string>
#include "definitions.h"

//> Helper Edge structure
struct Edge_GPU { 
    float location_x;
    float location_y; 
    float orientation;
};

void spatial_grid_mapping(
    int device_id,
    const Edge_GPU* d_edges,
    int num_of_edges,
    int num_of_grids_in_width,
    int* d_cell_ids);

// Thrust + kernel + sort pipeline (implemented in spatial_grid_mapping.cu)
float edge_spatial_map_thrust_pipeline(
    int device_id,
    Edge_GPU* d_left_edges,
    const Edge_GPU* h_left_edges,
    std::size_t num_left_edges,
    Edge_GPU* d_right_edges,
    const Edge_GPU* h_right_edges,
    std::size_t num_right_edges,
    int* d_cell_ids,
    int* d_edge_indices,
    int* d_cell_start_idx,
    int* d_num_of_edges_in_cell,
    int total_num_of_grid_cells,
    int num_of_grid_cells_in_width,
    cudaEvent_t start,
    cudaEvent_t stop);

#endif // GPU_KERNELS_H