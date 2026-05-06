#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#include <cuda_runtime.h>
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

#endif // GPU_KERNELS_H