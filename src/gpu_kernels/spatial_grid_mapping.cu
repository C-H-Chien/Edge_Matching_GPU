#ifndef SPATIAL_GRID_MAPPING_CU
#define SPATIAL_GRID_MAPPING_CU

#include <cuda_runtime.h>
#include "gpu_kernels.h"
#include "gpu_settings.h"

//> Assign each edge a grid cell ID based on its edge location
__global__ 
void 
spatial_grid_mapping_kernel(
    const Edge_GPU* d_edges, 
    int num_of_edges, 
    int num_of_grids_in_width,
    int* d_cell_ids) 
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_of_edges) {
        // Find 2D grid coordinates
        int cx = (int)(d_edges[tid].location_x / GRID_CELL_SIZE);
        int cy = (int)(d_edges[tid].location_y / GRID_CELL_SIZE);
        
        // Flatten 2D grid coordinates to a 1D Cell ID
        d_cell_ids[tid] = cx + cy * num_of_grids_in_width;
    }
}

void spatial_grid_mapping(
    int device_id,
    const Edge_GPU* d_edges,
    int num_of_edges,
    int num_of_grids_in_width,
    int* d_cell_ids)
{
    if (num_of_edges <= 0) {
        LOG_ERROR("Number of edges is less than or equal to 0 in spatial grid mapping");
        return;
    }

    cudacheck(cudaSetDevice(device_id));

    int nthreads_max = 1024;
    cudacheck(cudaDeviceGetAttribute(&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device_id));

    // One thread per edge: block size must stay within device limit (typically ≤1024).
    const int threads_per_block =
        (NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING < nthreads_max)
            ? NUM_OF_THREADBLOCKS_SPATIAL_GRID_MAPPING
            : nthreads_max;
    const int num_blocks =
        (num_of_edges + threads_per_block - 1) / threads_per_block;

    dim3 block_dim(threads_per_block, 1, 1);
    dim3 grid_dim(num_blocks, 1, 1);

    void *kernel_args[] = {
        reinterpret_cast<void*>(&d_edges),
        reinterpret_cast<void*>(&num_of_edges),
        reinterpret_cast<void*>(&num_of_grids_in_width),
        reinterpret_cast<void*>(&d_cell_ids)
    };

    cudacheck(cudaLaunchKernel(
        reinterpret_cast<const void*>(spatial_grid_mapping_kernel),
        grid_dim,
        block_dim,
        kernel_args,
        0,
        nullptr));
}

#endif // SPATIAL_GRID_MAPPING_CU