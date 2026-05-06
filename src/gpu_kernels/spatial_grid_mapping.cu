#ifndef SPATIAL_GRID_MAPPING_CU
#define SPATIAL_GRID_MAPPING_CU

#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

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
    cudaEvent_t stop)
{
    //> Copy edges data from host to device
    cudacheck( cudaSetDevice(device_id) );
    cudacheck( cudaMemcpy( d_left_edges,  h_left_edges,  num_left_edges  * sizeof(Edge_GPU), cudaMemcpyHostToDevice) );
    cudacheck( cudaMemcpy( d_right_edges, h_right_edges, num_right_edges * sizeof(Edge_GPU), cudaMemcpyHostToDevice) );

    //> Wrap raw pointers in thrust::device_ptr; this allows Thrust to interact with memory allocated via cudaMalloc
    thrust::device_ptr<int> t_cell_ids(d_cell_ids);
    thrust::device_ptr<int> t_edge_indices(d_edge_indices);
    thrust::device_ptr<int> t_cell_start_idx(d_cell_start_idx);
    thrust::device_ptr<int> t_num_of_edges_in_cell(d_num_of_edges_in_cell);

    //> Initialize arrays using Thrust. By default the cells are empty.
    thrust::fill(t_cell_start_idx, t_cell_start_idx + total_num_of_grid_cells, 0);
    thrust::fill(t_num_of_edges_in_cell, t_num_of_edges_in_cell + total_num_of_grid_cells, 0);

    //> Fill edge indices with incremental integers used to track original edges during sort
    thrust::sequence(t_edge_indices, t_edge_indices + num_right_edges);

    //> Start the CUDA event timer
    cudacheck(cudaEventRecord(start));

    //> Create and Launch the GPU kernel: assign each right edge a grid cell id based on its location
    spatial_grid_mapping( device_id, d_right_edges, static_cast<int>(num_right_edges), num_of_grid_cells_in_width, d_cell_ids );

    //> Synchronize the host and device
    cudacheck(cudaDeviceSynchronize());

    //> End the CUDA event timer
    float time_spatial_grid_mapping = 0.0f;
    cudacheck(cudaEventRecord(stop));
    cudacheck(cudaEventSynchronize(stop));
    cudacheck(cudaEventElapsedTime(&time_spatial_grid_mapping, start, stop));

    std::cout << "Time for spatial grid mapping: " << time_spatial_grid_mapping << " milliseconds" << std::endl;

    //> Sort the indices using the Cell IDs as the key. This groups all points in Cell 0 together, then Cell 1, etc.
    //> It rearranges 't_edge_indices' to match the sorted 't_cell_ids'.
    thrust::sort_by_key(t_cell_ids, t_cell_ids + num_right_edges, t_edge_indices);

    return time_spatial_grid_mapping;
}

#endif // SPATIAL_GRID_MAPPING_CU