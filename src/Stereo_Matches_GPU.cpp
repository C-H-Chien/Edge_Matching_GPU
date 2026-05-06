#ifndef STEREO_MATCHES_GPU_CPP
#define STEREO_MATCHES_GPU_CPP

#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "definitions.h"
#include "Stereo_Matches_GPU.h"
#include "gpu_kernels.h"
#include "Dataset.h"

Stereo_Matches_GPU_Pipeline::Stereo_Matches_GPU_Pipeline(Dataset::Ptr dataset, const StereoFrame& current_frame, int device)
    : device_id(device)
    ,img_left_height(mat_rows_prefer_undistorted(current_frame.left_image_undistorted, current_frame.left_image))
    ,img_right_height(mat_rows_prefer_undistorted(current_frame.right_image_undistorted, current_frame.right_image))
    ,img_left_width(mat_cols_prefer_undistorted(current_frame.left_image_undistorted, current_frame.left_image))
    ,img_right_width(mat_cols_prefer_undistorted(current_frame.right_image_undistorted, current_frame.right_image))
    ,dataset_(std::move(dataset))
{
    //> Fundamental matrix (left to right) is casted to float matrix
    F_left2right_ = dataset_->get_fund_mat_21().cast<float>();
    F_left2right = F_left2right_.data();
    num_left_edges_ = current_frame.left_edges.size();
    num_right_edges_ = current_frame.right_edges.size();

    //> convert edges to float (double -> float)
    h_left_edges = new Edge_GPU[num_left_edges_];
    h_right_edges = new Edge_GPU[num_right_edges_];

    //> Casting the edges to the proper format for GPU processing
    #pragma omp parallel for
    for (size_t i = 0; i < num_left_edges_; i++) {
        h_left_edges[i].location_x = current_frame.left_edges[i].location.x;
        h_left_edges[i].location_y = current_frame.left_edges[i].location.y;
        h_left_edges[i].orientation = current_frame.left_edges[i].orientation;
    }

    #pragma omp parallel for
    for (size_t i = 0; i < num_right_edges_; i++) {
        h_right_edges[i].location_x = current_frame.right_edges[i].location.x;
        h_right_edges[i].location_y = current_frame.right_edges[i].location.y;
        h_right_edges[i].orientation = current_frame.right_edges[i].orientation;
    }

    num_of_grid_cells_in_width = static_cast<int>(std::ceil(static_cast<double>(img_right_width) / GRID_CELL_SIZE));

    //> Allocate GPU memory
    cudacheck( cudaMalloc((void**)&d_left_edges,    num_left_edges_ * sizeof(Edge_GPU))   );
    cudacheck( cudaMalloc((void**)&d_right_edges,   num_right_edges_ * sizeof(Edge_GPU))  );
    cudacheck( cudaMalloc((void**)&d_cell_ids,      num_right_edges_ * sizeof(int))       );

    //> cuda event
    cudacheck( cudaEventCreate(&start) );
    cudacheck( cudaEventCreate(&stop) );
}

void Stereo_Matches_GPU_Pipeline::get_stereo_edge_matching_GPU() {

    //> Timer holders
    GPU_Timing_Statistics GPU_time_collection;

    //> For each edge on the right, construct the spatial grid mapping (result is stored in d_cell_ids)
    GPU_time_collection.time_spatial_grid_mapping = map_spatial_grid_to_edges();
}

float Stereo_Matches_GPU_Pipeline::map_spatial_grid_to_edges() {

    //> Copy edges data from host to device
    cudacheck( cudaMemcpy(d_left_edges,  h_left_edges,  num_left_edges_  * sizeof(Edge_GPU), cudaMemcpyHostToDevice) );
    cudacheck( cudaMemcpy(d_right_edges, h_right_edges, num_right_edges_ * sizeof(Edge_GPU), cudaMemcpyHostToDevice) );

    //> Start the CUDA event timer
    cudacheck( cudaEventRecord(start) );

    //> Create and Launch the GPU kernel
    spatial_grid_mapping(device_id, d_left_edges, static_cast<int>(num_left_edges_), num_of_grid_cells_in_width, d_cell_ids);

    //> Synchronize the host and device
    cudacheck( cudaDeviceSynchronize());

    //> End the CUDA event timer
    float time_spatial_grid_mapping = 0.0;
    cudacheck( cudaEventRecord(stop) );
	cudacheck( cudaEventSynchronize(stop) );
	cudacheck( cudaEventElapsedTime(&time_spatial_grid_mapping, start, stop) );

    std::cout << "Time for spatial grid mapping: " << time_spatial_grid_mapping << " milliseconds" << std::endl;
    return time_spatial_grid_mapping;
}

Stereo_Matches_GPU_Pipeline::~Stereo_Matches_GPU_Pipeline()
{
    delete[] h_left_edges;
    delete[] h_right_edges;
    delete[] h_cell_ids;

    cudacheck( cudaFree(d_left_edges) );
    cudacheck( cudaFree(d_right_edges) );
    cudacheck( cudaFree(d_cell_ids) );

    cudacheck(cudaEventDestroy(start));
    cudacheck(cudaEventDestroy(stop));
}

#endif // STEREO_MATCHES_GPU_CPP