#ifndef EVAL_EDGES_IN_GRID_CELLS_CUH
#define EVAL_EDGES_IN_GRID_CELLS_CUH

#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>
#include <math.h>
#include "gpu_kernels.h"

__device__ 
void 
eval_edges_in_grid_cells(
    Edge_GPU left_edge,
    int cx, int cy, int grid_width, 
    int left_edge_idx, float a, float b, float c,
    const Edge_GPU* d_right_edges, const int* d_edge_indices,
    const int* d_cell_start_idx, const int* d_cell_counts,
    Match_by_Edge_Index* d_matches, int* d_match_count, int max_matches)
{
    int cell_id = cy * grid_width + cx;
    int start_idx = d_cell_start_idx[cell_id];
    int count = d_cell_counts[cell_id];

    for (int i = 0; i < count; i++) {
        //> Memory Lookup: Get the original index, then the actual point
        int right_edge_idx = d_edge_indices[start_idx + i];
        Edge_GPU right_edge = d_right_edges[right_edge_idx];

        //> Skip if the left edge is to the left of the right edge which is not possible if a stereo rig is left-right configuration
        if (left_edge.location_x < right_edge.location_x) continue;

        //> Calculate the exact perpendicular distance to the epipolar line
        float dist_to_epipolar_line = fabsf(a * right_edge.location_x + b * right_edge.location_y + c);

        //> Calculate the squared distance between left and right edges
        float disparity_squared = (left_edge.location_x - right_edge.location_x) * (left_edge.location_x - right_edge.location_x) + \
                                    (left_edge.location_y - right_edge.location_y) * (left_edge.location_y - right_edge.location_y);

        //> EP and LP filters
        if (dist_to_epipolar_line < EPIPOLAR_LINE_DIST_THRESH && disparity_squared < MAX_DISPARITY * MAX_DISPARITY) {
            int insert_pos = atomicAdd(d_match_count, 1);
            if (insert_pos < max_matches) {
                d_matches[insert_pos].left_edge_idx = left_edge_idx;
                d_matches[insert_pos].right_edge_idx = right_edge_idx;
            }
        }
    }
}

#endif