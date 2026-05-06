#ifndef GPU_TOED_HPP
#define GPU_TOED_HPP

#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <Eigen/Core>

#include "Stereo_Iterator.h"
#include "gpu_settings.h"
#include "definitions.h"
#include "gpu_kernels.h"
#include "Dataset.h"

struct GPU_Timing_Statistics
{
    float time_spatial_grid_mapping;
    float time_find_grid_boundaries;
};

class Stereo_Matches_GPU_Pipeline {
    
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Stereo_Matches_GPU_Pipeline> Ptr;

    Stereo_Matches_GPU_Pipeline(Dataset::Ptr dataset, const StereoFrame& current_frame, int device);
    ~Stereo_Matches_GPU_Pipeline();

    //> main function for constructing stereo edge correspondences
    void get_stereo_edge_matching_GPU();

    float map_spatial_grid_to_edges();

    int device_id;
    int img_left_height, img_right_height;
    int img_left_width, img_right_width;    

    float *img, *dev_img;
    float *Ix, *dev_Ix; 
    float *Iy, *dev_Iy;

    // timing
    float time_EP;
    cudaEvent_t start, stop;

private:

    //> Edges casted to float (host buffers; free in destructor)
    Edge_GPU* h_left_edges = nullptr;
    Edge_GPU* h_right_edges = nullptr;
    Edge_GPU* d_left_edges = nullptr;
    Edge_GPU* d_right_edges = nullptr;

    int* h_cell_ids = nullptr;
    int* d_cell_ids = nullptr;

    int num_of_grid_cells_in_width;
    size_t num_left_edges_ = 0;
    size_t num_right_edges_ = 0;

    Dataset::Ptr dataset_;
    Eigen::Matrix3f F_left2right_;
    float* F_left2right = nullptr;

    //> TEST functions
    void test_spatial_grid_mapping() {
        //> Do a simplet test: randomly pick 10 right edge locations and print the cell ids
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, num_right_edges_ - 1);
        for (int i = 0; i < 10; i++) {
            int random_index = dis(gen);
            std::cout << "Cell id for right edge " << random_index << " at location (" << h_right_edges[random_index].location_x << ", " << h_right_edges[random_index].location_y << "): " << h_cell_ids[random_index] << std::endl;
        }
    }

};

namespace {
    int mat_rows_prefer_undistorted(const cv::Mat& undistorted, const cv::Mat& raw)
    {
        if (!undistorted.empty())
            return undistorted.rows;
        return raw.empty() ? 0 : raw.rows;
    }

    int mat_cols_prefer_undistorted(const cv::Mat& undistorted, const cv::Mat& raw)
    {
        if (!undistorted.empty())
            return undistorted.cols;
        return raw.empty() ? 0 : raw.cols;
    }
}

#endif