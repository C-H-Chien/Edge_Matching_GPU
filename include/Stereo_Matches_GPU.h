#ifndef STEREO_MATCHES_GPU_HPP
#define STEREO_MATCHES_GPU_HPP

#include <cmath>
#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <Eigen/Core>

#include "Stereo_Iterator.h"
#include "gpu_settings.h"
#include "definitions.h"
#include "gpu_kernels.h"
#include "Stereo_Matches_GPU_Validator.h"
#include "Dataset.h"

struct GPU_Timing_Statistics
{
    float time_spatial_grid_mapping;
    float time_epipolar_rasterization;
    float time_precompute_left_edge_patches_photometry;
    float time_apply_NCC_filter;
};

class Stereo_Matches_GPU_Pipeline {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Stereo_Matches_GPU_Pipeline> Ptr;

    Stereo_Matches_GPU_Pipeline(Dataset::Ptr dataset, const StereoFrame& current_frame, int device);
    ~Stereo_Matches_GPU_Pipeline();

    //> main function for constructing stereo edge correspondences
    void get_stereo_edge_matching_GPU();

    float build_edge_spatial_map();
    float rasterize_epipolar_band_for_matches();
    float precompute_left_edge_patches_photometry();
    float apply_NCC_filter();

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
    //> Pointers to the host memory
    float* h_left_img = nullptr;
    float* h_right_img = nullptr;
    Edge_GPU* h_left_edges = nullptr;
    Edge_GPU* h_right_edges = nullptr;
    int* h_cell_ids = nullptr;
    int h_match_count_passing_EP_LP_OR_filters = 0;
    float* h_F = nullptr;
    Match_by_Edge_Index* h_matches_out = nullptr;

    //> Pointers to the device memory
    Edge_GPU* d_left_edges = nullptr;
    Edge_GPU* d_right_edges = nullptr;
    int* d_cell_ids = nullptr;
    int* d_edge_indices = nullptr;
    int* d_cell_start_idx = nullptr;
    int* d_num_of_edges_in_cell = nullptr;
    int* d_match_count_passing_EP_LP_OR_filters = nullptr;
    Match_by_Edge_Index* d_final_matches_passing_EP_LP_OR_filters = nullptr;
    float* d_F = nullptr;
    Precomputed_Edge_Patches_Photometry* d_left_patches;
    int* d_final_count_passing_NCC_filter = nullptr;
    Match_by_Edge_Index* d_final_matches_passing_NCC_filter = nullptr;

    int num_of_grid_cells_in_width;
    int num_of_grid_cells_in_height;
    int total_num_of_grid_cells;
    int max_stereo_edge_matches_after_epipolar_rasterization;
    size_t num_left_edges_ = 0;
    size_t num_right_edges_ = 0;

    Dataset::Ptr dataset_;
    Eigen::Matrix3f h_F_;

    Stereo_Matches_GPU_Test_Context make_test_context() const;
    void test_spatial_grid_mapping() const;
    void test_epipolar_rasterization() const;
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
} // namespace

#endif
