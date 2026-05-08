#ifndef STEREO_MATCHES_GPU_VALIDATOR_H
#define STEREO_MATCHES_GPU_VALIDATOR_H

#include <cstddef>
#include <Eigen/Core>
#include <memory>

#include "Dataset.h"
#include "definitions.h"
#include "gpu_kernels.h"

//> The structure takes the pointers of the Stereo_Matches_GPU_Pipeline class for testing the results returned by GPU
struct Stereo_Matches_GPU_Test_Context {
    const Edge_GPU* h_left_edges = nullptr;
    const Edge_GPU* h_right_edges = nullptr;
    const int* h_cell_ids = nullptr;
    const float* h_F = nullptr;

    const Match_by_Edge_Index* d_final_matches_passing_EP_LP_OR_filters = nullptr;
    const int* d_match_count_passing_EP_LP_OR_filters = nullptr;
    const Match_by_Edge_Index* d_final_matches_passing_NCC_filter = nullptr;
    const int* d_final_count_passing_NCC_filter = nullptr;
    const Precomputed_Edge_Patches_Photometry* d_left_patches = nullptr;
    const int* d_edge_indices = nullptr;
    const float* h_right_img = nullptr;

    std::size_t num_left_edges = 0;
    std::size_t num_right_edges = 0;

    int img_right_width = 0;
    int img_right_height = 0;
    int num_of_grid_cells_in_width = 0;
    int num_of_grid_cells_in_height = 0;
    int max_matches = 0;

    /** When set together, enables cross-check against Stereo_Matches::collect_candidate_pairs_after_EP_LP_OR. */
    Dataset::Ptr dataset;
    const StereoFrame* stereo_frame_for_cpu = nullptr;
};

class Stereo_Matches_GPU_Validator {
public:
    static void test_spatial_grid_mapping(const Stereo_Matches_GPU_Test_Context& ctx);
    static void test_epipolar_rasterization(const Stereo_Matches_GPU_Test_Context& ctx);
    /** Compare GPU EP+LP+OR matches to Stereo_Matches.cpp (same filters as get_Stereo_Edge_Pairs before SIFT/NCC). */
    static void test_EP_LP_OR_vs_stereo_matches_cpu(const Stereo_Matches_GPU_Test_Context& ctx);
    static void test_ncc_filter(const Stereo_Matches_GPU_Test_Context& ctx);

private:
    static bool epipolar_line_intersects_image(float a, float b, float c, int W, int H);
};

#endif
