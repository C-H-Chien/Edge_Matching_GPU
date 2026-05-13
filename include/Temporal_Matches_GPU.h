#ifndef TEMPORAL_MATCHES_GPU_HPP
#define TEMPORAL_MATCHES_GPU_HPP

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
#include "Dataset.h"

// ---------------------------------------------------------------------------
// Timing breakdown for the temporal GPU pipeline
// ---------------------------------------------------------------------------
struct Temporal_Matches_GPU_Timing_Statistics
{
    float time_build_cf_left_spatial_grid;      //> CF-left edge spatial grid (reuses stereo grid kernel)
    float time_precompute_kf_left_patches;      //> photometry patches for KF left edges
    float time_precompute_kf_right_patches;     //> photometry patches for KF right edges
    float time_temporal_candidate_generation;   //> spatial-grid intersection + orientation filter
    float time_apply_ncc_left_filter;           //> NCC: KF-left patch vs CF-left sample
    float time_apply_ncc_right_filter;          //> NCC: KF-right patch vs CF-right sample (on ncc_left survivors)
    float time_apply_sift_left_filter;          //> SIFT: KF-left descriptor vs CF-left descriptor
    float time_apply_sift_right_filter;         //> SIFT: KF-right descriptor vs CF-right descriptor (on sift_left survivors)
    float time_apply_bnb_ncc_filter;            //> Best-nearly-best on NCC scores
    float time_apply_bnb_sift_filter;           //> Best-nearly-best on SIFT scores
    float total_time;
};

// ---------------------------------------------------------------------------
// GPU temporal quad matching pipeline.
//
// Matching unit: a "quad" is a (KF stereo mate, CF stereo mate) pair, where
// each stereo mate is a final_stereo_edge_pair holding a left and a right edge.
//
// Pipeline (mirrors run_temporal_quad_pipeline_filters in Temporal_Matches.cpp):
//   1. Build CF-left spatial grid
//   2. Precompute KF-left and KF-right photometry patches
//   3. Temporal candidate generation (spatial-grid intersection + orientation filter)
//   4. NCC left-left  → NCC right-right  (sequential; both must pass)
//   5. SIFT left-left → SIFT right-right  (sequential; both must pass)
//   6. BNB-NCC → BNB-SIFT
//
// Match_by_Edge_Index is reused as { left_edge_idx = kf_mate_idx,
//                                    right_edge_idx = cf_mate_idx }.
// ---------------------------------------------------------------------------
class Temporal_Matches_GPU_Pipeline {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Temporal_Matches_GPU_Pipeline> Ptr;

    Temporal_Matches_GPU_Pipeline(
        Dataset::Ptr                                   dataset,
        const StereoFrame&                             keyframe,
        const StereoFrame&                             current_frame,
        const std::vector<final_stereo_edge_pair>&     kf_stereo_mates,
        const std::vector<final_stereo_edge_pair>&     cf_stereo_mates,
        int                                            device);
    ~Temporal_Matches_GPU_Pipeline();

    //> Run the full GPU temporal quad filter pipeline for one KF–CF pair.
    Temporal_Matches_GPU_Timing_Statistics get_temporal_edge_matching_GPU(
        size_t kf_frame_idx, size_t cf_frame_idx);

    //> Copy the final (kf_mate_idx, cf_mate_idx) pairs to a host vector.
    void retrieve_final_matches(std::vector<Match_by_Edge_Index>& out_matches) const;

    //> Write final temporal matches to a text file for offline inspection.
    void write_finalized_matches_to_file(size_t kf_frame_idx, size_t cf_frame_idx) const;

    int device_id;
    int img_width;   //> shared width for all four images (stereo cameras same resolution)
    int img_height;

    cudaEvent_t start, stop;

private:
    // ----- sizes ---------------------------------------------------------
    size_t num_kf_mates_          = 0;
    size_t num_cf_mates_          = 0;
    int    num_of_grid_cells_in_width  = 0;
    int    num_of_grid_cells_in_height = 0;
    int    total_num_of_grid_cells     = 0;
    int    max_temporal_candidates     = 0;  //> = num_kf_mates_ * FACTOR_OF_NUM_OF_KF_MATES_FOR_TEMPORAL_CANDIDATES

    // ----- host image buffers (float, row-major) -------------------------
    float* h_kf_left_img  = nullptr;
    float* h_kf_right_img = nullptr;
    float* h_cf_left_img  = nullptr;
    float* h_cf_right_img = nullptr;

    // ----- host edge arrays (from stereo mates) --------------------------
    Edge_GPU* h_kf_left_edges  = nullptr;
    Edge_GPU* h_kf_right_edges = nullptr;
    Edge_GPU* h_cf_left_edges  = nullptr;
    Edge_GPU* h_cf_right_edges = nullptr;

    int* h_cf_left_cell_ids = nullptr;   //> output of spatial grid mapping (host-side cell id per CF-left edge)

    // ----- match counts (host-side bookkeeping) --------------------------
    int h_temporal_candidate_count    = 0;
    int h_match_count_after_ncc_left  = 0;
    int h_match_count_after_ncc_right = 0;
    int h_match_count_after_sift_left  = 0;
    int h_match_count_after_sift_right = 0;
    int h_match_count_after_bnb_ncc   = 0;
    int h_match_count_after_bnb_sift  = 0;

    // ----- device: KF and CF edge arrays ---------------------------------
    Edge_GPU* d_kf_left_edges  = nullptr;
    Edge_GPU* d_kf_right_edges = nullptr;
    Edge_GPU* d_cf_left_edges  = nullptr;
    Edge_GPU* d_cf_right_edges = nullptr;

    // ----- device: CF-left spatial grid (grid built on CF-left edge positions) ---
    int* d_cf_left_cell_ids         = nullptr;
    int* d_cf_left_edge_indices     = nullptr;   //> sorted by cell
    int* d_cf_left_cell_start_idx   = nullptr;
    int* d_cf_left_num_in_cell      = nullptr;

    // ----- device: temporal candidate buffer (pre-allocated) -------------
    Match_by_Edge_Index* d_temporal_candidates      = nullptr;
    int*                 d_temporal_candidate_count = nullptr;

    // ----- device: precomputed photometry patches ------------------------
    Precomputed_Edge_Patches_Photometry* d_kf_left_patches  = nullptr;
    Precomputed_Edge_Patches_Photometry* d_kf_right_patches = nullptr;

    // ----- device: NCC filter outputs (lazily allocated per stage) -------
    Match_by_Edge_Index* d_matches_after_ncc_left  = nullptr;
    int*                 d_count_after_ncc_left     = nullptr;
    float*               d_ncc_left_scores          = nullptr;
    Match_by_Edge_Index* d_matches_after_ncc_right  = nullptr;
    int*                 d_count_after_ncc_right    = nullptr;
    float*               d_ncc_right_scores         = nullptr;

    // ----- device: SIFT filter outputs (lazily allocated per stage) ------
    Match_by_Edge_Index* d_matches_after_sift_left  = nullptr;
    int*                 d_count_after_sift_left    = nullptr;
    float*               d_sift_left_scores         = nullptr;
    Match_by_Edge_Index* d_matches_after_sift_right = nullptr;
    int*                 d_count_after_sift_right   = nullptr;
    float*               d_sift_right_scores        = nullptr;

    // ----- device: BNB filter outputs (lazily allocated per stage) -------
    Match_by_Edge_Index* d_matches_after_bnb_ncc  = nullptr;
    float*               d_scores_after_bnb_ncc   = nullptr;
    int*                 d_count_after_bnb_ncc     = nullptr;
    Match_by_Edge_Index* d_matches_after_bnb_sift  = nullptr;
    float*               d_scores_after_bnb_sift   = nullptr;
    int*                 d_count_after_bnb_sift    = nullptr;

    // ----- image textures (four images: KF-left, KF-right, CF-left, CF-right) ---
    CUDA_Texture_Wrapper kf_left_image_texture_  = {nullptr, 0};
    CUDA_Texture_Wrapper kf_right_image_texture_ = {nullptr, 0};
    CUDA_Texture_Wrapper cf_left_image_texture_  = {nullptr, 0};
    CUDA_Texture_Wrapper cf_right_image_texture_ = {nullptr, 0};
    bool b_image_textures_ready_ = false;

    Dataset::Ptr dataset_;

    // ----- stage functions -----------------------------------------------
    float build_cf_left_spatial_grid();
    float precompute_kf_left_patches();
    float precompute_kf_right_patches();
    float generate_temporal_candidates();
    float apply_ncc_left_filter();
    float apply_ncc_right_filter();
    float apply_sift_left_filter();
    float apply_sift_right_filter();
    float apply_bnb_ncc_filter();
    float apply_bnb_sift_filter();

    void allocate_image_textures();
    void release_image_textures();
    void free_lazy_allocations();
};

#endif // TEMPORAL_MATCHES_GPU_HPP
