#ifndef TEMPORAL_EDGE_TRACKING_GPU_HPP
#define TEMPORAL_EDGE_TRACKING_GPU_HPP

#include <memory>
#include <vector>
#include <cuda_runtime.h>

#include "Stereo_Iterator.h"
#include "gpu_settings.h"
#include "definitions.h"
#include "gpu_kernels.h"
#include "Dataset.h"
#include "memory_stats.h"

// --------------------------------------------------------------------------------------------------
// Device-resident data of one frame for left-only temporal edge matching: the left image texture,
// the raw third-order edges, and edge-indexed photometry patches + SIFT descriptors.
// Built once per frame; a CF instance is reused as-is when it is promoted to KF, so edge indices
// stay valid across KF switches.
// --------------------------------------------------------------------------------------------------
class Mono_Frame_GPU {
public:
    typedef std::shared_ptr<Mono_Frame_GPU> Ptr;

    Mono_Frame_GPU(const StereoFrame& frame, size_t frame_idx, int device_id);
    ~Mono_Frame_GPU();
    Mono_Frame_GPU(const Mono_Frame_GPU&) = delete;
    Mono_Frame_GPU& operator=(const Mono_Frame_GPU&) = delete;

    size_t frame_idx  = 0;
    int    device_id  = 0;
    int    img_width  = 0;
    int    img_height = 0;
    int    num_edges  = 0;

    std::vector<Edge_GPU> h_edges;
    Edge_GPU*                             d_edges   = nullptr;
    Precomputed_Edge_Patches_Photometry*  d_patches = nullptr;
    Precomputed_Edge_SIFT_Descriptor_GPU* d_sift    = nullptr;
    CUDA_Texture_Wrapper                  texture   = {nullptr, 0};

    float time_precompute = 0.0f;   //> GPU time of patch + SIFT precomputation (ms)
};

//> Host-side final left-only temporal match (KF edge -> refined CF edge)
struct Temporal_Edge_Match {
    int   kf_edge_idx = -1;
    int   cf_edge_idx = -1;
    float cf_x = 0.f;
    float cf_y = 0.f;
    float cf_orientation = 0.f;
    float photometric_rms = 0.f;
};

struct Temporal_Edge_Tracking_GPU_Timing_Statistics
{
    float time_precompute_cf_frame = 0.0f;      //> CF patches + SIFT descriptors
    float time_build_cf_spatial_grid = 0.0f;
    float time_candidate_generation = 0.0f;     //> spatial grid search + orientation filter
    float time_apply_ncc_filter = 0.0f;
    float time_apply_sift_filter = 0.0f;
    float time_apply_bnb_ncc_filter = 0.0f;
    float time_apply_bnb_sift_filter = 0.0f;
    float time_apply_photometric_refine = 0.0f; //> 2D Gauss-Newton refinement + clustering
    float time_apply_lowe_ratio_filter = 0.0f;
    float total_time = 0.0f;
    Memory_Usage_Statistics memory;
};

// --------------------------------------------------------------------------------------------------
// GPU left-only temporal edge matching pipeline (KF left edges -> CF left edges; no stereo, no quads).
//
//   1. Build CF spatial grid from raw CF edges
//   2. Candidate generation: grid search within TEMPORAL_SEARCH_RADIUS_PIXELS + orientation filter
//   3. NCC (left-left)
//   4. SIFT (left-left)
//   5. BNB-NCC, then BNB-SIFT (best per KF edge)
//   6. 2D photometric refinement + clustering
//   7. Lowe's ratio on photometric_rms for KF edges with >1 surviving match
//
// Match_by_Edge_Index: { left_edge_idx = kf_edge_idx, right_edge_idx = cf_edge_idx }
// --------------------------------------------------------------------------------------------------
class Temporal_Edge_Tracking_GPU_Pipeline {
public:
    typedef std::shared_ptr<Temporal_Edge_Tracking_GPU_Pipeline> Ptr;

    Temporal_Edge_Tracking_GPU_Pipeline(Mono_Frame_GPU::Ptr kf_frame, Mono_Frame_GPU::Ptr cf_frame, int device);
    ~Temporal_Edge_Tracking_GPU_Pipeline();

    Temporal_Edge_Tracking_GPU_Timing_Statistics get_temporal_edge_matching_GPU();

    //> Final matches sorted by ascending photometric_rms
    void retrieve_final_matches(std::vector<Temporal_Edge_Match>& out_matches) const;

    int device_id;
    cudaEvent_t start, stop;

private:
    Mono_Frame_GPU::Ptr kf_;
    Mono_Frame_GPU::Ptr cf_;

    int num_of_grid_cells_in_width  = 0;
    int num_of_grid_cells_in_height = 0;
    int total_num_of_grid_cells     = 0;
    int max_temporal_candidates     = 0;  //> = kf_->num_edges * FACTOR_OF_NUM_OF_KF_MATES_FOR_TEMPORAL_CANDIDATES

    int h_candidate_count           = 0;
    int h_match_count_after_ncc     = 0;
    int h_match_count_after_sift    = 0;
    int h_match_count_after_bnb_ncc = 0;
    int h_match_count_after_bnb_sift = 0;
    int h_match_count_after_refine  = 0;
    int h_match_count_after_cluster = 0;

    // ----- device: CF spatial grid -----
    int* h_cf_cell_ids         = nullptr;
    int* d_cf_cell_ids         = nullptr;
    int* d_cf_edge_indices     = nullptr;   //> sorted by cell
    int* d_cf_cell_start_idx   = nullptr;
    int* d_cf_num_in_cell      = nullptr;

    // ----- device: stage outputs -----
    Match_by_Edge_Index* d_candidates        = nullptr;
    int*                 d_candidate_count   = nullptr;

    Match_by_Edge_Index* d_matches_after_ncc = nullptr;
    int*                 d_count_after_ncc   = nullptr;
    float*               d_ncc_scores        = nullptr;

    Match_by_Edge_Index* d_matches_after_sift     = nullptr;
    int*                 d_count_after_sift       = nullptr;
    float*               d_sift_scores            = nullptr;
    float*               d_ncc_scores_after_sift  = nullptr;

    Match_by_Edge_Index* d_matches_after_bnb_ncc       = nullptr;
    int*                 d_count_after_bnb_ncc         = nullptr;
    float*               d_sift_scores_after_bnb_ncc   = nullptr;

    Match_by_Edge_Index* d_matches_after_bnb_sift = nullptr;
    int*                 d_count_after_bnb_sift   = nullptr;

    Temporal_Refined_Match_GPU*      d_refined_matches   = nullptr;
    int*                             d_refined_count     = nullptr;
    Temporal_Refined_Quad_Match_GPU* d_clustered_matches = nullptr;   //> cf_right_* unused (0)
    int*                             d_clustered_count   = nullptr;
};

#endif // TEMPORAL_EDGE_TRACKING_GPU_HPP
