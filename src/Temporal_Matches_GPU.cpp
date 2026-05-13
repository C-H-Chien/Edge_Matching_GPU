#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "definitions.h"
#include "Temporal_Matches_GPU.h"
#include "gpu_kernels.h"
#include "Dataset.h"

// ---------------------------------------------------------------------------
// Helpers: pick the undistorted image when available, fall back to raw.
// Named with _tgpu suffix to avoid ODR conflicts with the anonymous-namespace
// versions in Stereo_Matches_GPU.h when both headers are included.
// ---------------------------------------------------------------------------
static const cv::Mat& pick_img_tgpu(const cv::Mat& undist, const cv::Mat& raw)
{
    return (!undist.empty()) ? undist : raw;
}

static int rows_tgpu(const cv::Mat& undist, const cv::Mat& raw)
{
    const cv::Mat& m = pick_img_tgpu(undist, raw);
    return m.empty() ? 0 : m.rows;
}

static int cols_tgpu(const cv::Mat& undist, const cv::Mat& raw)
{
    const cv::Mat& m = pick_img_tgpu(undist, raw);
    return m.empty() ? 0 : m.cols;
}

//> Helper: convert a cv::Mat to a contiguous float buffer (allocates with new[])
static float* mat_to_float_buf(const cv::Mat& undist, const cv::Mat& raw)
{
    const cv::Mat& src = pick_img_tgpu(undist, raw);
    cv::Mat f;
    src.convertTo(f, CV_32F);
    if (!f.isContinuous()) f = f.clone();
    const int n = f.rows * f.cols;
    float* buf = new float[static_cast<size_t>(n)];
    std::memcpy(buf, f.ptr<float>(), static_cast<size_t>(n) * sizeof(float));
    return buf;
}

Temporal_Matches_GPU_Pipeline::Temporal_Matches_GPU_Pipeline(
    Dataset::Ptr                                   dataset,
    const StereoFrame&                             keyframe,
    const StereoFrame&                             current_frame,
    const std::vector<final_stereo_edge_pair>&     kf_stereo_mates,
    const std::vector<final_stereo_edge_pair>&     cf_stereo_mates,
    int                                            device)
    : device_id(device)
    , img_width (cols_tgpu(current_frame.left_image_undistorted, current_frame.left_image))
    , img_height(rows_tgpu(current_frame.left_image_undistorted, current_frame.left_image))
    , dataset_(std::move(dataset))
{
    num_kf_mates_ = kf_stereo_mates.size();
    num_cf_mates_ = cf_stereo_mates.size();

    if (num_kf_mates_ == 0)
        LOG_WARNING("Temporal GPU: KF stereo mate list is empty");
    if (num_cf_mates_ == 0)
        LOG_WARNING("Temporal GPU: CF stereo mate list is empty");

    //> Flatten images to float buffers
    h_kf_left_img  = mat_to_float_buf(keyframe.left_image_undistorted,       keyframe.left_image);
    h_kf_right_img = mat_to_float_buf(keyframe.right_image_undistorted,      keyframe.right_image);
    h_cf_left_img  = mat_to_float_buf(current_frame.left_image_undistorted,  current_frame.left_image);
    h_cf_right_img = mat_to_float_buf(current_frame.right_image_undistorted, current_frame.right_image);

    //> Build host edge arrays from stereo mate positions
    h_kf_left_edges  = new Edge_GPU[num_kf_mates_ > 0 ? num_kf_mates_ : 1];
    h_kf_right_edges = new Edge_GPU[num_kf_mates_ > 0 ? num_kf_mates_ : 1];
    h_cf_left_edges  = new Edge_GPU[num_cf_mates_ > 0 ? num_cf_mates_ : 1];
    h_cf_right_edges = new Edge_GPU[num_cf_mates_ > 0 ? num_cf_mates_ : 1];
    h_cf_left_cell_ids = new int[num_cf_mates_ > 0 ? num_cf_mates_ : 1];

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(num_kf_mates_); ++i) {
        h_kf_left_edges[i]  = { static_cast<float>(kf_stereo_mates[i].left_edge.location.x),
                                 static_cast<float>(kf_stereo_mates[i].left_edge.location.y),
                                 static_cast<float>(kf_stereo_mates[i].left_edge.orientation) };
        h_kf_right_edges[i] = { static_cast<float>(kf_stereo_mates[i].right_edge.location.x),
                                 static_cast<float>(kf_stereo_mates[i].right_edge.location.y),
                                 static_cast<float>(kf_stereo_mates[i].right_edge.orientation) };
    }
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(num_cf_mates_); ++i) {
        h_cf_left_edges[i]  = { static_cast<float>(cf_stereo_mates[i].left_edge.location.x),
                                 static_cast<float>(cf_stereo_mates[i].left_edge.location.y),
                                 static_cast<float>(cf_stereo_mates[i].left_edge.orientation) };
        h_cf_right_edges[i] = { static_cast<float>(cf_stereo_mates[i].right_edge.location.x),
                                 static_cast<float>(cf_stereo_mates[i].right_edge.location.y),
                                 static_cast<float>(cf_stereo_mates[i].right_edge.orientation) };
    }

    // ----- Grid dimensions ------------------------------------------------
    num_of_grid_cells_in_height = static_cast<int>(std::ceil(static_cast<double>(img_height) / GRID_CELL_SIZE));
    num_of_grid_cells_in_width  = static_cast<int>(std::ceil(static_cast<double>(img_width)  / GRID_CELL_SIZE));
    total_num_of_grid_cells     = num_of_grid_cells_in_width * num_of_grid_cells_in_height;

    max_temporal_candidates =
        static_cast<int>(num_kf_mates_) * FACTOR_OF_NUM_OF_KF_MATES_FOR_TEMPORAL_CANDIDATES;
    if (max_temporal_candidates < 1) max_temporal_candidates = 1;

    // ----- CUDA device sanity checks --------------------------------------
    (void)cudaGetLastError();                     // clear any sticky error
    int n_devices = 0;
    cudaError_t ec = cudaGetDeviceCount(&n_devices);
    if (ec != cudaSuccess)
        throw std::runtime_error(std::string("Temporal GPU: cudaGetDeviceCount failed: ") + cudaGetErrorString(ec));
    if (n_devices == 0)
        throw std::runtime_error("Temporal GPU: no CUDA devices found");
    if (device_id < 0 || device_id >= n_devices)
        throw std::runtime_error("Temporal GPU: invalid device_id " + std::to_string(device_id));
    cudacheck(cudaSetDevice(device_id));

    // ----- Allocate device edge arrays ------------------------------------
    cudacheck(cudaMalloc((void**)&d_kf_left_edges,  num_kf_mates_ * sizeof(Edge_GPU)));
    cudacheck(cudaMalloc((void**)&d_kf_right_edges, num_kf_mates_ * sizeof(Edge_GPU)));
    cudacheck(cudaMalloc((void**)&d_cf_left_edges,  num_cf_mates_  * sizeof(Edge_GPU)));
    cudacheck(cudaMalloc((void**)&d_cf_right_edges, num_cf_mates_  * sizeof(Edge_GPU)));

    // ----- Allocate CF-left spatial grid buffers --------------------------
    cudacheck(cudaMalloc((void**)&d_cf_left_cell_ids,       num_cf_mates_ * sizeof(int)));
    cudacheck(cudaMalloc((void**)&d_cf_left_edge_indices,   num_cf_mates_ * sizeof(int)));
    cudacheck(cudaMalloc((void**)&d_cf_left_cell_start_idx, total_num_of_grid_cells * sizeof(int)));
    cudacheck(cudaMalloc((void**)&d_cf_left_num_in_cell,    total_num_of_grid_cells * sizeof(int)));

    // ----- Allocate temporal candidate buffer (pre-sized upper bound) -----
    cudacheck(cudaMalloc((void**)&d_temporal_candidates,
                         static_cast<size_t>(max_temporal_candidates) * sizeof(Match_by_Edge_Index)));
    cudacheck(cudaMalloc((void**)&d_temporal_candidate_count, sizeof(int)));

    // ----- Upload KF edge arrays to device --------------------------------
    if (num_kf_mates_ > 0) {
        cudacheck(cudaMemcpy(d_kf_left_edges,  h_kf_left_edges,  num_kf_mates_ * sizeof(Edge_GPU), cudaMemcpyHostToDevice));
        cudacheck(cudaMemcpy(d_kf_right_edges, h_kf_right_edges, num_kf_mates_ * sizeof(Edge_GPU), cudaMemcpyHostToDevice));
    }

    // ----- CUDA timing events ---------------------------------------------
    cudacheck(cudaEventCreate(&start));
    cudacheck(cudaEventCreate(&stop));

    // ----- Image textures (uploaded once, reused across all pipeline stages)
    allocate_image_textures();
}

// ---------------------------------------------------------------------------
// allocate_image_textures / release_image_textures
// ---------------------------------------------------------------------------
void Temporal_Matches_GPU_Pipeline::allocate_image_textures()
{
    if (b_image_textures_ready_) return;

    cudacheck(cudaSetDevice(device_id));
    kf_left_image_texture_  = create_CUDA_texture_object_for_img(h_kf_left_img,  img_width, img_height);
    kf_right_image_texture_ = create_CUDA_texture_object_for_img(h_kf_right_img, img_width, img_height);
    cf_left_image_texture_  = create_CUDA_texture_object_for_img(h_cf_left_img,  img_width, img_height);
    cf_right_image_texture_ = create_CUDA_texture_object_for_img(h_cf_right_img, img_width, img_height);
    b_image_textures_ready_ = true;
}

static void destroy_texture(CUDA_Texture_Wrapper& w)
{
    if (w.texObj != 0) { cudacheck(cudaDestroyTextureObject(w.texObj)); w.texObj = 0; }
    if (w.array  != nullptr) { cudacheck(cudaFreeArray(w.array));        w.array  = nullptr; }
}

void Temporal_Matches_GPU_Pipeline::release_image_textures()
{
    destroy_texture(kf_left_image_texture_);
    destroy_texture(kf_right_image_texture_);
    destroy_texture(cf_left_image_texture_);
    destroy_texture(cf_right_image_texture_);
    b_image_textures_ready_ = false;
}

// ---------------------------------------------------------------------------
// free_lazy_allocations — called in destructor and before a re-run
// ---------------------------------------------------------------------------
void Temporal_Matches_GPU_Pipeline::free_lazy_allocations()
{
    auto safe_free = [](auto*& p) { if (p) { cudacheck(cudaFree(p)); p = nullptr; } };

    safe_free(d_kf_left_patches);
    safe_free(d_kf_right_patches);

    safe_free(d_matches_after_ncc_left);
    safe_free(d_count_after_ncc_left);
    safe_free(d_ncc_left_scores);
    safe_free(d_matches_after_ncc_right);
    safe_free(d_count_after_ncc_right);
    safe_free(d_ncc_right_scores);

    safe_free(d_matches_after_sift_left);
    safe_free(d_count_after_sift_left);
    safe_free(d_sift_left_scores);
    safe_free(d_matches_after_sift_right);
    safe_free(d_count_after_sift_right);
    safe_free(d_sift_right_scores);

    safe_free(d_matches_after_bnb_ncc);
    safe_free(d_scores_after_bnb_ncc);
    safe_free(d_count_after_bnb_ncc);
    safe_free(d_matches_after_bnb_sift);
    safe_free(d_scores_after_bnb_sift);
    safe_free(d_count_after_bnb_sift);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
Temporal_Matches_GPU_Pipeline::~Temporal_Matches_GPU_Pipeline()
{
    release_image_textures();
    free_lazy_allocations();

    delete[] h_kf_left_img;
    delete[] h_kf_right_img;
    delete[] h_cf_left_img;
    delete[] h_cf_right_img;
    delete[] h_kf_left_edges;
    delete[] h_kf_right_edges;
    delete[] h_cf_left_edges;
    delete[] h_cf_right_edges;
    delete[] h_cf_left_cell_ids;

    if (d_kf_left_edges)  cudacheck(cudaFree(d_kf_left_edges));
    if (d_kf_right_edges) cudacheck(cudaFree(d_kf_right_edges));
    if (d_cf_left_edges)  cudacheck(cudaFree(d_cf_left_edges));
    if (d_cf_right_edges) cudacheck(cudaFree(d_cf_right_edges));

    if (d_cf_left_cell_ids)       cudacheck(cudaFree(d_cf_left_cell_ids));
    if (d_cf_left_edge_indices)   cudacheck(cudaFree(d_cf_left_edge_indices));
    if (d_cf_left_cell_start_idx) cudacheck(cudaFree(d_cf_left_cell_start_idx));
    if (d_cf_left_num_in_cell)    cudacheck(cudaFree(d_cf_left_num_in_cell));

    if (d_temporal_candidates)      cudacheck(cudaFree(d_temporal_candidates));
    if (d_temporal_candidate_count) cudacheck(cudaFree(d_temporal_candidate_count));

    cudacheck(cudaEventDestroy(start));
    cudacheck(cudaEventDestroy(stop));
}

// ===========================================================================
// Stage functions
// ===========================================================================

// ---------------------------------------------------------------------------
// Stage 1: Build the CF-left edge spatial grid.
//          Reuses the existing edge_spatial_map_thrust_pipeline entirely.
//          After this call d_cf_left_edge_indices / d_cf_left_cell_start_idx /
//          d_cf_left_num_in_cell are ready on device, and d_cf_left_edges
//          contains the uploaded CF-left edges.
// ---------------------------------------------------------------------------
float Temporal_Matches_GPU_Pipeline::build_cf_left_spatial_grid()
{
    return edge_spatial_map_thrust_pipeline(
        device_id,
        h_cf_left_edges,
        h_cf_left_cell_ids,
        d_cf_left_edges,
        d_cf_left_cell_ids,
        d_cf_left_edge_indices,
        d_cf_left_cell_start_idx,
        d_cf_left_num_in_cell,
        num_cf_mates_,
        total_num_of_grid_cells,
        num_of_grid_cells_in_width,
        num_of_grid_cells_in_height,
        start,
        stop);
}

// ---------------------------------------------------------------------------
// Stage 2a/2b: Precompute photometry patches for KF left / KF right edges.
//              Reuses precompute_left_edge_patches_photometry_pipeline directly.
//              d_kf_left_patches / d_kf_right_patches are allocated inside.
// ---------------------------------------------------------------------------
float Temporal_Matches_GPU_Pipeline::precompute_kf_left_patches()
{
    return precompute_left_edge_patches_photometry_pipeline(
        device_id,
        kf_left_image_texture_.texObj,
        d_kf_left_patches,
        d_kf_left_edges,
        num_kf_mates_,
        start, stop);
}

float Temporal_Matches_GPU_Pipeline::precompute_kf_right_patches()
{
    return precompute_left_edge_patches_photometry_pipeline(
        device_id,
        kf_right_image_texture_.texObj,
        d_kf_right_patches,
        d_kf_right_edges,
        num_kf_mates_,
        start, stop);
}

// ---------------------------------------------------------------------------
// Stage 3: Temporal candidate generation (spatial grid intersection + OR filter).
//          Also uploads d_cf_right_edges to device (needed by the kernel).
//          Outputs d_temporal_candidates with d_temporal_candidate_count.
// ---------------------------------------------------------------------------
float Temporal_Matches_GPU_Pipeline::generate_temporal_candidates()
{
    //> Upload CF-right edges (not uploaded by the spatial grid stage above)
    if (num_cf_mates_ > 0) {
        cudacheck(cudaMemcpy(d_cf_right_edges, h_cf_right_edges,
                             num_cf_mates_ * sizeof(Edge_GPU), cudaMemcpyHostToDevice));
    }

    return temporal_candidate_generation_pipeline(
        device_id,
        d_kf_left_edges,
        d_kf_right_edges,
        static_cast<int>(num_kf_mates_),
        d_cf_left_edges,
        d_cf_right_edges,
        d_cf_left_edge_indices,
        d_cf_left_cell_start_idx,
        d_cf_left_num_in_cell,
        num_of_grid_cells_in_width,
        num_of_grid_cells_in_height,
        TEMPORAL_SEARCH_RADIUS_PIXELS,
        TEMPORAL_ORIENTATION_THRESH_DEG,
        d_temporal_candidates,
        d_temporal_candidate_count,
        h_temporal_candidate_count,
        max_temporal_candidates,
        start, stop);
}

// ---------------------------------------------------------------------------
// Stage 4a: NCC left-left.
//   Reuses apply_NCC_filter_pipeline without modification.
//   Semantic mapping:  left_edge_idx = kf_mate_idx (→ d_kf_left_patches)
//                      right_edge_idx = cf_mate_idx (→ d_cf_left_edges position,
//                                                      sampled from cf_left_tex)
//   NOTE: apply_NCC_filter_kernel does NOT read d_left_edges in its body;
//         we pass d_kf_left_edges as a harmless placeholder.
// ---------------------------------------------------------------------------
float Temporal_Matches_GPU_Pipeline::apply_ncc_left_filter()
{
    if (h_temporal_candidate_count <= 0) {
        LOG_WARNING("Temporal GPU NCC-left: no candidates, skipping");
        h_match_count_after_ncc_left = 0;
        return 0.0f;
    }

    const int n = h_temporal_candidate_count;
    const size_t bytes = static_cast<size_t>(n) * sizeof(Match_by_Edge_Index);
    cudacheck(cudaMalloc(&d_matches_after_ncc_left, bytes > 0 ? bytes : sizeof(Match_by_Edge_Index)));
    cudacheck(cudaMalloc(&d_count_after_ncc_left, sizeof(int)));

    return apply_NCC_filter_pipeline(
        device_id,
        h_match_count_after_ncc_left,
        cf_left_image_texture_.texObj,     //> right_tex → sample CF-left image
        d_temporal_candidates,
        d_kf_left_edges,                   //> d_left_edges  (unused by kernel body)
        d_cf_left_edges,                   //> d_right_edges → CF-left edge positions
        d_matches_after_ncc_left,
        d_count_after_ncc_left,
        d_ncc_left_scores,
        d_kf_left_patches,                 //> precomputed KF-left patches
        n,
        start, stop,
        static_cast<float>(NCC_THRESH_TEMPORAL));
}

// ---------------------------------------------------------------------------
// Stage 4b: NCC right-right (applied to ncc_left survivors).
//   Semantic mapping:  left_edge_idx = kf_mate_idx (→ d_kf_right_patches)
//                      right_edge_idx = cf_mate_idx (→ d_cf_right_edges position,
//                                                      sampled from cf_right_tex)
// ---------------------------------------------------------------------------
float Temporal_Matches_GPU_Pipeline::apply_ncc_right_filter()
{
    if (h_match_count_after_ncc_left <= 0) {
        LOG_WARNING("Temporal GPU NCC-right: no NCC-left survivors, skipping");
        h_match_count_after_ncc_right = 0;
        return 0.0f;
    }

    const int n = h_match_count_after_ncc_left;
    const size_t bytes = static_cast<size_t>(n) * sizeof(Match_by_Edge_Index);
    cudacheck(cudaMalloc(&d_matches_after_ncc_right, bytes > 0 ? bytes : sizeof(Match_by_Edge_Index)));
    cudacheck(cudaMalloc(&d_count_after_ncc_right, sizeof(int)));

    return apply_NCC_filter_pipeline(
        device_id,
        h_match_count_after_ncc_right,
        cf_right_image_texture_.texObj,    //> right_tex → sample CF-right image
        d_matches_after_ncc_left,
        d_kf_right_edges,                  //> d_left_edges  (unused by kernel body)
        d_cf_right_edges,                  //> d_right_edges → CF-right edge positions
        d_matches_after_ncc_right,
        d_count_after_ncc_right,
        d_ncc_right_scores,
        d_kf_right_patches,                //> precomputed KF-right patches
        n,
        start, stop,
        static_cast<float>(NCC_THRESH_TEMPORAL));
}

// ---------------------------------------------------------------------------
// Stage 5a: SIFT left-left (applied to ncc_right survivors).
//   Reuses apply_SIFT_descriptor_filter_pipeline without modification.
//   left_tex = KF-left image, right_tex = CF-left image
//   d_left_edges = KF-left edges, d_right_edges = CF-left edges
// ---------------------------------------------------------------------------
float Temporal_Matches_GPU_Pipeline::apply_sift_left_filter()
{
    if (h_match_count_after_ncc_right <= 0) {
        LOG_WARNING("Temporal GPU SIFT-left: no NCC-right survivors, skipping");
        h_match_count_after_sift_left = 0;
        return 0.0f;
    }

    return apply_SIFT_descriptor_filter_pipeline(
        device_id,
        h_match_count_after_sift_left,
        kf_left_image_texture_.texObj,     //> left_tex  → KF-left image
        cf_left_image_texture_.texObj,     //> right_tex → CF-left image
        d_matches_after_ncc_right,
        d_kf_left_edges,                   //> left edge positions (kf_mate_idx → KF-left)
        d_cf_left_edges,                   //> right edge positions (cf_mate_idx → CF-left)
        d_matches_after_sift_left,
        d_count_after_sift_left,
        d_sift_left_scores,
        img_width,  img_height,            //> KF-left image dimensions
        img_width,  img_height,            //> CF-left image dimensions (same resolution)
        h_match_count_after_ncc_right,
        start, stop,
        static_cast<float>(SIFT_THRESH_TEMPORAL));
}

// ---------------------------------------------------------------------------
// Stage 5b: SIFT right-right (applied to sift_left survivors).
//   left_tex = KF-right image, right_tex = CF-right image
//   d_left_edges = KF-right edges, d_right_edges = CF-right edges
// ---------------------------------------------------------------------------
float Temporal_Matches_GPU_Pipeline::apply_sift_right_filter()
{
    if (h_match_count_after_sift_left <= 0) {
        LOG_WARNING("Temporal GPU SIFT-right: no SIFT-left survivors, skipping");
        h_match_count_after_sift_right = 0;
        return 0.0f;
    }

    return apply_SIFT_descriptor_filter_pipeline(
        device_id,
        h_match_count_after_sift_right,
        kf_right_image_texture_.texObj,    //> left_tex  → KF-right image
        cf_right_image_texture_.texObj,    //> right_tex → CF-right image
        d_matches_after_sift_left,
        d_kf_right_edges,                  //> KF-right edge positions
        d_cf_right_edges,                  //> CF-right edge positions
        d_matches_after_sift_right,
        d_count_after_sift_right,
        d_sift_right_scores,
        img_width,  img_height,
        img_width,  img_height,
        h_match_count_after_sift_left,
        start, stop,
        static_cast<float>(SIFT_THRESH_TEMPORAL));
}

// ---------------------------------------------------------------------------
// Stage 6a: BNB-NCC (applied to ncc_right survivors; groups by kf_mate_idx).
//           NCC score is "higher is better".
// ---------------------------------------------------------------------------
float Temporal_Matches_GPU_Pipeline::apply_bnb_ncc_filter()
{
    if (h_match_count_after_ncc_right <= 0 || d_ncc_right_scores == nullptr) {
        LOG_WARNING("Temporal GPU BNB-NCC: no candidates or scores, skipping");
        h_match_count_after_bnb_ncc = 0;
        return 0.0f;
    }

    return apply_best_nearly_best_filter_pipeline(
        device_id,
        h_match_count_after_bnb_ncc,
        d_matches_after_ncc_right,
        d_ncc_right_scores,
        d_matches_after_bnb_ncc,
        d_scores_after_bnb_ncc,
        d_count_after_bnb_ncc,
        h_match_count_after_ncc_right,
        static_cast<int>(num_kf_mates_),   //> num_left_edges == num KF mates (BNB groups by kf_mate_idx)
        static_cast<float>(BNB_NCC_TEMPORAL),
        true,                              //> higher NCC is better
        start, stop);
}

// ---------------------------------------------------------------------------
// Stage 6b: BNB-SIFT (applied to sift_right survivors; groups by kf_mate_idx).
//           SIFT L2 distance is "lower is better".
// ---------------------------------------------------------------------------
float Temporal_Matches_GPU_Pipeline::apply_bnb_sift_filter()
{
    if (h_match_count_after_sift_right <= 0 || d_sift_right_scores == nullptr) {
        LOG_WARNING("Temporal GPU BNB-SIFT: no candidates or scores, skipping");
        h_match_count_after_bnb_sift = 0;
        return 0.0f;
    }

    return apply_best_nearly_best_filter_pipeline(
        device_id,
        h_match_count_after_bnb_sift,
        d_matches_after_sift_right,
        d_sift_right_scores,
        d_matches_after_bnb_sift,
        d_scores_after_bnb_sift,
        d_count_after_bnb_sift,
        h_match_count_after_sift_right,
        static_cast<int>(num_kf_mates_),
        static_cast<float>(BNB_SIFT_TEMPORAL),
        false,                             //> lower SIFT distance is better
        start, stop);
}

// ===========================================================================
// Main pipeline orchestrator
// ===========================================================================
Temporal_Matches_GPU_Timing_Statistics Temporal_Matches_GPU_Pipeline::get_temporal_edge_matching_GPU( size_t kf_frame_idx, size_t cf_frame_idx )
{
    //> Release any allocations left over from a previous run before re-running.
    free_lazy_allocations();

    Temporal_Matches_GPU_Timing_Statistics GPU_time_collection;
    GPU_time_collection.total_time = 0.0f;

    //> CF-left spatial grid
    GPU_time_collection.time_build_cf_left_spatial_grid = build_cf_left_spatial_grid();
    GPU_time_collection.total_time += GPU_time_collection.time_build_cf_left_spatial_grid;

    //> Precompute KF-left patches
    GPU_time_collection.time_precompute_kf_left_patches = precompute_kf_left_patches();
    GPU_time_collection.total_time += GPU_time_collection.time_precompute_kf_left_patches;

    //> Precompute KF-right patches
    GPU_time_collection.time_precompute_kf_right_patches = precompute_kf_right_patches();
    GPU_time_collection.total_time += GPU_time_collection.time_precompute_kf_right_patches;

    //> Temporal candidate generation (spatial intersection + OR filter)
    GPU_time_collection.time_temporal_candidate_generation = generate_temporal_candidates();
    GPU_time_collection.total_time += GPU_time_collection.time_temporal_candidate_generation;
    std::cout << "[Temporal GPU] Candidates after spatial+OR filter: " << h_temporal_candidate_count << std::endl;

    //> NCC left-left
    GPU_time_collection.time_apply_ncc_left_filter = apply_ncc_left_filter();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_ncc_left_filter;
    std::cout << "[Temporal GPU] Candidates after NCC left-left:  " << h_match_count_after_ncc_left  << std::endl;

    //> NCC right-right (on ncc_left survivors)
    GPU_time_collection.time_apply_ncc_right_filter = apply_ncc_right_filter();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_ncc_right_filter;
    std::cout << "[Temporal GPU] Candidates after NCC right-right: " << h_match_count_after_ncc_right << std::endl;

    //> SIFT left-left (on ncc_right survivors)
    GPU_time_collection.time_apply_sift_left_filter = apply_sift_left_filter();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_sift_left_filter;
    std::cout << "[Temporal GPU] Candidates after SIFT left-left:  " << h_match_count_after_sift_left  << std::endl;

    //> SIFT right-right (on sift_left survivors)
    GPU_time_collection.time_apply_sift_right_filter = apply_sift_right_filter();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_sift_right_filter;
    std::cout << "[Temporal GPU] Candidates after SIFT right-right: " << h_match_count_after_sift_right << std::endl;

    //> BNB-NCC (on ncc_right survivors, grouped by kf_mate_idx)
    GPU_time_collection.time_apply_bnb_ncc_filter = apply_bnb_ncc_filter();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_bnb_ncc_filter;
    std::cout << "[Temporal GPU] Candidates after BNB-NCC: " << h_match_count_after_bnb_ncc << std::endl;

    //> BNB-SIFT (on sift_right survivors, grouped by kf_mate_idx)
    GPU_time_collection.time_apply_bnb_sift_filter = apply_bnb_sift_filter();
    GPU_time_collection.total_time += GPU_time_collection.time_apply_bnb_sift_filter;
    std::cout << "[Temporal GPU] Final matches after BNB-SIFT: " << h_match_count_after_bnb_sift << std::endl;

#if SHOW_TIMING_BREAKDOWN
    std::cout.precision(3);
    std::cout << "Temporal GPU timing breakdown" << std::endl;
    std::cout << "  build_cf_left_spatial_grid:      " << GPU_time_collection.time_build_cf_left_spatial_grid      << " ms" << std::endl;
    std::cout << "  precompute_kf_left_patches:      " << GPU_time_collection.time_precompute_kf_left_patches      << " ms" << std::endl;
    std::cout << "  precompute_kf_right_patches:     " << GPU_time_collection.time_precompute_kf_right_patches     << " ms" << std::endl;
    std::cout << "  temporal_candidate_generation:   " << GPU_time_collection.time_temporal_candidate_generation   << " ms" << std::endl;
    std::cout << "  NCC left-left filter:            " << GPU_time_collection.time_apply_ncc_left_filter           << " ms" << std::endl;
    std::cout << "  NCC right-right filter:          " << GPU_time_collection.time_apply_ncc_right_filter          << " ms" << std::endl;
    std::cout << "  SIFT left-left filter:           " << GPU_time_collection.time_apply_sift_left_filter          << " ms" << std::endl;
    std::cout << "  SIFT right-right filter:         " << GPU_time_collection.time_apply_sift_right_filter         << " ms" << std::endl;
    std::cout << "  BNB-NCC filter:                  " << GPU_time_collection.time_apply_bnb_ncc_filter            << " ms" << std::endl;
    std::cout << "  BNB-SIFT filter:                 " << GPU_time_collection.time_apply_bnb_sift_filter           << " ms" << std::endl;
    std::cout << "  TOTAL:                           " << GPU_time_collection.total_time                           << " ms" << std::endl;
#endif

    LOG_INFO("Temporal GPU total time: " + std::to_string(GPU_time_collection.total_time) + " (ms)");
    std::cout << "Final quad matches: " << h_match_count_after_bnb_sift << std::endl;

    return GPU_time_collection;
}

// ===========================================================================
// retrieve_final_matches
// ===========================================================================
void Temporal_Matches_GPU_Pipeline::retrieve_final_matches(
    std::vector<Match_by_Edge_Index>& out_matches) const
{
    out_matches.clear();
    if (d_matches_after_bnb_sift == nullptr || h_match_count_after_bnb_sift <= 0) return;

    out_matches.resize(static_cast<size_t>(h_match_count_after_bnb_sift));
    cudacheck(cudaMemcpy(out_matches.data(), d_matches_after_bnb_sift,
                         static_cast<size_t>(h_match_count_after_bnb_sift) * sizeof(Match_by_Edge_Index),
                         cudaMemcpyDeviceToHost));
}

// ===========================================================================
// write_finalized_matches_to_file
// ===========================================================================
void Temporal_Matches_GPU_Pipeline::write_finalized_matches_to_file(
    size_t kf_frame_idx, size_t cf_frame_idx) const
{
    if (d_matches_after_bnb_sift == nullptr || h_match_count_after_bnb_sift <= 0) {
        LOG_WARNING("write_finalized_matches_to_file: no final temporal matches to write");
        return;
    }

    std::vector<Match_by_Edge_Index> matches(static_cast<size_t>(h_match_count_after_bnb_sift));
    cudaError_t err = cudaMemcpy(matches.data(), d_matches_after_bnb_sift,
                                 static_cast<size_t>(h_match_count_after_bnb_sift) * sizeof(Match_by_Edge_Index),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_ERROR("write_finalized_matches_to_file: cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
        return;
    }

    const std::string out_dir = (dataset_ != nullptr) ? dataset_->get_output_path() : std::string("output_files");
    const std::string fname   = out_dir + "/gpu_temporal_quad_matches_kf"
                                + std::to_string(kf_frame_idx) + "_cf"
                                + std::to_string(cf_frame_idx) + ".txt";
    std::ofstream ofs(fname);
    if (!ofs.is_open()) {
        LOG_ERROR("write_finalized_matches_to_file: cannot open " + fname);
        return;
    }

    ofs << "kf_mate_idx cf_mate_idx "
           "kf_left_x kf_left_y kf_left_orientation "
           "kf_right_x kf_right_y kf_right_orientation "
           "cf_left_x cf_left_y cf_left_orientation "
           "cf_right_x cf_right_y cf_right_orientation\n";
    ofs.precision(8);

    for (int i = 0; i < h_match_count_after_bnb_sift; ++i) {
        const int kf_idx = matches[static_cast<size_t>(i)].left_edge_idx;
        const int cf_idx = matches[static_cast<size_t>(i)].right_edge_idx;
        if (kf_idx < 0 || cf_idx < 0) continue;
        const Edge_GPU& kl = h_kf_left_edges[kf_idx];
        const Edge_GPU& kr = h_kf_right_edges[kf_idx];
        const Edge_GPU& cl = h_cf_left_edges[cf_idx];
        const Edge_GPU& cr = h_cf_right_edges[cf_idx];
        ofs << kf_idx << " " << cf_idx << " "
            << kl.location_x << " " << kl.location_y << " " << kl.orientation << " "
            << kr.location_x << " " << kr.location_y << " " << kr.orientation << " "
            << cl.location_x << " " << cl.location_y << " " << cl.orientation << " "
            << cr.location_x << " " << cr.location_y << " " << cr.orientation << "\n";
    }
    ofs.close();
    std::cout << "[Temporal GPU] Wrote " << h_match_count_after_bnb_sift
              << " temporal quad matches to " << fname << std::endl;
}
