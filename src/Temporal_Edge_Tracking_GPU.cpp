#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include "definitions.h"
#include "Temporal_Edge_Tracking_GPU.h"
#include "gpu_kernels.h"

// ===========================================================================
// Mono_Frame_GPU
// ===========================================================================
Mono_Frame_GPU::Mono_Frame_GPU(const StereoFrame& frame, size_t frame_idx_in, int device)
    : frame_idx(frame_idx_in)
    , device_id(device)
{
    cudacheck(cudaSetDevice(device_id));

    //> Same image source as the edge detector: undistorted when available, else raw
    const cv::Mat& src = frame.left_image_undistorted.empty() ? frame.left_image : frame.left_image_undistorted;
    cv::Mat img_float;
    src.convertTo(img_float, CV_32F);
    if (!img_float.isContinuous())
        img_float = img_float.clone();
    img_width  = img_float.cols;
    img_height = img_float.rows;
    texture = create_CUDA_texture_object_for_img(img_float.ptr<float>(), img_width, img_height);

    num_edges = static_cast<int>(frame.left_edges.size());
    h_edges.resize(static_cast<size_t>(num_edges));
    for (int i = 0; i < num_edges; ++i) {
        h_edges[i].location_x  = static_cast<float>(frame.left_edges[i].location.x);
        h_edges[i].location_y  = static_cast<float>(frame.left_edges[i].location.y);
        h_edges[i].orientation = static_cast<float>(frame.left_edges[i].orientation);
    }
    if (num_edges <= 0) {
        LOG_WARNING("Mono_Frame_GPU: frame " + std::to_string(frame_idx) + " has no left edges");
        return;
    }

    cudacheck(cudaMalloc(&d_edges, static_cast<size_t>(num_edges) * sizeof(Edge_GPU)));
    cudacheck(cudaMemcpy(d_edges, h_edges.data(), static_cast<size_t>(num_edges) * sizeof(Edge_GPU), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudacheck(cudaEventCreate(&start));
    cudacheck(cudaEventCreate(&stop));
    time_precompute  = precompute_left_edge_patches_photometry_pipeline(
        device_id, texture.texObj, d_patches, d_edges, static_cast<size_t>(num_edges), start, stop);
    time_precompute += precompute_left_edge_sift_descriptors_pipeline(
        device_id, texture.texObj, d_sift, d_edges, img_width, img_height, static_cast<size_t>(num_edges), start, stop);
    cudacheck(cudaEventDestroy(start));
    cudacheck(cudaEventDestroy(stop));
}

Mono_Frame_GPU::~Mono_Frame_GPU()
{
    safe_free(d_edges);
    safe_free(d_patches);
    safe_free(d_sift);
    if (texture.texObj != 0) {
        cudacheck_noexcept(cudaDestroyTextureObject(texture.texObj));
        texture.texObj = 0;
    }
    if (texture.array != nullptr) {
        cudacheck_noexcept(cudaFreeArray(texture.array));
        texture.array = nullptr;
    }
}

// ===========================================================================
// Temporal_Edge_Tracking_GPU_Pipeline
// ===========================================================================
Temporal_Edge_Tracking_GPU_Pipeline::Temporal_Edge_Tracking_GPU_Pipeline(
    Mono_Frame_GPU::Ptr kf_frame, Mono_Frame_GPU::Ptr cf_frame, int device)
    : device_id(device)
    , kf_(std::move(kf_frame))
    , cf_(std::move(cf_frame))
{
    cudacheck(cudaSetDevice(device_id));

    num_of_grid_cells_in_height = static_cast<int>(std::ceil(static_cast<double>(cf_->img_height) / GRID_CELL_SIZE));
    num_of_grid_cells_in_width  = static_cast<int>(std::ceil(static_cast<double>(cf_->img_width)  / GRID_CELL_SIZE));
    total_num_of_grid_cells     = num_of_grid_cells_in_width * num_of_grid_cells_in_height;
    max_temporal_candidates     = kf_->num_edges * FACTOR_OF_NUM_OF_KF_MATES_FOR_TEMPORAL_CANDIDATES;

    const size_t num_cf = static_cast<size_t>(std::max(cf_->num_edges, 1));
    cudacheck(cudaMalloc((void**)&d_cf_cell_ids,       num_cf * sizeof(int)));
    cudacheck(cudaMalloc((void**)&d_cf_edge_indices,   num_cf * sizeof(int)));
    cudacheck(cudaMalloc((void**)&d_cf_cell_start_idx, static_cast<size_t>(total_num_of_grid_cells) * sizeof(int)));
    cudacheck(cudaMalloc((void**)&d_cf_num_in_cell,    static_cast<size_t>(total_num_of_grid_cells) * sizeof(int)));

    cudacheck(cudaMalloc((void**)&d_candidates, static_cast<size_t>(std::max(max_temporal_candidates, 1)) * sizeof(Match_by_Edge_Index)));
    cudacheck(cudaMalloc((void**)&d_candidate_count, sizeof(int)));

    cudacheck(cudaEventCreate(&start));
    cudacheck(cudaEventCreate(&stop));
}

Temporal_Edge_Tracking_GPU_Timing_Statistics Temporal_Edge_Tracking_GPU_Pipeline::get_temporal_edge_matching_GPU()
{
    Temporal_Edge_Tracking_GPU_Timing_Statistics t;
#if MEASURE_MEMORY
    MemoryMonitor memory_monitor(device_id);
#define MONO_SAMPLE_MEMORY() memory_monitor.sample()
#else
#define MONO_SAMPLE_MEMORY() ((void)0)
#endif

    if (kf_->num_edges <= 0 || cf_->num_edges <= 0) {
        LOG_WARNING("Temporal Mono GPU: KF or CF has no edges, skipping");
        return t;
    }
    const int num_kf = kf_->num_edges;
    const int num_cf = cf_->num_edges;

    //> 1. CF spatial grid
    t.time_build_cf_spatial_grid = edge_spatial_map_thrust_pipeline(
        device_id, cf_->h_edges.data(), h_cf_cell_ids,
        cf_->d_edges, d_cf_cell_ids, d_cf_edge_indices, d_cf_cell_start_idx, d_cf_num_in_cell,
        static_cast<size_t>(num_cf), total_num_of_grid_cells,
        num_of_grid_cells_in_width, num_of_grid_cells_in_height, start, stop);
    MONO_SAMPLE_MEMORY();

    //> 2. Candidate generation
    t.time_candidate_generation = mono_temporal_candidate_generation_pipeline(
        device_id, kf_->d_edges, cf_->d_edges,
        d_cf_edge_indices, d_cf_cell_start_idx, d_cf_num_in_cell,
        num_of_grid_cells_in_width, num_of_grid_cells_in_height, num_kf,
        d_candidates, d_candidate_count, h_candidate_count, max_temporal_candidates, start, stop);
    MONO_SAMPLE_MEMORY();
    std::cout << "[Temporal Mono GPU] Candidates after spatial+OR filter: " << h_candidate_count << std::endl;

    //> 3. NCC
    t.time_apply_ncc_filter = mono_temporal_ncc_filter_pipeline(
        device_id, h_match_count_after_ncc, d_candidates,
        kf_->d_patches, cf_->d_patches,
        d_matches_after_ncc, d_count_after_ncc, d_ncc_scores,
        h_candidate_count, num_kf, num_cf, start, stop);
    MONO_SAMPLE_MEMORY();
    std::cout << "[Temporal Mono GPU] Candidates after NCC:  " << h_match_count_after_ncc << std::endl;

    //> 4. SIFT
    t.time_apply_sift_filter = mono_temporal_sift_filter_pipeline(
        device_id, h_match_count_after_sift, d_matches_after_ncc,
        kf_->d_sift, cf_->d_sift,
        d_matches_after_sift, d_count_after_sift, d_sift_scores,
        d_ncc_scores, d_ncc_scores_after_sift,
        h_match_count_after_ncc, num_kf, num_cf, start, stop);
    MONO_SAMPLE_MEMORY();
    std::cout << "[Temporal Mono GPU] Candidates after SIFT: " << h_match_count_after_sift << std::endl;

    //> 5. BNB-NCC (rank by NCC, carry SIFT) then BNB-SIFT (rank by SIFT)
    t.time_apply_bnb_ncc_filter = mono_temporal_bnb_filter_pipeline(
        device_id, h_match_count_after_bnb_ncc, d_matches_after_sift,
        d_ncc_scores_after_sift, d_sift_scores,
        d_matches_after_bnb_ncc, d_sift_scores_after_bnb_ncc, d_count_after_bnb_ncc,
        h_match_count_after_sift, num_kf, static_cast<float>(TEMPORAL_BNB_NCC_THRESHOLD), true, start, stop);
    MONO_SAMPLE_MEMORY();
    std::cout << "[Temporal Mono GPU] Candidates after BNB-NCC:  " << h_match_count_after_bnb_ncc << std::endl;

    float* sift_carry_out = nullptr;
    t.time_apply_bnb_sift_filter = mono_temporal_bnb_filter_pipeline(
        device_id, h_match_count_after_bnb_sift, d_matches_after_bnb_ncc,
        d_sift_scores_after_bnb_ncc, nullptr,
        d_matches_after_bnb_sift, sift_carry_out, d_count_after_bnb_sift,
        h_match_count_after_bnb_ncc, num_kf, static_cast<float>(TEMPORAL_BNB_SIFT_THRESHOLD), false, start, stop);
    MONO_SAMPLE_MEMORY();
    std::cout << "[Temporal Mono GPU] Candidates after BNB-SIFT: " << h_match_count_after_bnb_sift << std::endl;

    //> 6. Photometric refinement + clustering
    t.time_apply_photometric_refine = mono_temporal_photometric_refine_and_cluster_pipeline(
        device_id, d_matches_after_bnb_sift, h_match_count_after_bnb_sift,
        kf_->d_patches, kf_->d_edges, cf_->d_edges,
        cf_->texture.texObj, cf_->img_width, cf_->img_height,
        d_refined_matches, d_refined_count, h_match_count_after_refine,
        d_clustered_matches, d_clustered_count, h_match_count_after_cluster, start, stop);
    MONO_SAMPLE_MEMORY();
    std::cout << "[Temporal Mono GPU] Candidates after photometric refinement: " << h_match_count_after_refine << std::endl;
    std::cout << "[Temporal Mono GPU] Matches after clustering:               " << h_match_count_after_cluster << std::endl;

    //> 7. Lowe's ratio
    if (h_match_count_after_cluster > 0 && d_clustered_matches != nullptr) {
        t.time_apply_lowe_ratio_filter = apply_temporal_quad_lowe_ratio_filter_pipeline(
            device_id, d_clustered_matches, h_match_count_after_cluster, d_clustered_count,
            static_cast<float>(LOWES_RATIO), start, stop);
    }
    MONO_SAMPLE_MEMORY();
    std::cout << "[Temporal Mono GPU] Final matches after Lowe's ratio:      " << h_match_count_after_cluster << std::endl;
#undef MONO_SAMPLE_MEMORY

#if MEASURE_MEMORY
    t.memory = memory_monitor.peaks();
#endif

    t.total_time = t.time_build_cf_spatial_grid + t.time_candidate_generation
                 + t.time_apply_ncc_filter + t.time_apply_sift_filter
                 + t.time_apply_bnb_ncc_filter + t.time_apply_bnb_sift_filter
                 + t.time_apply_photometric_refine + t.time_apply_lowe_ratio_filter;

#if SHOW_TIMING_BREAKDOWN
    std::cout.precision(3);
    std::cout << "Temporal Mono GPU timing breakdown" << std::endl;
    std::cout << "  build_cf_spatial_grid:         " << t.time_build_cf_spatial_grid    << " ms" << std::endl;
    std::cout << "  candidate_generation:          " << t.time_candidate_generation     << " ms" << std::endl;
    std::cout << "  NCC filter:                    " << t.time_apply_ncc_filter         << " ms" << std::endl;
    std::cout << "  SIFT filter:                   " << t.time_apply_sift_filter        << " ms" << std::endl;
    std::cout << "  BNB-NCC filter:                " << t.time_apply_bnb_ncc_filter     << " ms" << std::endl;
    std::cout << "  BNB-SIFT filter:               " << t.time_apply_bnb_sift_filter    << " ms" << std::endl;
    std::cout << "  photometric refine + cluster:  " << t.time_apply_photometric_refine << " ms" << std::endl;
    std::cout << "  Lowe's ratio filter:           " << t.time_apply_lowe_ratio_filter  << " ms" << std::endl;
    std::cout << "  TOTAL (excl. CF precompute):   " << t.total_time                    << " ms" << std::endl;
#endif

    return t;
}

void Temporal_Edge_Tracking_GPU_Pipeline::retrieve_final_matches(std::vector<Temporal_Edge_Match>& out_matches) const
{
    out_matches.clear();
    if (d_clustered_matches == nullptr || h_match_count_after_cluster <= 0)
        return;

    std::vector<Temporal_Refined_Quad_Match_GPU> gpu_matches(static_cast<size_t>(h_match_count_after_cluster));
    cudacheck(cudaMemcpy(gpu_matches.data(), d_clustered_matches,
                         gpu_matches.size() * sizeof(Temporal_Refined_Quad_Match_GPU), cudaMemcpyDeviceToHost));

    out_matches.reserve(gpu_matches.size());
    for (const auto& g : gpu_matches) {
        Temporal_Edge_Match m;
        m.kf_edge_idx     = g.kf_mate_idx;
        m.cf_edge_idx     = g.cf_mate_idx;
        m.cf_x            = g.cf_left_x;
        m.cf_y            = g.cf_left_y;
        m.cf_orientation  = g.cf_left_orientation;
        m.photometric_rms = g.photometric_rms;
        out_matches.push_back(m);
    }
    std::stable_sort(out_matches.begin(), out_matches.end(),
        [](const Temporal_Edge_Match& a, const Temporal_Edge_Match& b) {
            if (a.photometric_rms != b.photometric_rms) return a.photometric_rms < b.photometric_rms;
            if (a.kf_edge_idx != b.kf_edge_idx) return a.kf_edge_idx < b.kf_edge_idx;
            return a.cf_edge_idx < b.cf_edge_idx;
        });
}

Temporal_Edge_Tracking_GPU_Pipeline::~Temporal_Edge_Tracking_GPU_Pipeline()
{
    safe_free(d_cf_cell_ids);
    safe_free(d_cf_edge_indices);
    safe_free(d_cf_cell_start_idx);
    safe_free(d_cf_num_in_cell);
    safe_free(d_candidates);
    safe_free(d_candidate_count);

    safe_free(d_matches_after_ncc);
    safe_free(d_count_after_ncc);
    safe_free(d_ncc_scores);

    safe_free(d_matches_after_sift);
    safe_free(d_count_after_sift);
    safe_free(d_sift_scores);
    safe_free(d_ncc_scores_after_sift);

    safe_free(d_matches_after_bnb_ncc);
    safe_free(d_count_after_bnb_ncc);
    safe_free(d_sift_scores_after_bnb_ncc);
    safe_free(d_matches_after_bnb_sift);
    safe_free(d_count_after_bnb_sift);

    safe_free(d_refined_matches);
    safe_free(d_refined_count);
    safe_free(d_clustered_matches);
    safe_free(d_clustered_count);

    cudacheck_noexcept(cudaEventDestroy(start));
    cudacheck_noexcept(cudaEventDestroy(stop));
}
