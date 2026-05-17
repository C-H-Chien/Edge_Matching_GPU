#ifndef PIPELINE_CPP
#define PIPELINE_CPP

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "Pipeline.h"
#include "definitions.h"

Pipeline::Pipeline(Dataset::Ptr dataset) : dataset_(dataset)
{
    //> Loading the dataset
    dataset_->load_dataset(dataset_->get_dataset_type(), left_ref_disparity_maps, right_ref_disparity_maps, left_occlusion_masks, right_occlusion_masks);

    //> Initialize the pointers to the classes
    stereo_matches_engine = std::make_shared<Stereo_Matches>();
    temporal_matches_engine = std::make_shared<Temporal_Matches>(dataset_);
    utility_tool = std::make_shared<Utility>();

    //> Open the file stream for writing timing statistics
    std::string timing_statistics_file_name = dataset_->get_output_path() + "/timing_statistics.txt";
    std::string stereo_matches_timing_statistics_file_name = dataset_->get_output_path() + "/stereo_matches_timing_statistics.txt";
    std::string temporal_matches_timing_statistics_file_name = dataset_->get_output_path() + "/temporal_matches_timing_statistics.txt";
    timing_statistics_file.open(timing_statistics_file_name);
    stereo_matches_timing_statistics_file.open(stereo_matches_timing_statistics_file_name);
    temporal_matches_timing_statistics_file.open(temporal_matches_timing_statistics_file_name);
}

bool Pipeline::Add_Stereo_Frame()
{
    do
    {
        if (b_end_pipeline_for_debug) {
            break;
        }
        switch (status_)
        {
        case PipelineStatus::STATUS_IMG_PREPARATION:
            //> Preparing images
            LOG_STATUS("IMG_PREPARATION");
            prepare_Stereo_Images();
            break;
        case PipelineStatus::STATUS_GET_STEREO_EDGE_CORRESPONDENCES:
            //> Get stereo edge correspondences
            LOG_STATUS("GET_STEREO_EDGE_CORRESPONDENCES");
            // get_Stereo_Edge_Correspondences();
            get_Stereo_Edge_Correspondences_GPU();
            break;
        case PipelineStatus::STATUS_GET_TEMPORAL_EDGE_CORRESPONDENCES:
            //> Get temporal edge correspondences (keyframe <-> current frame)
            LOG_STATUS("GET_TEMPORAL_EDGE_CORRESPONDENCES");
            // get_Temporal_Edge_Correspondences();
            // get_Temporal_Edge_Correspondences_GPU();
            b_end_pipeline_for_debug = true;
            break;
        case PipelineStatus::STATUS_SYSTEM_EXIT:
            //> System exit
            LOG_STATUS("SYSTEM_EXIT");
            break;
        }
    } while (!send_control_to_main);

    return true;
}

void Pipeline::ProcessEdges(const cv::Mat &image, std::vector<Edge> &edges, bool is_left)
{
    std::cout << "Running third-order edge detector..." << std::endl;
    if (is_left) {
        TOED_left->get_Third_Order_Edges(image);
    }
    else {
        TOED_right->get_Third_Order_Edges(image);
    }
    edges = (is_left) ? TOED_left->toed_edges : TOED_right->toed_edges;
}

void Pipeline::prepare_Stereo_Images()
{
    if (dataset_->has_gt()) {
        current_frame.left_disparity_map = (stereo_current_frame_idx < left_ref_disparity_maps.size()) ? left_ref_disparity_maps[stereo_current_frame_idx] : cv::Mat();
        current_frame.right_disparity_map = (stereo_current_frame_idx < right_ref_disparity_maps.size()) ? right_ref_disparity_maps[stereo_current_frame_idx] : cv::Mat();
    }

    std::cout << std::endl << "Stereo Image Pair #" << stereo_current_frame_idx << std::endl;

    cv::Mat left_cur_undistorted, right_cur_undistorted;
    cv::undistort(current_frame.left_image, left_cur_undistorted, dataset_->get_left_calib_matrix_cvMat(), dataset_->get_left_dist_coeff_mat());
    cv::undistort(current_frame.right_image, right_cur_undistorted, dataset_->get_right_calib_matrix_cvMat(), dataset_->get_right_dist_coeff_mat());
    current_frame.left_image_undistorted = left_cur_undistorted;
    current_frame.right_image_undistorted = right_cur_undistorted;

    util_compute_Img_Gradients(current_frame.left_image_undistorted, current_frame.left_image_gradients_x, current_frame.left_image_gradients_y);
    util_compute_Img_Gradients(current_frame.right_image_undistorted, current_frame.right_image_gradients_x, current_frame.right_image_gradients_y);

    //> initialize the pointers of the third-order edge detector and the spatial grids
    if (get_Current_Frame_Index() == 0)
        initialize_TOED_and_Spatial_Grids();

    auto start_time_left = std::chrono::high_resolution_clock::now();
    ProcessEdges(left_cur_undistorted, dataset_->left_edges, true);
    auto end_time_left = std::chrono::high_resolution_clock::now();
    auto duration_left = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_left - start_time_left);
    // std::cout << "Time taken by the third-order edge detector for the left image: " << duration_left.count() << " milliseconds" << std::endl;
    std::cout << "Number of edges on the left image: " << dataset_->left_edges.size() << std::endl;
    current_frame.left_edges = dataset_->left_edges;

    auto start_time_right = std::chrono::high_resolution_clock::now();
    ProcessEdges(right_cur_undistorted, dataset_->right_edges, false);
    auto end_time_right = std::chrono::high_resolution_clock::now();
    auto duration_right = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_right - start_time_right);
    // std::cout << "Time taken by the third-order edge detector for the right image: " << duration_right.count() << " milliseconds" << std::endl;
    std::cout << "Number of edges on the right image: " << dataset_->right_edges.size() << std::endl;
    current_frame.right_edges = dataset_->right_edges;

    timing_statistics_file << duration_left.count() + duration_right.count() << " ";

    //> Shift to the next status
    status_ = PipelineStatus::STATUS_GET_STEREO_EDGE_CORRESPONDENCES;
    send_control_to_main = false;
}

void Pipeline::get_Stereo_Edge_Correspondences_GPU() 
{
    //> Set up the flattened fundamental matrix, left and right images, and the GPU device
    prepare_for_GPU_engine = std::make_shared<Prepare_For_GPU_Pipeline>(dataset_, current_frame, DEVICE_ID);

    //> Reset before allocate: frees the previous frame's GPU pipeline before this frame's cudaMalloc traffic (avoids a brief 2x VRAM spike).
    stereo_matches_engine_GPU.reset();

    Stereo_Matches_GPU_Timing_Statistics stereo_GPU_times;

    //> h_F, left_image_texture_, right_image_texture_ are created and passed to the stereo matches engine
    stereo_matches_engine_GPU = std::make_shared<Stereo_Matches_GPU_Pipeline>(dataset_, prepare_for_GPU_engine, current_frame, DEVICE_ID);
    stereo_GPU_times = stereo_matches_engine_GPU->get_stereo_edge_matching_GPU(stereo_current_frame_idx);
    collect_stereo_GPU_times.push_back(stereo_GPU_times);

    //> Convert GPU final stereo matches to the `current_frame_stereo_edge_mates` structure
    //  for the preparation of feeding it into the temporal edge matching pipeline
    //> CH: This should be optional
    stereo_matches_engine_GPU->retrieve_stereo_mates(current_frame, current_frame_stereo_edge_mates);
    std::cout << "Number of stereo edge matches: " << current_frame_stereo_edge_mates.size() << std::endl;

    //> Populate patches + SIFT descriptors so the CPU temporal pipeline can run on these mates.
    //  (The GPU temporal pipeline computes NCC/SIFT from textures on-the-fly and doesn't need this.)
    // stereo_matches_engine->enrich_stereo_mates_for_cpu_temporal(current_frame_stereo_edge_mates, current_frame);

    //> output the results to files
    stereo_matches_engine_GPU->write_finalized_matches_to_file(stereo_current_frame_idx);

    //> Release GPU immediately; nothing else uses stereo_matches_engine_GPU until the next stereo pass.
    //> CH Notes: this is subject to change in the future.
    stereo_matches_engine_GPU.reset();

    //> If the current frame is the first frame, make current frame the keyframe
    if (get_Current_Frame_Index() == 0)
    {
        set_Keyframe();
        increment_Current_Stereo_Frame_Index();
        status_ = PipelineStatus::STATUS_IMG_PREPARATION;
        send_control_to_main = true;
    }
    else
    {
        // increment_Current_Stereo_Frame_Index();
        //> Shift to the next status
        status_ = PipelineStatus::STATUS_GET_TEMPORAL_EDGE_CORRESPONDENCES;
        send_control_to_main = false;
        b_end_pipeline_for_debug = true;
    }
}

void Pipeline::get_Stereo_Edge_Correspondences()
{
    //> Set the stereo left constructor
    set_Stereo_Left_Constructor();
    Timing_Statistics stereo_timing_statistics;

    //> For each left edge, get the corresponding GT location (not right edge) on the right image, and the triangulated 3D point in the left camera coordinate
    //> For dataset without GT, we still need these functions to set up `current_frame_stereo_left_constructor` but we will not use the GT locations
    stereo_matches_engine->Find_Stereo_GT_Locations(dataset_, current_frame.left_disparity_map, current_frame, current_frame_stereo_left_constructor, true);
    //> Construct a GT stereo edge pool
    stereo_matches_engine->get_Stereo_Edge_GT_Pairs(dataset_, current_frame, current_frame_stereo_left_constructor, true);

    //> construct stereo edge correspondences for the current_frame
    Frame_Evaluation_Metrics metrics = stereo_matches_engine->get_Stereo_Edge_Pairs(dataset_, current_frame_stereo_left_constructor, stereo_current_frame_idx, stereo_timing_statistics);
    //> Only accumulate evaluation metrics when GT is available
    if (dataset_->has_gt()) {
        all_stereo_matches_metrics.push_back(metrics);
    }

    //> Finalize the stereo edge pairs for the current_frame
    stereo_matches_engine->finalize_stereo_edge_mates(current_frame_stereo_left_constructor, current_frame_stereo_edge_mates);
    
    // stereo_matches_engine->write_finalized_stereo_edge_pairs_to_file(dataset_, current_frame_stereo_edge_mates, stereo_current_frame_idx);

    //> If the current frame is the first frame, make current frame the keyframe
    if (get_Current_Frame_Index() == 0)
    {
        set_Keyframe();
        increment_Current_Stereo_Frame_Index();
        status_ = PipelineStatus::STATUS_IMG_PREPARATION;
        send_control_to_main = true;
    }
    else
    {
        increment_Current_Stereo_Frame_Index();
        //> Shift to the next status
        status_ = PipelineStatus::STATUS_GET_TEMPORAL_EDGE_CORRESPONDENCES;
        send_control_to_main = false;
        b_end_pipeline_for_debug = true;
    }
}

void Pipeline::get_Temporal_Edge_Correspondences_GPU()
{
    exit(1);
    std::cout << "Finding temporal edge correspondences (GPU) (" << get_Keyframe_Index() << "->" << get_Current_Frame_Index() << ")" << std::endl;

    //> Reset before allocate: avoids a brief 2x VRAM spike
    temporal_matches_engine_GPU.reset();

    Temporal_Matches_GPU_Timing_Statistics temporal_GPU_times;
    temporal_matches_engine_GPU = std::make_shared<Temporal_Matches_GPU_Pipeline>(
        dataset_,
        keyframe,                        //> StereoFrame for KF images
        current_frame,                   //> StereoFrame for CF images
        keyframe_stereo_edge_mates,      //> populated by retrieve_stereo_mates at KF time
        current_frame_stereo_edge_mates, //> populated by retrieve_stereo_mates this frame
        DEVICE_ID);

    //> Main temporal edge matching pipeline in GPU
    temporal_GPU_times = temporal_matches_engine_GPU->get_temporal_edge_matching_GPU( stereo_key_frame_idx, stereo_current_frame_idx);

    //> Retrieve final (kf_mate_idx, cf_mate_idx) pairs for downstream use
    std::vector<Match_by_Edge_Index> final_quad_matches;
    temporal_matches_engine_GPU->retrieve_final_matches(final_quad_matches);

    //> output the results to files
    temporal_matches_engine_GPU->write_finalized_matches_to_file(stereo_key_frame_idx, stereo_current_frame_idx);

    collect_temporal_GPU_times.push_back(temporal_GPU_times);
    temporal_matches_engine_GPU.reset();

    //> final_quad_matches[i].left_edge_idx  = index into keyframe_stereo_edge_mates
    //> final_quad_matches[i].right_edge_idx = index into current_frame_stereo_edge_mates

    set_Keyframe();

    status_ = PipelineStatus::STATUS_IMG_PREPARATION;
    send_control_to_main = true;
    b_end_pipeline_for_debug = true;
}

void Pipeline::get_Temporal_Edge_Correspondences()
{
    std::cout << "Finding temporal edge correspondences (CPU) (" << get_Keyframe_Index() << "->" << get_Current_Frame_Index() << ")" << std::endl;
    temporal_quads_by_kf.clear();
    temporal_quads_by_kf.shrink_to_fit();

    //> Reset spatial grids so they contain only the current frame's edges (otherwise candidates accumulate across frames and memory explodes)
    reset_spatial_grids();
    Timing_Statistics temporal_timing_statistics;

    //> construct spatial grids for the current stereo edge mates
    temporal_matches_engine->add_edges_to_spatial_grid(current_frame_stereo_edge_mates, left_spatial_grids, right_spatial_grids);

    bool used_chained_veridical_quads = false;
#if ENABLE_QUAD_PROPAGATION
    if (dataset_->has_gt() && stereo_current_frame_idx > stereo_key_frame_idx + 1)
        used_chained_veridical_quads = try_propagate_multihop_veridical_quads();
#endif

    //> `temporal_quads_by_kf` is a struct that stores quads from KF stereo edge pairs
    //> One KF stereo edge pair could pair up with multiple veridical CF stereo edge pairs.
    //> `build_Veridical_Quads` fills veridical quads when reference disparity GT exists; otherwise it still runs and seeds one
    //> `KF_Temporal_Edge_Quads` per KF stereo mate (see Temporal_Matches::build_Veridical_Quads).
    if (!used_chained_veridical_quads)
        temporal_matches_engine->build_Veridical_Quads(temporal_quads_by_kf, keyframe_stereo_edge_mates, current_frame_stereo_edge_mates, keyframe_stereo_left_constructor, current_frame_stereo_left_constructor, left_spatial_grids, right_spatial_grids);

    //> Store adjacent quads for multi-hop propagation (propagation path still requires has_gt() today).
    if (!used_chained_veridical_quads && stereo_current_frame_idx == stereo_key_frame_idx + 1)
        push_adjacent_veridical_snapshot();

    //> Quad-centric pipeline: build veridical quads, apply filters, optionally convert to temporal pairs for backward compatibility
    Frame_Evaluation_Metrics metrics = temporal_matches_engine->get_Temporal_Edge_Pairs_from_Quads(
        temporal_quads_by_kf,
        keyframe_stereo_edge_mates,
        current_frame_stereo_edge_mates,
        left_spatial_grids, right_spatial_grids,
        keyframe_stereo_left_constructor, current_frame_stereo_left_constructor,
        keyframe, current_frame,
        stereo_key_frame_idx, stereo_current_frame_idx, temporal_timing_statistics);
    //> Only accumulate temporal evaluation metrics when GT is available
    if (dataset_->has_gt())
        all_temporal_matches_metrics.push_back(metrics);

    //> write quads to a file
    // temporal_matches_engine->write_quads_to_file(temporal_quads_by_kf, stereo_key_frame_idx, stereo_current_frame_idx);

    // timing_statistics_file << temporal_timing_statistics.total_time << " ";
    // temporal_matches_engine->write_timings_to_file(temporal_matches_timing_statistics_file, temporal_timing_statistics);

    //> Count KF groups that still have at least one surviving candidate (= accepted temporal matches)
    size_t final_match_count = 0;
    for (const auto &kvq : temporal_quads_by_kf)
        if (!kvq.candidate_quads.empty()) ++final_match_count;
    std::cout << "[Temporal CPU] Final matches: " << final_match_count << std::endl;

    set_Keyframe();

    status_ = PipelineStatus::STATUS_IMG_PREPARATION;
    send_control_to_main = true;
    b_end_pipeline_for_debug = true;
}

// void Pipeline::free_stereo_temporal_pointers()
// {
//     stereo_matches_engine_GPU.reset();
//     temporal_matches_engine_GPU.reset();
// }

const Pipeline::AdjacentQuadSnapshot *Pipeline::find_adjacent_snapshot(size_t kf_frame, size_t cf_frame) const
{
    for (const auto &s : adjacent_quad_history_)
    {
        if (s.kf_frame_idx == kf_frame && s.cf_frame_idx == cf_frame)
            return &s;
    }
    return nullptr;
}

void Pipeline::push_adjacent_veridical_snapshot()
{
    AdjacentQuadSnapshot snap;
    snap.kf_frame_idx = stereo_key_frame_idx;
    snap.cf_frame_idx = stereo_current_frame_idx;
    snap.kf_stereo_mates = keyframe_stereo_edge_mates;
    snap.cf_stereo_mates = current_frame_stereo_edge_mates;
    snap.quads = temporal_quads_by_kf;
    for (auto &kvq : snap.quads)
    {
        ptrdiff_t ki = kvq.KF_stereo_mate - keyframe_stereo_edge_mates.data();
        if (ki >= 0 && ki < static_cast<ptrdiff_t>(keyframe_stereo_edge_mates.size()))
            kvq.KF_stereo_mate = snap.kf_stereo_mates.data() + ki;
    }
    adjacent_quad_history_.push_back(std::move(snap));
    while (adjacent_quad_history_.size() > QUAD_PROPAGATION_HISTORY_MAX)
        adjacent_quad_history_.pop_front();
}

bool Pipeline::try_propagate_multihop_veridical_quads()
{
#if !ENABLE_QUAD_PROPAGATION
    return false;
#else
    if (!dataset_->has_gt())
        return false;
    const size_t gap = stereo_current_frame_idx - stereo_key_frame_idx;
    if (gap <= 1)
        return false;

    std::vector<const AdjacentQuadSnapshot *> chain;
    chain.reserve(gap);
    for (size_t f = stereo_key_frame_idx; f < stereo_current_frame_idx; ++f)
    {
        const AdjacentQuadSnapshot *s = find_adjacent_snapshot(f, f + 1);
        if (!s)
            return false;
        chain.push_back(s);
    }

    std::vector<KF_Temporal_Edge_Quads> acc = chain[0]->quads;

    for (size_t i = 1; i < chain.size(); ++i)
    {
        std::vector<KF_Temporal_Edge_Quads> next_out;
        Temporal_Matches::propagate_veridical_quads_one_hop(
            acc,
            chain[0]->kf_stereo_mates,
            chain[i]->kf_stereo_mates,
            chain[i]->quads,
            chain[i]->cf_stereo_mates,
            next_out);
        acc = std::move(next_out);
    }

    if (acc.empty())
        return false;

    temporal_quads_by_kf = std::move(acc);

    for (auto &kvq : temporal_quads_by_kf)
    {
        if (!kvq.KF_stereo_mate)
            continue;
        ptrdiff_t ki = kvq.KF_stereo_mate - chain[0]->kf_stereo_mates.data();
        if (ki >= 0 && ki < static_cast<ptrdiff_t>(keyframe_stereo_edge_mates.size()))
            kvq.KF_stereo_mate = keyframe_stereo_edge_mates.data() + ki;
    }

    size_t total_v = 0;
    for (const auto &k : temporal_quads_by_kf)
        total_v += k.veridical_quads.size();
    std::cout << "Quad propagation (" << stereo_key_frame_idx << "->" << stereo_current_frame_idx << "): "
              << temporal_quads_by_kf.size() << " KF groups, " << total_v << " chained veridical quads" << std::endl;

    return true;
#endif
}

//> Destructor
Pipeline::~Pipeline()
{
    timing_statistics_file.close();
    stereo_matches_timing_statistics_file.close();
    temporal_matches_timing_statistics_file.close();
}



#endif