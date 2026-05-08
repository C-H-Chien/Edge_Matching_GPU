#ifndef STEREO_MATCHES_GPU_CPP
#define STEREO_MATCHES_GPU_CPP

#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cstring>
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
    ,stereo_frame_ptr_(&current_frame)
{
    //> Fundamental matrix (left to right) is casted to float matrix.
    //> Eigen stores Matrix3f column-major by default, but kernels index h_F as row-major:
    //> [0..2] = first row, [3..5] = second row, [6..8] = third row.
    h_F_ = dataset_->get_fund_mat_21().cast<float>();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            h_F_row_major_[r * 3 + c] = h_F_(r, c);
        }
    }
    h_F = h_F_row_major_;
    num_left_edges_ = current_frame.left_edges.size();
    num_right_edges_ = current_frame.right_edges.size();

    //> Sanity check
    if (num_left_edges_ <= 0) {
        LOG_ERROR("Number of edges on the left image is less than or equal to 0");
    }
    if (num_right_edges_ <= 0) {
        LOG_ERROR("Number of edges on the right image is less than or equal to 0");
    }

    //> CAUTION: this is subject to the scene complexity
    max_stereo_edge_matches_after_epipolar_rasterization = num_left_edges_ * FACTOR_OF_NUM_OF_LEFT_EDGES_FOR_STEREO_EDGE_MATCHES_AFTER_EPIPOLAR_RASTERIZATION;

    //> convert edges to float (double -> float)
    h_left_edges = new Edge_GPU[num_left_edges_];
    h_right_edges = new Edge_GPU[num_right_edges_];
    h_cell_ids = new int[num_right_edges_];
    h_matches_out = new Match_by_Edge_Index[max_stereo_edge_matches_after_epipolar_rasterization];

    //> Flatten images as float. Note that the data type needs to be CV_32F
    //> Prefer the undistorted image (the same image edges are detected on and CPU SIFT reads),
    //> If the undistorted image is empty, fall back to the raw distorted image
    //> The same chain feeds img_*_width/height above, so the buffer dimensions stay in sync with the source.
    const cv::Mat& left_src  = pick_image_for_gpu( current_frame.left_image_undistorted, current_frame.left_image );
    const cv::Mat& right_src = pick_image_for_gpu( current_frame.right_image_undistorted, current_frame.right_image );
    cv::Mat left_float;
    cv::Mat right_float;
    left_src.convertTo(left_float, CV_32F);
    right_src.convertTo(right_float, CV_32F);
    if (!left_float.isContinuous())
        left_float = left_float.clone();
    if (!right_float.isContinuous())
        right_float = right_float.clone();

    h_left_img = new float[img_left_width * img_left_height];
    h_right_img = new float[img_right_width * img_right_height];
    std::memcpy(h_left_img, left_float.ptr<float>(), static_cast<size_t>(img_left_width) * static_cast<size_t>(img_left_height) * sizeof(float));
    std::memcpy(h_right_img, right_float.ptr<float>(), static_cast<size_t>(img_right_width) * static_cast<size_t>(img_right_height) * sizeof(float));

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

    num_of_grid_cells_in_height = static_cast<int>(std::ceil(static_cast<double>(img_right_height) / GRID_CELL_SIZE));
    num_of_grid_cells_in_width = static_cast<int>(std::ceil(static_cast<double>(img_right_width) / GRID_CELL_SIZE));
    total_num_of_grid_cells = num_of_grid_cells_in_width * num_of_grid_cells_in_height;

    std::cout << "Right image size: " << img_right_width << " x " << img_right_height << std::endl;
    std::cout << "Number of grid cells in width: " << num_of_grid_cells_in_width << std::endl;

    //> Bind a valid CUDA device before any allocations (required on multi-GPU systems and some job schedulers)
    //> Discard any sticky CUDA error from earlier API use in this process (e.g. OpenCV built with CUDA).
    (void)cudaGetLastError();

    int driver_ver = 0;
    (void)cudaDriverGetVersion(&driver_ver);
    int runtime_ver = 0;
    (void)cudaRuntimeGetVersion(&runtime_ver);

    int n_cuda_devices = 0;
    cudaError_t err_count = cudaGetDeviceCount(&n_cuda_devices);
    if (err_count != cudaSuccess) {
        const char* cvd = ::getenv("CUDA_VISIBLE_DEVICES");
        std::string cvd_str = cvd ? std::string(" CUDA_VISIBLE_DEVICES=\"") + cvd + "\"." : "";
        throw std::runtime_error(
            std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(err_count) +
            " (error code " + std::to_string(static_cast<int>(err_count)) + "). "
            "cudaDriverGetVersion=" + std::to_string(driver_ver) + ", cudaRuntimeGetVersion=" +
            std::to_string(runtime_ver) + "." + cvd_str +
            " Error 999 (cudaErrorUnknown) is commonly a **NVIDIA driver vs CUDA toolkit mismatch**: the top of "
            "nvidia-smi shows the **maximum CUDA version the driver supports**; this binary uses libcudart 12.x "
            "(see runtime number). Rebuild and run with a **CUDA toolkit module that is <= that driver max**, "
            "or use a node with a newer driver. Check `ldd` on your binary to see which libcudart is loaded.");
    }
    if (n_cuda_devices == 0) {
        throw std::runtime_error(
            "cudaGetDeviceCount returned success but reported 0 CUDA devices (runtime version " +
            std::to_string(runtime_ver) +
            "). If nvidia-smi lists a GPU, common causes are: (1) running outside your GPU Slurm allocation "
            "in a shell that cannot access /dev/nvidia*; (2) CUDA_VISIBLE_DEVICES empty or invalid; "
            "(3) cluster cgroup denies GPU to this process—launch main_VO from the same interactive GPU session "
            "where nvidia-smi works.");
    }
    if (device_id < 0 || device_id >= n_cuda_devices) {
        throw std::runtime_error(
            "CUDA: invalid device_id " + std::to_string(device_id) + " (visible devices: 0-" +
            std::to_string(n_cuda_devices - 1) + "). Adjust DEVICE_ID or CUDA_VISIBLE_DEVICES.");
    }
    cudacheck(cudaSetDevice(device_id));

    //> Allocate GPU memory
    cudacheck( cudaMalloc((void**)&d_F,                            9 * sizeof(float))    );
    cudacheck( cudaMalloc((void**)&d_left_edges,    num_left_edges_  * sizeof(Edge_GPU)) );
    cudacheck( cudaMalloc((void**)&d_right_edges,   num_right_edges_ * sizeof(Edge_GPU)) );
    cudacheck( cudaMalloc((void**)&d_cell_ids,      num_right_edges_ * sizeof(int))      );

    cudacheck( cudaMalloc((void**)&d_edge_indices,          num_right_edges_ * sizeof(int))        );
    cudacheck( cudaMalloc((void**)&d_cell_start_idx,        total_num_of_grid_cells * sizeof(int)) );
    cudacheck( cudaMalloc((void**)&d_num_of_edges_in_cell,  total_num_of_grid_cells * sizeof(int)) );

    //> Output returned from the kernels processing EP, LP, and OR filters
    //> These can be allocated first since there is no dependency on the preceding kernels
    cudacheck( cudaMalloc((void**)&d_final_matches_passing_EP_LP_OR_filters,  max_stereo_edge_matches_after_epipolar_rasterization * sizeof(Match_by_Edge_Index)) );
    cudacheck( cudaMalloc((void**)&d_match_count_passing_EP_LP_OR_filters, sizeof(int)) );

    //> cuda event
    cudacheck( cudaEventCreate(&start) );
    cudacheck( cudaEventCreate(&stop) );

    //> Upload image data once per frame and keep texture objects alive for all image-sampling GPU stages
    //> This avoids re-uploading the same undistorted images in each pipeline function.
    allocate_image_textures();
}

void Stereo_Matches_GPU_Pipeline::allocate_image_textures()
{
    if (b_image_textures_ready_)
        return;

    cudacheck(cudaSetDevice(device_id));
    left_image_texture_  = create_CUDA_texture_object_for_img(h_left_img,  img_left_width,  img_left_height);
    right_image_texture_ = create_CUDA_texture_object_for_img(h_right_img, img_right_width, img_right_height);
    b_image_textures_ready_ = true;
}

void Stereo_Matches_GPU_Pipeline::release_image_textures()
{
    if (left_image_texture_.texObj != 0) {
        cudacheck(cudaDestroyTextureObject(left_image_texture_.texObj));
        left_image_texture_.texObj = 0;
    }
    if (left_image_texture_.array != nullptr) {
        cudacheck(cudaFreeArray(left_image_texture_.array));
        left_image_texture_.array = nullptr;
    }
    if (right_image_texture_.texObj != 0) {
        cudacheck(cudaDestroyTextureObject(right_image_texture_.texObj));
        right_image_texture_.texObj = 0;
    }
    if (right_image_texture_.array != nullptr) {
        cudacheck(cudaFreeArray(right_image_texture_.array));
        right_image_texture_.array = nullptr;
    }
    b_image_textures_ready_ = false;
}

void Stereo_Matches_GPU_Pipeline::get_stereo_edge_matching_GPU() {

    //> Timer holders
    GPU_Timing_Statistics GPU_time_collection;

    //> For each edge on the right, construct the spatial grid mapping (result is stored in d_cell_ids)
    GPU_time_collection.time_spatial_grid_mapping = build_edge_spatial_map();
    #if COPY_DATA_FROM_HOST_TO_DEVICE_FOR_TESTING
    test_spatial_grid_mapping();
    #endif

    //> Do epipolar line rasterization to find right candidate edges that pass EP, LP, and OR filters
    GPU_time_collection.time_epipolar_rasterization = rasterize_epipolar_band_for_matches();
    #if COPY_DATA_FROM_HOST_TO_DEVICE_FOR_TESTING
    test_epipolar_rasterization();
    test_EP_LP_OR_vs_stereo_matches_cpu();
    #endif

    //> Precompute the left edge patches photometry
    GPU_time_collection.time_precompute_left_edge_patches_photometry = precompute_left_edge_patches_photometry();

    //> Apply NCC filter
    GPU_time_collection.time_apply_NCC_filter = apply_NCC_filter();
    std::cout << "Number of matches passing NCC filter: " << h_match_count_passing_NCC_filter << std::endl;
    #if COPY_DATA_FROM_HOST_TO_DEVICE_FOR_TESTING
    test_ncc_filter();
    #endif

    //> Apply GPU-only SIFT-style descriptor filter
    GPU_time_collection.time_apply_SIFT_descriptor_filter = apply_SIFT_descriptor_filter();
    std::cout << "Number of matches passing GPU SIFT descriptor filter: " << h_match_count_passing_SIFT_filter << std::endl;
    #if COPY_DATA_FROM_HOST_TO_DEVICE_FOR_TESTING
    test_SIFT_descriptor_filter();
    #endif
}

float Stereo_Matches_GPU_Pipeline::build_edge_spatial_map() {

    //> Build edge spatial map
    return edge_spatial_map_thrust_pipeline(
        device_id,
        h_right_edges,
        h_cell_ids,
        d_right_edges,
        d_cell_ids,
        d_edge_indices,
        d_cell_start_idx,
        d_num_of_edges_in_cell,
        num_right_edges_,
        total_num_of_grid_cells,
        num_of_grid_cells_in_width,
        num_of_grid_cells_in_height,
        start,
        stop);
}

float Stereo_Matches_GPU_Pipeline::rasterize_epipolar_band_for_matches() {

    return epipolar_rasterization_thick_band_pipeline(
        device_id,
        h_F,
        h_left_edges,
        h_match_count_passing_EP_LP_OR_filters,
        h_matches_out,
        d_left_edges,
        d_right_edges,
        d_edge_indices,
        d_cell_start_idx,
        d_num_of_edges_in_cell,
        d_F,
        d_final_matches_passing_EP_LP_OR_filters,
        d_match_count_passing_EP_LP_OR_filters,
        num_left_edges_,
        img_right_width,
        img_right_height,
        num_of_grid_cells_in_width,
        num_of_grid_cells_in_height,
        max_stereo_edge_matches_after_epipolar_rasterization,
        start,
        stop);
}

float Stereo_Matches_GPU_Pipeline::precompute_left_edge_patches_photometry() {

    return precompute_left_edge_patches_photometry_pipeline(
        device_id,
        left_image_texture_.texObj,
        d_left_patches,
        d_left_edges,
        num_left_edges_,
        start,
        stop
    );
}

float Stereo_Matches_GPU_Pipeline::apply_NCC_filter() {

    //> Use device match count: host h_match_count_* must match d_match_count_* or the NCC kernel launches with
    //> num_candidates == 0 while candidates still live on device (validator then sees CPU passes / GPU empty).
    int num_candidates = 0;
    cudacheck(cudaMemcpy(&num_candidates, d_match_count_passing_EP_LP_OR_filters, sizeof(int), cudaMemcpyDeviceToHost));
    if (num_candidates > max_stereo_edge_matches_after_epipolar_rasterization) {
        LOG_WARNING("NCC: clamping candidate count to epipolar buffer (" + std::to_string(num_candidates) + " -> " +
                    std::to_string(max_stereo_edge_matches_after_epipolar_rasterization) + ")");
        num_candidates = max_stereo_edge_matches_after_epipolar_rasterization;
    }
    if (num_candidates < 0) {
        LOG_ERROR("NCC: invalid negative candidate count from device; treating as 0");
        num_candidates = 0;
    }
    h_match_count_passing_EP_LP_OR_filters = num_candidates;

    size_t ncc_out_bytes = static_cast<size_t>(num_candidates) * sizeof(Match_by_Edge_Index);
    if (ncc_out_bytes == 0)
        ncc_out_bytes = sizeof(Match_by_Edge_Index);
    cudacheck(cudaMalloc(&d_final_matches_passing_NCC_filter, ncc_out_bytes));
    cudacheck(cudaMalloc(&d_final_count_passing_NCC_filter, sizeof(int)));

    std::cout << "Number of matches passing EP, LP, and OR filters: " << num_candidates << std::endl;
    return apply_NCC_filter_pipeline(
        device_id,
        h_match_count_passing_NCC_filter,
        right_image_texture_.texObj,
        d_final_matches_passing_EP_LP_OR_filters,
        d_left_edges,
        d_right_edges,
        d_final_matches_passing_NCC_filter,
        d_final_count_passing_NCC_filter,
        d_left_patches,
        num_candidates,
        start,
        stop
    );
}

float Stereo_Matches_GPU_Pipeline::apply_SIFT_descriptor_filter() {

    //> Just a sanity check
    if (h_match_count_passing_NCC_filter < 0) {
        LOG_ERROR("GPU SIFT: invalid negative NCC count from device; treating as 0");
        h_match_count_passing_NCC_filter = 0;
        return -1.0;
    }

    return apply_SIFT_descriptor_filter_pipeline(
        device_id,
        h_match_count_passing_SIFT_filter,
        left_image_texture_.texObj,
        right_image_texture_.texObj,
        d_final_matches_passing_NCC_filter,
        d_left_edges,
        d_right_edges,
        d_final_matches_passing_SIFT_filter,
        d_final_count_passing_SIFT_filter,
        img_left_width,
        img_left_height,
        img_right_width,
        img_right_height,
        h_match_count_passing_NCC_filter,
        start,
        stop
    );
}

Stereo_Matches_GPU_Test_Context Stereo_Matches_GPU_Pipeline::make_test_context() const
{
    Stereo_Matches_GPU_Test_Context ctx;
    ctx.h_left_edges = h_left_edges;
    ctx.h_right_edges = h_right_edges;
    ctx.h_cell_ids = h_cell_ids;
    ctx.h_F = h_F;
    ctx.d_final_matches_passing_EP_LP_OR_filters = d_final_matches_passing_EP_LP_OR_filters;
    ctx.d_match_count_passing_EP_LP_OR_filters = d_match_count_passing_EP_LP_OR_filters;
    ctx.d_final_matches_passing_NCC_filter = d_final_matches_passing_NCC_filter;
    ctx.d_final_count_passing_NCC_filter = d_final_count_passing_NCC_filter;
    ctx.d_final_matches_passing_SIFT_filter = d_final_matches_passing_SIFT_filter;
    ctx.d_final_count_passing_SIFT_filter = d_final_count_passing_SIFT_filter;
    ctx.d_left_patches = d_left_patches;
    ctx.d_edge_indices = d_edge_indices;
    ctx.h_right_img = h_right_img;
    ctx.num_left_edges = num_left_edges_;
    ctx.num_right_edges = num_right_edges_;
    ctx.img_right_width = img_right_width;
    ctx.img_right_height = img_right_height;
    ctx.num_of_grid_cells_in_width = num_of_grid_cells_in_width;
    ctx.num_of_grid_cells_in_height = num_of_grid_cells_in_height;
    ctx.max_matches = max_stereo_edge_matches_after_epipolar_rasterization;
    ctx.dataset = dataset_;
    ctx.stereo_frame_for_cpu = stereo_frame_ptr_;
    return ctx;
}

void Stereo_Matches_GPU_Pipeline::test_spatial_grid_mapping() const {
    Stereo_Matches_GPU_Validator::test_spatial_grid_mapping(make_test_context());
}

void Stereo_Matches_GPU_Pipeline::test_epipolar_rasterization() const {
    Stereo_Matches_GPU_Validator::test_epipolar_rasterization(make_test_context());
}

void Stereo_Matches_GPU_Pipeline::test_EP_LP_OR_vs_stereo_matches_cpu() const {
    Stereo_Matches_GPU_Validator::test_EP_LP_OR_vs_stereo_matches_cpu(make_test_context());
}

void Stereo_Matches_GPU_Pipeline::test_ncc_filter() const {
    Stereo_Matches_GPU_Validator::test_ncc_filter(make_test_context());
}

void Stereo_Matches_GPU_Pipeline::test_SIFT_descriptor_filter() const {
    Stereo_Matches_GPU_Validator::test_SIFT_descriptor_filter(make_test_context());
}

void Stereo_Matches_GPU_Pipeline::write_sift_matches_to_file(size_t frame_idx) const
{
    if (d_final_matches_passing_SIFT_filter == nullptr || d_final_count_passing_SIFT_filter == nullptr) {
        LOG_WARNING("write_sift_matches_to_file: GPU SIFT outputs not allocated; skipping");
        return;
    }
    int sift_count = 0;
    cudaError_t err = cudaMemcpy(&sift_count, d_final_count_passing_SIFT_filter, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_ERROR(std::string("write_sift_matches_to_file: cudaMemcpy d_final_count_passing_SIFT_filter failed: ") + cudaGetErrorString(err));
        return;
    }
    if (sift_count < 0) {
        LOG_ERROR("write_sift_matches_to_file: invalid negative SIFT match count from device");
        return;
    }

    std::vector<Match_by_Edge_Index> sift_matches(static_cast<size_t>(sift_count));
    if (sift_count > 0) {
        err = cudaMemcpy(sift_matches.data(), d_final_matches_passing_SIFT_filter,
                         static_cast<size_t>(sift_count) * sizeof(Match_by_Edge_Index),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            LOG_ERROR(std::string("write_sift_matches_to_file: cudaMemcpy d_final_matches_passing_SIFT_filter failed: ") + cudaGetErrorString(err));
            return;
        }
    }

    const std::string output_dir = (dataset_ != nullptr) ? dataset_->get_output_path() : std::string("output_files");
    const std::string filename = output_dir + "/gpu_sift_matches_frame_" + std::to_string(frame_idx) + ".txt";
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        LOG_ERROR("write_sift_matches_to_file: cannot open output file " + filename);
        return;
    }
    outfile << "left_idx left_x left_y left_orientation right_idx right_x right_y right_orientation\n";
    outfile.precision(8);

    size_t skipped_invalid = 0;
    for (const auto& m : sift_matches) {
        if (m.left_edge_idx < 0 || static_cast<size_t>(m.left_edge_idx) >= num_left_edges_ ||
            m.right_edge_idx < 0 || static_cast<size_t>(m.right_edge_idx) >= num_right_edges_) {
            ++skipped_invalid;
            continue;
        }
        const Edge_GPU& le = h_left_edges[m.left_edge_idx];
        const Edge_GPU& re = h_right_edges[m.right_edge_idx];
        outfile << m.left_edge_idx  << " " << le.location_x << " " << le.location_y << " " << le.orientation << " "
                << m.right_edge_idx << " " << re.location_x << " " << re.location_y << " " << re.orientation << "\n";
    }
    outfile.close();

    std::cout << "Wrote " << (sift_matches.size() - skipped_invalid) << " GPU SIFT matches to "
              << filename;
    if (skipped_invalid > 0)
        std::cout << " (skipped " << skipped_invalid << " out-of-range entries)";
    std::cout << std::endl;
}

Stereo_Matches_GPU_Pipeline::~Stereo_Matches_GPU_Pipeline()
{
    //> h_F aliases fixed storage h_F_row_major_; no need to free it.
    release_image_textures();

    delete[] h_left_img;
    delete[] h_right_img;
    delete[] h_left_edges;
    delete[] h_right_edges;
    delete[] h_cell_ids;
    delete[] h_matches_out;

    cudacheck( cudaFree(d_F) );
    cudacheck( cudaFree(d_left_edges) );
    cudacheck( cudaFree(d_right_edges) );
    cudacheck( cudaFree(d_cell_ids) );
    cudacheck( cudaFree(d_edge_indices) );
    cudacheck( cudaFree(d_cell_start_idx) );
    cudacheck( cudaFree(d_num_of_edges_in_cell) );
    cudacheck( cudaFree(d_final_matches_passing_EP_LP_OR_filters) );
    cudacheck( cudaFree(d_match_count_passing_EP_LP_OR_filters) );
    cudacheck( cudaFree(d_left_patches) );
    cudacheck( cudaFree(d_final_matches_passing_NCC_filter) );
    cudacheck( cudaFree(d_final_count_passing_NCC_filter) );
    if (d_final_matches_passing_SIFT_filter) cudacheck( cudaFree(d_final_matches_passing_SIFT_filter) );
    if (d_final_count_passing_SIFT_filter) cudacheck( cudaFree(d_final_count_passing_SIFT_filter) );

    cudacheck(cudaEventDestroy(start));
    cudacheck(cudaEventDestroy(stop));
}

#endif // STEREO_MATCHES_GPU_CPP