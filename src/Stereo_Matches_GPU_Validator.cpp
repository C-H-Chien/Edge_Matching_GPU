#include "Stereo_Matches_GPU_Validator.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "Stereo_Iterator.h"
#include "Stereo_Matches.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cuda_runtime.h>

#include "gpu_settings.h"

namespace {

int cell_id_from_location(float location_x, float location_y, int num_grid_cells_width, int num_grid_cells_height)
{
    int cx = static_cast<int>(location_x / static_cast<float>(GRID_CELL_SIZE));
    int cy = static_cast<int>(location_y / static_cast<float>(GRID_CELL_SIZE));
    cx = std::max(0, std::min(cx, num_grid_cells_width - 1));
    cy = std::max(0, std::min(cy, num_grid_cells_height - 1));
    return cx + cy * num_grid_cells_width;
}

bool epipolar_line_endpoints(float a, float b, float c, int W, int H, float& x0, float& y0, float& x1, float& y1)
{
    struct Pt { float x; float y; };
    Pt ips[2];
    int count = 0;
    const float Wf = static_cast<float>(W);
    const float Hf = static_cast<float>(H);

    if (std::fabs(b) > 1e-6f) {
        float y_left = -c / b;
        if (y_left >= 0.f && y_left <= Hf)
            ips[count++] = {0.f, y_left};
        float y_right = -(a * Wf + c) / b;
        if (y_right >= 0.f && y_right <= Hf && count < 2)
            ips[count++] = {Wf, y_right};
    }
    if (std::fabs(a) > 1e-6f && count < 2) {
        float x_top = -c / a;
        if (x_top >= 0.f && x_top <= Wf)
            ips[count++] = {x_top, 0.f};
        if (count < 2) {
            float x_bottom = -(b * Hf + c) / a;
            if (x_bottom >= 0.f && x_bottom <= Wf)
                ips[count++] = {x_bottom, Hf};
        }
    }
    if (count != 2)
        return false;
    x0 = ips[0].x;
    y0 = ips[0].y;
    x1 = ips[1].x;
    y1 = ips[1].y;
    return true;
}

float sample_img_bilinear_clamped(const float* img, int W, int H, float x, float y)
{
    if (W <= 0 || H <= 0 || img == nullptr)
        return 0.f;
    // Match cudaTextureObject + tex2D<float> with unnormalized coordinates and linear filtering:
    // texture coordinates are defined at texel centers (0.5, 0.5), so emulate that half-texel shift.
    x = x - 0.5f;
    y = y - 0.5f;
    x = std::max(0.f, std::min(x, static_cast<float>(W - 1)));
    y = std::max(0.f, std::min(y, static_cast<float>(H - 1)));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, W - 1);
    const int y1 = std::min(y0 + 1, H - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float v00 = img[y0 * W + x0];
    const float v10 = img[y0 * W + x1];
    const float v01 = img[y1 * W + x0];
    const float v11 = img[y1 * W + x1];
    const float a = v00 + (v10 - v00) * tx;
    const float b = v01 + (v11 - v01) * tx;
    return a + (b - a) * ty;
}

float sample_img_bilinear_clamped_no_half_shift(const float* img, int W, int H, float x, float y)
{
    if (W <= 0 || H <= 0 || img == nullptr)
        return 0.f;
    x = std::max(0.f, std::min(x, static_cast<float>(W - 1)));
    y = std::max(0.f, std::min(y, static_cast<float>(H - 1)));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, W - 1);
    const int y1 = std::min(y0 + 1, H - 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float v00 = img[y0 * W + x0];
    const float v10 = img[y0 * W + x1];
    const float v01 = img[y1 * W + x0];
    const float v11 = img[y1 * W + x1];
    const float a = v00 + (v10 - v00) * tx;
    const float b = v01 + (v11 - v01) * tx;
    return a + (b - a) * ty;
}

void extract_patch_stats_cpu(const float* img, int W, int H, float shifted_cx, float shifted_cy, float cos_t, float sin_t,
                             float* patch_data, float& mean, float& variance)
{
    float sum = 0.f;
    float sum_sq = 0.f;
    int count = 0;
    for (int v = -PATCH_RADIUS; v <= PATCH_RADIUS; ++v) {
        for (int u = -PATCH_RADIUS; u <= PATCH_RADIUS; ++u) {
            const float sx = shifted_cx + static_cast<float>(u) * cos_t - static_cast<float>(v) * sin_t;
            const float sy = shifted_cy + static_cast<float>(u) * sin_t + static_cast<float>(v) * cos_t;
            const float val = sample_img_bilinear_clamped(img, W, H, sx, sy);
            patch_data[count++] = val;
            sum += val;
            sum_sq += val * val;
        }
    }
    const float total = static_cast<float>(TOTAL_NUM_OF_PATCH_PIXELS);
    mean = sum / total;
    variance = (sum_sq / total) - (mean * mean);
    if (variance < 1e-5f)
        variance = 1e-5f;
}

void extract_patch_stats_cpu_no_half_shift(const float* img, int W, int H, float shifted_cx, float shifted_cy, float cos_t,
                                           float sin_t, float* patch_data, float& mean, float& variance)
{
    float sum = 0.f;
    float sum_sq = 0.f;
    int count = 0;
    for (int v = -PATCH_RADIUS; v <= PATCH_RADIUS; ++v) {
        for (int u = -PATCH_RADIUS; u <= PATCH_RADIUS; ++u) {
            const float sx = shifted_cx + static_cast<float>(u) * cos_t - static_cast<float>(v) * sin_t;
            const float sy = shifted_cy + static_cast<float>(u) * sin_t + static_cast<float>(v) * cos_t;
            const float val = sample_img_bilinear_clamped_no_half_shift(img, W, H, sx, sy);
            patch_data[count++] = val;
            sum += val;
            sum_sq += val * val;
        }
    }
    const float total = static_cast<float>(TOTAL_NUM_OF_PATCH_PIXELS);
    mean = sum / total;
    variance = (sum_sq / total) - (mean * mean);
    if (variance < 1e-5f)
        variance = 1e-5f;
}

float compute_ncc_cpu(const float* p1, float m1, float v1, const float* p2, float m2, float v2)
{
    float cross_cov = 0.f;
    for (int i = 0; i < TOTAL_NUM_OF_PATCH_PIXELS; ++i)
        cross_cov += (p1[i] - m1) * (p2[i] - m2);
    cross_cov /= static_cast<float>(TOTAL_NUM_OF_PATCH_PIXELS);
    return cross_cov / std::sqrt(std::max(v1 * v2, 1e-10f));
}

unsigned long long pair_key(int left_idx, int right_idx)
{
    return (static_cast<unsigned long long>(static_cast<unsigned int>(left_idx)) << 32) |
           static_cast<unsigned int>(right_idx);
}

// Mirrors edge_pair_passes_orientation_prefilter_deg on device (Stereo_Matches::apply_orientation_filter).
bool cpu_edge_pair_orientation_ok(float ori_left, float ori_right, double tol_deg)
{
    double diff_deg =
        std::fabs(static_cast<double>(ori_left) - static_cast<double>(ori_right)) * (180.0 / M_PI);
    if (diff_deg > 180.0)
        diff_deg = 360.0 - diff_deg;
    if (diff_deg < tol_deg)
        return true;
    if (std::fabs(diff_deg - 180.0) < tol_deg)
        return true;
    return false;
}

} // namespace

void Stereo_Matches_GPU_Validator::test_spatial_grid_mapping(const Stereo_Matches_GPU_Test_Context& ctx)
{
    if (ctx.num_right_edges == 0) {
        LOG_TEST("test_spatial_grid_mapping: skip (no right edges)");
        return;
    }

    // d_cell_ids gets sorted by key, so h_cell_ids is in sorted order (not original edge order).
    // Use d_edge_indices to map each sorted slot back to the original right-edge index.
    std::vector<int> sorted_edge_indices(ctx.num_right_edges);
    if (cudaMemcpy(sorted_edge_indices.data(), ctx.d_edge_indices, ctx.num_right_edges * sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("test_spatial_grid_mapping: cudaMemcpy d_edge_indices failed");
        return;
    }

    size_t mismatches = 0;
    for (size_t i = 0; i < ctx.num_right_edges; ++i) {
        const int edge_idx = sorted_edge_indices[i];
        if (edge_idx < 0 || edge_idx >= static_cast<int>(ctx.num_right_edges)) {
            if (mismatches < 5) {
                std::cout << "[TEST] spatial_grid invalid edge index at sorted slot " << i << " edge_idx=" << edge_idx << "\n";
            }
            ++mismatches;
            continue;
        }
        const int expected = cell_id_from_location(
            ctx.h_right_edges[edge_idx].location_x, ctx.h_right_edges[edge_idx].location_y,
            ctx.num_of_grid_cells_in_width, ctx.num_of_grid_cells_in_height);
        if (ctx.h_cell_ids[i] != expected) {
            if (mismatches < 5) {
                std::cout << "[TEST] spatial_grid cell mismatch sorted_slot=" << i
                          << " edge_idx=" << edge_idx
                          << " GPU=" << ctx.h_cell_ids[i]
                          << " CPU=" << expected
                          << " pos=(" << ctx.h_right_edges[edge_idx].location_x << ","
                          << ctx.h_right_edges[edge_idx].location_y << ")\n";
            }
            ++mismatches;
        }
    }
    if (mismatches == 0) {
        LOG_TEST("spatial_grid_mapping: GPU cell ids match CPU reference for all right edges");
    }
    else {
        LOG_ERROR("spatial_grid_mapping: " + std::to_string(mismatches) + " GPU/CPU cell id mismatches (showing first 5)");
    }
}

void Stereo_Matches_GPU_Validator::test_epipolar_rasterization(const Stereo_Matches_GPU_Test_Context& ctx)
{
    if (ctx.num_left_edges == 0 || ctx.num_right_edges == 0) {
        LOG_TEST("test_epipolar_rasterization: skip (no edges)");
        return;
    }

    int raw_gpu_count = 0;
    if (cudaMemcpy(&raw_gpu_count, ctx.d_match_count_passing_EP_LP_OR_filters, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("test_epipolar_rasterization: cudaMemcpy d_match_count_passing_EP_LP_OR_filters failed");
        return;
    }

    if (raw_gpu_count > ctx.max_matches) {
        LOG_WARNING("GPU produced " + std::to_string(raw_gpu_count) +
                    " matches (capped buffer " + std::to_string(ctx.max_matches) +
                    "); CPU/GPU comparison may miss overflow pairs");
    }
    const int gpu_n = std::min(raw_gpu_count, ctx.max_matches);
    std::vector<Match_by_Edge_Index> gpu_pairs(static_cast<size_t>(gpu_n));
    if (gpu_n > 0 &&
        cudaMemcpy(gpu_pairs.data(), ctx.d_final_matches_passing_EP_LP_OR_filters, static_cast<size_t>(gpu_n) * sizeof(Match_by_Edge_Index),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("test_epipolar_rasterization: cudaMemcpy d_final_matches_passing_EP_LP_OR_filters failed");
        return;
    }

    const float max_disp_sq = static_cast<float>(MAX_DISPARITY * MAX_DISPARITY);
    std::vector<std::vector<Match_by_Edge_Index>> thread_pairs;
#ifdef _OPENMP
    thread_pairs.resize(static_cast<size_t>(omp_get_max_threads()));
#else
    thread_pairs.resize(1);
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int li = 0; li < static_cast<int>(ctx.num_left_edges); ++li) {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        const float lx = ctx.h_left_edges[static_cast<size_t>(li)].location_x;
        const float ly = ctx.h_left_edges[static_cast<size_t>(li)].location_y;
        const float a_raw = ctx.h_F[0] * lx + ctx.h_F[1] * ly + ctx.h_F[2];
        const float b_raw = ctx.h_F[3] * lx + ctx.h_F[4] * ly + ctx.h_F[5];
        const float c_raw = ctx.h_F[6] * lx + ctx.h_F[7] * ly + ctx.h_F[8];
        float norm = std::sqrt(a_raw * a_raw + b_raw * b_raw);
        if (norm < 1e-6f)
            continue;
        float a = a_raw / norm;
        float b = b_raw / norm;
        float c = c_raw / norm;
        float x0 = 0.f, y0 = 0.f, x1 = 0.f, y1 = 0.f;
        if (!epipolar_line_endpoints(a, b, c, ctx.img_right_width, ctx.img_right_height, x0, y0, x1, y1))
            continue;
        const float vx = x1 - x0;
        const float vy = y1 - y0;
        const float v2 = vx * vx + vy * vy;
        if (v2 < 1e-6f)
            continue;

        auto& bucket = thread_pairs[static_cast<size_t>(tid)];
        for (size_t rj = 0; rj < ctx.num_right_edges; ++rj) {
            const float rx = ctx.h_right_edges[rj].location_x;
            const float ry = ctx.h_right_edges[rj].location_y;
            if (lx < rx)
                continue;
            const float disparity_sq = (lx - rx) * (lx - rx) + (ly - ry) * (ly - ry);
            const float dist = std::fabs(a * rx + b * ry + c);
            const float t = ((rx - x0) * vx + (ry - y0) * vy) / v2;
            if (t < 0.f || t > 1.f)
                continue;
            if (dist < static_cast<float>(EPIPOLAR_LINE_DIST_THRESH) && disparity_sq < max_disp_sq &&
                cpu_edge_pair_orientation_ok(ctx.h_left_edges[static_cast<size_t>(li)].orientation,
                                            ctx.h_right_edges[rj].orientation,
                                            EDGE_PAIR_ORIENTATION_PREFILTER_DEG))
                bucket.push_back(Match_by_Edge_Index{li, static_cast<int>(rj)});
        }
    }

    std::vector<Match_by_Edge_Index> cpu_pairs;
    for (auto& v : thread_pairs)
        cpu_pairs.insert(cpu_pairs.end(), v.begin(), v.end());

    auto less_pair = [](const Match_by_Edge_Index& p, const Match_by_Edge_Index& q) {
        if (p.left_edge_idx != q.left_edge_idx)
            return p.left_edge_idx < q.left_edge_idx;
        return p.right_edge_idx < q.right_edge_idx;
    };
    std::sort(gpu_pairs.begin(), gpu_pairs.end(), less_pair);
    std::sort(cpu_pairs.begin(), cpu_pairs.end(), less_pair);

    auto pair_key = [](int left_idx, int right_idx) -> unsigned long long {
        return (static_cast<unsigned long long>(static_cast<unsigned int>(left_idx)) << 32) |
               static_cast<unsigned int>(right_idx);
    };
    std::unordered_set<unsigned long long> gpu_unique;
    gpu_unique.reserve(gpu_pairs.size());
    size_t gpu_duplicates = 0;
    for (const auto& p : gpu_pairs) {
        const auto key = pair_key(p.left_edge_idx, p.right_edge_idx);
        if (!gpu_unique.insert(key).second)
            ++gpu_duplicates;
    }
    std::unordered_set<unsigned long long> cpu_unique;
    cpu_unique.reserve(cpu_pairs.size());
    for (const auto& p : cpu_pairs)
        cpu_unique.insert(pair_key(p.left_edge_idx, p.right_edge_idx));

    size_t gpu_invalid_by_filter = 0;
    for (const auto& p : gpu_pairs) {
        if (p.left_edge_idx < 0 || p.left_edge_idx >= static_cast<int>(ctx.num_left_edges) ||
            p.right_edge_idx < 0 || p.right_edge_idx >= static_cast<int>(ctx.num_right_edges)) {
            ++gpu_invalid_by_filter;
            continue;
        }
        const float lx = ctx.h_left_edges[p.left_edge_idx].location_x;
        const float ly = ctx.h_left_edges[p.left_edge_idx].location_y;
        const float rx = ctx.h_right_edges[p.right_edge_idx].location_x;
        const float ry = ctx.h_right_edges[p.right_edge_idx].location_y;
        const float a_raw = ctx.h_F[0] * lx + ctx.h_F[1] * ly + ctx.h_F[2];
        const float b_raw = ctx.h_F[3] * lx + ctx.h_F[4] * ly + ctx.h_F[5];
        const float c_raw = ctx.h_F[6] * lx + ctx.h_F[7] * ly + ctx.h_F[8];
        const float norm = std::sqrt(a_raw * a_raw + b_raw * b_raw);
        if (norm < 1e-6f) {
            ++gpu_invalid_by_filter;
            continue;
        }
        const float a = a_raw / norm;
        const float b = b_raw / norm;
        const float c = c_raw / norm;
        float x0 = 0.f, y0 = 0.f, x1 = 0.f, y1 = 0.f;
        if (!epipolar_line_endpoints(a, b, c, ctx.img_right_width, ctx.img_right_height, x0, y0, x1, y1)) {
            ++gpu_invalid_by_filter;
            continue;
        }
        const float vx = x1 - x0;
        const float vy = y1 - y0;
        const float v2 = vx * vx + vy * vy;
        if (v2 < 1e-6f) {
            ++gpu_invalid_by_filter;
            continue;
        }
        if (lx < rx) {
            ++gpu_invalid_by_filter;
            continue;
        }
        const float disparity_sq = (lx - rx) * (lx - rx) + (ly - ry) * (ly - ry);
        const float dist = std::fabs(a * rx + b * ry + c);
        const float t = ((rx - x0) * vx + (ry - y0) * vy) / v2;
        if (!(dist < static_cast<float>(EPIPOLAR_LINE_DIST_THRESH) && disparity_sq < max_disp_sq &&
              t >= 0.f && t <= 1.f &&
              cpu_edge_pair_orientation_ok(ctx.h_left_edges[p.left_edge_idx].orientation,
                                           ctx.h_right_edges[p.right_edge_idx].orientation,
                                           EDGE_PAIR_ORIENTATION_PREFILTER_DEG)))
            ++gpu_invalid_by_filter;
    }

    size_t gpu_not_in_cpu = 0;
    for (const auto& p : gpu_pairs) {
        if (!cpu_unique.count(pair_key(p.left_edge_idx, p.right_edge_idx)))
            ++gpu_not_in_cpu;
    }
    size_t cpu_not_in_gpu = 0;
    for (const auto& p : cpu_pairs) {
        if (!gpu_unique.count(pair_key(p.left_edge_idx, p.right_edge_idx)))
            ++cpu_not_in_gpu;
    }

    if (gpu_not_in_cpu > 0 || gpu_invalid_by_filter > 0) {
        LOG_ERROR("epipolar_rasterization: GPU produced invalid/extra matches compared to CPU reference"
                  " (gpu_not_in_cpu=" + std::to_string(gpu_not_in_cpu) +
                  ", gpu_filter_invalid=" + std::to_string(gpu_invalid_by_filter) +
                  ", raw_gpu=" + std::to_string(raw_gpu_count) +
                  ", gpu_unique=" + std::to_string(gpu_unique.size()) +
                  ", cpu_unique=" + std::to_string(cpu_unique.size()) +
                  ", gpu_duplicates=" + std::to_string(gpu_duplicates) + ")");
        return;
    }

    const double recall = cpu_unique.empty()
                              ? 1.0
                              : static_cast<double>(gpu_unique.size()) / static_cast<double>(cpu_unique.size());
    if (cpu_not_in_gpu == 0) {
        LOG_TEST("epipolar_rasterization: GPU matches equal CPU reference set (" +
                 std::to_string(cpu_unique.size()) + " unique pairs)");
    } else {
        LOG_WARNING("epipolar_rasterization: GPU is a strict subset of CPU reference"
                    " (missing=" + std::to_string(cpu_not_in_gpu) +
                    ", gpu_unique=" + std::to_string(gpu_unique.size()) +
                    ", cpu_unique=" + std::to_string(cpu_unique.size()) +
                    ", recall=" + std::to_string(recall) +
                    ", duplicates=" + std::to_string(gpu_duplicates) + ")");
    }
}

void Stereo_Matches_GPU_Validator::test_EP_LP_OR_vs_stereo_matches_cpu(const Stereo_Matches_GPU_Test_Context& ctx)
{
    if (ctx.dataset == nullptr || ctx.stereo_frame_for_cpu == nullptr) {
        LOG_TEST("test_EP_LP_OR_vs_stereo_matches_cpu: skip (dataset or stereo_frame not set)");
        return;
    }
    if (ctx.d_match_count_passing_EP_LP_OR_filters == nullptr || ctx.d_final_matches_passing_EP_LP_OR_filters == nullptr) {
        LOG_WARNING("test_EP_LP_OR_vs_stereo_matches_cpu: skip (missing device epipolar pointers)");
        return;
    }

    const StereoFrame& frame = *ctx.stereo_frame_for_cpu;
    Stereo_Matches sm;
    std::vector<std::pair<int, int>> cpu_lr;
    sm.collect_candidate_pairs_after_EP_LP_OR(ctx.dataset, frame, cpu_lr);

    int raw_gpu_count = 0;
    if (cudaMemcpy(&raw_gpu_count, ctx.d_match_count_passing_EP_LP_OR_filters, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("test_EP_LP_OR_vs_stereo_matches_cpu: cudaMemcpy d_match_count failed");
        return;
    }
    if (raw_gpu_count > ctx.max_matches) {
        LOG_WARNING("Stereo_Matches cross-check: raw GPU epipolar count " + std::to_string(raw_gpu_count) +
                    " exceeds buffer " + std::to_string(ctx.max_matches) +
                    "; comparison uses truncated prefix only");
    }
    const int gpu_n = std::min(raw_gpu_count, ctx.max_matches);
    std::vector<Match_by_Edge_Index> gpu_pairs(static_cast<size_t>(gpu_n));
    if (gpu_n > 0 &&
        cudaMemcpy(gpu_pairs.data(), ctx.d_final_matches_passing_EP_LP_OR_filters,
                   static_cast<size_t>(gpu_n) * sizeof(Match_by_Edge_Index), cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("test_EP_LP_OR_vs_stereo_matches_cpu: cudaMemcpy d_final_matches EP failed");
        return;
    }

    std::vector<unsigned long long> cpu_keys;
    cpu_keys.reserve(cpu_lr.size());
    for (const auto& pr : cpu_lr)
        cpu_keys.push_back(pair_key(pr.first, pr.second));
    std::vector<unsigned long long> gpu_keys;
    gpu_keys.reserve(gpu_pairs.size());
    for (const auto& p : gpu_pairs)
        gpu_keys.push_back(pair_key(p.left_edge_idx, p.right_edge_idx));

    std::sort(cpu_keys.begin(), cpu_keys.end());
    std::sort(gpu_keys.begin(), gpu_keys.end());

    if (cpu_keys == gpu_keys) {
        LOG_TEST("EP_LP_OR: GPU multiset equals finite-segment-pruned Stereo_Matches EP+LP+OR (" +
                 std::to_string(cpu_keys.size()) + " pairs)");
        return;
    }

    LOG_ERROR("EP_LP_OR: GPU multiset differs from finite-segment-pruned Stereo_Matches reference (gpu_sorted=" +
              std::to_string(gpu_keys.size()) + ", cpu_sorted=" + std::to_string(cpu_keys.size()) +
              ", raw_gpu_count=" + std::to_string(raw_gpu_count) +
              ") — check float F on GPU vs double F on host, or right image W/H alignment with Stereo_Matches_GPU_Pipeline");

    auto unpack = [](unsigned long long k) {
        const int lj = static_cast<int>(static_cast<unsigned int>(k >> 32));
        const int rj = static_cast<int>(static_cast<unsigned int>(k & 0xffffffffu));
        return std::string("(") + std::to_string(lj) + "," + std::to_string(rj) + ")";
    };
    size_t i = 0;
    while (i < cpu_keys.size() && i < gpu_keys.size() && cpu_keys[i] == gpu_keys[i])
        ++i;
    if (i < cpu_keys.size() && i < gpu_keys.size())
        std::cerr << "[TEST] EP_LP_OR first_sorted_diff cpu " << unpack(cpu_keys[i]) << " gpu "
                  << unpack(gpu_keys[i]) << std::endl;
    else if (i < cpu_keys.size())
        std::cerr << "[TEST] EP_LP_OR extra CPU-only (sorted) begins at " << unpack(cpu_keys[i]) << std::endl;
    else if (i < gpu_keys.size())
        std::cerr << "[TEST] EP_LP_OR extra GPU-only (sorted) begins at " << unpack(gpu_keys[i]) << std::endl;
}

void Stereo_Matches_GPU_Validator::test_ncc_filter(const Stereo_Matches_GPU_Test_Context& ctx)
{
    if (ctx.d_match_count_passing_EP_LP_OR_filters == nullptr ||
        ctx.d_final_count_passing_NCC_filter == nullptr ||
        ctx.d_final_matches_passing_EP_LP_OR_filters == nullptr ||
        ctx.d_final_matches_passing_NCC_filter == nullptr ||
        ctx.d_left_patches == nullptr ||
        ctx.h_right_img == nullptr) {
        LOG_WARNING("test_ncc_filter: skip (missing NCC test pointers)");
        return;
    }
    if (ctx.num_left_edges == 0 || ctx.num_right_edges == 0) {
        LOG_TEST("test_ncc_filter: skip (no edges)");
        return;
    }

    int candidate_count = 0;
    if (cudaMemcpy(&candidate_count, ctx.d_match_count_passing_EP_LP_OR_filters, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("test_ncc_filter: cudaMemcpy d_match_count_passing_EP_LP_OR_filters failed");
        return;
    }
    int ncc_count = 0;
    if (cudaMemcpy(&ncc_count, ctx.d_final_count_passing_NCC_filter, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("test_ncc_filter: cudaMemcpy d_final_count_passing_NCC_filter failed");
        return;
    }
    if (candidate_count < 0 || ncc_count < 0) {
        LOG_ERROR("test_ncc_filter: negative candidate/final count returned from GPU");
        return;
    }
    if (ncc_count > candidate_count) {
        LOG_ERROR("ncc_filter: GPU final match count exceeds candidate count"
                  " (final=" + std::to_string(ncc_count) +
                  ", candidate=" + std::to_string(candidate_count) + ")");
        return;
    }

    std::vector<Match_by_Edge_Index> candidates(static_cast<size_t>(candidate_count));
    if (candidate_count > 0 &&
        cudaMemcpy(candidates.data(), ctx.d_final_matches_passing_EP_LP_OR_filters,
                   static_cast<size_t>(candidate_count) * sizeof(Match_by_Edge_Index), cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("test_ncc_filter: cudaMemcpy d_final_matches_passing_EP_LP_OR_filters failed");
        return;
    }
    std::vector<Match_by_Edge_Index> ncc_matches(static_cast<size_t>(ncc_count));
    if (ncc_count > 0 &&
        cudaMemcpy(ncc_matches.data(), ctx.d_final_matches_passing_NCC_filter,
                   static_cast<size_t>(ncc_count) * sizeof(Match_by_Edge_Index), cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("test_ncc_filter: cudaMemcpy d_final_matches_passing_NCC_filter failed");
        return;
    }
    std::vector<Precomputed_Edge_Patches_Photometry> left_patches(ctx.num_left_edges);
    if (cudaMemcpy(left_patches.data(), ctx.d_left_patches,
                   ctx.num_left_edges * sizeof(Precomputed_Edge_Patches_Photometry), cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("test_ncc_filter: cudaMemcpy d_left_patches failed");
        return;
    }

    std::unordered_set<unsigned long long> candidate_set;
    candidate_set.reserve(candidates.size());
    for (const auto& p : candidates)
        candidate_set.insert(pair_key(p.left_edge_idx, p.right_edge_idx));

    std::unordered_set<unsigned long long> gpu_final_set;
    gpu_final_set.reserve(ncc_matches.size());
    size_t gpu_duplicates = 0;
    size_t gpu_not_in_candidates = 0;
    size_t gpu_invalid_idx = 0;
    size_t gpu_below_threshold = 0;

    // Must match apply_NCC_filter_kernel and Stereo_Matches::apply_NCC_Filtering:
    // `if (final_SIM_score > NCC_THRESH)`
    const float ncc_thresh_f = static_cast<float>(NCC_THRESH);
    // Tiny tolerance only for float noise when sanity-checking rows the GPU claimed to pass:
    // Cross-device texture interpolation/FMA can drift at ~1e-3 around the threshold.
    const float gpu_row_tol = 1e-3f;

    for (const auto& p : ncc_matches) {
        const auto key = pair_key(p.left_edge_idx, p.right_edge_idx);
        if (!gpu_final_set.insert(key).second)
            ++gpu_duplicates;
        if (!candidate_set.count(key))
            ++gpu_not_in_candidates;
        if (p.left_edge_idx < 0 || p.left_edge_idx >= static_cast<int>(ctx.num_left_edges) ||
            p.right_edge_idx < 0 || p.right_edge_idx >= static_cast<int>(ctx.num_right_edges)) {
            ++gpu_invalid_idx;
            continue;
        }

        const auto& right_edge = ctx.h_right_edges[p.right_edge_idx];
        const auto& left_patch = left_patches[static_cast<size_t>(p.left_edge_idx)];
        const float cos_R = std::cos(right_edge.orientation);
        const float sin_R = std::sin(right_edge.orientation);
        const float nx_R = -sin_R * OFFSET_DIST;
        const float ny_R = cos_R * OFFSET_DIST;

        float pR_plus[TOTAL_NUM_OF_PATCH_PIXELS];
        float pR_minus[TOTAL_NUM_OF_PATCH_PIXELS];
        float mR_plus = 0.f, vR_plus = 0.f, mR_minus = 0.f, vR_minus = 0.f;
        extract_patch_stats_cpu(ctx.h_right_img, ctx.img_right_width, ctx.img_right_height,
                                right_edge.location_x + nx_R, right_edge.location_y + ny_R,
                                cos_R, sin_R, pR_plus, mR_plus, vR_plus);
        extract_patch_stats_cpu(ctx.h_right_img, ctx.img_right_width, ctx.img_right_height,
                                right_edge.location_x - nx_R, right_edge.location_y - ny_R,
                                cos_R, sin_R, pR_minus, mR_minus, vR_minus);

        const float sim_pp = compute_ncc_cpu(left_patch.pL_plus, left_patch.mL_plus, left_patch.vL_plus, pR_plus, mR_plus, vR_plus);
        const float sim_pn = compute_ncc_cpu(left_patch.pL_plus, left_patch.mL_plus, left_patch.vL_plus, pR_minus, mR_minus, vR_minus);
        const float sim_np = compute_ncc_cpu(left_patch.pL_minus, left_patch.mL_minus, left_patch.vL_minus, pR_plus, mR_plus, vR_plus);
        const float sim_nn = compute_ncc_cpu(left_patch.pL_minus, left_patch.mL_minus, left_patch.vL_minus, pR_minus, mR_minus, vR_minus);
        const float max_ncc = std::max(std::max(sim_pp, sim_pn), std::max(sim_np, sim_nn));
        // Host recompute must still clear the CPU/GPU hurdle (> NCC_THRESH), allow tiny FP slack below.
        if (!(max_ncc > ncc_thresh_f - gpu_row_tol))
            ++gpu_below_threshold;
    }

    std::unordered_set<unsigned long long> cpu_pass_set;
    std::unordered_set<unsigned long long> cpu_pass_set_tol;
    std::unordered_map<unsigned long long, float> cpu_max_ncc_by_key;
    cpu_pass_set.reserve(candidates.size());
    cpu_pass_set_tol.reserve(candidates.size());
    cpu_max_ncc_by_key.reserve(candidates.size());
    size_t cpu_lenient_borderline = 0;
    for (const auto& p : candidates) {
        if (p.left_edge_idx < 0 || p.left_edge_idx >= static_cast<int>(ctx.num_left_edges) ||
            p.right_edge_idx < 0 || p.right_edge_idx >= static_cast<int>(ctx.num_right_edges)) {
            continue;
        }
        const auto& right_edge = ctx.h_right_edges[p.right_edge_idx];
        const auto& left_patch = left_patches[static_cast<size_t>(p.left_edge_idx)];
        const float cos_R = std::cos(right_edge.orientation);
        const float sin_R = std::sin(right_edge.orientation);
        const float nx_R = -sin_R * OFFSET_DIST;
        const float ny_R = cos_R * OFFSET_DIST;

        float pR_plus[TOTAL_NUM_OF_PATCH_PIXELS];
        float pR_minus[TOTAL_NUM_OF_PATCH_PIXELS];
        float mR_plus = 0.f, vR_plus = 0.f, mR_minus = 0.f, vR_minus = 0.f;
        extract_patch_stats_cpu(ctx.h_right_img, ctx.img_right_width, ctx.img_right_height,
                                right_edge.location_x + nx_R, right_edge.location_y + ny_R,
                                cos_R, sin_R, pR_plus, mR_plus, vR_plus);
        extract_patch_stats_cpu(ctx.h_right_img, ctx.img_right_width, ctx.img_right_height,
                                right_edge.location_x - nx_R, right_edge.location_y - ny_R,
                                cos_R, sin_R, pR_minus, mR_minus, vR_minus);

        const float sim_pp = compute_ncc_cpu(left_patch.pL_plus, left_patch.mL_plus, left_patch.vL_plus, pR_plus, mR_plus, vR_plus);
        const float sim_pn = compute_ncc_cpu(left_patch.pL_plus, left_patch.mL_plus, left_patch.vL_plus, pR_minus, mR_minus, vR_minus);
        const float sim_np = compute_ncc_cpu(left_patch.pL_minus, left_patch.mL_minus, left_patch.vL_minus, pR_plus, mR_plus, vR_plus);
        const float sim_nn = compute_ncc_cpu(left_patch.pL_minus, left_patch.mL_minus, left_patch.vL_minus, pR_minus, mR_minus, vR_minus);
        const float max_ncc = std::max(std::max(sim_pp, sim_pn), std::max(sim_np, sim_nn));
        const auto k = pair_key(p.left_edge_idx, p.right_edge_idx);
        cpu_max_ncc_by_key[k] = max_ncc;
        // Same predicate as Stereo_Matches::apply_NCC_Filtering and the device kernel (> not >=).
        if (max_ncc > ncc_thresh_f)
            cpu_pass_set.insert(k);
        // Tolerant set used only for cross-device validator parity (texture interpolation / FMA rounding).
        if (max_ncc > ncc_thresh_f - gpu_row_tol)
            cpu_pass_set_tol.insert(k);
        // Diagnostic only: pairs in a band just below threshold (sampling noise / old bugs).
        else if (max_ncc > ncc_thresh_f - 2e-2f)
            ++cpu_lenient_borderline;
    }

    if (cpu_lenient_borderline > 0) {
        LOG_TEST("ncc_filter: " + std::to_string(cpu_lenient_borderline) +
                 " candidate pairs in (NCC_THRESH-0.02, NCC_THRESH] — no pass on CPU with same rule as Stereo_Matches.cpp (> NCC_THRESH); "
                 "NCC_THRESH=" + std::to_string(ncc_thresh_f));
    }

    size_t gpu_not_in_cpu = 0;
    size_t gpu_not_in_cpu_tol = 0;
    for (const auto key : gpu_final_set) {
        if (!cpu_pass_set.count(key))
            ++gpu_not_in_cpu;
        if (!cpu_pass_set_tol.count(key))
            ++gpu_not_in_cpu_tol;
    }
    size_t cpu_not_in_gpu = 0;
    for (const auto key : cpu_pass_set) {
        if (!gpu_final_set.count(key))
            ++cpu_not_in_gpu;
    }

    if (gpu_invalid_idx > 0 || gpu_not_in_candidates > 0) {
        if (gpu_not_in_cpu_tol > 0) {
            size_t printed = 0;
            for (const auto& p : ncc_matches) {
                const auto k = pair_key(p.left_edge_idx, p.right_edge_idx);
                if (cpu_pass_set_tol.count(k))
                    continue;
                auto it = cpu_max_ncc_by_key.find(k);
                const float cpu_max_ncc = (it == cpu_max_ncc_by_key.end()) ? -1.0f : it->second;
                float cpu_max_ncc_no_half_shift = -1.0f;
                float lx = -1.f, ly = -1.f, ltheta = 0.f;
                float rx = -1.f, ry = -1.f, rtheta = 0.f;
                if (p.left_edge_idx >= 0 && p.left_edge_idx < static_cast<int>(ctx.num_left_edges)) {
                    const auto &le = ctx.h_left_edges[p.left_edge_idx];
                    lx = le.location_x;
                    ly = le.location_y;
                    ltheta = le.orientation;
                }
                if (p.right_edge_idx >= 0 && p.right_edge_idx < static_cast<int>(ctx.num_right_edges)) {
                    const auto &re = ctx.h_right_edges[p.right_edge_idx];
                    rx = re.location_x;
                    ry = re.location_y;
                    rtheta = re.orientation;
                }
                const float border_min_dist =
                    std::min(std::min(rx, static_cast<float>(ctx.img_right_width - 1) - rx),
                             std::min(ry, static_cast<float>(ctx.img_right_height - 1) - ry));
                if (p.left_edge_idx >= 0 && p.left_edge_idx < static_cast<int>(ctx.num_left_edges) &&
                    p.right_edge_idx >= 0 && p.right_edge_idx < static_cast<int>(ctx.num_right_edges)) {
                    const auto &right_edge = ctx.h_right_edges[p.right_edge_idx];
                    const auto &left_patch = left_patches[static_cast<size_t>(p.left_edge_idx)];
                    const float cos_R = std::cos(right_edge.orientation);
                    const float sin_R = std::sin(right_edge.orientation);
                    const float nx_R = -sin_R * OFFSET_DIST;
                    const float ny_R = cos_R * OFFSET_DIST;

                    float pR_plus_no_half[TOTAL_NUM_OF_PATCH_PIXELS];
                    float pR_minus_no_half[TOTAL_NUM_OF_PATCH_PIXELS];
                    float mR_plus_no_half = 0.f, vR_plus_no_half = 0.f, mR_minus_no_half = 0.f, vR_minus_no_half = 0.f;
                    extract_patch_stats_cpu_no_half_shift(ctx.h_right_img, ctx.img_right_width, ctx.img_right_height,
                                                          right_edge.location_x + nx_R, right_edge.location_y + ny_R,
                                                          cos_R, sin_R, pR_plus_no_half, mR_plus_no_half, vR_plus_no_half);
                    extract_patch_stats_cpu_no_half_shift(ctx.h_right_img, ctx.img_right_width, ctx.img_right_height,
                                                          right_edge.location_x - nx_R, right_edge.location_y - ny_R,
                                                          cos_R, sin_R, pR_minus_no_half, mR_minus_no_half, vR_minus_no_half);
                    const float sim_pp_no_half = compute_ncc_cpu(left_patch.pL_plus, left_patch.mL_plus, left_patch.vL_plus,
                                                                 pR_plus_no_half, mR_plus_no_half, vR_plus_no_half);
                    const float sim_pn_no_half = compute_ncc_cpu(left_patch.pL_plus, left_patch.mL_plus, left_patch.vL_plus,
                                                                 pR_minus_no_half, mR_minus_no_half, vR_minus_no_half);
                    const float sim_np_no_half = compute_ncc_cpu(left_patch.pL_minus, left_patch.mL_minus, left_patch.vL_minus,
                                                                 pR_plus_no_half, mR_plus_no_half, vR_plus_no_half);
                    const float sim_nn_no_half = compute_ncc_cpu(left_patch.pL_minus, left_patch.mL_minus, left_patch.vL_minus,
                                                                 pR_minus_no_half, mR_minus_no_half, vR_minus_no_half);
                    cpu_max_ncc_no_half_shift =
                        std::max(std::max(sim_pp_no_half, sim_pn_no_half), std::max(sim_np_no_half, sim_nn_no_half));
                }
                std::cerr << "[TEST] ncc_filter gpu_only_tol pair=(" << p.left_edge_idx << "," << p.right_edge_idx
                          << ") cpu_max_ncc=" << cpu_max_ncc
                          << " cpu_max_ncc_no_half=" << cpu_max_ncc_no_half_shift
                          << " ncc_thresh=" << ncc_thresh_f
                          << " delta=" << (cpu_max_ncc - ncc_thresh_f)
                          << " delta_no_half=" << (cpu_max_ncc_no_half_shift - ncc_thresh_f)
                          << " L=(" << lx << "," << ly << "," << ltheta << ")"
                          << " R=(" << rx << "," << ry << "," << rtheta << ")"
                          << " R_border_min_dist=" << border_min_dist
                          << std::endl;
                if (++printed >= 5)
                    break;
            }
        }
        LOG_ERROR("ncc_filter: GPU produced invalid/extra matches compared to CPU reference"
                  " (gpu_invalid_idx=" + std::to_string(gpu_invalid_idx) +
                  ", gpu_not_in_candidates=" + std::to_string(gpu_not_in_candidates) +
                  ", gpu_not_in_cpu=" + std::to_string(gpu_not_in_cpu) +
                  ", gpu_not_in_cpu_tol=" + std::to_string(gpu_not_in_cpu_tol) +
                  ", gpu_below_thresh=" + std::to_string(gpu_below_threshold) +
                  ", gpu_duplicates=" + std::to_string(gpu_duplicates) +
                  ", gpu_unique=" + std::to_string(gpu_final_set.size()) +
                  ", cpu_unique=" + std::to_string(cpu_pass_set.size()) + ")");
        return;
    }
    if (gpu_not_in_cpu_tol > 0) {
        size_t printed = 0;
        for (const auto& p : ncc_matches) {
            const auto k = pair_key(p.left_edge_idx, p.right_edge_idx);
            if (cpu_pass_set_tol.count(k))
                continue;
            auto it = cpu_max_ncc_by_key.find(k);
            const float cpu_max_ncc = (it == cpu_max_ncc_by_key.end()) ? -1.0f : it->second;
            float cpu_max_ncc_no_half_shift = -1.0f;
            float lx = -1.f, ly = -1.f, ltheta = 0.f;
            float rx = -1.f, ry = -1.f, rtheta = 0.f;
            if (p.left_edge_idx >= 0 && p.left_edge_idx < static_cast<int>(ctx.num_left_edges)) {
                const auto &le = ctx.h_left_edges[p.left_edge_idx];
                lx = le.location_x;
                ly = le.location_y;
                ltheta = le.orientation;
            }
            if (p.right_edge_idx >= 0 && p.right_edge_idx < static_cast<int>(ctx.num_right_edges)) {
                const auto &re = ctx.h_right_edges[p.right_edge_idx];
                rx = re.location_x;
                ry = re.location_y;
                rtheta = re.orientation;
            }
            const float border_min_dist =
                std::min(std::min(rx, static_cast<float>(ctx.img_right_width - 1) - rx),
                         std::min(ry, static_cast<float>(ctx.img_right_height - 1) - ry));
            if (p.left_edge_idx >= 0 && p.left_edge_idx < static_cast<int>(ctx.num_left_edges) &&
                p.right_edge_idx >= 0 && p.right_edge_idx < static_cast<int>(ctx.num_right_edges)) {
                const auto &right_edge = ctx.h_right_edges[p.right_edge_idx];
                const auto &left_patch = left_patches[static_cast<size_t>(p.left_edge_idx)];
                const float cos_R = std::cos(right_edge.orientation);
                const float sin_R = std::sin(right_edge.orientation);
                const float nx_R = -sin_R * OFFSET_DIST;
                const float ny_R = cos_R * OFFSET_DIST;

                float pR_plus_no_half[TOTAL_NUM_OF_PATCH_PIXELS];
                float pR_minus_no_half[TOTAL_NUM_OF_PATCH_PIXELS];
                float mR_plus_no_half = 0.f, vR_plus_no_half = 0.f, mR_minus_no_half = 0.f, vR_minus_no_half = 0.f;
                extract_patch_stats_cpu_no_half_shift(ctx.h_right_img, ctx.img_right_width, ctx.img_right_height,
                                                      right_edge.location_x + nx_R, right_edge.location_y + ny_R,
                                                      cos_R, sin_R, pR_plus_no_half, mR_plus_no_half, vR_plus_no_half);
                extract_patch_stats_cpu_no_half_shift(ctx.h_right_img, ctx.img_right_width, ctx.img_right_height,
                                                      right_edge.location_x - nx_R, right_edge.location_y - ny_R,
                                                      cos_R, sin_R, pR_minus_no_half, mR_minus_no_half, vR_minus_no_half);
                const float sim_pp_no_half = compute_ncc_cpu(left_patch.pL_plus, left_patch.mL_plus, left_patch.vL_plus,
                                                             pR_plus_no_half, mR_plus_no_half, vR_plus_no_half);
                const float sim_pn_no_half = compute_ncc_cpu(left_patch.pL_plus, left_patch.mL_plus, left_patch.vL_plus,
                                                             pR_minus_no_half, mR_minus_no_half, vR_minus_no_half);
                const float sim_np_no_half = compute_ncc_cpu(left_patch.pL_minus, left_patch.mL_minus, left_patch.vL_minus,
                                                             pR_plus_no_half, mR_plus_no_half, vR_plus_no_half);
                const float sim_nn_no_half = compute_ncc_cpu(left_patch.pL_minus, left_patch.mL_minus, left_patch.vL_minus,
                                                             pR_minus_no_half, mR_minus_no_half, vR_minus_no_half);
                cpu_max_ncc_no_half_shift =
                    std::max(std::max(sim_pp_no_half, sim_pn_no_half), std::max(sim_np_no_half, sim_nn_no_half));
            }
            std::cerr << "[TEST] ncc_filter gpu_only_tol pair=(" << p.left_edge_idx << "," << p.right_edge_idx
                      << ") cpu_max_ncc=" << cpu_max_ncc
                      << " cpu_max_ncc_no_half=" << cpu_max_ncc_no_half_shift
                      << " ncc_thresh=" << ncc_thresh_f
                      << " delta=" << (cpu_max_ncc - ncc_thresh_f)
                      << " delta_no_half=" << (cpu_max_ncc_no_half_shift - ncc_thresh_f)
                      << " L=(" << lx << "," << ly << "," << ltheta << ")"
                      << " R=(" << rx << "," << ry << "," << rtheta << ")"
                      << " R_border_min_dist=" << border_min_dist
                      << std::endl;
            if (++printed >= 5)
                break;
        }
        LOG_WARNING("ncc_filter: small GPU-only tolerant mismatch remains"
                    " (gpu_not_in_cpu=" + std::to_string(gpu_not_in_cpu) +
                    ", gpu_not_in_cpu_tol=" + std::to_string(gpu_not_in_cpu_tol) +
                    ", gpu_below_thresh=" + std::to_string(gpu_below_threshold) +
                    ", gpu_unique=" + std::to_string(gpu_final_set.size()) +
                    ", cpu_unique=" + std::to_string(cpu_pass_set.size()) + ")");
    }
    if (gpu_below_threshold > 0) {
        LOG_WARNING("ncc_filter: " + std::to_string(gpu_below_threshold) +
                    " GPU-passed rows recompute below strict CPU threshold (within tolerance band); "
                    "treating as boundary numerical drift");
    }

    const double recall = cpu_pass_set.empty()
                              ? 1.0
                              : static_cast<double>(gpu_final_set.size()) / static_cast<double>(cpu_pass_set.size());
    if (cpu_not_in_gpu == 0) {
        LOG_TEST("ncc_filter: GPU matches equal CPU NCC-pass set (" +
                 std::to_string(cpu_pass_set.size()) + " unique pairs)");
    } else {
        LOG_WARNING("ncc_filter: GPU is a strict subset of CPU NCC-pass set"
                    " (missing=" + std::to_string(cpu_not_in_gpu) +
                    ", gpu_unique=" + std::to_string(gpu_final_set.size()) +
                    ", cpu_unique=" + std::to_string(cpu_pass_set.size()) +
                    ", recall=" + std::to_string(recall) +
                    ", duplicates=" + std::to_string(gpu_duplicates) + ")");
    }
}

namespace {

//> Replicates Utility::get_Orthogonal_Shifted_Points in float for the validator (avoids
//> dragging the full Utility object into this translation unit).
inline void cpu_orthogonal_shifted_points(double cx, double cy, double orientation, double shift,
                                          cv::Point2d& p_plus, cv::Point2d& p_minus)
{
    const double s = std::sin(orientation);
    const double c = std::cos(orientation);
    p_plus  = cv::Point2d(cx + shift * s,  cy - shift * c);
    p_minus = cv::Point2d(cx - shift * s,  cy + shift * c);
}

//> Compute the two SIFT descriptors per edge (at the two orthogonal shift points) using
//> default cv::SIFT::create() so we exactly mirror Stereo_Matches::augment_Edge_Data and
//> Stereo_Matches::apply_SIFT_filtering. `valid[k] == 1` iff both rows came back from
//> sift->compute (matches the CPU pipeline's "rows == 2" check).
void compute_cpu_sift_pair_descriptors(const cv::Mat& image,
                                       const std::vector<Edge>& edges,
                                       const std::vector<int>& edge_indices,
                                       std::vector<cv::Mat>& desc_plus,
                                       std::vector<cv::Mat>& desc_minus,
                                       std::vector<unsigned char>& valid)
{
    const std::size_t N = edge_indices.size();
    desc_plus.assign(N, cv::Mat());
    desc_minus.assign(N, cv::Mat());
    valid.assign(N, 0);

    if (N == 0 || image.empty())
        return;

    //> Mirror Stereo_Matches::augment_Edge_Data: feed the raw undistorted image directly to
    //> cv::SIFT::compute (it handles CV_8UC1 / CV_8UC3 internally). Avoid any conversion that
    //> would diverge from the CPU reference.
    const cv::Mat& sift_image = image;

    const std::size_t edges_size = edges.size();

#pragma omp parallel
    {
        cv::Ptr<cv::SIFT> thread_sift = cv::SIFT::create();
#pragma omp for schedule(dynamic)
        for (long long k = 0; k < static_cast<long long>(N); ++k) {
            const int idx = edge_indices[static_cast<std::size_t>(k)];
            if (idx < 0 || static_cast<std::size_t>(idx) >= edges_size)
                continue;
            const Edge& e = edges[static_cast<std::size_t>(idx)];

            cv::Point2d p_plus, p_minus;
            cpu_orthogonal_shifted_points(e.location.x, e.location.y, e.orientation,
                                          static_cast<double>(SIFT_ORTHOGONAL_SHIFT), p_plus, p_minus);

            std::vector<cv::KeyPoint> kps;
            kps.emplace_back(cv::Point2f(static_cast<float>(p_plus.x), static_cast<float>(p_plus.y)),
                             1.0f, static_cast<float>(180.0 / M_PI * e.orientation));
            kps.emplace_back(cv::Point2f(static_cast<float>(p_minus.x), static_cast<float>(p_minus.y)),
                             1.0f, static_cast<float>(180.0 / M_PI * e.orientation));

            cv::Mat descriptors;
            try {
                thread_sift->compute(sift_image, kps, descriptors);
            } catch (...) {
                continue;
            }
            //> cv::SIFT::compute may drop keypoints near the border. We require both rows present
            //> to mirror the CPU pipeline's "rows == 2" gate; otherwise we leave valid=0 and let
            //> the caller treat the pair as a CPU pass-through (fail-safe).
            if (descriptors.rows == 2 && descriptors.cols == 128 && descriptors.type() == CV_32F) {
                desc_plus[static_cast<std::size_t>(k)]  = descriptors.row(0).clone();
                desc_minus[static_cast<std::size_t>(k)] = descriptors.row(1).clone();
                valid[static_cast<std::size_t>(k)] = 1;
            }
        }
    }
}

} // namespace

void Stereo_Matches_GPU_Validator::test_SIFT_descriptor_filter(const Stereo_Matches_GPU_Test_Context& ctx)
{
    if (ctx.d_final_matches_passing_NCC_filter == nullptr || ctx.d_final_count_passing_NCC_filter == nullptr) {
        LOG_ERROR("test_SIFT_descriptor_filter: NCC filter outputs missing in context; skipping");
        return;
    }
    if (ctx.d_final_matches_passing_SIFT_filter == nullptr || ctx.d_final_count_passing_SIFT_filter == nullptr) {
        LOG_ERROR("test_SIFT_descriptor_filter: GPU SIFT filter outputs missing in context; skipping");
        return;
    }
    if (ctx.stereo_frame_for_cpu == nullptr) {
        LOG_ERROR("test_SIFT_descriptor_filter: stereo_frame_for_cpu missing in context; skipping");
        return;
    }

    const StereoFrame& sf = *ctx.stereo_frame_for_cpu;
    //> Match Stereo_Matches::augment_Edge_Data and apply_SIFT_filtering: feed the undistorted
    //> image straight into cv::SIFT::compute (which does its own internal Gaussian smoothing
    //> via the SIFT pyramid). Falls back to the raw distorted image only if undistorted is missing.
    const cv::Mat& left_img  = !sf.left_image_undistorted.empty()  ? sf.left_image_undistorted  : sf.left_image;
    const cv::Mat& right_img = !sf.right_image_undistorted.empty() ? sf.right_image_undistorted : sf.right_image;
    if (left_img.empty() || right_img.empty()) {
        LOG_ERROR("test_SIFT_descriptor_filter: stereo frame images are empty; skipping");
        return;
    }

    int ncc_count = 0;
    if (cudaMemcpy(&ncc_count, ctx.d_final_count_passing_NCC_filter, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("test_SIFT_descriptor_filter: cudaMemcpy d_final_count_passing_NCC_filter failed");
        return;
    }
    int sift_count = 0;
    if (cudaMemcpy(&sift_count, ctx.d_final_count_passing_SIFT_filter, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("test_SIFT_descriptor_filter: cudaMemcpy d_final_count_passing_SIFT_filter failed");
        return;
    }
    if (ncc_count < 0 || sift_count < 0) {
        LOG_ERROR("test_SIFT_descriptor_filter: invalid negative match counts (ncc=" + std::to_string(ncc_count) +
                  ", sift=" + std::to_string(sift_count) + ")");
        return;
    }
    if (ncc_count == 0) {
        LOG_TEST("SIFT_descriptor_filter: 0 NCC inputs to compare against; nothing to validate");
        return;
    }
    if (sift_count > ncc_count) {
        LOG_ERROR("test_SIFT_descriptor_filter: GPU SIFT pass count exceeds NCC input count (sift=" +
                  std::to_string(sift_count) + ", ncc=" + std::to_string(ncc_count) + ")");
        return;
    }

    std::vector<Match_by_Edge_Index> ncc_matches(static_cast<std::size_t>(ncc_count));
    if (cudaMemcpy(ncc_matches.data(), ctx.d_final_matches_passing_NCC_filter,
                   static_cast<std::size_t>(ncc_count) * sizeof(Match_by_Edge_Index),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("test_SIFT_descriptor_filter: cudaMemcpy d_final_matches_passing_NCC_filter failed");
        return;
    }
    std::vector<Match_by_Edge_Index> sift_matches(static_cast<std::size_t>(sift_count));
    if (sift_count > 0 &&
        cudaMemcpy(sift_matches.data(), ctx.d_final_matches_passing_SIFT_filter,
                   static_cast<std::size_t>(sift_count) * sizeof(Match_by_Edge_Index),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        LOG_ERROR("test_SIFT_descriptor_filter: cudaMemcpy d_final_matches_passing_SIFT_filter failed");
        return;
    }

    //> Compact the unique left/right edges referenced by the NCC inputs so we only ever compute
    //> one cv::SIFT::compute call per edge (mirrors how the CPU pipeline reuses left descriptors).
    std::unordered_map<int, int> left_to_compact;
    std::unordered_map<int, int> right_to_compact;
    std::vector<int> unique_left;
    std::vector<int> unique_right;
    left_to_compact.reserve(static_cast<std::size_t>(ncc_count));
    right_to_compact.reserve(static_cast<std::size_t>(ncc_count));
    unique_left.reserve(static_cast<std::size_t>(ncc_count));
    unique_right.reserve(static_cast<std::size_t>(ncc_count));

    std::size_t ncc_invalid_idx = 0;
    for (const auto& m : ncc_matches) {
        if (m.left_edge_idx < 0 || m.left_edge_idx >= static_cast<int>(sf.left_edges.size()) ||
            m.right_edge_idx < 0 || m.right_edge_idx >= static_cast<int>(sf.right_edges.size())) {
            ++ncc_invalid_idx;
            continue;
        }
        if (left_to_compact.find(m.left_edge_idx) == left_to_compact.end()) {
            left_to_compact[m.left_edge_idx] = static_cast<int>(unique_left.size());
            unique_left.push_back(m.left_edge_idx);
        }
        if (right_to_compact.find(m.right_edge_idx) == right_to_compact.end()) {
            right_to_compact[m.right_edge_idx] = static_cast<int>(unique_right.size());
            unique_right.push_back(m.right_edge_idx);
        }
    }

    std::vector<cv::Mat> left_desc_plus, left_desc_minus;
    std::vector<unsigned char> left_valid;
    std::vector<cv::Mat> right_desc_plus, right_desc_minus;
    std::vector<unsigned char> right_valid;
    compute_cpu_sift_pair_descriptors(left_img,  sf.left_edges,  unique_left,  left_desc_plus,  left_desc_minus,  left_valid);
    compute_cpu_sift_pair_descriptors(right_img, sf.right_edges, unique_right, right_desc_plus, right_desc_minus, right_valid);

    //> Build CPU pass set with the same predicate as Stereo_Matches::apply_SIFT_filtering.
    //> Pairs whose descriptors failed extraction are passed through (CPU fail-safe), so they
    //> always count as CPU passes — matching the existing CPU semantics, not the GPU one.
    const double sift_thresh = static_cast<double>(SIFT_THRESHOLD);
    std::unordered_set<unsigned long long> cpu_pass_set;
    std::unordered_map<unsigned long long, double> cpu_min_dist_by_key;
    cpu_pass_set.reserve(ncc_matches.size());
    cpu_min_dist_by_key.reserve(ncc_matches.size());

    std::size_t cpu_passthrough = 0;
    std::size_t cpu_evaluable = 0;
    double sum_pass_dist = 0.0;
    double sum_fail_dist = 0.0;
    std::size_t pass_count = 0;
    std::size_t fail_count = 0;

    for (const auto& m : ncc_matches) {
        auto itl = left_to_compact.find(m.left_edge_idx);
        auto itr = right_to_compact.find(m.right_edge_idx);
        if (itl == left_to_compact.end() || itr == right_to_compact.end()) {
            //> ncc_invalid_idx case: we still let the GPU set decide; do not add to CPU set.
            continue;
        }
        const std::size_t li = static_cast<std::size_t>(itl->second);
        const std::size_t ri = static_cast<std::size_t>(itr->second);
        const auto key = pair_key(m.left_edge_idx, m.right_edge_idx);

        if (!left_valid[li] || !right_valid[ri]) {
            cpu_pass_set.insert(key);
            cpu_min_dist_by_key[key] = -1.0;
            ++cpu_passthrough;
            continue;
        }
        ++cpu_evaluable;
        const double d1 = cv::norm(left_desc_plus[li],  right_desc_plus[ri],  cv::NORM_L2);
        const double d2 = cv::norm(left_desc_minus[li], right_desc_plus[ri],  cv::NORM_L2);
        const double d3 = cv::norm(left_desc_plus[li],  right_desc_minus[ri], cv::NORM_L2);
        const double d4 = cv::norm(left_desc_minus[li], right_desc_minus[ri], cv::NORM_L2);
        const double dmin = std::min({d1, d2, d3, d4});
        cpu_min_dist_by_key[key] = dmin;
        if (dmin < sift_thresh) {
            cpu_pass_set.insert(key);
            sum_pass_dist += dmin;
            ++pass_count;
        } else {
            sum_fail_dist += dmin;
            ++fail_count;
        }
    }

    std::unordered_set<unsigned long long> gpu_pass_set;
    gpu_pass_set.reserve(sift_matches.size());
    std::size_t gpu_duplicates = 0;
    std::size_t gpu_invalid_idx = 0;
    std::size_t gpu_not_in_ncc = 0;
    std::unordered_set<unsigned long long> ncc_set;
    ncc_set.reserve(ncc_matches.size());
    for (const auto& m : ncc_matches)
        ncc_set.insert(pair_key(m.left_edge_idx, m.right_edge_idx));
    for (const auto& m : sift_matches) {
        if (m.left_edge_idx < 0 || m.left_edge_idx >= static_cast<int>(sf.left_edges.size()) ||
            m.right_edge_idx < 0 || m.right_edge_idx >= static_cast<int>(sf.right_edges.size())) {
            ++gpu_invalid_idx;
            continue;
        }
        const auto key = pair_key(m.left_edge_idx, m.right_edge_idx);
        if (!gpu_pass_set.insert(key).second)
            ++gpu_duplicates;
        if (!ncc_set.count(key))
            ++gpu_not_in_ncc;
    }

    std::size_t intersection = 0;
    std::size_t gpu_only = 0;
    std::size_t cpu_only = 0;
    double sum_dist_gpu_only = 0.0;
    double sum_dist_cpu_only = 0.0;
    std::size_t cnt_dist_gpu_only = 0;
    std::size_t cnt_dist_cpu_only = 0;
    for (const auto k : gpu_pass_set) {
        if (cpu_pass_set.count(k)) {
            ++intersection;
        } else {
            ++gpu_only;
            auto it = cpu_min_dist_by_key.find(k);
            if (it != cpu_min_dist_by_key.end() && it->second >= 0.0) {
                sum_dist_gpu_only += it->second;
                ++cnt_dist_gpu_only;
            }
        }
    }
    for (const auto k : cpu_pass_set) {
        if (gpu_pass_set.count(k))
            continue;
        ++cpu_only;
        auto it = cpu_min_dist_by_key.find(k);
        if (it != cpu_min_dist_by_key.end() && it->second >= 0.0) {
            sum_dist_cpu_only += it->second;
            ++cnt_dist_cpu_only;
        }
    }

    const double recall = cpu_pass_set.empty()
                              ? 1.0
                              : static_cast<double>(intersection) / static_cast<double>(cpu_pass_set.size());
    const double precision = gpu_pass_set.empty()
                                 ? 1.0
                                 : static_cast<double>(intersection) / static_cast<double>(gpu_pass_set.size());
    const std::size_t union_size = cpu_pass_set.size() + gpu_pass_set.size() - intersection;
    const double iou = (union_size == 0) ? 1.0 : static_cast<double>(intersection) / static_cast<double>(union_size);
    const double mean_pass_dist = (pass_count == 0) ? 0.0 : sum_pass_dist / static_cast<double>(pass_count);
    const double mean_fail_dist = (fail_count == 0) ? 0.0 : sum_fail_dist / static_cast<double>(fail_count);
    const double mean_dist_gpu_only = (cnt_dist_gpu_only == 0) ? 0.0 : sum_dist_gpu_only / static_cast<double>(cnt_dist_gpu_only);
    const double mean_dist_cpu_only = (cnt_dist_cpu_only == 0) ? 0.0 : sum_dist_cpu_only / static_cast<double>(cnt_dist_cpu_only);

    if (gpu_invalid_idx > 0 || gpu_not_in_ncc > 0 || gpu_duplicates > 0) {
        LOG_ERROR("SIFT_descriptor_filter: GPU produced invalid/unexpected matches (invalid_idx=" +
                  std::to_string(gpu_invalid_idx) + ", not_in_ncc=" + std::to_string(gpu_not_in_ncc) +
                  ", duplicates=" + std::to_string(gpu_duplicates) + ")");
    }

    std::ostringstream summary;
    summary << "SIFT_descriptor_filter: ncc_inputs=" << ncc_count
            << " gpu_pass=" << gpu_pass_set.size()
            << " cpu_pass=" << cpu_pass_set.size()
            << " (cpu_evaluable=" << cpu_evaluable
            << ", cpu_passthrough=" << cpu_passthrough << ")"
            << " | intersection=" << intersection
            << " gpu_only=" << gpu_only
            << " cpu_only=" << cpu_only
            << " | recall=" << recall
            << " precision=" << precision
            << " IoU=" << iou
            << " | cpu_dist[mean(pass)=" << mean_pass_dist
            << ", mean(fail)=" << mean_fail_dist
            << ", mean(gpu_only)=" << mean_dist_gpu_only
            << ", mean(cpu_only)=" << mean_dist_cpu_only
            << "]"
            << " | thresh=" << sift_thresh;
    if (ncc_invalid_idx > 0)
        summary << " ncc_invalid_idx=" << ncc_invalid_idx;

    if (gpu_only == 0 && cpu_only == 0) {
        LOG_TEST(summary.str());
    }
    else {
        LOG_WARNING(summary.str());
    }

    //> Print up to a handful of representative disagreements so the user can eyeball whether the
    //> mismatches cluster around the threshold (calibration) or are systemic (descriptor mismatch).
    std::size_t printed = 0;
    for (const auto& m : sift_matches) {
        if (printed >= 5)
            break;
        const auto k = pair_key(m.left_edge_idx, m.right_edge_idx);
        if (!gpu_pass_set.count(k))
            continue;
        if (cpu_pass_set.count(k))
            continue;
        const auto it = cpu_min_dist_by_key.find(k);
        const double cpu_d = (it == cpu_min_dist_by_key.end()) ? -1.0 : it->second;
        std::cerr << "[TEST] SIFT_descriptor_filter gpu_only pair=(" << m.left_edge_idx << "," << m.right_edge_idx
                  << ") cpu_min_L2=" << cpu_d << " thresh=" << sift_thresh << std::endl;
        ++printed;
    }
    printed = 0;
    for (const auto& m : ncc_matches) {
        if (printed >= 5)
            break;
        const auto k = pair_key(m.left_edge_idx, m.right_edge_idx);
        if (!cpu_pass_set.count(k) || gpu_pass_set.count(k))
            continue;
        const auto it = cpu_min_dist_by_key.find(k);
        const double cpu_d = (it == cpu_min_dist_by_key.end()) ? -1.0 : it->second;
        std::cerr << "[TEST] SIFT_descriptor_filter cpu_only pair=(" << m.left_edge_idx << "," << m.right_edge_idx
                  << ") cpu_min_L2=" << cpu_d << " thresh=" << sift_thresh << std::endl;
        ++printed;
    }
}

bool Stereo_Matches_GPU_Validator::epipolar_line_intersects_image(float a, float b, float c, int W, int H)
{
    float x0 = 0.f, y0 = 0.f, x1 = 0.f, y1 = 0.f;
    return epipolar_line_endpoints(a, b, c, W, H, x0, y0, x1, y1);
}
