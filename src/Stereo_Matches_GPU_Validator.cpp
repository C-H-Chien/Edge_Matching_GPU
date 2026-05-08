#include "Stereo_Matches_GPU_Validator.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

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
            if (dist < static_cast<float>(EPIPOLAR_LINE_DIST_THRESH) && disparity_sq < max_disp_sq)
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
              t >= 0.f && t <= 1.f))
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

bool Stereo_Matches_GPU_Validator::epipolar_line_intersects_image(float a, float b, float c, int W, int H)
{
    float x0 = 0.f, y0 = 0.f, x1 = 0.f, y1 = 0.f;
    return epipolar_line_endpoints(a, b, c, W, H, x0, y0, x1, y1);
}
