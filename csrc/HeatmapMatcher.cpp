#include "HeatmapMatcher.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <utility>

// Helper: Bresenham Line Algorithm
std::vector<std::pair<int, int>> get_pixels(int x0, int y0, int x1, int y1, int H, int W) {
    std::vector<std::pair<int, int>> pixels;
    int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1, sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    while (true) {
        if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H) {
            pixels.push_back({y0, x0});
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx)  { err += dx; y0 += sy; }
    }
    return pixels;
}

HeatmapMatcher::HeatmapMatcher(int height, int width) : H(height), W(width) {}

py::tuple HeatmapMatcher::evaluate_sequence(
    py::array_t<float> dt_lines, 
    py::array_t<float> gt_lines
) {
    auto r_dt = dt_lines.unchecked<3>();
    auto r_gt = gt_lines.unchecked<3>();
    ssize_t N_dt = r_dt.shape(0);
    ssize_t N_gt = r_gt.shape(0);

    // 1. Rasterize GT Mask (for this specific image)
    std::vector<bool> gt_mask(H * W, false);
    long total_gt_pixels = 0;

    for (ssize_t i = 0; i < N_gt; ++i) {
        // FIX: Access values using (), not .data()
        auto pixels = get_pixels((int)r_gt(i,0,0), (int)r_gt(i,0,1),
                                 (int)r_gt(i,1,0), (int)r_gt(i,1,1), H, W);
        for (auto& p : pixels) {
            int idx = p.first * W + p.second;
            if (!gt_mask[idx]) {
                gt_mask[idx] = true;
                total_gt_pixels++;
            }
        }
    }

    // 2. Rasterize DT Incrementally
    std::vector<bool> dt_claimed(H * W, false);
    
    // We return arrays of "newly added" TP/FP pixels for each line
    auto tp_counts = py::array_t<long>(N_dt);
    auto fp_counts = py::array_t<long>(N_dt);
    
    auto ptr_tp = tp_counts.mutable_unchecked<1>();
    auto ptr_fp = fp_counts.mutable_unchecked<1>();

    for (ssize_t i = 0; i < N_dt; ++i) {
        auto pixels = get_pixels((int)r_dt(i,0,0), (int)r_dt(i,0,1),
                                 (int)r_dt(i,1,0), (int)r_dt(i,1,1), H, W);
        
        long step_tp = 0;
        long step_fp = 0;

        for (auto& p : pixels) {
            int idx = p.first * W + p.second;
            
            // Only count if this pixel hasn't been claimed by a higher-score line
            if (!dt_claimed[idx]) {
                dt_claimed[idx] = true;
                if (gt_mask[idx]) {
                    step_tp++;
                } else {
                    step_fp++;
                }
            }
        }
        ptr_tp(i) = step_tp;
        ptr_fp(i) = step_fp;
    }

    return py::make_tuple(tp_counts, fp_counts, total_gt_pixels);
}