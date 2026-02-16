#include "HeatmapMatcher.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <utility>
#include <cstdint> // Added for uint8_t

// OPTIMIZATION 3: Hyper-fast float-to-int rounding cast
inline int fast_round(float val) {
    return static_cast<int>(val + 0.5f);
}

inline void get_pixels_linspace(float y1, float x1, float y2, float x2, int H, int W, float offset, std::vector<std::pair<int, int>>& pixels) {
    pixels.clear(); 
    
    float dist = std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    int vn = std::ceil(dist);

    if (vn <= 1) {
        int xx = std::max(0, std::min(fast_round(x1 + offset) - 1, W - 1));
        int yy = std::max(0, std::min(fast_round(y1 + offset) - 1, H - 1));
        pixels.push_back({yy, xx});
        return;
    }

    float dx = x2 - x1;
    float dy = y2 - y1;
    float step = 1.0f / (vn - 1);

    for (int i = 0; i < vn; ++i) {
        float t = i * step; // Avoid division in the loop
        int xx = std::max(0, std::min(fast_round((x1 + t * dx) + offset) - 1, W - 1)); 
        int yy = std::max(0, std::min(fast_round((y1 + t * dy) + offset) - 1, H - 1));
        pixels.push_back({yy, xx});
    }
}

HeatmapMatcher::HeatmapMatcher() {}

py::tuple HeatmapMatcher::evaluate_sequence(
    py::array_t<float> dt_lines, 
    py::array_t<float> gt_lines,
    int H,
    int W
) {
    auto r_dt = dt_lines.unchecked<3>();
    auto r_gt = gt_lines.unchecked<3>();
    ssize_t N_dt = r_dt.shape(0);
    ssize_t N_gt = r_gt.shape(0);

    float tol = 0.01f * std::sqrt(H * H + W * W);
    int window = std::ceil(tol);
    
    int stride = W + 2 * window;
    int padded_size = (H + 2 * window) * stride;

    struct Offset { int dy, dx; float dist; };
    std::vector<Offset> offsets;
    for(int dy = -window; dy <= window; ++dy) {
        for(int dx = -window; dx <= window; ++dx) {
            float d = std::sqrt(dx * dx + dy * dy);
            if (d <= tol) offsets.push_back({dy, dx, d});
        }
    }
    std::sort(offsets.begin(), offsets.end(), [](const Offset& a, const Offset& b) {
        return a.dist < b.dist;
    });

    std::vector<int> flat_offsets;
    flat_offsets.reserve(offsets.size());
    for (const auto& off : offsets) {
        flat_offsets.push_back(off.dy * stride + off.dx);
    }

    // OPTIMIZATION 4: Switch from bit-packed bools to byte-addressable uint8_t
    std::vector<uint8_t> gt_mask(padded_size, 0);
    std::vector<uint8_t> gt_matched(padded_size, 0);
    long total_gt_pixels = 0;

    std::vector<std::pair<int, int>> pixels;
    pixels.reserve(2000); 

    for (ssize_t i = 0; i < N_gt; ++i) {
        get_pixels_linspace(r_gt(i,0,0), r_gt(i,0,1), r_gt(i,1,0), r_gt(i,1,1), H, W, 0.0f, pixels);
        for (auto& p : pixels) {
            int p_idx = (p.first + window) * stride + (p.second + window); 
            if (!gt_mask[p_idx]) {
                gt_mask[p_idx] = 1;
                total_gt_pixels++;
            }
        }
    }

    std::vector<uint8_t> dt_claimed(H * W, 0);
    auto tp_counts = py::array_t<long>(N_dt);
    auto fp_counts = py::array_t<long>(N_dt);
    auto ptr_tp = tp_counts.mutable_unchecked<1>();
    auto ptr_fp = fp_counts.mutable_unchecked<1>();

    for (ssize_t i = 0; i < N_dt; ++i) {
        get_pixels_linspace(r_dt(i,0,0), r_dt(i,0,1), r_dt(i,1,0), r_dt(i,1,1), H, W, -0.5f, pixels);
        long step_tp = 0, step_fp = 0;

        for (auto& p : pixels) {
            int yy = p.first, xx = p.second;
            int dt_idx = yy * W + xx; 
            
            if (!dt_claimed[dt_idx]) {
                dt_claimed[dt_idx] = 1;
                bool matched = false;

                int center_idx = (yy + window) * stride + (xx + window);

                for (int flat_off : flat_offsets) {
                    int n_idx = center_idx + flat_off;
                    if (gt_mask[n_idx] && !gt_matched[n_idx]) {
                        gt_matched[n_idx] = 1; 
                        matched = true;
                        break; 
                    }
                }

                if (matched) step_tp++;
                else step_fp++;
            }
        }
        ptr_tp(i) = step_tp;
        ptr_fp(i) = step_fp;
    }

    return py::make_tuple(tp_counts, fp_counts, total_gt_pixels);
}
