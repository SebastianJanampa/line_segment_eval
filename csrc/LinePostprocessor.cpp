#include "LinePostprocessor.h"
#include <vector>
#include <algorithm>
#include <cmath>

// Helper 1: pline
inline float pline(float x1, float y1, float x2, float y2, float x, float y) {
    float px = x2 - x1;
    float py = y2 - y1;
    float dd = px * px + py * py;
    float den = std::max(1e-9f, dd);
    float u = ((x - x1) * px + (y - y1) * py) / den;
    float dx = x1 + u * px - x;
    float dy = y1 + u * py - y;
    return dx * dx + dy * dy;
}

// Helper 2: plambda
inline float plambda(float x1, float y1, float x2, float y2, float x, float y) {
    float px = x2 - x1;
    float py = y2 - y1;
    float dd = px * px + py * py;
    float den = std::max(1e-9f, dd);
    return ((x - x1) * px + (y - y1) * py) / den;
}

py::tuple postprocess_cpp(py::array_t<float> lines, py::array_t<float> scores, float threshold, float tol, bool do_clip) {
    auto r_lines = lines.unchecked<3>();
    auto r_scores = scores.unchecked<1>();
    ssize_t N = r_lines.shape(0);

    struct Line { float x1, y1, x2, y2; };
    std::vector<Line> nlines;
    std::vector<float> nscores;

    float thresh_sq = threshold * threshold;

    for (ssize_t i = 0; i < N; ++i) {
        float p_x = r_lines(i, 0, 0), p_y = r_lines(i, 0, 1);
        float q_x = r_lines(i, 1, 0), q_y = r_lines(i, 1, 1);
        float score = r_scores(i);

        float start = 0.0f;
        float end = 1.0f;

        for (const auto& ab : nlines) {
            float a_x = ab.x1, a_y = ab.y1;
            float b_x = ab.x2, b_y = ab.y2;

            float max_pq_to_ab = std::max(pline(a_x, a_y, b_x, b_y, p_x, p_y),
                                          pline(a_x, a_y, b_x, b_y, q_x, q_y));
            float max_ab_to_pq = std::max(pline(p_x, p_y, q_x, q_y, a_x, a_y),
                                          pline(p_x, p_y, q_x, q_y, b_x, b_y));

            if (std::min(max_pq_to_ab, max_ab_to_pq) > thresh_sq) continue;

            float lambda_a = plambda(p_x, p_y, q_x, q_y, a_x, a_y);
            float lambda_b = plambda(p_x, p_y, q_x, q_y, b_x, b_y);

            if (lambda_a > lambda_b) std::swap(lambda_a, lambda_b);
            
            lambda_a -= tol;
            lambda_b += tol;

            // case 1: skip (if not do_clip)
            if (!do_clip && start < lambda_a && lambda_b < end) continue;

            // not intersect
            if (lambda_b < start || lambda_a > end) continue;

            // cover
            if (lambda_a <= start && end <= lambda_b) {
                start = 10.0f; 
                break;
            }

            // case 2 & 3:
            if (lambda_a <= start && start <= lambda_b) start = lambda_b;
            if (lambda_a <= end && end <= lambda_b) end = lambda_a;

            if (start >= end) break;
        }

        if (start >= end) continue;

        Line clipped;
        clipped.x1 = p_x + (q_x - p_x) * start;
        clipped.y1 = p_y + (q_y - p_y) * start;
        clipped.x2 = p_x + (q_x - p_x) * end;
        clipped.y2 = p_y + (q_y - p_y) * end;

        nlines.push_back(clipped);
        nscores.push_back(score);
    }

    // Prepare outputs back to NumPy arrays
    ssize_t out_N = nlines.size();
    auto out_lines = py::array_t<float>({out_N, (ssize_t)2, (ssize_t)2});
    auto out_scores = py::array_t<float>(out_N);

    auto ptr_lines = out_lines.mutable_unchecked<3>();
    auto ptr_scores = out_scores.mutable_unchecked<1>();

    for (ssize_t i = 0; i < out_N; ++i) {
        ptr_lines(i, 0, 0) = nlines[i].x1;
        ptr_lines(i, 0, 1) = nlines[i].y1;
        ptr_lines(i, 1, 0) = nlines[i].x2;
        ptr_lines(i, 1, 1) = nlines[i].y2;
        ptr_scores(i) = nscores[i];
    }

    return py::make_tuple(out_lines, out_scores);
}
