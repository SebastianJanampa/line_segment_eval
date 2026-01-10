#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace py = pybind11;

// --------------------------------------------------------------------------
// METRIC: Line Distance
// --------------------------------------------------------------------------
inline float compute_line_distance(const float* p, const float* g) {
    // Direct match (p1->g1, p2->g2)
    float d_direct = std::pow(p[0]-g[0], 2) + std::pow(p[1]-g[1], 2) +
                     std::pow(p[2]-g[2], 2) + std::pow(p[3]-g[3], 2);

    // Flipped match (p1->g2, p2->g1)
    float d_flipped = std::pow(p[0]-g[2], 2) + std::pow(p[1]-g[3], 2) +
                      std::pow(p[2]-g[0], 2) + std::pow(p[3]-g[1], 2);

    return std::min(d_direct, d_flipped);
}

// --------------------------------------------------------------------------
// ENGINE: Threshold Agnostic
// --------------------------------------------------------------------------
class LineMatcher {
public:
    LineMatcher() {}

    // FIX: We accept 'thresholds' as an argument.
    // This removes the need for the static class member that caused your link error.
    py::dict match_lines(
        py::array_t<float> dt_lines,
        py::array_t<float> gt_lines,
        std::vector<float> thresholds
    ) {
        auto r_dt = dt_lines.unchecked<3>();
        auto r_gt = gt_lines.unchecked<3>();

        ssize_t N = r_dt.shape(0);
        ssize_t M = r_gt.shape(0);

        py::dict results;

        // Loop over the thresholds provided by Python
        for (float thresh : thresholds) {

            // Prepare Output Arrays
            auto tp = py::array_t<int>(N);
            auto fp = py::array_t<int>(N);

            // Zero Initialization
            std::memset(tp.mutable_data(), 0, N * sizeof(int));
            std::memset(fp.mutable_data(), 0, N * sizeof(int));

            auto ptr_tp = tp.mutable_unchecked<1>();
            auto ptr_fp = fp.mutable_unchecked<1>();

            std::vector<bool> gt_used(M, false);

            // Greedy Matching Loop
            for (ssize_t i = 0; i < N; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                ssize_t best_gt_idx = -1;
                const float* p_line = r_dt.data(i, 0, 0);

                for (ssize_t j = 0; j < M; ++j) {
                    const float* g_line = r_gt.data(j, 0, 0);
                    float dist = compute_line_distance(p_line, g_line);

                    if (dist < min_dist) {
                        min_dist = dist;
                        best_gt_idx = j;
                    }
                }

                if (min_dist <= thresh && best_gt_idx != -1 && !gt_used[best_gt_idx]) {
                    ptr_tp(i) = 1;
                    gt_used[best_gt_idx] = true;
                } else {
                    ptr_fp(i) = 1;
                }
            }

            // Key is the threshold as a string (e.g., "5.0")
            // converting float to int string to match Python keys "5", "10"
            std::string key = std::to_string((int)thresh);
            results[key.c_str()] = py::make_tuple(tp, fp);
        }
        return results;
    }
};

PYBIND11_MODULE(_C, m) {
    py::class_<LineMatcher>(m, "LineMatcher")
        .def(py::init<>())
        .def("match_lines", &LineMatcher::match_lines);
}