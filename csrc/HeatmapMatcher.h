#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

class HeatmapMatcher {
public:
    HeatmapMatcher();
    py::tuple evaluate_sequence(
        py::array_t<float> dt_lines, 
        py::array_t<float> gt_lines,
        int H, 
        int W
    );
};
