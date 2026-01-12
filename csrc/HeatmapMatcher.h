#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

class HeatmapMatcher {
public:
    int H, W;
    HeatmapMatcher(int height, int width);

    py::tuple evaluate_sequence(
        py::array_t<float> dt_lines, 
        py::array_t<float> gt_lines
    );
};