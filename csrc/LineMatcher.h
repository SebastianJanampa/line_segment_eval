#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

class LineMatcher {
public:
    LineMatcher();
    
    py::dict match_lines(
        py::array_t<float> dt_lines, 
        py::array_t<float> gt_lines,
        py::array_t<int> dt_labels,
        py::array_t<int> gt_labels,
        std::vector<float> thresholds 
    );
};