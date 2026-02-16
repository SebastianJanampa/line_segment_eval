#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::tuple postprocess_cpp(
    py::array_t<float> lines, 
    py::array_t<float> scores, 
    float threshold, 
    float tol, 
    bool do_clip
);
