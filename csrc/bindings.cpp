#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "LineMatcher.h"
#include "HeatmapMatcher.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    py::class_<LineMatcher>(m, "LineMatcher")
        .def(py::init<>())
        .def("match_lines", &LineMatcher::match_lines);

    py::class_<HeatmapMatcher>(m, "HeatmapMatcher")
        .def(py::init<int, int>())
        .def("evaluate_sequence", &HeatmapMatcher::evaluate_sequence);
}