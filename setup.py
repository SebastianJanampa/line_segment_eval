from setuptools import setup, Extension, find_packages
import sys


# Delayed imports to ensure build-system requirements are installed first
class BuildExt(object):
    def __init__(self):
        self.extensions = []


def get_ext_modules():
    import pybind11
    import numpy as np

    # Define the Extension
    ext_modules = [
        Extension(
            "line_seg_eval._C",  # The import name in Python
            ["csrc/_line_seg_eval.cpp"],  # Source file
            include_dirs=[
                pybind11.get_include(),
                np.get_include()
            ],
            extra_compile_args=["-O3", "-Wall", "-std=c++14"],
            language="c++",
        ),
    ]
    return ext_modules


# We wrap the extension generation to handle the 'setup' call gracefully
try:
    setup(
        name="line_seg_eval",
        version="0.1.0",
        packages=find_packages(),
        ext_modules=get_ext_modules(),
        zip_safe=False,
    )
except ImportError:
    # Fallback if dependencies aren't present during a non-build call
    print("Build dependencies missing. Assuming this is a metadata-only call.")
    setup(
        name="faster_line_eval",
        packages=find_packages(),
    )