import os
import sys

from setuptools import setup, Extension, find_packages


# Delayed imports to ensure build-system requirements are installed first
class BuildExt(object):
    def __init__(self):
        self.extensions = []

# Read the contents of your README file
def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content

def get_ext_modules():
    import pybind11
    import numpy as np

    # Define the Extension
    ext_modules = [
        Extension(
            "line_seg_eval._C",
            [
                "csrc/bindings.cpp",
                "csrc/LineMatcher.cpp",
                "csrc/HeatmapMatcher.cpp",
                "csrc/LinePostprocessor.cpp"
            ],
            include_dirs=[
                pybind11.get_include(),
                np.get_include(),
                "csrc"  # Add 'csrc' so they can find their own headers
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
        version="0.1.1",
        packages=find_packages(),
        ext_modules=get_ext_modules(),
        zip_safe=False,
        url='https://github.com/SebastianJanampa/line_segment_eval',
        long_description=readme(),  # <--- Use the string variable here
        long_description_content_type="text/markdown",
    )
except ImportError:
    # Fallback if dependencies aren't present during a non-build call
    print("Build dependencies missing. Assuming this is a metadata-only call.")
    setup(
        name="faster_line_eval",
        packages=find_packages(),
    )
