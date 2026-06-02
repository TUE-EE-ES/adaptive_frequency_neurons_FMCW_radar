from setuptools import setup, Extension
import pybind11
import sys

ext_modules = [
    Extension(
        "adaptive_oscillator",
        ["AdaptiveOscillator.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            "/usr/include/eigen3"
        ],
        language="c++",
        extra_compile_args=["-O3","-march=native","-std=c++17","-ffast-math"]
    ),
]

setup(
    name="adaptive_oscillator",
    ext_modules=ext_modules,
)