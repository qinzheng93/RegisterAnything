import os.path as osp
from glob import glob

import torch
from setuptools import Extension, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

extension_dir = "vision3d/csrc"


def get_cpu_ext_module() -> Extension:
    sources = []
    sources += glob(osp.join(extension_dir, "cpu", "ops", "*", "*.cpp"))
    sources += glob(osp.join(extension_dir, "cpu", "pybind.cpp"))

    include_dirs = []
    include_dirs.append(osp.abspath(osp.join(extension_dir, "cpu", "external", "eigen3")))
    include_dirs.append(osp.abspath(osp.join(extension_dir, "cpu", "external", "nanoflann")))
    include_dirs.append(osp.abspath(osp.join(extension_dir, "cpu", "common")))

    cpu_ext = CppExtension(
        name="vision3d.ext_cpu",
        sources=sources,
        include_dirs=include_dirs,
    )
    return cpu_ext


def get_gpu_ext_module() -> Extension:
    sources = []
    sources += glob(osp.join(extension_dir, "cuda", "ops", "*", "*.cpp"))
    sources += glob(osp.join(extension_dir, "cuda", "ops", "*", "*.cu"))
    sources += glob(osp.join(extension_dir, "cuda", "pybind.cpp"))

    include_dirs = []
    include_dirs.append(osp.abspath(osp.join(extension_dir, "cuda", "common")))

    gpu_ext = CUDAExtension(
        name="vision3d.ext_gpu",
        sources=sources,
        include_dirs=include_dirs,
    )
    return gpu_ext


def get_ext_modules() -> list[Extension]:
    ext_modules = []
    ext_modules.append(get_cpu_ext_module())
    if torch.cuda.is_available():
        print("Building extensions: CPU, GPU.")
        ext_modules.append(get_gpu_ext_module())
    else:
        print("Building extensions: CPU.")
    return ext_modules


setup(
    ext_modules=get_ext_modules(),
    cmdclass={"build_ext": BuildExtension},
)
