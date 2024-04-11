import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cwd = Path(os.path.dirname(os.path.abspath(__file__)))

nvcc_flags = [
    "-std=c++17",  # NOTE: CUTLASS requires c++17
    "-gencode=arch=compute_90a,code=sm_90a",
]

ext_modules = [
    CUDAExtension(
        "fp8_gmm_backend",
        ["csrc/ops.cu", "csrc/fp8_gmm.cu"],
        libraries=["cuda"],
        include_dirs=[
            f"{cwd}/third_party/cutlass/include/",
            f"{cwd}/third_party/cutlass/tools/util/include/",
            f"{cwd}/csrc",
        ],
        extra_compile_args={
            "cxx": ["-fopenmp", "-fPIC", "-Wno-strict-aliasing"],
            "nvcc": nvcc_flags,
        },
    )
]

extra_deps = {}

extra_deps["dev"] = [
    "absl-py",
]

extra_deps["all"] = set(dep for deps in extra_deps.values() for dep in deps)

setup(
    name="fp8_gmm",
    version="0.0.1",
    author="Vincent Wang",
    author_email="weicwang@microsoft.com",
    description="FP8 Grouped GEMM",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    extras_require=extra_deps,
)
