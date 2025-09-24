# setup_sparse.py  ── auto‑build when run with no args
import sys
# If the script is launched with no command‑line arguments (e.g. runfile in Spyder),
# append the usual build arguments so setuptools doesn't bail out.
if len(sys.argv) == 1:
    sys.argv += ["build_ext", "--inplace"]

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np
import sys

# -------- platform‑specific compiler flags ----------------------------
if sys.platform == "darwin":                     # Apple‑clang
    omp_compile, omp_link = ["-Xpreprocessor", "-fopenmp"], ["-lomp"]
elif sys.platform.startswith("win"):             # MSVC / clang‑cl
    omp_compile, omp_link = ["/openmp"], ["/openmp"]
else:                                            # GCC / Clang on Linux
    omp_compile, omp_link = ["-fopenmp"], ["-fopenmp"]

opt = ["/O2", "/arch:AVX2", "/fp:fast"] if sys.platform.startswith("win") \
      else ["-O3", "-march=native", "-ffast-math"]

ext_modules = [
    Extension(
        "ising_mcmc_sparse",
        ["ising_mcmc_sparse.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=opt + omp_compile,
        extra_link_args=omp_link,
        language="c",
    )
]

setup(
    name="ising_mcmc_sparse",
    ext_modules=cythonize(
        ext_modules,
        language_level=3,
        compiler_directives=dict(
            boundscheck=False,
            wraparound=False,
            initializedcheck=False,
            cdivision=True,
        ),
    ),
    zip_safe=False,
)
