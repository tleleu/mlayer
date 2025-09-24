# python setup_sa_sparse_ising.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
ext = Extension(
    "sa_sparse_ising",
    ["sa_sparse_ising.pyx"],
    extra_compile_args=["-O3", "-march=native"],  # add "-fopenmp" if you enable prange
    extra_link_args=[]                            # add "-fopenmp" if you enable prange
)
setup(
    name="sa_sparse_ising",
    ext_modules=cythonize([ext], language_level=3),
    include_dirs=[np.get_include()],
)
