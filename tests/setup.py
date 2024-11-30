import numpy as np
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

import eigency

extensions = [
    Extension(
        "eigency_tests.eigency_tests",
        sources=["eigency_tests/eigency_tests.pyx"],
        include_dirs=[".", "eigency_tests"] + eigency.get_includes(),
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
    extensions.append(
        Extension(
            "eigency_tests.eigency_np2",
            sources=["eigency_tests/eigency_np2.pyx"],
            include_dirs=[".", "eigency_tests"] + eigency.get_includes(),
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    )

setup(
    name="eigency-tests",
    version="0.0.0",
    ext_modules=cythonize(
        extensions,
        compiler_directives=dict(
            language_level="3",
        ),
    ),
    packages=["eigency_tests"],
)
