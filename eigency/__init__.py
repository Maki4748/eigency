import os.path
from importlib.resources import files, as_file

import numpy as np

__eigen_dir__ = (files(__name__) / "eigen").__str__()


def get_includes(include_eigen=True):
    root = os.path.dirname(__file__)
    path = [root, np.get_include()]
    if include_eigen:
        path.append(os.path.join(root, __eigen_dir__))
    return path
