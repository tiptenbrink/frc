"""The types that the functions of frc take and return."""

# Copyright (C) 2021                Department of Imaging Physics
# All rights reserved               Faculty of Applied Sciences
#                                   TU Delft
# Tip ten Brink

from collections.abc import Callable

import numpy as np

__all__ = ['Img', 'Func1D', 'NoIntersectionException']

# An image is a NumPy array. Before 0.3, frc also took DIPlib images.
Img = np.ndarray

# A function of one variable: it maps a float to a float, and an array of
# floats to an array of floats.
Func1D = Callable[[np.ndarray | float], np.ndarray | float]


class NoIntersectionException(ValueError):
    """The error of a curve that does not cross its threshold."""

