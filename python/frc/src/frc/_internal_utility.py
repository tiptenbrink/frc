"""Helpers for the other modules of frc."""

# Copyright (C) 2021                Department of Imaging Physics
# All rights reserved               Faculty of Applied Sciences
#                                   TU Delft
# Tip ten Brink

import numpy as np


def are_axes_equal(a: np.ndarray) -> bool:
    """Returns whether all axes of a have the same length."""
    return len(set(a.shape)) <= 1


def are_sizes_equal(arrs: list[np.ndarray], check_equal_axes=True) -> bool:
    """Returns whether all arrays have the same shape and, if
    check_equal_axes, all axes of that shape have the same length."""
    same_shape = all(a.shape == arrs[0].shape for a in arrs)
    return same_shape and (not check_equal_axes or are_axes_equal(arrs[0]))


def counts_of(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Returns img as counts: img itself if it holds integers, and otherwise
    its values truncated to int32. Also returns whether img holds whole
    numbers only, so that the counts equal img."""
    if np.issubdtype(img.dtype, np.integer):
        return img, True
    counts = img.astype(np.int32)
    return counts, np.array_equal(counts, img)


def separate_noncontiguous(a: np.ndarray) -> list[np.ndarray]:
    """Splits a sorted array of integers into runs of consecutive ones."""
    run_starts = np.asarray(np.diff(a) > 1).nonzero()[0] + 1
    return np.split(a, run_starts)
