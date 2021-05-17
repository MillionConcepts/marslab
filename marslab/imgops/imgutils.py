"""
image processing utility functions
"""
from collections.abc import (
    Callable,
    Sequence,
    Mapping,
    MutableMapping,
    Collection,
)
import gc
from functools import partial
from operator import methodcaller
import sys
from typing import Union, Any

import numpy as np


def get_from_all(key, mappings, default=None):
    """
    get all values of "key" from each dict or whatever in "mappings"
    """
    if isinstance(mappings, Mapping):
        view = mappings.values()
    else:
        view = mappings
    return list(map(methodcaller("get", key, default), view))


def absolutely_destroy(thing):
    if isinstance(thing, MutableMapping):
        keys = list(thing.keys())
        for key in keys:
            del thing[key]
    elif isinstance(thing, Sequence):
        for _ in thing:
            del _
    del thing
    if "matplotlib.pyplot" in sys.modules.keys():
        sys.modules["matplotlib.pyplot"].close("all")
    gc.collect()


def eightbit(array, cheat_low=0, cheat_high=0):
    """return an eight-bit version of an array"""
    return np.round(
        normalize_range(array, 0, 255, cheat_low, cheat_high)
    ).astype(np.uint8)


def crop(array: np.ndarray, bounds=None, **_) -> np.ndarray:
    """
    :param array: array to be cropped
    :param bounds: tuple of (left, right, top, bottom) pixels to crop
    """
    if bounds is None:
        return array
    assert len(bounds) == 4  # test for bad inputs
    pixels = [side if side != 0 else None for side in bounds]
    for value in (1, 3):
        if isinstance(pixels[value], int):
            if pixels[value] > 0:
                pixels[value] = pixels[value] * -1
    return array[pixels[2] : pixels[3], pixels[0] : pixels[1]]


def crop_all(
    arrays: Collection[np.ndarray], bounds=None, **_
) -> Union[list[np.ndarray], np.ndarray]:
    """applies crop() to every array in the passed collection"""
    # if you _didn't_ pass it a collection, just overload / dispatch to crop()
    if isinstance(arrays, np.ndarray):
        return crop(arrays, bounds)
    # otherwise map crop()
    return [crop(array, bounds) for array in arrays]


def split_filter(
    filter_function: Callable, axis: int = -1
) -> Callable[[np.ndarray, Any], np.ndarray]:
    """
    produce a 'split' version of a filter that applies itself to slices across
    a particular axis -- e.g., take a gaussian blur function, return a function
    that applies a gaussian blur to R / G / B channels separately and then
    recomposes them
    """

    def multi(
        array, *, set_axis: int = axis, **kwargs
    ) -> np.ndarray:
        filt = partial(filter_function, **kwargs)
        filtered = map(filt, np.split(array, array.shape[set_axis], set_axis))
        return np.concatenate(tuple(filtered), axis=set_axis)
    return multi


def map_filter(filter_function: Callable) -> Callable:
    """
    returns a version of a function that automatically maps itself across all
    elements of a collection
    """

    def mapped_filter(arrays, *args, **kwargs):
        return [filter_function(array, *args, **kwargs) for array in arrays]

    return mapped_filter


def normalize_range(
    image, range_min=0, range_max=1, cheat_low=None, cheat_high=None
):
    """
    simple linear min-max scaler that optionally cuts off low and high
    percentiles of the input
    """
    working_image = image.copy()
    if cheat_low is not None:
        minimum = np.percentile(image, cheat_low).astype(image.dtype)
    else:
        minimum = image.min()
    if cheat_high is not None:
        maximum = np.percentile(image, 100 - cheat_high).astype(image.dtype)
    else:
        maximum = image.max()
    if not ((cheat_high is None) and (cheat_low is None)):
        working_image = np.clip(working_image, minimum, maximum)
    return range_min + (working_image - minimum) * (range_max - range_min) / (
        maximum - minimum
    )


def enhance_color(
    image, range_min=0, range_max=1, cheat_low=None, cheat_high=None
):
    """
    wrapper for normalize_range -- normalize each channel individually,
    conventional "enhanced color" operation
    """
    return split_filter(normalize_range)(
        image, range_min=range_min, range_max=range_max, cheat_low=cheat_low, cheat_high=cheat_high
    )


def std_clip(image, sigma=1):
    """
    simple clipping function that clips at multiples of an array's standard
    deviation offset from its mean
    """
    mean = np.mean(image)
    std = np.std(image)
    return np.clip(image, *(mean - std * sigma, mean + std * sigma))


def minmax_clip(image, cheat_low=0, cheat_high=0):
    """
    simple minmax clip that optionally cheats 0 up and 1 down at multiples
    of an array's dynamic range
    """
    dynamic_range = image.max() - image.min()
    return np.clip(
        image,
        image.min() + dynamic_range * cheat_low,
        image.max() - dynamic_range * cheat_high,
    )


def bilinear_interpolate(
    input_array: np.ndarray,
    output_shape: tuple[int, int],
    y_coords: np.ndarray,
    x_coords: np.ndarray,
) -> np.ndarray:
    """
    TODO: would it be better for these functions to use flat mask arrays
        rather than coordinate grids? (performance is the same, question is
        interface convenience

    upsample a 2D array, gridding its pixels at y-coordinates given in y_coords
    and x-coordinates given in x_coords and linearly interpolating from there,
    first in the x direction and then in the y direction. y_coords and x_coords
    are 2D arrays of the same size like those created by np.meshgrid,
    defining locations in N**2
    This is the general (gridded) case and should work correctly for any
    pixel class defined as a partition of a grid. Use the faster
    bilinear_interpolate_subgrid() for pixel classes that can be defined as
    locations relative to a regular tiling of a grid.
    """
    horizontal = np.zeros(output_shape, dtype=input_array.dtype)
    vertical = np.zeros(output_shape, dtype=input_array.dtype)
    rows = np.unique(y_coords)
    for row_ix in rows:
        coordinate_reference_indices = np.where(y_coords == row_ix)
        column_indices = x_coords[coordinate_reference_indices]
        horizontal[row_ix] = np.interp(
            np.arange(output_shape[1]),
            column_indices,
            input_array[coordinate_reference_indices],
        )
    # column indices are immaterial now, because we've interpolated the rows.
    for column_ix in np.arange(output_shape[1]):
        vertical[:, column_ix] = np.interp(
            np.arange(output_shape[0]), rows, horizontal[rows, column_ix]
        )
    return vertical


def bilinear_interpolate_subgrid(
    rows: np.ndarray,
    columns: np.ndarray,
    input_array: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """
    interpolate 2D values to a 2D array, gridding those values
    according to a regular pattern.

    this is a special case for pixel classes that are defined as unique
    positions within m x n subgrids that tile the coordinate space.
    in particular, it will work for any conventional Bayer pattern,
    when each pattern cell is treated as a separate pixel 'class' (even if
    pixels are of the 'same' color), as well as a variety of more complex
    patterns. use the slower bilinear_interpolate() for general gridded-data
    cases.
    """
    horizontal = np.empty(output_shape, dtype=input_array.dtype)
    vertical = np.empty(output_shape, dtype=input_array.dtype)
    # because of the 'subgrid' assumption, in any row that contains pixels,
    # their column indices are the same (and vice versa)
    for row_ix, row in enumerate(rows):
        horizontal[row] = np.interp(
            np.arange(output_shape[1]),
            columns,
            input_array[row_ix, :],
        )
    for output_column in np.arange(output_shape[1]):
        vertical[:, output_column] = np.interp(
            np.arange(output_shape[0]), rows, horizontal[rows, output_column]
        )
    return vertical


def apply_image_filter(image, image_filter=None):
    """
    unpacking / pseudo-dispatch function

    `image_filter` is a dict() that contains 'params' and 'function'
    keys. The optional 'params' value is a parameters dict() and the 'function'
    value is the corresponding function to run on `image`.
    """
    if image_filter is None:
        return image
    return image_filter["function"](image, **image_filter.get("params", {}))
