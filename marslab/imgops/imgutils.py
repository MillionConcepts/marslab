"""
image processing utility functions
"""
import gc
import sys
from functools import partial
from itertools import chain
from typing import (
    Callable,
    Sequence,
    MutableMapping,
    Collection,
    Mapping,
)
from typing import Union

import numpy as np
from dustgoggles.structures import dig_for_values
from numpy.ma import MaskedArray


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
        getattr(sys.modules["matplotlib.pyplot"], "close")("all")
    gc.collect()


def eightbit(array, stretch=(0, 0)):
    """return an eight-bit version of an array"""
    return np.round(normalize_range(array, (0, 255), stretch)).astype(np.uint8)


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
) -> Union[Mapping[str, list[np.ndarray]], list[np.ndarray], np.ndarray]:
    """applies crop() to every array in the passed collection"""
    # if you _didn't_ pass it a collection, just overload / dispatch to crop()
    if isinstance(arrays, np.ndarray):
        return crop(arrays, bounds)
    # otherwise map crop()
    elif isinstance(arrays, Mapping):
        return {
            name: [crop(array, bounds) for array in name_arrays]
            for name, name_arrays in arrays.items()
        }
    return [crop(array, bounds) for array in arrays]


def split_filter(filter_function: Callable, axis: int = -1) -> Callable:
    """
    produce a 'split' version of a filter that applies itself to slices across
    a particular axis -- e.g., take a gaussian blur function, return a function
    that applies a gaussian blur to R / G / B channels separately and then
    recomposes them
    """

    def multi(array, *, set_axis: int = axis, **kwargs) -> np.ndarray:
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


# noinspection PyArgumentList
def normalize_range(
    image: np.ndarray,
    bounds: Sequence[int] = (0, 1),
    stretch: Union[float, Sequence[float]] = None,
) -> np.ndarray:
    """
    simple linear min-max scaler that optionally cuts off low and high

    percentiles of the input
    """
    working_image = image.copy()
    if isinstance(stretch, Sequence):
        cheat_low, cheat_high = stretch
    else:
        cheat_low, cheat_high = (stretch, stretch)
    range_min, range_max = bounds
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


def enhance_color(image: np.ndarray, bounds, stretch):
    """
    wrapper for normalize_range -- normalize each channel individually,
    conventional "enhanced color" operation
    """
    return split_filter(normalize_range)(image, bounds=bounds, stretch=stretch)


def std_clip(image, sigma=1):
    """
    simple clipping function that clips at multiples of an array's standard
    deviation offset from its mean
    """
    finite = np.ma.masked_invalid(image)
    mean = np.ma.mean(finite)
    std = np.ma.std(finite)
    return np.ma.clip(finite, *(mean - std * sigma, mean + std * sigma)).data


def centile_clip(image, centiles=(1, 99)):
    """
    simple clipping function that clips values above and below a given
    percentile range
    """
    finite = np.ma.masked_invalid(image)
    bounds = np.percentile(finite[~finite.mask].data, centiles)
    return np.ma.clip(finite, *bounds).data


def minmax_clip(image, stretch=(0, 0)):
    """
    simple minmax clip that optionally cheats 0 up and 1 down at multiples
    of an array's dynamic range
    """
    if stretch != (0, 0):
        dynamic_range = image.max() - image.min()
        cheat_low, cheat_high = stretch
    else:
        dynamic_range, cheat_low, cheat_high = (0, 0, 0)
    return np.clip(
        image,
        image.min() + dynamic_range * cheat_low,
        image.max() - dynamic_range * cheat_high,
    )


# TODO: would it be better for these functions to use flat mask arrays
#  rather than coordinate grids? (performance is the same, question is
#  interface convenience)
def bilinear_interpolate(
    input_array: np.ndarray,
    output_shape: tuple[int, int],
    y_coords: np.ndarray,
    x_coords: np.ndarray,
) -> np.ndarray:
    """
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
    according to a regular pattern. input array must be of an
    integer or float dtype.

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


def mapfilter(predicate, key, map_sequence):
    new_sequence = []
    for mapping in map_sequence:
        obj = mapping if key is None else mapping.get(key)
        if predicate(obj):
            new_sequence.append(mapping)
    return new_sequence


def mask_below(array, value):
    return MaskedArray(array, array <= value)


def make_mask_passer(func, mask_nans=True):
    def mask_passer(array, *args, **kwargs):
        transformed = func(array, *args, **kwargs)
        if isinstance(array, np.ma.masked_array):
            mask = array.mask
            if mask_nans:
                mask[np.isnan(transformed)] = True
            return np.ma.MaskedArray(transformed, mask)
        return transformed

    return mask_passer


def get_all_bands(instruction: Mapping):
    """
    helper function for look set analysis: get all bands mentioned in an
    instructions, including an instructions with nested bands
    """
    return chain.from_iterable(dig_for_values(instruction, "bands"))


def get_all_bands_from_all(instructions: Collection[Mapping]):
    """
    helper function for look set analysis: get all bands mentioned in all
    instructions, including from instructions with nested bands
    """
    return set(chain.from_iterable(map(get_all_bands, instructions)))