"""
image processing utility functions
"""
import gc
import sys
from functools import partial, reduce
from itertools import chain
from operator import mul
from typing import (
    Callable,
    Sequence,
    MutableMapping,
    Collection,
    Mapping,
    Union,
    MutableSequence,
)

from dustgoggles.structures import dig_for_values
import numpy as np


def absolutely_destroy(thing, delay_collect=False):
    if isinstance(thing, MutableMapping):
        keys = list(thing.keys())
        for key in keys:
            del thing[key]
    elif isinstance(thing, MutableSequence):
        [thing.pop() for _ in range(len(thing))]
    del thing
    if "matplotlib.pyplot" in sys.modules.keys():
        getattr(sys.modules["matplotlib.pyplot"], "close")("all")
    if delay_collect is False:
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


def clip_unmasked(array: np.ma.MaskedArray):
    return np.clip(array.data, array.min(), array.max())


def split_filter(filter_function: Callable, axis: int = -1) -> Callable:
    """
    produce a 'split' version of a filter that applies itself to slices across
    a particular axis -- e.g., take a gaussian blur function, return a function
    that applies a gaussian blur to R / G / B channels separately and then
    recomposes them
    """

    def multi(array, *, set_axis: int = axis, **kwargs) -> np.ndarray:
        filt = partial(filter_function, **kwargs)
        filtered = tuple(
            map(filt, np.split(array, array.shape[set_axis], set_axis))
        )
        if isinstance(filtered[0], np.ma.MaskedArray):
            return np.ma.concatenate(filtered, axis=set_axis)
        return np.concatenate(filtered, axis=set_axis)

    return multi


def map_filter(filter_function: Callable) -> Callable:
    """
    returns a version of a function that automatically maps itself across all
    elements of a collection
    """

    def mapped_filter(arrays, *args, **kwargs):
        return [filter_function(array, *args, **kwargs) for array in arrays]

    return mapped_filter


def find_masked_bounds(image, cheat_low, cheat_high):
    """
    relatively memory-efficient way to perform bound calculations for
    normalize_range on a masked array.
    """
    if isinstance(image.mask, np.bool_):
        if image.mask:
            valid = np.array([])
        else:
            valid = image.data
    else:
        valid = image[~image.mask].data
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return None, None
    if (cheat_low != 0) and (cheat_high != 0):
        minimum, maximum = np.percentile(
            valid, [cheat_low, 100 - cheat_high], overwrite_input=True
        ).astype(image.dtype)
    elif cheat_low != 0:
        maximum = valid.max()
        minimum = np.percentile(valid, cheat_low, overwrite_input=True).astype(
            image.dtype
        )
    elif cheat_high != 0:
        minimum = valid.min()
        maximum = np.percentile(
            valid, 100 - cheat_high, overwrite_input=True
        ).astype(image.dtype)
    else:
        minimum = valid.min()
        maximum = valid.max()
    return minimum, maximum


# noinspection PyArgumentList
def find_unmasked_bounds(image, cheat_low, cheat_high):
    """straightforward way to find unmasked array bounds for normalize_range"""
    if cheat_low != 0:
        minimum = np.percentile(image, cheat_low).astype(image.dtype)
    else:
        minimum = image.min()
    if cheat_high != 0:
        maximum = np.percentile(image, 100 - cheat_high).astype(image.dtype)
    else:
        maximum = image.max()
    return minimum, maximum


# TODO: address rollover issues
def normalize_range(
    image: np.ndarray,
    bounds: Sequence[int] = (0, 1),
    stretch: Union[float, tuple[float, float]] = 0,
    inplace: bool = False,
) -> np.ndarray:
    """
    simple linear min-max scaler that optionally percentile-clips the input at
    stretch = (low_percentile, 100 - high_percentile). if inplace is True,
    may transform the original array, with attendant memory savings and
    destructive effects.
    """
    if isinstance(stretch, Sequence):
        cheat_low, cheat_high = stretch
    else:
        cheat_low, cheat_high = (stretch, stretch)
    range_min, range_max = bounds
    if isinstance(image, np.ma.MaskedArray):
        minimum, maximum = find_masked_bounds(image, cheat_low, cheat_high)
        if minimum is None:
            return image
    else:
        minimum, maximum = find_unmasked_bounds(image, cheat_low, cheat_high)
    if not ((cheat_high is None) and (cheat_low is None)):
        if inplace is True:
            image = np.clip(image, minimum, maximum, out=image)
        else:
            image = np.clip(image, minimum, maximum)
    if inplace is True:
        # perform the operation in-place
        image -= minimum
        image *= range_max - range_min
        if image.dtype.char in np.typecodes["AllInteger"]:
            # this loss of precision is probably better than
            # automatically typecasting it.
            # TODO: detect rollover cases, etc.
            image //= maximum - minimum
        else:
            image /= maximum - minimum
        image += range_min
        return image
    return (image - minimum) * (range_max - range_min) / (
        maximum - minimum
    ) + range_min


def enhance_color(image: np.ndarray, bounds, stretch):
    """
    wrapper for normalize_range -- normalize each channel individually,
    conventional "enhanced color" operation
    """
    return split_filter(normalize_range)(image, bounds=bounds, stretch=stretch)


def clip_finite(image, a_min: int, a_max: int):
    finite = np.ma.masked_invalid(image)
    result = np.ma.clip(finite, a_min, a_max)
    if isinstance(image, np.ma.MaskedArray):
        return result
    return result.data


def std_clip(image, sigma=1):
    """
    simple clipping function that clips at multiples of an array's standard
    deviation offset from its mean
    """
    finite = np.ma.masked_invalid(image)
    mean = np.ma.mean(finite)
    std = np.ma.std(finite)
    result = np.ma.clip(finite, *(mean - std * sigma, mean + std * sigma))
    if isinstance(image, np.ma.MaskedArray):
        return result
    return result.data


def centile_clip(image, centiles=(1, 99)):
    """
    simple clipping function that clips values above and below a given
    percentile range
    """
    finite = np.ma.masked_invalid(image)
    bounds = np.percentile(finite[~finite.mask].data, centiles)
    result = np.ma.clip(finite, *bounds)
    if isinstance(image, np.ma.MaskedArray):
        return result
    return result.data


def minmax_clip(image, stretch=(0, 0)):
    """
    simple minmax clip that optionally cheats 0 up and 1 down at multiples
    of an array's dynamic range
    """
    finite = np.ma.masked_invalid(image)
    if stretch != (0, 0):
        dynamic_range = finite.max() - finite.min()
        cheat_low, cheat_high = stretch
    else:
        dynamic_range, cheat_low, cheat_high = (0, 0, 0)
    result = np.clip(
        image,
        image.min() + dynamic_range * cheat_low,
        image.max() - dynamic_range * cheat_high,
    )
    if isinstance(image, np.ma.MaskedArray):
        return result
    return result.data


def sinh_scale(array, bounds=(0, 1), stretch=1):
    minimum, maximum = array.min(), array.max()
    scaled = (array - minimum) / (maximum - minimum) - 0.5
    expanded = np.sinh(scaled * stretch)
    return normalize_range(expanded, bounds)


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


def make_mask_passer(func, mask_nans=True):
    def mask_passer(array, *args, **kwargs):
        transformed = func(array, *args, **kwargs)
        if isinstance(array, np.ma.MaskedArray):
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


def closest_ratio(integer, target):
    from sympy import factorint

    factors = tuple(
        chain.from_iterable(
            [[f] * count for f, count in factorint(integer).items()]
        )
    )
    factors = tuple(reversed(factors))
    factorizations = []
    possibilities = []
    for unique in set(factors):
        left = [unique]
        right = [f for f in factors if f not in left]
        quantity = factors.count(unique)
        right += [unique] * (quantity - 1)
        ratio = target
        new_left, new_right = left, right
        while ratio <= target:
            for check_ix in range(len(right)):
                new_left = left + [right[check_ix]]
                for f in new_left:
                    if f in new_right:
                        intermediate = [r for r in right if r != f]
                        quantity = right.count(f)
                        intermediate += [f] * (quantity - 1)
                        new_right = intermediate
                ratio = reduce(mul, new_left) / reduce(mul, new_right)
            if ratio <= target:
                left, right = new_left, new_right
        possibilities.append(reduce(mul, left) / reduce(mul, right))
        factorizations.append((left, right))
        possibilities.append(ratio)
        factorizations.append((new_left, new_right))
    differences = [abs(p - target) for p in possibilities]
    closest = min(differences)
    for p, diff, fact in zip(possibilities, differences, factorizations):
        if diff == closest:
            return p, fact


def closest_aspect(new_size, aspect_ratio):
    _, factors = closest_ratio(new_size, aspect_ratio)
    left, right = factors
    return reduce(mul, right), reduce(mul, left)


def strict_reshape(array, aspect_ratio):
    return array.reshape(closest_aspect(array.size, aspect_ratio))


def ravel_valid(array, copy=True):
    values = array.ravel()
    if isinstance(values, np.ma.MaskedArray):
        if copy is True:
            values = values.copy()
        values[values.mask] = np.nan
        return values[np.isfinite(values)].data
    return values[np.isfinite(values)]


def setmask(array, value, copy=True):
    if not isinstance(array, np.ma.MaskedArray):
        return array
    if copy is True:
        data = array.data.copy()
    else:
        data = array.data
    data[array.mask] = value
    return data


def zero_mask(array, copy=True):
    return setmask(array, 0, copy)


def nanmask(array, copy=True):
    return setmask(array, np.nan, copy)


def make_mask_canvas(image, unmasked_alpha=1, masked_alpha=0):
    canvas = np.full(image.shape[:2], unmasked_alpha, dtype="float16")
    if len(image.shape) > 2:
        mask = image.mask.sum(axis=-1).astype(bool)
    else:
        mask = image.mask
    canvas[mask] = masked_alpha
    return canvas


def colorfill_maskedarray(
    masked_array: np.ma.MaskedArray,
    color: Union[float, tuple[float, float, float]] = 0.45,
    mask_alpha=None,
    unmasked_alpha=None,
):
    """
    masked_array: 2-D masked array or a 3-D masked array with last axis of
    length 3. should be float normalized to 0-1.
    color: optionally-specified RGB color or single integer for gray
    (default medium gray).
    if both alpha parameters are None, return a 3-D array with masked values
    filled with color.
    otherwise, return a 4-D array (last axis intended as an alpha plane),
    with alpha value equal to unmasked_alpha outside the mask and masked_alpha
    inside the mask (either defaults to 0 if not set)
    """
    if (mask_alpha is None) and (unmasked_alpha is not None):
        mask_alpha = 0
    if (unmasked_alpha is None) and (mask_alpha is not None):
        unmasked_alpha = 0
    if len(masked_array.shape) == 2:
        if isinstance(color, (int, float)):
            color = (color, color, color)
        stack = [masked_array.filled(color[ix]) for ix in range(3)]
        if mask_alpha is not None:
            stack.append(
                make_mask_canvas(masked_array, unmasked_alpha, mask_alpha)
            )
        return np.dstack(stack)
    if masked_array.shape[-1] != 3:
        raise ValueError("3-D arrays must have last axis of length = 3")
    if isinstance(color, (int, float)):
        stack = masked_array.filled(color)
    else:
        stack = np.dstack(
            [masked_array[:, :, ix].filled(color[ix]) for ix in range(3)]
        )
    if mask_alpha is None:
        return stack
    return np.dstack(
        [stack, make_mask_canvas(stack, unmasked_alpha, mask_alpha)]
    )


def maskwhere(images, constants):
    return reduce(np.logical_or, [np.isin(i, constants) for i in images])


def pick_mask_constructors(region):
    if region == "inner":
        return np.ma.masked_outside, np.logical_or
    elif region == "outer":
        return np.ma.masked_inside, np.logical_and
    raise ValueError(f"region={region}; region must be 'inner' or 'outer'")


def centered_indices(array):
    y, x = np.indices(array.shape)
    y0, x0 = (array.shape[0] - 1) / 2, (array.shape[1] - 1) / 2
    return y - y0, x - x0


def radial_index(array):
    y_ix, x_ix = centered_indices(array)
    return np.sqrt(y_ix**2 + x_ix**2)


def join_cut_mask(array, cut_mask, copy=True):
    if isinstance(array, np.ma.MaskedArray):
        if copy is True:
            array = array.copy()
        array.mask = np.logical_or(cut_mask, array.mask)
    else:
        array = np.ma.MaskedArray(array, mask=cut_mask)
    return array


def cut_annulus(array, bounds, region="inner", copy=True):
    mask_method, _ = pick_mask_constructors(region)
    distance = radial_index(array)
    pass_min, pass_max = bounds
    pass_min = pass_min if pass_min is not None else distance.min()
    pass_max = pass_max if pass_max is not None else distance.max()
    cut_mask = mask_method(distance, pass_min, pass_max).mask
    return join_cut_mask(array, cut_mask, copy)


def cut_rectangle(array, bounds, region="inner", center=True, copy=True):
    mask_method, op = pick_mask_constructors(region)
    if center is True:
        y_dist, x_dist = centered_indices(array)
    else:
        y_dist, x_dist = np.indices(array.shape)
    masks = []
    x_bounds, y_bounds = bounds
    for bound, dist in zip((x_bounds, y_bounds), (x_dist, y_dist)):
        pass_min, pass_max = bound
        pass_min = pass_min if pass_min is not None else dist.min()
        pass_max = pass_max if pass_max is not None else dist.max()
        masks.append(mask_method(dist, pass_min, pass_max).mask)
    cut_mask = op(*masks)
    return join_cut_mask(array, cut_mask, copy)


def zerocut(*args, **kwargs):
    return zero_mask(cut_rectangle(*args, **kwargs))
