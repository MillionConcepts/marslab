"""
image processing utility functions
"""
import gc
import sys
from functools import partial, reduce
from itertools import chain
from numbers import Number, Real
from operator import mul
from typing import (
    Callable,
    Sequence,
    MutableMapping,
    Collection,
    Mapping,
    Union,
    MutableSequence, Any, Hashable, Optional, Literal,
)

from dustgoggles.structures import dig_for_values
import numpy as np
from numpy.typing import ArrayLike


# TODO: maybe does not belong in this module
# TODO, maybe: add even more stuff -- try to run .close(), etc.
def absolutely_destroy(
    thing: Union[MutableMapping, MutableSequence], delay_collect: bool = False
) -> None:
    """
    Object-clearing utility intended primarily but not exclusively for mutable
    collections of matplotlib objects.

    Clear the contents of `thing`, which may be a list-like or dict-like
    object; then, close all matplotlib figures (if matplotlib.pyplot is
    loaded); then (if `delay_collect` is False), run gc.collect().

    This throw-things-at-the-wall approach is intended to trigger various
    destructor behaviors for collections and elements of collections that may
    not respond as expected to methods such as .clear() and .close(), and may
    enjoy creating reference cycles.
    """
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


def eightbit(
    array: np.ndarray, stretch: Union[float, tuple[float, float], None] = None
) -> np.ndarray:
    """
    Return an 'eight-bit' (single-byte unsigned integer, uint8, u1, etc.)
    version of an array scaled to (0, 255); optionally, stretch (range-clip)
    it inline. Intended primarily for preparing arrays to be interpreted as
    images (or image channels) in a conventional colorspace of some kind.

    If `array` is already uint8, does not modify it unless stretch is not None,
    in which case it will map its min/max values to 0/255 (optionally
    stretched) as with any other input array.

    WARNING: If `array` is a MaskedArray, this function automatically fills
      corresponding masked values of the output array with 0s to prevent
      undesirable typecasting behaviors.
    """
    if array.dtype.char == 'B' and stretch is None:
        return array
    if array.dtype.char == 'B':
        return normalize_range(array, (0, 255), stretch)
    output = normalize_range(array, (0, 255), stretch)
    if isinstance(output, np.ma.MaskedArray):
        output[output.mask] = np.uint8(0)
    if output.dtype.char not in np.typecodes["AllInteger"]:
        output = np.round(output)
    return output.astype(np.uint8)


def cropmask(
    arr: np.ndarray,
    bounds: Optional[tuple[int, int, int, int]] = None,
    copy: bool = True,
    **_
) -> np.ndarray:
    """
    Return a MaskedArray based on `arr` with all elements that fall within a
    rectangular 'border' masked (along with any already-masked elements, if
    `arr` is a MaskedArray).
    `bounds` is a tuple of  (left, right, top, bottom) pixels to crop, or
    None; if `bounds` is None; simply return `arr`.

    If `copy` is False and `arr` is a MaskedArray, modify `arr.mask` inplace.
    The `copy` argument has no effect if `arr` is not a MaskedArray.
    """
    if bounds is None:
        return arr
    assert len(bounds) == 4  # test for bad inputs
    pixels = [side if side != 0 else None for side in bounds]
    canvas = np.ones(arr.shape)
    for value in (1, 3):
        if isinstance(pixels[value], int):
            if pixels[value] > 0:
                pixels[value] = pixels[value] * -1
    canvas[pixels[2]: pixels[3], pixels[0]: pixels[1]] = 0
    if isinstance(arr, np.ma.MaskedArray):
        if copy is True:
            arr.mask = np.logical_or(canvas, arr.mask)
            return arr
        return np.ma.masked_array(
            arr.data, np.logical_or(canvas, arr.mask)
        )
    return np.ma.masked_array(arr, canvas)


def crop(
    arr: np.ndarray, bounds: Optional[tuple[int, int, int, int]] = None, **_
) -> np.ndarray:
    """
    return a copy of `arr` cropped to specified rectangular bounds.
    `bounds` is a tuple of (left, right, top, bottom) pixels to crop, or None;
    if `bounds` is None, simply return `arr`.
    """
    if bounds is None:
        return arr
    assert len(bounds) == 4  # test for bad inputs
    pixels = [side if side != 0 else None for side in bounds]
    for value in (1, 3):
        if isinstance(pixels[value], int):
            if pixels[value] > 0:
                pixels[value] = pixels[value] * -1
    return arr[pixels[2]: pixels[3], pixels[0]: pixels[1]]


def crop_all(
    arrays: Collection[np.ndarray], bounds=None, **_
) -> Union[Mapping[str, list[np.ndarray]], list[np.ndarray], np.ndarray]:
    """applies crop() to every array in `arrays`."""
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


def clip_unmasked(arr: np.ma.MaskedArray) -> np.ndarray:
    """
    np.clip() does not always respect masks properly. Return an unmasked
    version of `arr` clipped to the min/max values of its unmasked elements.
    """
    return np.clip(arr.data, arr.min(), arr.max())


# TODO: this may no longer be necessary for some functions due to upstream
#  API changes (notably, the referred-to gaussian blur function).
def split_filter(
    filter_function: Callable[[np.ndarray, ...], np.ndarray], axis: int = -1
) -> Callable:
    """
    produce a 'split' version of a function -- notionally an image filter --
    that applies itself to slices across a particular axis. e.g., if passed a
    gaussian blur function, returns a function that applies a gaussian blur to
    R / G / B channels separately and then recomposes them.
    """

    def multi(
        array: np.ndarray, *, set_axis: int = axis, **kwargs: Any
    ) -> np.ndarray:
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
    returns a version of a function -- notionally an image filter -- that
    automatically maps itself across all elements of a passed collection.
    """

    def mapped_filter(arrays, *args, **kwargs):
        return [filter_function(array, *args, **kwargs) for array in arrays]

    return mapped_filter


def find_masked_bounds(
    image: np.ma.MaskedArray, cheat_low: float, cheat_high: float
) -> Union[tuple[Real, Real], tuple[None, None]]:
    """
    relatively memory-efficient way to perform bound calculations for
    normalize_range on a masked array.
    """
    if image.mask is np.False_:
        valid = image.data
    else:
        valid = image[~image.mask].data
    # NOTE: unlike its relative in `pdr.browsify`, this function does not
    #  assume that the masked array it receives has already been masked
    #  where invalid.
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
def find_unmasked_bounds(
    image: np.ndarray, cheat_low: float, cheat_high: float
) -> tuple[Real, Real]:
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


def normalize_range(
    arr: np.ndarray,
    bounds: tuple[Real, Real] = (0., 1.),
    stretch: Union[float, tuple[float, float], None] = None,
    inplace: bool = False,
) -> Union[np.ndarray, np.ma.MaskedArray]:
    """
    simple linear min-max scaler that optionally percentile-clips the input at
    stretch = (low_percentile, 100 - high_percentile). if inplace is True,
    may transform the original array, with attendant memory savings and
    destructive effects.

    Setting bounds[0] >= bounds[1] may cause undesired effects.

    Attempts to maintain dtype of the input array, but will cast if required
    to match bounds. Note that the inplace argument is effectively ignored in
    that case.

    Always ignores masked data in an array when computing scale.
    Arrays with nan values will always return an array of np.nan. Arrays with
    -inf and/or inf values will clip those values to bounds[0] and bounds[1]
    respectively.
    """
    if not isinstance(stretch, Sequence):
        stretch = (stretch, stretch)
    stretch = np.array([0 if i is None else i for i in stretch])
    stretch = np.clip(stretch, 0, 100)
    do_clip = max(stretch) > 0
    if isinstance(arr, np.ma.MaskedArray):
        inrange = find_masked_bounds(arr, *stretch)
    else:
        inrange = find_unmasked_bounds(arr, *stretch)
    inrange = tuple(
        map(lambda i: arr.dtype.type(i) if i is not None else i, inrange)
    )
    if do_clip and set(inrange) != {None}:
        if inplace is True:
            arr = np.clip(arr, *inrange, out=arr)
        else:
            arr = np.clip(arr, *inrange)
    mintype = reduce(
        np.promote_types,
        map(np.min_scalar_type, filter(None, (*inrange, *bounds)))
    )
    if not np.can_cast(mintype, arr.dtype):
        arr = arr.astype(mintype)
    # Note that this case will only occur for all-masked input array
    if set(inrange) == {None}:
        return np.clip(arr, *bounds)
    inrange, bounds = map(
        lambda s: np.array(s, arr.dtype.type), (inrange, bounds)
    )
    scale_up, scale_down = map(np.ptp, (bounds, inrange))
    # TODO, maybe: messy
    if arr.dtype.char in np.typecodes["AllFloat"]:
        # nonsensical-looking but effective floating point OOB cheat
        smax = (inrange[1] - inrange[0]) * scale_up / scale_down + bounds[0]
        if smax > bounds[1]:
            scale_up *= (1 + (bounds[1] - smax))
    if inplace is True:
        arr -= inrange[0]
        arr *= scale_up
        if arr.dtype.char in np.typecodes["AllInteger"]:
            # this loss of precision is probably better than
            # automatically typecasting it.
            # TODO, maybe: cute inplace rounding
            arr //= scale_down
        else:
            arr /= scale_down
        arr += bounds[0]
        return arr
    elif arr.dtype.char in np.typecodes["AllInteger"]:
        return (
            ((arr - inrange[0]) * scale_up) // scale_down + bounds[0]
        )
    return (
        (arr - inrange[0]) * scale_up / scale_down + bounds[0]
    )


def enhance_color(image: np.ndarray, bounds, stretch):
    """
    wrapper for normalize_range -- normalize each channel individually,
    conventional "enhanced color" operation
    """
    return split_filter(normalize_range)(image, bounds=bounds, stretch=stretch)


# TODO: probably pointless cruft. verify.
def clip_finite(image, a_min: int, a_max: int):
    finite = np.ma.masked_invalid(image)
    result = np.ma.clip(finite, a_min, a_max)
    if isinstance(image, np.ma.MaskedArray):
        return result
    return result.data


def std_clip(arr: np.ndarray, sigma: Number = 1) -> np.ndarray:
    """
    simple clipping function that clips at multiples of an array's standard
    deviation offset from its mean
    """
    finite = np.ma.masked_invalid(arr)
    mean, shift = np.ma.mean(finite), np.ma.std(finite) * sigma
    clips = np.array([mean - shift, mean + shift], dtype=arr.dtype)
    result = np.ma.clip(finite, *clips)
    if isinstance(arr, np.ma.MaskedArray):
        return np.ma.masked_array(result.data, mask=(arr.mask + result.mask))
    # NOTE: np typestubs suggest MaskedArray.data is a memoryview, but it's not
    # noinspection PyTypeChecker
    return result.data


def centile_clip(
    arr: np.ndarray, centiles: tuple[Number, Number] = (1, 99)
) -> np.ndarray:
    """
    simple clipping function that clips values above and below a given
    percentile range
    """
    finite = np.ma.masked_invalid(arr)
    bounds = np.percentile(finite[~finite.mask].data, centiles)
    result = np.ma.clip(finite, *(bounds.astype(arr.dtype)))
    if isinstance(arr, np.ma.MaskedArray):
        return np.ma.masked_array(result.data, mask=(arr.mask + result.mask))
    # NOTE: np typestubs suggest MaskedArray.data is a memoryview, but it's not
    # noinspection PyTypeChecker
    return result.data


def minmax_clip(arr: np.ndarray, stretch: tuple[float, float] = (0, 0)):
    """
    simple minmax clip to cheat min and max up and down at multiples of an
    array's dynamic range. always returns a copy.
    """
    if max(stretch) == 0:
        return arr.copy()
    if max(stretch) > 1:
        raise ValueError("Stretch bounds must be between 0 and 1 inclusive.")
    finite = np.ma.masked_invalid(arr)
    fmin, fmax = finite.min(), finite.max()
    ptp = fmax - fmin
    result = np.ma.clip(
        finite,
        finite.dtype.type(fmin + stretch[0] * ptp),
        finite.dtype.type(fmax - stretch[1] * ptp)
    )
    if isinstance(arr, np.ma.MaskedArray):
        return np.ma.masked_array(result.data, mask=(arr.mask + result.mask))
    return result.data


def sinh_scale(
    arr: np.ndarray,
    bounds: tuple[Number, Number] = (0., 1.),
    stretch: float = 1
) -> np.ndarray:
    """
    Applies a simple hyperbolic sine stretch to the input array. Maintains
    the dtype of the input array if it is floating-point; otherwise, converts
    it to float64.
    """
    finite = np.ma.masked_invalid(arr)
    fmin, fmax = finite.min(), finite.max()
    scaling = 1 / (fmax - fmin)
    if arr.dtype.char in np.typecodes["AllFloat"]:
        scaling = arr.dtype.type(scaling)
    normed = normalize_range(
        np.sinh((finite - fmin) * scaling * stretch - 0.5), bounds
    )
    if isinstance(arr, np.ma.masked_array):
        return np.ma.masked_array(normed, mask=normed.mask + arr.mask)
    # noinspection PyTypeChecker
    return normed.data


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


# TODO, maybe: doesn't really go in this module
def mapfilter(
    predicate: Callable[[Any], bool],
    key: Optional[Hashable],
    map_sequence: Sequence[Mapping]
) -> list[Mapping]:
    """
    Return a list containing all elements of map_sequence such that, if `key`
    is None, `predicate(element)` is True, or if `key` is not None,
    `predicate(element.get(key))` is True.
    Intended primarily as a helper function for picking looks.
    """
    new_sequence = []
    for mapping in map_sequence:
        obj = mapping if key is None else mapping.get(key)
        if predicate(obj):
            new_sequence.append(mapping)
    return new_sequence


def make_mask_passer(
    func: Callable[[np.ndarray, ...], np.ndarray],
    mask_nans=True
) -> Callable[[np.ndarray, ...], np.ndarray]:
    """
    Function decorator for functions that take at least a single positional
    ndarray argument. Returns a function that attempts to restore the masks of
    any MaskedArray passed as the first argument after passing it (and all
    other arguments) to the decorated function. Will of course fail if `func`
    mutates the mask inplace or similar. Optionally (and by default), also
    mask NaNs inline.
    """
    def mask_passer(array, *args, **kwargs):
        transformed = func(array, *args, **kwargs)
        if isinstance(array, np.ma.MaskedArray):
            mask = array.mask
            if mask_nans:
                mask[np.isnan(transformed)] = True
            return np.ma.MaskedArray(transformed, mask)
        return transformed

    return mask_passer


# TODO, maybe: doesn't really go in this module.
# TODO, maybe: this probably shouldn't actually return an iterator.
def get_all_bands(instruction: Mapping) -> chain:
    """
    helper function for look set analysis: get all bands mentioned in an
    instructions, including an instructions with nested bands
    """
    return chain.from_iterable(dig_for_values(instruction, "bands"))


# TODO, maybe: doesn't really go in this module.
def get_all_bands_from_all(instructions: Collection[Mapping]) -> set:
    """
    helper function for look set analysis: get all bands mentioned in all
    instructions, including from instructions with nested bands
    """
    return set(chain.from_iterable(map(get_all_bands, instructions)))


def closest_ratio(integer: int, target: float) -> tuple[int, int]:
    """
    Return (x, y) satisfying:

    (a, b ∈ ℕ) ∧ (a * b ==`integer`)
     → abs((x / y) - `target`) <= abs((a / b) - `target`)
    """
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


def closest_aspect(size: int, aspect_ratio: float) -> tuple[int, int]:
    """
    Given an array size and a desired aspect ratio, find the integer values
    for width and height that most closely approximate the ratio.
    """
    _, factors = closest_ratio(size, aspect_ratio)
    left, right = factors
    return reduce(mul, right), reduce(mul, left)


def strict_reshape(array: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """
    Return a copy of `array` reshaped such that its aspect ratio is as close
    as possible to `aspect_ratio` (without discarding or adding any elements).
    """
    return array.reshape(closest_aspect(array.size, aspect_ratio))


def ravel_valid(arr: np.ndarray) -> np.ndarray:
    """
    Return a raveled (i.e. flattened, 1-D) copy of `arr` with all 'invalid'
    elements discarded, where 'invalid' means nonfinite and/or (if `arr`
    is a `MaskedArray`) masked.

    Intended primarily as a convenience for aggregate statistical operations,
    particularly operations that use library functions that don't like inf/nan
    and/or don't respect masks.
    """
    raveled = arr.ravel()
    if (
        arr.dtype.char in np.typecodes["AllInteger"]
        and not isinstance(arr, np.ma.MaskedArray)
    ):
        # this case can't actually have 'invalid' values and it is a waste of
        # time to check for them
        return raveled
    if isinstance(raveled, np.ma.MaskedArray):
        return raveled[np.isfinite(raveled) + raveled.mask].data
    return raveled[np.isfinite(raveled)]


rv = ravel_valid
"""alias for `ravel_valid()`."""


def setmask(arr: np.ndarray, value: Any, copy: bool = True) -> np.ndarray:
    """
    Convenience wrapper/handler for MaskedArray 'filling' behaviors that
    politely ignores non-masked arrays.

    Returns a (non-masked) array like `arr`, but with all masked elements
    replaced with `value`. If `copy`  is True, calls `arr.filled()` (which
    creates a copy of `arr`). Otherwise, modifies `arr.data` inplace and
    returns it.

    If `arr` is _not_ a MaskedArray, simply returns `arr` if `copy` is False,
    and `arr.copy()` if `copy` is True.

    Note: Does not attempt to typecast `arr` to match `value` or vice versa.
     Will throw an exception on numpy>=2.0.0 if `value` is incompatible with
     `arr.dtype`.
    """
    if not isinstance(arr, np.ma.MaskedArray):
        return arr if copy is False else arr.copy()
    if copy is True:
        return arr.filled(value)
    arr.data[arr.mask] = value
    return arr.data


def zero_mask(arr: np.ndarray, copy: bool = True) -> np.ndarray:
    """Alias for `setmask(arr, 0, copy)`"""
    return setmask(arr, 0, copy)


def nanmask(arr: np.ndarray, copy: bool = True) -> np.ndarray:
    """Alias for `setmask(arr, np.nan, copy)`"""
    return setmask(arr, np.nan, copy)


def make_mask_canvas(
    image: np.ma.MaskedArray,
    unmasked_alpha: float = 1,
    mask_alpha: float = 0
) -> np.ndarray:
    """
    Creates a half-float array shaped like the first two dimensions of `image`,
    which must be a 2D or 3D ndarray.

    The 'canvas' array's elements are `masked_alpha` where `image.mask` is True
    (True in any 'channel'/'band', if `image` is 3D), and 'unmasked_alpha'
    elsewhere.

    This can be used to create an alpha channel or as a sort of stencil.
    """
    if image.ndim == 3:
        mask = image.mask.sum(axis=-1).astype(bool)
    elif image.ndim == 2:
        mask = image.mask
    else:
        raise ValueError("`image` must be 2D or 3D")
    return np.where(mask, np.float16(mask_alpha), np.float16(unmasked_alpha))


def colorfill_maskedarray(
    masked_array: np.ma.MaskedArray,
    color: Union[float, tuple[float, float, float]] = 0.45,
    mask_alpha=None,
    unmasked_alpha=None,
) -> np.ndarray:
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


def maskwhere(
    images: Sequence[np.ndarray], constants: Collection[ArrayLike]
) -> list[np.ndarray]:
    """
    Return a list of boolean ndarrays corresponding to `images` such that
    the i-th element of that list is True where `images[i]` is in `constants`
    and False elsewhere.

    Intended primarily as a convenience function for producing special constant
    masks for a group of related images.
    """
    # noinspection PyTypeChecker
    return reduce(np.logical_or, [np.isin(i, constants) for i in images])


def _pick_mask_constructors(
    region: Literal['inner', 'outer']
) -> tuple[
    Callable[[np.ndarray, Any, Any], np.ma.MaskedArray],
    Callable[[np.ndarray, np.ndarray], np.ndarray]
]:
    """
    Helper function for cut_annulus() and cut_rectangle(). Selects a relation
    and a logical operator that facilitate masking everything outside or inside
    a set of bounds.
    """
    if region == "inner":
        return np.ma.masked_outside, np.logical_or
    elif region == "outer":
        return np.ma.masked_inside, np.logical_and
    raise ValueError(f"region={region}; region must be 'inner' or 'outer'")


def centered_indices(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a pair of arrays giving offsets from center, in index units, for
    the 2D array `arr` (or any other array of its shape).

    NOTE: These cannot actually be used as index arrays! They represent
        'distance' along each axis from center.
    """
    if arr.ndim != 2:
        raise ValueError("This function only takes 2D arrays.")
    y, x = np.indices(arr.shape)
    y0, x0 = (arr.shape[0] - 1) / 2, (arr.shape[1] - 1) / 2
    return y - y0, x - x0


def radial_index(arr: np.ndarray) -> np.ndarray:
    """
    Return an array giving Euclidean distance from center, in index units, for
    the 2D array `arr` (or any other array of its shape).
    """
    y_ix, x_ix = centered_indices(arr)
    return np.sqrt(y_ix**2 + x_ix**2)


def join_cut_mask(
    arr: np.ndarray, cut_mask: np.ndarray, copy: bool = True
) -> np.ma.MaskedArray:
    """
    Convenience function that masks `arr` with `cut_mask` if `arr` is not a
    `MaskedArray`, or adds `cut_mask` to `arr.mask` if `arr` is a MaskedArray.

    If `copy` is False and `arr` is a MaskedArray, this modifies `arr.mask`
    inplace. `copy` has no effect if `arr` is not a MaskedArray.

    `cut_mask` must be the same shape as `arr`; for best results, it should be
    boolean.
    """
    if not isinstance(arr, np.ma.MaskedArray):
        return np.ma.MaskedArray(arr, mask=cut_mask)
    if copy is True:
        arr = arr.copy()
    arr.mask = np.logical_or(cut_mask, arr.mask)
    return arr


def cut_annulus(
    arr: np.ndarray,
    bounds: tuple[Real, Real],
    region: Literal["inner", "outer"] = "inner",
    copy: bool = True
) -> np.ma.MaskedArray:
    """
    Return a version of the 2D ndarray `arr` with all values
    inside or outside of an annulus masked. `bounds[0]` defines the inner
    bound of the annulus; `bounds[1]` the outer. These definitions are given
    in units of array-index distance (which is to say 'pixels' or
    'picture-plane distance' for most images). The Euclidean norm is assumed.

    For example:
    `cut_annulus(arr, (100, 200), 'inner')` will return a version of `arr` that
    is masked everywhere except within a ring bounded by 100 array-index units
    from the center of `arr` and 200 array-index units from the center of
    `arr`. `cut_annulus(arr, (100, 200), 'outer')` will do the same, except
    that the array it returns will be masked _only_ within that ring.
    """
    mask_method, _ = _pick_mask_constructors(region)
    distance = radial_index(arr)
    pass_min = bounds[0] if bounds[0] is not None else distance.min()
    pass_max = bounds[1] if bounds[1] is not None else distance.max()
    cut_mask = mask_method(distance, pass_min, pass_max).mask
    return join_cut_mask(arr, cut_mask, copy)


def cut_rectangle(
    arr,
    bounds: tuple[tuple[Real, Real], tuple[Real, Real]],
    region: Literal["inner", "outer"] = "inner",
    center: bool = True,
    copy: bool = True
) -> np.ma.MaskedArray:
    """
    Return a version of the 2D ndarray `arr` with all values
    inside or outside of a rectangle masked. `bounds` must be a tuple of
    two tuples, each of which contains two numbers defining extrema of the
    rectangle along the x and y axes respectively. These definitions are given
    in units of array-index distance (which is to say 'pixels' for most
    images). If `center` is True (which it is by default), `bounds` defines
    edges with reference to offsets from the center of `arr`. If `center` is
    False, `bounds` defines edges with reference to the upperleft element of
    `arr` (in other words, standard numpy indexing).

    For example:
    `cut_rectangle(arr, ((-100, 100), (-200, 100)), 'inner')` will return a
    version of `arr` that is masked everywhere except within a rectangle
    that extends 100 pixels left, 100 pixels right, 200 pixels down, and 100
    pixels up from the its center element.
    """
    mask_method, op = _pick_mask_constructors(region)
    if center is True:
        y_dist, x_dist = centered_indices(arr)
    else:
        y_dist, x_dist = np.indices(arr.shape)
    masks = []
    x_bounds, y_bounds = bounds
    for bound, dist in zip((x_bounds, y_bounds), (x_dist, y_dist)):
        pass_min, pass_max = bound
        pass_min = pass_min if pass_min is not None else dist.min()
        pass_max = pass_max if pass_max is not None else dist.max()
        masks.append(mask_method(dist, pass_min, pass_max).mask)
    cut_mask = op(*masks)
    return join_cut_mask(arr, cut_mask, copy)


def zerocut(*args, **kwargs) -> np.ndarray:
    """
    Convenience wrapper for `cut_rectangle()` that automatically sets all
    masked elements to 0. Intended primarily for quick renders and spectral
    analyses.
    """
    return zero_mask(cut_rectangle(*args, **kwargs))
