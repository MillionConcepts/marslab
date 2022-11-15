from _operator import truediv, floordiv
from functools import reduce
from inspect import signature
from types import MappingProxyType
from typing import Collection

import numpy as np
import scipy.ndimage as ndi
from more_itertools import windowed
from skimage.measure import label

from marslab.imgops.imgutils import (
    ravel_valid,
    colorfill_maskedarray,
    cut_annulus,
    eightbit,
    normalize_range,
)


def threshold_mask(
    arrays: Collection[np.ndarray], percentiles=(1, 99), operator="or"
):
    masks = []
    if operator == "mean":
        arrays = [_get_flat_mean(arrays, None)]
    for array in arrays:
        low, high = np.percentile(ravel_valid(array), percentiles)
        masks.append(np.ma.masked_outside(array, low, high).mask)
    if operator == "mean":
        return masks[0]
    if operator == "or":
        return reduce(np.logical_or, masks)
    return reduce(np.logical_and, masks)


def ink(
    images,
    median=5,
    minimum=5,
    maximum=3,
    edge_thresholds=(60, 110),
    erosion=3,
    input_stretch=(10, 1)
):
    mean = maybe_filter(
        _get_flat_mean(images, None, normalize=input_stretch), median
    )
    edges = outline(mean, edge_thresholds, maximum, erosion=erosion)
    canvas = np.ones(images[0].shape)
    canvas[edges] = 0
    return maybe_filter(canvas, minimum, filt=ndi.minimum_filter)


def centile_threshold(array, percentile):
    return array > np.percentile(ravel_valid(array), percentile)


def pluck_label(array, connectivity=1, label_ix=1):
    mask = label(array, connectivity=connectivity) == label_ix
    return mask


def floodfill_mask(mask):
    import cv2

    cvmask = mask.astype("uint8")
    contours, hierarchy = cv2.findContours(
        cvmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )
    flood = cv2.fillPoly(cvmask, contours, 1)
    return flood.astype("bool")


def crop_bounds(mask):
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    return mask[rows][:, cols], rows, cols


def find_widest_row(mask):
    cropped, rows, cols = crop_bounds(mask)
    contiguous = []
    for row in cropped:
        truthy = [
            a
            for a in np.split(row, np.nonzero(np.diff(row))[0] + 1)
            if a.any()
        ]
        contiguous.append(max([t.size for t in truthy]))
    contiguous = np.array(contiguous)
    return np.nonzero(rows)[0][
        np.nonzero(contiguous == contiguous.max())[0][-1]
    ]


def get_widest_section(row):
    split_indices = np.nonzero(np.diff(row))[0] + 1
    length_indices = {}
    for ix, section in enumerate(np.split(row, split_indices)):
        if section.any():
            length_indices[ix] = section
    widest = max([len(v) for v in length_indices.values()])
    widest_section = [
        k for k, v in length_indices.items() if len(v) == widest
    ][0]
    return tuple(windowed([0, *split_indices, len(row) - 1], 2))[
        widest_section
    ]


def test_axes(mask, shape):
    vertical, horizontal = shape
    if (vertical is None) and (horizontal is None):
        return True
    col_indices, row_ix = widest_part(mask)
    if vertical is not None:
        above = mask[:row_ix, col_indices[0] : col_indices[1]]
        try:
            if (np.nonzero(above)[0].size / above.size) < vertical:
                return False
        except ZeroDivisionError:
            pass
    if horizontal is not None:
        if ((col_indices[1] - col_indices[0]) / mask.shape[1]) < horizontal:
            return False
    return True


def test_coverage(mask, cutoff):
    if cutoff is None:
        return True
    mask_extent = mask[np.nonzero(mask)].size
    filled_extent = mask[np.nonzero(floodfill_mask(mask))].size
    if (mask_extent / filled_extent) < cutoff:
        return False
    return True


def test_extent(mask, cutoff):
    if cutoff is None:
        return True
    if (mask[np.nonzero(mask)].size / mask.size) < cutoff:
        return False
    return True


def _blank(shape):
    return np.full(shape, False)


def maybe_filter(array, size=None, footprint=None, filt=ndi.median_filter):
    if (size is None) and (footprint is None):
        return array
    if footprint is None:
        footprint = np.ones((size, size))
    sig = signature(filt)
    for kwarg in ("structure", "footprint"):
        if kwarg in sig.parameters.keys():
            break
    return filt(array, **{kwarg: footprint})


def widest_part(mask):
    widest_row_ix = find_widest_row(mask)
    widest_row = mask[widest_row_ix]
    w_col_indices = get_widest_section(widest_row)
    return w_col_indices, widest_row_ix


def check_mask_validity(mask, extent=None, coverage=None, v=None, h=None):
    for check, threshold in zip(
        (test_extent, test_coverage, test_axes), (extent, coverage, (v, h))
    ):
        if check(mask, threshold) is False:
            return False
    return True


def outline(array, edge_thresholds=(60, 100), maximum=5, erosion=3):
    import cv2

    if array.dtype != np.uint8:
        array = eightbit(array)
    edges = cv2.Canny(array, *edge_thresholds)
    edges = maybe_filter(edges, maximum, filt=ndi.maximum_filter)
    return maybe_filter(edges, erosion, filt=ndi.binary_erosion)


def _get_flat_mean(arrays, dilate_mean=None, normalize=None):
    if normalize is not None:
        arrays = [
            normalize_range(array, stretch=normalize) for array in arrays
        ]
    if all([isinstance(a, np.ma.MaskedArray) for a in arrays]):
        mean = _flat_masked_mean(arrays, dilate_mean)
    else:
        mean = np.ma.mean(np.ma.dstack(arrays), axis=-1)
    return mean


def clear_overhead_pixels(mask, row, start, stop):
    """warning: mutates mask inplace"""
    mask[:row, start:stop] = 1
    return mask


def trim_infiltrate_pixels(mask, row=0, trim_depth=10, depth_sigma=7):
    not_mask = ~floodfill_mask(mask).copy()
    not_mask[:row] = 0
    depth = ndi.gaussian_filter(np.cumsum(not_mask, axis=0), depth_sigma)
    depth_mask = depth <= trim_depth
    return np.logical_and(mask, depth_mask)


def refit_mask(
    mask, clear=True, trim=False, trim_depth=10, depth_sigma=7
):
    if (clear is False) and (trim is False):
        return mask
    w_col_indices, w_row_ix = widest_part(floodfill_mask(mask))
    if clear is True:
        mask = clear_overhead_pixels(
            mask, w_row_ix, w_col_indices[0], w_col_indices[-1]
        )
    if trim is True:
        mask = trim_infiltrate_pixels(
            mask, w_row_ix, trim_depth, depth_sigma
        )
    return pluck_label(mask)


def pick_valid_label(array, cutoffs):
    labels = label(array)
    for label_ix in np.unique(labels[0][labels[0] >= 1]):
        mask = labels == label_ix
        if check_mask_validity(mask, **cutoffs) is True:
            return mask
    return None


def superpixel_edges(
    images,
    stretch=(10, 1),
    slic_kwargs=MappingProxyType(
        {"n_segments": 5, "sigma": 2, "compactness": 3}
    ),
):
    from skimage.segmentation import find_boundaries, slic

    cube = np.dstack(
        [normalize_range(image, stretch=stretch) for image in images]
    )
    return find_boundaries(slic(cube, **slic_kwargs))


def skymask(
    arrays,
    percentile=75,
    edge_params=MappingProxyType({"sigma": 3}),
    input_median=5,
    trace_maximum=5,
    cutoffs=MappingProxyType(
        {"extent": 0.05, "coverage": None, "v": 0.9, "h": None}
    ),
    input_mask_dilation=None,
    input_stretch=(10, 1),
    floodfill=True,
    trim_params=MappingProxyType({"trim": False}),
    clear=True,
    colorblock=False,
):
    mean = maybe_filter(
        _get_flat_mean(arrays, input_mask_dilation, normalize=input_stretch),
        input_median
    )
    trace = ~outline(mean, **edge_params)
    if percentile is not None:
        trace = np.logical_and(trace, centile_threshold(mean, percentile))
    if colorblock is True:
        trace[superpixel_edges(arrays)] = 0
    if trace_maximum is not None:
        trace = ndi.minimum_filter(trace, trace_maximum)
    if (mask := pick_valid_label(trace, cutoffs)) is None:
        return _blank(mean.shape)
    mask = refit_mask(mask, clear, **trim_params)
    if floodfill is True:
        return floodfill_mask(mask)
    return mask


def _flat_masked_mean(arrays, dilate=None):
    if dilate is not None:
        mask = flatmask(arrays, dilate=dilate)
    else:
        mask = reduce(np.logical_or, [a.mask for a in arrays])
    mean = np.ma.MaskedArray(np.mean(np.dstack(arrays), axis=-1), mask=mask)
    return mean


def extract_masks(images, instructions=None):
    if instructions is None:
        return images, None, []
    passmasks, sendmasks = [], []
    if any((inst.get("colorfill") is not None for inst in instructions)):
        canvas = np.zeros(images[0].shape)
    for inst in instructions:
        do_pass, do_send = inst.get("pass", False), inst.get("send", True)
        if (do_pass is False) and (do_send is False):
            continue
        func, params = inst["function"], inst["params"]
        mask = func(images, **params)
        if do_pass is True:
            passmasks.append(mask)
        if do_send is True:
            if (color_kwargs := inst.get("colorfill")) is not None:
                sendmasks.append(
                    colorfill_maskedarray(
                        np.ma.masked_array(canvas, mask), **color_kwargs
                    )
                )
            else:
                sendmasks.append(mask)
    flattened_mask = (
        None if len(passmasks) == 0 else reduce(np.logical_or, passmasks)
    )
    return images, flattened_mask, sendmasks


def flatmask(images, dilate=None, sharp=False, square=False):
    try:
        mask = reduce(
            np.logical_or,
            [i.mask for i in images if isinstance(i, np.ma.MaskedArray)],
        )
    except TypeError as te:
        if "empty iterable" not in str(te):
            raise
        return np.full(images[0].shape, False)
    if dilate is not None:
        # noinspection PyTypeChecker
        return dilate_mask(mask, dilate, sharp, square)


def dilation_kernel(size=2, sharp=False, square=False):
    if size < 2:
        raise ValueError("Kernel size must be at least 2.")
    kernel = np.ones((size, size))
    if square is True:
        return kernel
    op = truediv if sharp is False else floordiv
    distance = op(size, 2)
    return np.ma.filled(cut_annulus(kernel, (0, distance)), 0)


def dilate_mask(image, size, sharp=False, square=False, copy=True):
    if not isinstance(image, np.ma.MaskedArray):
        return ndi.binary_dilation(
            image.astype(bool), structure=dilation_kernel(size, sharp, square)
        )
    if copy is True:
        image = image.copy()
    image.mask = ndi.binary_dilation(
        image.mask, structure=dilation_kernel(size, sharp, square)
    )
    return image
