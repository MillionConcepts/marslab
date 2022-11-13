from _operator import truediv, floordiv
from functools import reduce
from typing import Collection

import numpy as np
import scipy.ndimage as ndi
from more_itertools import windowed

from marslab.imgops.imgutils import (
    ravel_valid,
    colorfill_maskedarray,
    cut_annulus,
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


def centile_threshold(array, percentile):
    return array > np.percentile(ravel_valid(array), percentile)


def pluck_label(array, connectivity=1, label_ix=1):
    from skimage.measure import label

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
    return tuple(
        windowed([0, *split_indices, len(row) - 1], 2)
    )[widest_section]


def test_vertical_coverage(mask, row_ix, col_indices, cutoff):
    above = mask[:row_ix, col_indices[0] : col_indices[1]]
    if (result := (np.nonzero(above)[0].size / above.size)) < cutoff:
        return False, result
    return True, result


def test_coverage(mask, cutoff, extent=None, filled=None):
    if filled is None:
        filled = floodfill_mask(mask)
    if extent is None:
        extent = mask[np.nonzero(mask)].size
    fill_extent = filled[np.nonzero(filled)].size
    if (result := (extent / fill_extent)) < cutoff:
        return False, extent, filled, result
    return True, extent, filled, result


def test_extent(mask, cutoff, extent=None):
    if extent is None:
        extent = mask[np.nonzero(mask)].size
    if (result := (extent / mask.size)) < cutoff:
        return False, extent, result
    return True, extent, result


def _blank(shape):
    return np.full(shape, False)


def skymask(
    arrays,
    dilate_mean=None,
    opening_radius=None,
    percentile=90,
    extent_cutoff=0.03,
    coverage_cutoff=0.9,
    vertical_coverage_cutoff=0.9,
    floodfill=True,
    clear_above=True,
    cut_below=True,
):
    mean = _get_flat_mean(arrays, dilate_mean)
    segments = centile_threshold(mean, percentile)
    if opening_radius is not None:
        segments = ndi.binary_opening(
            segments, dilation_kernel(opening_radius * 2, sharp=True)
        )
    mask = pluck_label(segments)
    ok, filled, extent = True, None, None
    if extent_cutoff is not None:
        ok, extent, result = test_extent(mask, extent_cutoff, extent)
        if ok is False:
            return _blank(mask.shape)
    if coverage_cutoff is not None:
        ok, extent, filled, result = test_coverage(
            mask, coverage_cutoff, extent
        )
        if ok is False:
            return _blank(mask.shape)
    if not (
        (vertical_coverage_cutoff is not None) or clear_above or cut_below
    ):
        if floodfill is True:
            if filled is None:
                return floodfill_mask(mask)
            else:
                return filled
        return mask
    widest_row_ix = find_widest_row(mask)
    widest_row = mask[widest_row_ix]
    w_col_indices = get_widest_section(widest_row)
    if vertical_coverage_cutoff is not None:
        ok, result = test_vertical_coverage(
            mask, widest_row_ix, w_col_indices, vertical_coverage_cutoff
        )
        if ok is False:
            return _blank(mask.shape)
    # show(mask)
    if clear_above is True:
        mask[:widest_row_ix, w_col_indices[0] : w_col_indices[1]] = 1
    # show(mask)
    if cut_below is True:
        below = ~mask[widest_row_ix:]
        row_x = np.arange(w_col_indices[0] + 1, w_col_indices[1])
        bounds = [np.nonzero(below[:, x])[0][0] for x in row_x]
        for bound, x in zip(bounds, row_x):
            mask[(widest_row_ix + bound):, x] = 0
        mask = pluck_label(ndi.binary_erosion(mask, dilation_kernel(6)))
    if floodfill is True:
        return floodfill_mask(mask)
    return mask


def _get_flat_mean(arrays, dilate_mean):
    if all([isinstance(a, np.ma.MaskedArray) for a in arrays]):
        mean = _flat_masked_mean(arrays, dilate_mean)
    else:
        mean = np.ma.mean(np.ma.dstack(arrays), axis=-1)
    return mean


def _flat_masked_mean(arrays, dilate):
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
