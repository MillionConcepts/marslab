from _operator import truediv, floordiv
from functools import reduce
from typing import Collection

import numpy as np
import scipy.ndimage as ndi

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


def skymask(
    arrays,
    dilate_mean=None,
    opening_radius=5,
    percentile=90,
    extent_ratio=0.9,
    floodfill=True,
):
    mean = _get_flat_mean(arrays, dilate_mean)
    segments = centile_threshold(mean, percentile)
    if opening_radius is not None:
        segments = ndi.binary_opening(
            segments, dilation_kernel(opening_radius * 2, sharp=True)
        )
    mask = pluck_label(segments)
    if (floodfill is False) and (extent_ratio is None):
        return mask
    filled = floodfill_mask(mask)
    if extent_ratio is not None:
        mask_extent = np.nonzero(mask)[0].size
        fill_extent = np.nonzero(filled)[0].size
        if (mask_extent / fill_extent) < extent_ratio:
            return np.full(mask.shape, False)
    if floodfill is False:
        return mask
    return filled


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
