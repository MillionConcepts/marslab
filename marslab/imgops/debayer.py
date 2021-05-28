from collections.abc import (
    Sequence,
    Mapping,
)
from typing import Union

import numpy as np

# bayer pattern definitions
from marslab.imgops.imgutils import bilinear_interpolate_subgrid

RGGB_PATTERN = {
    "red": (0, 0),
    "green_1": (0, 1),
    "green_2": (1, 0),
    "blue": (1, 1),
}


def make_pattern_masks(
    array_shape: tuple[int, int],
    bayer_pattern: Mapping[str, tuple[int]],
    pattern_shape: tuple[int, int],
) -> dict[str, tuple]:
    """
    given (y, x) array shape and a dict of tuples defining grid class names
    and positions,
    generate a dict of array pairs containing y, x coordinates for each
    named grid class in a m x n pattern beginning at the upper-left-hand corner
    of an array of shape shape and extending across the entirety of that array,


    supports only m x n patterns (like conventional bayer patterns); not a
    'generalized n-d discrete
    frequency class sorter' or whatever
    """

    y_coord, x_coord = np.meshgrid(
        np.arange(array_shape[0]), np.arange(array_shape[1])
    )
    masks = {}
    for name, position in bayer_pattern.items():
        y_slice = slice(position[0], None, pattern_shape[0])
        x_slice = slice(position[1], None, pattern_shape[1])
        masks[name] = (
            np.ravel(y_coord[y_slice, x_slice]),
            np.ravel(x_coord[y_slice, x_slice]),
        )
    return masks


def make_bayer(
    array_shape: Sequence[int, int],
    bayer_pattern: Mapping[str, Sequence[int]],
) -> dict[str, tuple]:
    """
    alias to make_pattern_masks
    """
    return make_pattern_masks(array_shape, bayer_pattern, (2, 2))


def debayer_upsample(
    image: np.ndarray,
    pixel: Union[str, Sequence[str]],
    pattern: Mapping[str, tuple] = None,
    masks: Mapping[str, tuple] = None,
    row_column: Mapping[str, tuple] = None,
    **_kwargs  # TODO: hacky!
) -> np.ndarray:
    """
    debayer and upsample an image, given a bayer pixel name or names
    and either a bayer pattern or explicitly precalculated sets of absolute (and
    optionally relative) coordinates for those pixels.
    averages arrays if more than one pixel name is given.
    TODO: the preamble to this may be excessively convoluted.
    """
    assert not (pattern is None and masks is None), (
        "debayer_upsample() must be passed either a bayer pattern or "
        "precalculated bayer masks."
    )
    if isinstance(pixel, str):
        pixel = [pixel]
    if masks is None:
        masks = make_bayer(image.shape, pattern)
    if row_column is None:
        row_column = {
            pixel: (np.unique(mask[0]), np.unique(mask[1]))
            for pixel, mask in masks.items()
        }
    upsampled_images = []
    for pix in pixel:
        mask = masks[pix]
        if row_column is None:
            rows, columns = np.unique(mask[0]), np.unique(mask[1])
        else:
            rows, columns = row_column[pix]
        subframe = image[mask].reshape(rows.size, columns.size, order="F")
        upsampled_images.append(
            bilinear_interpolate_subgrid(rows, columns, subframe, image.shape)
        )
    if len(upsampled_images) == 1:
        return upsampled_images[0]
    return np.mean(np.dstack(upsampled_images), axis=-1)


# TODO: extract methods, clean up
def mask_bayer_pixels(
    image: np.ndarray,
    pixel: Union[str, Sequence[str]],
    pattern: Mapping[str, tuple] = None,
    masks: Mapping[str, tuple] = None,
    default = 0,
    **_kwargs  # TODO: hacky!
) -> np.ndarray:
    """
    return a version of an image with non-matching bayer pixels set to default
    """
    assert not (pattern is None and masks is None), (
        "debayer_blank() must be passed either a bayer pattern or "
        "precalculated bayer masks."
    )
    masked = image.copy()
    if isinstance(pixel, str):
        pixel = [pixel]
    if masks is None:
        masks = make_bayer(image.shape, pattern)
    for pix in masks.keys():
        if pix in pixel:
            continue
        mask = masks[pix]
        masked[mask] = default
    return masked