"""
inline rendering functions for look pipelines. can also be called on their own.
"""

import io
from collections.abc import (
    Sequence,
)
from functools import reduce
from itertools import repeat
from typing import Union

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy.typing import ArrayLike

from marslab.imgops.debayer import make_bayer, debayer_upsample
from marslab.imgops.imgutils import (
    normalize_range,
    eightbit,
    enhance_color,
)
from marslab.imgops.pltutils import (
    set_colorbar_font,
    get_mpl_image, attach_axis, strip_axes,
)


def decorrelation_stretch(
    channels: Sequence[ArrayLike],
    *,
    contrast_stretch=None,
    special_constants=None,
    sigma=None,
):
    """
    decorrelation stretch of passed array on last axis of array. see
    Gillespie et al. 1986, etc., etc.

    This is partly an adaptation of the MATLAB DCS implementation.
    Work towards this adaptation is partly due to Lukáš Brabec
    (github.com/lbrabec/decorrstretch) and Christian Tate (unreleased).
    """
    working_array = np.dstack(channels)
    # TODO: this is not a good general case solution, might need to use masked
    #  arrays or some handrolled analog
    if special_constants is not None:
        working_array = np.where(
            np.isin(working_array, special_constants), 0, working_array
        )
    input_shape = working_array.shape
    channel_vectors = working_array.reshape(-1, input_shape[-1])
    if channel_vectors.dtype.char in np.typecodes["AllInteger"]:
        working_dtype = np.float32
    else:
        working_dtype = channel_vectors.dtype
    channel_covariance = np.cov(channel_vectors.T, dtype=working_dtype)
    # target per-channel standard deviation as a diagonalized matrix.
    # set equal to sigma if sigma is passed; otherwise simply set
    # equal to per-channel input standard deviation
    if sigma is not None:
        channel_sigmas = np.diag(
            np.array([sigma for _ in range(len(channels))])
        )
    else:
        channel_sigmas = np.diag(np.sqrt(channel_covariance.diagonal()))
    eigenvalues, eigenvectors = np.linalg.eig(channel_covariance)
    # diagonal matrix containing per-band "stretch factors"
    stretch_matrix = np.diag(1 / np.sqrt(eigenvalues))
    # mean values for each channel
    channel_means = np.mean(channel_vectors, axis=0, dtype=working_dtype)
    # full transformation matrix:
    # rotates into eigenspace of covariance matrix, applies stretch,
    # rotates back to channelspace, applies sigma scaling
    transformation_matrix = reduce(
        np.dot, [channel_sigmas, eigenvectors, stretch_matrix, eigenvectors.T]
    )
    # TODO: this 'offset' term does not explicitly exist in the matlab
    #  implementation. check against reference algorithm. it rarely does
    #  anything, though.
    offset = channel_means - np.dot(channel_means, transformation_matrix)
    # remove mean from each channel, transform, replace mean and add offset
    dcs_vectors = (
        np.dot((channel_vectors - channel_means), transformation_matrix)
        + channel_means
        + offset
    )
    dcs_array = dcs_vectors.reshape(input_shape)

    # special limiter included ensuite
    if contrast_stretch is None:
        return normalize_range(dcs_array)
    return enhance_color(dcs_array, (0, 1), contrast_stretch)


def render_overlay(
    overlay_image,
    base_image,
    *,
    overlay_cmap=cm.get_cmap("viridis"),
    base_cmap=cm.get_cmap("Greys_r"),
    overlay_opacity=0.5,
    mpl_settings=None,
):
    """
    TODO: this is a bit of a hack. consider finding a cleaner way to do the
      compositing without necessarily returning a Figure, even if mpl is used
      as an intermediate step sometimes -- although to later make a colorbar
      correctly, if there is no ScalarMappable, range state will have to be
      stored separately, which is ugly and circuitous. so maybe no intermediate
      possibility, or at least intent
    """
    if mpl_settings is None:
        mpl_settings = {}
    norm = plt.Normalize(vmin=overlay_image.min(), vmax=overlay_image.max())
    fig = plt.figure()
    ax = fig.add_subplot()

    base_image = normalize_range(base_image, (0, 1), 1)
    if isinstance(base_cmap, str):
        base_cmap = cm.get_cmap(base_cmap)
    if isinstance(overlay_cmap, str):
        overlay_cmap = cm.get_cmap(overlay_cmap)
    ax.imshow(
        base_cmap(base_image) * (1 - overlay_opacity)
        + overlay_cmap(norm(overlay_image)) * overlay_opacity
    )
    strip_axes(ax)
    cax = attach_axis(ax, size="3%", pad="0.5%")
    colorbar = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=overlay_cmap),
        alpha=overlay_opacity,
        cax=cax
    )
    if mpl_settings.get("colorbar_fp"):
        set_colorbar_font(colorbar, mpl_settings["colorbar_fp"])
    return fig


def render_rgb_composite(channels, *, special_constants=None):
    """
    render a composited image from three input channels. this is a good basis
    for producing both "true-color" and "enhanced-color" images from most
    filter sets.

    TODO: following is no longer true. move this as an interpretation step to
      Look.compile_from_instruction()
    this assumes normalization as a default option because you're presumably
    going to want to view these as "normal" RGB images, not scaled to an
    arbitrary colormap (although aren't they all?).
    """
    assert len(channels) == 3
    composed = np.dstack(channels)
    if special_constants is not None:
        composed = np.where(np.isin(composed, special_constants), 0, composed)
    return composed


def spectop_look(
    images,
    *,
    spectop=None,
    wavelengths=None,
    special_constants=None,
    default_value=0,
):
    look = spectop(images, None, wavelengths)[0]
    # ignoring nans and special constants in scaling
    look[np.where(np.isnan(look))] = default_value
    look[np.where(np.isinf(look))] = default_value
    if special_constants is not None:
        for image in images:
            look[np.where(np.isin(image, special_constants))] = default_value
    return look


# TODO: cruft, this should be handled by RGBset
def rgb_from_bayer(
    image,
    bayer_pattern,
    bayer_pixels=("red", ("green_1", "green_2"), "blue"),
):
    """
    assemble m x n x 3 array from specified bayer pixels of passed
    single m x n array or 3 m x n arrays
    """
    assert len(bayer_pixels) == 3, "this function only makes 3-channel images"
    if not isinstance(image, Sequence):
        image = repeat(image)
    elif len(image) == 1:
        image = repeat(image[0])
    else:
        assert (
            len(image) == 3
        ), "this function operates on either one or three input arrays"
    channels = []
    bayer_masks = None
    bayer_row_column = None
    for image, bayer_pixel in zip(image, bayer_pixels):
        if bayer_masks is None:  # i.e., the first time
            bayer_masks = make_bayer(image.shape, bayer_pattern)
            bayer_row_column = {
                pixel: (np.unique(mask[0]), np.unique(mask[1]))
                for pixel, mask in bayer_masks.items()
            }
        channels.append(
            debayer_upsample(
                image,
                bayer_pixel,
                masks=bayer_masks,
                row_column=bayer_row_column,
            )
        )
    return np.dstack(channels)


def make_thumbnail(
    image_array,
    thumbnail_size=(256, 256),
    file_or_path_or_buffer=None,
    filetype=None,
):
    """
    makes thumbnails from arrays or matplotlib images or PIL.Images
    """
    if file_or_path_or_buffer is None:
        file_or_path_or_buffer = io.BytesIO()
    if filetype is None:
        filetype = "jpeg"
    if isinstance(image_array, mpl.figure.Figure):
        thumbnail_array = get_mpl_image(image_array).convert("RGB")
    elif isinstance(image_array, np.ndarray):
        image_array = eightbit(image_array)
        thumbnail_array = Image.fromarray(image_array).convert("RGB")
    else:
        thumbnail_array = image_array
    thumbnail_array.thumbnail(thumbnail_size)
    thumbnail_array.save(file_or_path_or_buffer, filetype)
    return file_or_path_or_buffer


def colormapped_plot(
    array: np.ndarray,
    *,
    cmap=None,
    render_colorbar=False,
    no_ticks=True,
    colorbar_fp=None,
    special_constants = None
):
    """generate a colormapped plot, optionally with colorbar, from 2D array"""
    # TODO: hacky bailout if this is stuck on the end of a pipeline it
    #   shouldn't be, remove this or something
    if isinstance(array, mpl.figure.Figure):
        return array
    if special_constants is not None:
        not_special = array[~np.isin(array, special_constants)]
    else:
        not_special = array
    norm = plt.Normalize(vmin=not_special.min(), vmax=not_special.max())
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    if cmap is None:
        cmap = cm.get_cmap("Greys_r")
    array[np.isin(array, special_constants)] = np.nan
    array = cmap(norm(array))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(array)
    if render_colorbar:
        cax = attach_axis(ax, size="3%", pad="0.5%")
        colorbar = plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
        )
        if colorbar_fp:
            set_colorbar_font(colorbar, colorbar_fp)
    if no_ticks:
        strip_axes(ax)
    return fig


def simple_figure(image: Union[ArrayLike, Image.Image]) -> mpl.figure.Figure:
    """
    wrap an array up in a matplotlib subplot and not much else
    """
    fig, ax = plt.subplots()
    ax.imshow(image)
    strip_axes(ax)
    return fig
