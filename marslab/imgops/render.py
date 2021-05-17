"""
inline rendering functions for look pipelines. can also be called on their own.
"""

import io
from collections.abc import (
    Sequence,
)
from functools import reduce
from itertools import repeat

import PIL.Image
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from marslab.imgops.debayer import make_bayer, debayer_upsample
from marslab.imgops.imgutils import (
    apply_image_filter,
    normalize_range,
    eightbit, enhance_color,
)
from marslab.imgops.pltutils import (
    set_colorbar_font,
    remove_ticks,
    get_mpl_image,
)


def decorrelation_stretch(
    channels,
    *,
    contrast_stretch=None,
    special_constants=None,
    sigma=None,
):
    """
    decorrelation stretch of passed array on last axis of array. see
    Gillespie et al. 1986, etc., etc.

    This is partly an adaptation of the MATLAB DCS implementation.
    Work towards this adaptation is partly due to lbrabec
    (github.com/lbrabec/decorrstretch) and Christian Tate (unreleased).
    """
    working_array = np.dstack(channels)
    # TODO: this is not a good general case solution
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
    if contrast_stretch is None:
        return normalize_range(dcs_array)
    return enhance_color(dcs_array, 0, 1, contrast_stretch, contrast_stretch)


def render_overlay(
    overlay_image,
    base_image,
    *,
    overlay_cmap="viridis",
    base_cmap="Greys_r",
    overlay_opacity=0.5,
    mpl_settings=None,
):
    norm = plt.Normalize(vmin=overlay_image.min(), vmax=overlay_image.max())
    fig = plt.figure()
    ax = fig.add_subplot()
    colorbar = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=overlay_cmap),
        ax=ax,
        fraction=0.03,
        pad=0.04,
        alpha=overlay_opacity,
    )
    base_image = normalize_range(base_image, 0, 1, 1, 1)
    if isinstance(base_cmap, str):
        base_cmap = cm.get_cmap(base_cmap)
    if isinstance(overlay_cmap, str):
        overlay_cmap = cm.get_cmap(overlay_cmap)
    ax.imshow(
        base_cmap(base_image) * (1 - overlay_opacity)
        + overlay_cmap(norm(overlay_image)) * overlay_opacity
    )
    remove_ticks(ax)
    if mpl_settings is not None:
        if mpl_settings.get("colorbar_fp"):
            set_colorbar_font(colorbar, mpl_settings["colorbar_fp"])
    return fig


def render_rgb_composite(
    channels,
    *,
    special_constants=None,
    normalize=(0, 1, 0, 0),
    image_filter=None,
    render_mpl=False,
    prefilter=None,
):
    """
    render a composited image from three input channels. this is a good basis
    for producing both "true-color" and "enhanced-color" images from most
    filter sets.

    this assumes normalization as a default option because you're presumably
    going to want to view these as "normal" RGB images, not scaled to an
    arbitrary colormap (although aren't they all?).
    """
    assert len(channels) == 3
    composed = [
        apply_image_filter(channel.copy(), prefilter) for channel in channels
    ]
    composed = np.dstack(composed)
    if special_constants is not None:
        composed = np.where(np.isin(composed, special_constants), 0, composed)
    if normalize not in (False, None):
        composed = normalize_range(composed, *normalize)
    composed = apply_image_filter(composed, image_filter)

    if render_mpl is True:
        return simple_mpl_figure(composed)
    return composed


def spectop_look(
    images,
    *,
    spectop=None,
    wavelengths=None,
    special_constants=None,
    clip=None,
    default_value=0,
):
    rapidlook = spectop(images, None, wavelengths)[0]
    # ignoring nans and special constants in scaling
    rapidlook[np.where(np.isnan(rapidlook))] = default_value
    rapidlook[np.where(np.isinf(rapidlook))] = default_value
    if special_constants is not None:
        for image in images:
            rapidlook[
                np.where(np.isin(image, special_constants))
            ] = default_value
    if clip is not None:
        if "params" in clip.keys():
            params = clip["params"]
        else:
            params = {}
        rapidlook = clip["function"](rapidlook, **params)
    return rapidlook


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
        thumbnail_array = PIL.Image.fromarray(image_array).convert("RGB")
    else:
        thumbnail_array = image_array
    thumbnail_array.thumbnail(thumbnail_size)
    thumbnail_array.save(file_or_path_or_buffer, filetype)
    return file_or_path_or_buffer


def colormapped_plot(
    array,
    *,
    cmap=None,
    render_colorbar=False,
    no_ticks=True,
    colorbar_fp=None,
):
    """generate a colormapped plot, optionally with colorbar, from 2D array"""
    # TODO: hacky bailout if this is stuck on the end of a pipeline it
    #   shouldn't be, remove this or something
    if isinstance(array, mpl.figure.Figure):
        return array
    norm = plt.Normalize(vmin=array.min(), vmax=array.max())
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    if cmap is None:
        cmap = cm.get_cmap("Greys_r")
    array = cmap(norm(array))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(array)
    if render_colorbar:
        colorbar = plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            fraction=0.03,
            pad=0.04,
        )
        if colorbar_fp:
            set_colorbar_font(colorbar, colorbar_fp)
    if no_ticks:
        remove_ticks(ax)
    return fig


def simple_mpl_figure(
    image,
):
    """
    wrap an image up in a matplotlib subplot and not much else
    """
    fig = plt.figure()
    plt.tight_layout()
    ax = fig.add_subplot()
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig
