"""
inline rendering functions for look pipelines. can also be called on their own.
"""

import io
from itertools import repeat
from functools import reduce
from typing import Union, Optional, Sequence, Collection

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy.typing import ArrayLike

from marslab.imgops.debayer import make_bayer, debayer_upsample
from marslab.imgops.imgutils import (
    normalize_range, eightbit, enhance_color, threshold_mask, skymask
)
from marslab.imgops.pltutils import (
    set_colorbar_font,
    get_mpl_image,
    attach_axis,
    strip_axes,
)


def decorrelation_stretch(
    channels: Sequence[np.ndarray],
    contrast_stretch: Optional[Union[Sequence[float], float]] = None,
    special_constants: Optional[Collection[float]] = None,
    sigma: Optional[float] = 1,
    threshold: Optional[tuple[float, float]] = None,
):
    """
    decorrelation stretch of passed array on last axis of array. see
    Gillespie et al. 1986, etc., etc.

    This is partly an adaptation of the MATLAB DCS implementation.
    Work towards this adaptation is partly due to Lukáš Brabec
    (github.com/lbrabec/decorrstretch) and Christian Tate (unreleased).

    channels: must be at least two ndarrays, numeric dtypes. each ndarray
    should be 2D (size > 1). no infs or nans. no channel may have entirely
    equal elements. extremely large variation in array means between channels
    relative to data type may result in errors.
    integer arrays will be cast to float32. float arrays of depth < 32 will
    be cast to float32.
    contrast_stretch: float or sequence of two floats. each must be
    between 0 and 100 inclusive. percentile ranges to use as max/min bounds
    of output array.
    sigma: fixed target standard deviation for each channel. must be >= 0;
    ranges between 0 and 1 recommended. None means that the original standard
    deviation per channel is used as the target (this is a 'classic'
    decorrelation stretch). Doesn't matter if you're applying a contrast
    stretch.
    """
    working_array = np.dstack(channels)
    if special_constants is not None:
        working_array = np.ma.masked_where(
            np.isin(working_array, special_constants), working_array
        )
    else:
        working_array = np.ma.masked_array(working_array)
    if threshold is not None:
        tmask = threshold_mask(channels)
        tmask = np.dstack([tmask for _ in range(3)])
        working_array.mask = np.logical_or(
            working_array.mask, tmask
        )
    input_shape = working_array.shape
    channel_vectors = working_array.reshape(-1, input_shape[-1])
    if channel_vectors.dtype not in [np.float32, np.float64]:
        working_dtype = np.float32
    else:
        working_dtype = channel_vectors.dtype
    if (cmask := channel_vectors.mask).size > 1:
        selector = cmask[:, 0] | cmask[:, 1] | cmask[:, 2]
        valid_vectors = channel_vectors[~selector]
    else:
        valid_vectors = channel_vectors
    channel_covariance = np.ma.cov(valid_vectors.T).astype(working_dtype)
    eigenvalues, eigenvectors = np.linalg.eig(channel_covariance)
    # diagonal matrix containing per-band "stretch factors"
    stretch_matrix = np.diag(1 / np.sqrt(eigenvalues))
    # mean values for each channel
    channel_means = np.mean(valid_vectors, axis=0, dtype=working_dtype)
    # full transformation matrix:
    # rotates into eigenspace of covariance matrix, applies stretch,
    # rotates back to channelspace, applies sigma scaling
    transformation_matrix = reduce(
        np.dot, [eigenvectors, stretch_matrix, eigenvectors.T]
    )
    # target per-channel standard deviation as a diagonalized matrix.
    # set equal to sigma if sigma is passed; otherwise simply set
    # equal to per-channel input standard deviation. if it's 1, skip the step.
    if sigma is None:
        channel_sigmas = np.diag(np.sqrt(channel_covariance.diagonal()))
        transformation_matrix = np.dot(transformation_matrix, channel_sigmas)
    elif sigma != 1:
        channel_sigmas = np.diag(
            np.array([sigma for _ in range(len(channels))])
        )
        transformation_matrix = np.dot(transformation_matrix, channel_sigmas)
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
    fill_mask=True,
    colorbar_fp=None
):
    """
    TODO: this is a bit of a hack. consider finding a cleaner way to do the
      compositing without necessarily returning a Figure, even if mpl is used
      as an intermediate step sometimes -- although to later make a colorbar
      correctly, if there is no ScalarMappable, range state will have to be
      stored separately, which is ugly and circuitous. so maybe no intermediate
      possibility, or at least intent
    """
    norm = plt.Normalize(vmin=overlay_image.min(), vmax=overlay_image.max())
    fig = plt.figure()
    ax = fig.add_subplot()
    if isinstance(base_cmap, str):
        base_cmap = cm.get_cmap(base_cmap)
    if isinstance(overlay_cmap, str):
        overlay_cmap = cm.get_cmap(overlay_cmap)
    base = base_cmap(normalize_range(base_image))
    overlay = overlay_cmap(norm(overlay_image))
    blend_array = np.full(base_image.shape, overlay_opacity)
    if fill_mask is True:
        blend_array = np.where(
            overlay_image.mask, np.zeros(base_image.shape), blend_array
        )
    base[:, :, 3] = 1 - blend_array
    overlay[:, :, 3] = blend_array
    ax.imshow(base)
    ax.imshow(overlay)
    strip_axes(ax)
    cax = attach_axis(ax, size="3%", pad="0.5%")
    colorbar = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=overlay_cmap),
        alpha=overlay_opacity,
        cax=cax,
    )
    if colorbar_fp is not None:
        set_colorbar_font(colorbar, colorbar_fp)
    return fig


# TODO: I think masking in this is too late and we should be doing it up
#  front, adding an optional median fill step for cases where we might have
#  undesirable white pixels everywhere or whatever.
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
    if isinstance(channels[0], np.ma.MaskedArray):
        composed = np.ma.dstack(channels)
    else:
        composed = np.dstack(channels)
    if special_constants is None:
        return composed
    special_mask = np.isin(composed, special_constants)
    if isinstance(composed, np.ma.MaskedArray):
        composed.mask = np.logical_or(composed.mask, special_mask)
    else:
        composed = np.ma.MaskedArray(composed, special_mask)
    return composed


def spectop_look(
    images,
    *,
    spectop=None,
    wavelengths=None,
    special_constants=None,
    threshold: Optional[tuple[float, float]] = None,
    skymask_threshold: Optional[float] = None
):
    mask = np.full(images[0].shape, False, bool)
    if special_constants is not None:
        for image in images:
            mask[np.nonzero(np.isin(image, special_constants))] = True
    if threshold is not None:
        mask = np.logical_or(mask, threshold_mask(images, threshold))
    if skymask_threshold is not None:
        mask = np.logical_or(mask, skymask(images, skymask_threshold))
    try:
        look = np.ma.masked_array(
            spectop(images, None, wavelengths)[0], mask=mask
        )
    except AssertionError:
        # print(f"spectop is {spectop}, wavelengths are {wavelengths}")
        raise
    return look


# TODO: cruft, this should be handled by BandSet -- but is it?
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


# TODO: probably all of this special-case special constants handling is
#  becoming hacky -- maybe solve it more consistently with masks?:
def colormapped_plot(
    array: np.ndarray,
    *,
    cmap=None,
    render_colorbar=False,
    no_ticks=True,
    colorbar_fp=None,
    special_constants=None,
    drop_mask_for_display=True,
    threshold_mask_array=None
):
    """generate a colormapped plot, optionally with colorbar, from 2D array"""
    # TODO: hacky bailout if this is stuck on the end of an overlay pipeline,
    #  this can be cleaned up much more effectively
    if isinstance(array, mpl.figure.Figure):
        return array
    normalization_array = array.copy()
    if threshold_mask is not None:
        normalization_array[np.nonzero(threshold_mask_array)] = np.nan
    if special_constants is not None:
        normalization_array = normalization_array[
            ~np.isin(normalization_array, special_constants)
        ]
    # TODO: hmm, we should be using this maybe?
    not_special = normalization_array[np.isfinite(normalization_array)]
    # this now should be using masks appropriately...but perhaps it is not
    norm = plt.Normalize(
        vmin=normalization_array.min(), vmax=normalization_array.max()
    )
    del normalization_array
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    if cmap is None:
        cmap = cm.get_cmap("Greys_r")
    # if drop_mask_for_display:
    #     array = array.data
    # TODO: this probably remains hacky...but it might not
    #  if it isn't we might want a _separate mask_
    #  for things like partials etc.
    # array[np.isin(array, special_constants)] = np.nan

    array = cmap(norm(array))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(array)
    if render_colorbar:
        cax = attach_axis(ax, size="3%", pad="0.5%")
        colorbar = plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
        )
        if colorbar_fp:
            set_colorbar_font(colorbar, colorbar_fp)
    if no_ticks:
        strip_axes(ax)
    return fig


def simple_figure(
    image: Union[ArrayLike, Image.Image], zero_mask=True, **imshow_kwargs
) -> mpl.figure.Figure:
    """
    wrap an array up in a matplotlib subplot and not much else
    """
    if (zero_mask is True) and isinstance(image, np.ma.MaskedArray):
        image = np.ma.filled(image, 0)
    fig, ax = plt.subplots()
    ax.imshow(image, **imshow_kwargs)
    strip_axes(ax)
    return fig


def render_nested_rgb_composite(
    channel_images,
    *,
    metadata,
    special_constants=None,
    norm_kwargs=None,
    **channel_instructions,
):
    # TODO: is there a cleaner way to handle this import?
    from marslab.imgops.look import Look
    rendered_channels = {}
    for channel, instruction in channel_instructions.items():
        pipeline = Look.compile_from_instruction(
            instruction, metadata, special_constants
        )
        rendered_channels[channel] = pipeline.execute(channel_images[channel])
    norm_kwargs = {} if norm_kwargs is None else norm_kwargs
    norm_channels = [
        normalize_range(rendered_channels[name], **norm_kwargs)
        for name in ("red", "green", "blue")
    ]
    return render_rgb_composite(
        norm_channels, special_constants=special_constants
    )
