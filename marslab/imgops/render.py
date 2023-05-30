"""
inline rendering functions for look pipelines. can also be called on their own.
"""
import io
from collections import defaultdict
from functools import reduce
from itertools import repeat, chain
from typing import Union, Optional, Sequence

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from dustgoggles.structures import separate_by
from numpy.typing import ArrayLike

from marslab.imgops.debayer import make_bayer, debayer_upsample
from marslab.imgops.imgutils import normalize_range, eightbit, enhance_color, \
    clip_unmasked, colorfill_maskedarray
from marslab.imgops.pltutils import (
    set_colorbar_font,
    get_mpl_image,
    attach_axis,
    strip_axes,
)


def decorrelation_stretch(
    channels: Sequence[np.ndarray],
    flat_mask=None,
    contrast_stretch: Optional[Union[Sequence[float], float]] = None,
    sigma: Optional[float] = 1,
    mask_fill_tone=None,
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
    if any(isinstance(c, np.ma.MaskedArray) for c in channels):
        working_array = np.ma.dstack(channels)
    else:
        working_array = np.dstack(channels)
    if flat_mask is not None:
        flat_mask = np.dstack([flat_mask] * 3)
        if not isinstance(working_array, np.ma.MaskedArray):
            working_array = np.ma.masked_array(working_array, mask=flat_mask)
        else:
            working_array.mask += flat_mask
    input_shape = working_array.shape
    channel_vectors = working_array.reshape(-1, input_shape[-1])
    if channel_vectors.dtype not in [np.float32, np.float64]:
        working_dtype = np.float32
    else:
        working_dtype = channel_vectors.dtype
    if (
        isinstance(channel_vectors, np.ma.MaskedArray)
        and (cmask := channel_vectors.mask).size > 1
    ):
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
        image = normalize_range(dcs_array)
    else:
        image = enhance_color(dcs_array, (0, 1), contrast_stretch)
    if (isinstance(image, np.ma.MaskedArray)) and (mask_fill_tone is not None):
        image = colorfill_maskedarray(image, mask_fill_tone)
    return image


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
    flat_mask=None,
    *,
    spectop=None,
    wavelengths=None,
):
    look = spectop(images, None, wavelengths)[0]
    if flat_mask is None:
        return look
    if isinstance(look, np.ma.MaskedArray):
        look.mask += flat_mask
    else:
        look = np.ma.MaskedArray(look, mask=flat_mask)
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
    cmap=None,
    render_colorbar=False,
    no_ticks=True,
    colorbar_fp=None,
    special_constants=None,
    mask_fill_color=0.45,
    drop_mask=True,
    alpha=None,
    layers=None,
    n_ticks=3
):
    """
    generate a colormapped plot, optionally with colorbar, from 2D array.
    render as many overlays and underlays as you like.
    layers may either be ndarrays, dicts of the format:
    {'layer_ix': int, 'image': np.ndarray},
    (where the colormapped array is always at layer_ix 0),
    or lists or tuples of the same.
    ndarrays without a layer_ix field will always be treated as overlays,
    rendered in argument order after the layer with the highest specified
    layer_ix.
    3D layers will be treated as RGB(A); 2D layers will be treated as
    grayscale.
    Note that if opacity is not set and cmap doesn't define any
    transparency, you won't see any underlays.
    """
    # TODO: hacky bailout if this is stuck on the end of an overlay pipeline,
    #  this can be cleaned up much more effectively
    if isinstance(array, mpl.figure.Figure):
        return array
    normalization_array = array.copy()
    if isinstance(array, np.ma.masked_array):
        normalization_array = normalization_array[~normalization_array.mask]
    if special_constants is not None:
        normalization_array = normalization_array[
            ~np.isin(normalization_array, special_constants)
        ]
    normalization_array = normalization_array[np.isfinite(normalization_array)]
    # this now should be using masks appropriately...but perhaps it is not
    norm = plt.Normalize(
        vmin=normalization_array.min(), vmax=normalization_array.max()
    )
    del normalization_array
    if isinstance(array, np.ma.masked_array):
        if drop_mask is True:
            array = array.data
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    if cmap is None:
        cmap = cm.get_cmap("Greys_r")
    mapped = cmap(norm(array))
    if isinstance(array, np.ma.MaskedArray) and (drop_mask is False):
        mapped[array.mask] = mask_fill_color
    del array
    if alpha is not None:
        mapped[:, :, 3] = alpha
    if layers is not None:
        fig, ax = flatten_into_figure(
            list(layers) + [{"layer_ix": 0, "image": mapped}]
        )
    else:
        fig, ax = plt.subplots()
        ax.imshow(mapped)
    if render_colorbar:
        attach_colorbar(ax, cmap, colorbar_fp, norm, n_ticks)
    if no_ticks:
        strip_axes(ax)
    return fig


def _tformat(number, order, precision):
    if order > 2:
        formatted = ("{:." + str(precision) + "e}").format(number)
        return "".join([formatted[:-2], formatted[-1]])
    if order + precision < 0:
        return round(number)
    return round(number, order + precision)


def attach_colorbar(ax, cmap, colorbar_fp, norm, n_ticks=3):
    cax = attach_axis(ax, size="3%", pad="0.5%")
    colorbar = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax
    )
    ymin, ymax = colorbar.ax.get_ylim()
    extent = ymax - ymin
    gauge = np.linspace(ymin, ymax, n_ticks * 4 - 1)
    order = int(1 - np.floor(np.log10(extent)))
    precision, ticks, labels = 0, [], []
    while len(labels) < (n_ticks - 1):
        ticks, labels = _trylabel(order, precision, gauge, n_ticks)
        precision += 1
    colorbar.ax.set_yticks(ticks)
    colorbar.ax.set_yticklabels(labels)
    if colorbar_fp:
        set_colorbar_font(colorbar, colorbar_fp)


def _trylabel(order, precision, gauge, n_ticks):
    ticks, labels = [], []
    for ix, t in enumerate(gauge):
        formatted = _tformat(t, order, precision)
        if formatted in labels:
            continue
        ticks.append(t)
        labels.append(formatted)
    if len(labels) > n_ticks:
        indices = []
        n_ticks = 10
        needed = n_ticks
        for cutoff in np.percentile(ticks, np.linspace(0, 100, needed)):
            indices.append(np.argmin(np.abs(ticks - cutoff)))
        ticks = [ticks[ix] for ix in indices]
        labels = [labels[ix] for ix in indices]
    return ticks, labels


def _duck_alpha(rgb, a):
    return np.einsum('ijk,ij->ijk', rgb, a)


def merge_layers(lower, upper):
    if upper.shape[2] == 3:
        return upper
    alpha_upper = upper[:, :, 3]
    if lower.shape[2] == 3:
        alpha_lower = 1 - alpha_upper
    else:
        alpha_lower = np.min([1 - alpha_upper, lower[:, :, 3]], axis=0)
    rgb = (
        _duck_alpha(upper[:, :, :3], alpha_upper)
        + _duck_alpha(lower[:, :, :3], alpha_lower)
    )
    return np.dstack([rgb, alpha_upper + alpha_lower])


def flatten_into_figure(layers, **imshow_kwargs):
    seq, single = separate_by(layers, lambda l: isinstance(l, (tuple, list)))
    layers = list(chain.from_iterable(seq)) + single
    dicts, arrays = separate_by(layers, lambda l: isinstance(l, dict))
    order = [d.get("layer_ix") for d in dicts]
    assert len(set(order)) == len(order)
    dicts.sort(key=lambda d: d.get("layer_ix"))
    images = []
    for image in list(map(lambda d: d.get("image"), dicts)) + arrays:
        if len(image.shape) != 3:
            image = cm.get_cmap("Greys_r")(image)
        images.append(image)
    fig, ax = plt.subplots()
    ax.imshow(reduce(merge_layers, images), **imshow_kwargs)
    return fig, ax


def simple_figure(
    image: Union[ArrayLike, Image.Image],
    zero_mask=True,
    layers=None,
    **imshow_kwargs,
) -> mpl.figure.Figure:
    """
    wrap an array + optional layers up in a matplotlib subplot
    """
    if (zero_mask is True) and isinstance(image, np.ma.MaskedArray):
        image = np.ma.filled(image, 0)
    if layers is not None:
        fig, ax = flatten_into_figure(
            list(layers) + [{"layer_ix": 0, "image": image}], **imshow_kwargs
        )
    else:
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
