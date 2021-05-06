import io
from functools import reduce, partial
from itertools import repeat
from operator import contains
from typing import Sequence, Union, Mapping, Any
from warnings import warn

import matplotlib.figure
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import pdr
from astropy.io import fits
from scipy.ndimage import sobel, distance_transform_edt

import marslab.spectops


def simple_mpl_figure(image):
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


def remove_ticks_and_style_colorbar(fig, ax, colorbar, mpl_options):
    ax.set_xticks([])
    ax.set_yticks([])
    if "tick_fp" in mpl_options:
        for tick in colorbar.ax.get_yticklabels():
            tick.set_font_properties(mpl_options["tick_fp"])
        colorbar.ax.get_yaxis().get_offset_text().set_font_properties(
            mpl_options["tick_fp"]
        )
    return fig


def apply_image_filter(image, image_filter=None):
    """
    unpacking / pseudo-dispatch function
    """
    if image_filter is None:
        return image
    if "params" in image_filter.keys():
        params = image_filter["params"]
    else:
        params = {}
    return image_filter["function"](image, **params)


def make_roi_hdu(input_array, roi_name, metadata_dict):
    """
    make an individual HDU for a marslab .roi file
    """
    roi_hdu = fits.PrimaryHDU(input_array)
    # this parameter is separate and mandatory because we really
    # need every ROI to have some kind of distinguishing name
    roi_hdu.header["NAME"] = roi_name
    # not being strict right now
    for field, value in metadata_dict.items():
        roi_hdu.header[field] = value
    roi_hdu.name = roi_name
    return roi_hdu


def roi_values(array):
    return {
        "mean": array.mean(),
        "err": array.std(),
        "min": array.min(),
        "max": array.max(),
        "total": array.sum(),
    }


def count_rois_on_image(
    roi_arrays, roi_names, image, detector_mask=None, special_constants=None
):
    """
    expects ROIs 'unpacked' from their FITS format -- this is basically because
    we want applications to be able to prefilter the list without sending a
    bunch of additional information to this function, and also to count arrays
    made in other ways without rolling them into FITS files!
    """
    count_dict = {}
    if special_constants is not None:
        special_mask = np.full(image.shape, True)
        special_mask[np.isin(image, special_constants)] = False
    for roi_mask, roi_name in zip(roi_arrays, roi_names):
        if detector_mask is not None:
            roi_mask = np.logical_and(roi_mask, detector_mask)
        if special_constants is not None:
            roi_mask = np.logical_and(roi_mask, detector_mask)
        count_dict[roi_name] = roi_values(image[roi_mask])
    return count_dict


def make_edgemap(image, threshold=0.01):
    edge = np.hypot(sobel(image, 0), sobel(image, 1))
    edge[np.where(edge > threshold)] = 1
    return edge / edge.max()


def furthest_from_edge(image):
    distances = distance_transform_edt(image)
    return np.unravel_index(distances.argmax(), distances.shape)


def make_roi_edgemaps(roi_fits, calculate_centers=True):
    roi_edgemaps = {}
    for roi_ix in roi_fits:
        roi_array = roi_fits[roi_ix].data
        name = roi_fits[roi_ix].header["EXTNAME"].lower()
        edgemap = make_edgemap(roi_array).astype("uint8")
        roi_edgemaps[name] = {
            "edgemap": edgemap,
        }
        if calculate_centers:
            center_y, center_x = furthest_from_edge(roi_array)
            roi_edgemaps[name]["center"] = (center_x, center_y)
    return roi_edgemaps


def draw_edgemaps_on_image(
    image, edgemap_dict, inscribe_names=False, fontproperties=None, width=4
):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(image)
    for roi_name, edgemap in edgemap_dict.items():
        if "color" in edgemap.keys():
            color = edgemap["color"]
        else:
            color = "white"
        edge_y, edge_x = np.nonzero(edgemap["edgemap"])
        plt.scatter(
            edge_x, edge_y, marker=".", s=width, color=color, alpha=0.4
        )
        if inscribe_names:
            plt.annotate(
                roi_name,
                (edgemap["center"][0] - 15, edgemap["center"][1]),
                color="white",
                fontproperties=fontproperties,
                alpha=0.8,
            )
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def read_from_pointing(
    pointing_df: pd.DataFrame, filter_name: str, just_image: bool = True
) -> Union[pdr.Data, np.ndarray]:
    # TODO: maybe too specific? yes. ImageSet or whatever should be a class.
    if filter_name[-1] in ("R", "G", "B"):
        filter_name = filter_name[:-1]
    filter_files = pointing_df.loc[
        pointing_df["FILTER"] == filter_name, "PATH"
    ]
    if len(filter_files) == 0:
        raise FileNotFoundError
    if len(filter_files) > 1:
        warn("there appears to be more than one " + filter_name + " image")
    pdr_data = pdr.read(filter_files.iloc[0])
    if just_image:
        return pdr_data.IMAGE
    return pdr_data


def border_crop(array: np.ndarray, crop=None) -> np.ndarray:
    """
    crop: tuple of (left, right, top, bottom) pixels to trim
    """
    if crop is None:
        return array
    pixels = [side if side != 0 else None for side in crop]
    for value in (1, 3):
        if isinstance(pixels[value], int):
            if pixels[value] > 0:
                pixels[value] = pixels[value] * -1
    return array[pixels[2] : pixels[3], pixels[0] : pixels[1]]


def check_one_to_one(series: pd.Series, strings: Sequence[str]) -> bool:
    """
    verifies a 1-1 mapping from strings in strings to elements of series
    TODO: this is cruft but I can't cut it because I think it's clever
    """
    matches = series.str.contains("|".join(strings), regex=True)
    if len(matches.loc[matches]) == len(strings):
        return True
    return False


# bayer pattern definitions
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
    array_shape: tuple[int, int],
    bayer_pattern: Mapping[str, tuple[int]],
) -> dict[str, tuple]:
    """
    alias to make_pattern_masks
    """
    return make_pattern_masks(array_shape, bayer_pattern, (2, 2))


def depth_stack(images):
    return np.moveaxis(np.array(images), (0, 1), (2, 0))


def bilinear_interpolate(
    input_array: np.ndarray,
    output_shape: tuple[int, int],
    y_coords: np.ndarray,
    x_coords: np.ndarray,
) -> np.ndarray:
    """
    TODO: would it be better for these functions to use flat mask arrays
        rather than coordinate grids? (performance is the same, question is
        interface convenience

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
    according to a regular pattern.

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
    return np.mean(depth_stack(upsampled_images), axis=-1)


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
    return depth_stack(channels)


def preprocess_image(image_array, debayer=None, crop_bounds=None, filt=None):
    perform_debayer = False
    if debayer is not None:
        perform_debayer = True
        if "eschew_filters" in debayer.keys():
            if filt in debayer["eschew_filters"]:
                perform_debayer = False
    if perform_debayer is True:
        if filt is not None:
            debayer["pixel"] = debayer["mapping"][filt]
        image_array = debayer_upsample(image_array, **debayer)
    if crop_bounds is not None:
        image_array = border_crop(image_array, crop_bounds)
    return image_array


def norm_clip(image, sigma=1):
    mean = np.mean(image)
    std = np.std(image)
    return np.clip(image, *(mean - std * sigma, mean + std * sigma))


def minmax_clip(image, cheat_low=0, cheat_high=0):
    dynamic_range = image.max() - image.min()
    return np.clip(
        image,
        image.min() + dynamic_range * cheat_low,
        image.max() - dynamic_range * cheat_high,
    )


def normalize_range(
    image, range_min=0, range_max=1, cheat_low=None, cheat_high=None
):
    working_image = image.copy()
    if cheat_low is not None:
        minimum = np.percentile(image, cheat_low).astype(image.dtype)
    else:
        minimum = image.min()
    if cheat_high is not None:
        maximum = np.percentile(image, 100 - cheat_high).astype(image.dtype)
    else:
        maximum = image.max()
    if not ((cheat_high is None) and (cheat_low is None)):
        working_image = np.clip(working_image, minimum, maximum)
    return range_min + (working_image - minimum) * (range_max - range_min) / (
        maximum - minimum
    )


def make_spectral_rapidlook(
    spectop,
    op_images,
    op_wavelengths=None,
    special_constants=None,
    clip=None,
    image_filter=None,
    mpl_options=None,
    default_value=0,
    prefilter=None,
):
    if prefilter is not None:
        if "params" in prefilter.keys():
            params = prefilter["params"]
        else:
            params = {}
        op_images = [
            prefilter["function"](op_image.copy(), **params)
            for op_image in op_images
        ]
    rapidlook = spectop(op_images, None, op_wavelengths)[0]
    # TODO: add watermark
    # ignoring nans and special constants in scaling
    rapidlook[np.where(np.isnan(rapidlook))] = default_value
    rapidlook[np.where(np.isinf(rapidlook))] = default_value
    if special_constants is not None:
        for op_image in op_images:
            rapidlook[
                np.where(np.isin(op_image, special_constants))
            ] = default_value
    if clip is not None:
        if "params" in clip.keys():
            params = clip["params"]
        else:
            params = {}
        rapidlook = clip["function"](rapidlook, **params)
    if image_filter is not None:
        if "params" in image_filter.keys():
            params = image_filter["params"]
        else:
            params = {}
        rapidlook = image_filter["function"](rapidlook, **params)
    if mpl_options is not None:
        norm = plt.Normalize(vmin=rapidlook.min(), vmax=rapidlook.max())
        cmap = mpl_options["cmap"]
        rapidlook = cmap(norm(rapidlook))
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(rapidlook)
        # TODO: make this editable i guess although it might just be a bad idea
        colorbar = plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            fraction=0.03,
            pad=0.04,
        )
        return remove_ticks_and_style_colorbar(fig, ax, colorbar, mpl_options)

    return rapidlook


def make_three_channel_filter(image_filter):
    def apply_per_channel(image, *args, **kwargs):
        output_channels = []
        for channel_ix in range(3):
            output_channels.append(
                image_filter(image[:, :, channel_ix], *args, **kwargs)
            )
        return depth_stack(output_channels)

    return apply_per_channel


def render_enhanced(
    channels,
    special_constants=None,
    normalize=(0, 1, 0, 0),
    image_filter=None,
    render_mpl=False,
    prefilter=None,
):
    """
    render an 'enhanced-color' image from three input channels

    this assumes normalization as a default option because you're presumably
    going to want to view these as "normal" RGB images, not scaled to an
    arbitrary colormap (although aren't they all?)
    """
    assert len(channels) == 3
    enhanced = [
        apply_image_filter(channel.copy(), prefilter) for channel in channels
    ]
    enhanced = depth_stack(enhanced)
    if special_constants is not None:
        enhanced = np.where(np.isin(enhanced, special_constants), 0, enhanced)
    if normalize not in (False, None):
        # or should we be using mpl's linear scaler as above? i don't think we
        # need its masking features in this case, it's mostly to make it play
        # nicely with cmaps
        enhanced = normalize_range(enhanced, *normalize)
    enhanced = apply_image_filter(enhanced, image_filter)

    if render_mpl is True:
        return simple_mpl_figure(enhanced)
    return enhanced


def decorrelation_stretch(
    input_array,
    contrast_stretch,
    special_constants=None,
    image_filter=None,
    render_mpl=False,
):
    """
    decorrelation stretch of passed array on last axis of array

    see MATLAB decorrstretch, etc.
    """
    working_array = input_array.copy()
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
    # target per-channel standard deviation as a diagonalized matrix,
    # here simply set equal to per-channel input standard deviation
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
    #  implementation.
    #  check against reference algorithm.
    offset = channel_means - np.dot(channel_means, transformation_matrix)
    # remove mean from each channel, transform, replace mean and add offset
    dcs_vectors = (
        np.dot((channel_vectors - channel_means), transformation_matrix)
        + channel_means
        + offset
    )
    dcs_array = dcs_vectors.reshape(input_shape)
    if contrast_stretch is not None:
        # optionally apply linear contrast stretch
        for channel_ix in range(dcs_array.shape[-1]):
            channel = dcs_array[..., channel_ix]
            dcs_array[..., channel_ix] = normalize_range(
                channel, 0, 1, contrast_stretch, contrast_stretch
            )
    dcs_array = apply_image_filter(dcs_array, image_filter)
    if render_mpl is True:
        return simple_mpl_figure(dcs_array)
    return dcs_array


def render_overlay(
    overlay_image,
    base_image,
    overlay_cmap,
    base_cmap,
    overlay_opacity,
    mpl_options,
):
    # TODO: put a filter inline on _just the overlay_
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
    base_image = normalize_range(base_image, 0, 1)
    ax.imshow(
        base_cmap(base_image) * (1 - overlay_opacity)
        + overlay_cmap(norm(overlay_image)) * overlay_opacity
    )
    return remove_ticks_and_style_colorbar(fig, ax, colorbar, mpl_options)


def db_masks_for_rapidlooks(debayer_options, shape):
    # great, they have been precomputed, or so we trust
    if "masks" in debayer_options.keys():
        return debayer_options
    debayer_options["masks"] = make_bayer(shape, debayer_options["pattern"])
    debayer_options["row_column"] = {
        pixel: (np.unique(mask[0]), np.unique(mask[1]))
        for pixel, mask in debayer_options["masks"].items()
    }
    return debayer_options


def rapidlooks_from_pointing(
    pointing_df: pd.DataFrame,
    rapidlook_instructions: Mapping,
    filter_dict: Mapping,
    preprocess_options: dict,
    preloaded_images=None,
    return_preprocessed_images=False,
) -> Any:
    """
    makes rapidlooks for compatible instruments given a dataframe containing
    images-per-filter and paths to those images, a mapping with operations
    and filters and options (e.g. "dcs": {
        "operation": "dcs",
        "filters": ("L2", "L5", "L6"),   if normalize not in (False, None):
        # or should we be using mpl's linear scaler as above? i don't think we
        # need its masking features in this case, it's mostly to make it play
        # nicely with cmaps
        enhanced = normalize_range(enhanced, *normalize)
        "options": STRETCHY_RAPIDLOOK_OPTIONS,
    }), and a dict containing per-filter
    wavelength information (e.g., output of make_xcam_filter_dict)

    TODO: this function may be too big?

    TODO: for things that don't have filters, this needs other rules

    TODO: maybe replace print statements here with some kind of fancier pass-
     through...although it is intended to be chatty, maybe the chattiness
     shouldn't live in imgops? maybe this is really an xcam-specific function?
    """
    image_stash = {}
    rapidlooks = {}
    # don't mess with the literal from larger scope
    pp_options = preprocess_options.copy()
    if preloaded_images is not None:
        available_filters = set(list(preloaded_images.keys()) + pointing_df["FILTER"].tolist())
    else:
        available_filters = set(pointing_df["FILTER"].tolist())
    availability = partial(contains, available_filters)
    for instruction in rapidlook_instructions.values():
        filters = instruction["filters"]
        operation = instruction["operation"]
        if "name" in instruction.keys():
            op_name = instruction["name"]
        else:
            op_name = operation
        if "overlay" in instruction.keys():
            all_filters = list(filters) + [instruction["overlay"]["filter"]]
        else:
            all_filters = filters
        # filters_present = map()
        if not all(map(availability, all_filters)):
            print(
                "Skipping "
                + op_name
                + " "
                + str(filters)
                + " due to missing images"
            )
            continue
        print("generating " + op_name + " " + str(filters))
        op_images = []
        if preloaded_images is None:
            preloaded_images = {}
        for filt in all_filters:
            if image_stash.get(filt) is not None:
                # TODO: this prevents you from overlaying a processed filter
                #  on itself
                if filt in filters:
                    op_images.append(image_stash.get(filt))
            else:
                if filt in preloaded_images.keys():
                    # TODO: presently you are responsible for passing it preloaded
                    #  bayer-named images if you want them -- L0R, etc. --
                    #  although they can and should just be copies of the
                    #  raw L0 / R0 bayer. if this function is xcam-specific
                    #  maybe we can just explicitly add special handling.
                    filt_image = preloaded_images[filt].copy()
                else:
                    filt_image = read_from_pointing(pointing_df, filt)
                # precompute debayer masks (small but real time savings)
                if "debayer" in pp_options.keys():
                    pp_options["debayer"] = db_masks_for_rapidlooks(
                        pp_options["debayer"], filt_image.shape
                    )
                pp_options["filt"] = filt
                filt_image = preprocess_image(filt_image, **pp_options)
                image_stash[filt] = filt_image
                if filt in filters:
                    op_images.append(filt_image)
        if operation in marslab.spectops.SPECTOP_NAMES:
            rapidlook = make_spectral_rapidlook(
                spectop=getattr(marslab.spectops, operation),
                op_images=op_images,
                op_wavelengths=list(map(filter_dict.get, filters)),
                **instruction["options"]
            )
        elif operation == "enhanced color":
            rapidlook = render_enhanced(op_images, **instruction["options"])
        elif operation == "dcs":
            rapidlook = decorrelation_stretch(
                depth_stack(op_images), **instruction["options"]
            )
        else:
            raise ValueError("unknown rapidlook operation " + operation)
        if "overlay" in instruction.keys():
            rapidlook = render_overlay(
                base_image=image_stash.get(instruction["overlay"]["filter"]),
                overlay_image=rapidlook,
                **instruction["overlay"]["options"]
            )
        rapidlooks[op_name + " " + "_".join(filters)] = rapidlook

    if return_preprocessed_images is True:
        return rapidlooks, image_stash
    return rapidlooks


def get_mpl_image(fig):
    """
    tries to get the first axis from a mpl figure.
    cranky and fault-intolerant
    """
    ax = fig.axes[0]
    buffer = io.BytesIO()
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(buffer, bbox_inches=extent)
    return PIL.Image.open(buffer)


# def get_mpl_image(fig):
#     """
#     tries to get the first AxesImage from a mpl figure.
#     cranky and fault-intolerant
#     """
#
#     axis_image = fig.axes[0].get_images()[0]
#     axis_array = axis_image.get_array()
#     if len(axis_array.shape) == 3:
#         return axis_array
#     return axis_image.cmap(axis_image.norm(axis_array))


def make_thumbnail(
    array, thumbnail_size=(256, 256), file_or_path_or_buffer=None, filetype=None
):
    """
    makes thumbnails from arrays or matplotlib images or PIL.Images
    """
    if file_or_path_or_buffer is None:
        file_or_path_or_buffer = io.BytesIO()
    if filetype is None:
        filetype = "jpeg"
    if isinstance(array, matplotlib.figure.Figure):
        output_image = get_mpl_image(array).convert("RGB")
    elif isinstance(array, np.ndarray):
        image_array = normalize_range(array, 0, 255).astype("uint8")
        output_image = PIL.Image.fromarray(image_array).convert("RGB")
    else:
        output_image = array
    output_image.thumbnail(thumbnail_size)
    output_image.save(file_or_path_or_buffer, filetype)
    return file_or_path_or_buffer
