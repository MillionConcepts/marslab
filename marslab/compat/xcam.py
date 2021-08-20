"""
mappings between filter characteristic wavelengths and designations,
along with a bunch of derived related values for "XCAM" 
instruments (PCAM, MCAM, ZCAM...), affording consistent interpretation 
of operations on individual spectra
"""
from collections.abc import Mapping, Sequence
from itertools import chain, combinations
from math import floor
from statistics import mean
from typing import Optional

import numpy as np
import pandas.api.types
import pandas as pd
from more_itertools import windowed

WAVELENGTH_TO_FILTER = {
    "CCAM": {
        400: '400',
        440: '440',
        535: '535',
        600: '600',
        670: '670',
        750: '750',
        840: '840'
    },
    "ZCAM": {
        "L": {
            630: "L0R",
            544: "L0G",
            480: "L0B",
            800: "L1",
            754: "L2",
            677: "L3",
            605: "L4",
            528: "L5",
            442: "L6",
        },
        "R": {
            631: "R0R",
            544: "R0G",
            480: "R0B",
            800: "R1",
            866: "R2",
            910: "R3",
            939: "R4",
            978: "R5",
            1022: "R6",
        },
    },
    "MCAM": {
        "L": {
            482: "L0B",  #
            493: "L0B",  # Accepted value of L0B has changed over time
            495: "L0B",  #
            554: "L0G",
            640: "L0R",
            527: "L1",
            445: "L2",
            751: "L3",
            676: "L4",
            867: "L5",
            1012: "L6",
        },
        "R": {
            482: "R0B",  #
            493: "R0B",  # Accepted value of R0B has changed over time
            495: "R0B",  #
            551: "R0G",
            638: "R0R",
            527: "R1",
            447: "R2",  #
            805: "R3",
            908: "R4",
            937: "R5",
            1013: "R6",  #
        },
    },
}


# rules currently in use:
# set of virtual filters === the set of pairs of real filters with nominal
# band centers within 5 nm of one another
# the virtual mean reflectance in an ROI for a virtual filter is the
# arithmetic mean of the mean reflectance values in that ROI for the two real
# filters in its associated pair.
# the nominal band center of a virtual filter is the arithmetic mean of the
# nominal band centers of the two real filters in its associated pair.


def make_xcam_filter_dict(abbreviation):
    """
    form filter: wavelength dictionary for mastcam-family instruments
    """
    if abbreviation == "CCAM":
        return { name: wavelength
                 for wavelength, name in sorted(
                    WAVELENGTH_TO_FILTER[abbreviation].items(), key=lambda item: item[1]
                )
        }
    left = {
        name: wavelength
        for wavelength, name in WAVELENGTH_TO_FILTER[abbreviation]["L"].items()
    }
    right = {
        name: wavelength
        for wavelength, name in WAVELENGTH_TO_FILTER[abbreviation]["R"].items()
    }
    return {
        name: wavelength
        for name, wavelength in sorted(
            {**left, **right}.items(), key=lambda item: item[1]
        )
    }


def make_xcam_filter_pairs(abbreviation):
    """
    form list of pairs of close filters for mastcam-family instruments
    """
    filter_dict = make_xcam_filter_dict(abbreviation)
    pairs = []
    for filter_1, filter_2 in combinations(filter_dict, 2):
        if abs(filter_dict[filter_1] - filter_dict[filter_2]) <= 5:
            pairs.append(tuple(sorted([filter_1, filter_2])))
    return tuple(pairs)


def make_virtual_filters(abbreviation):
    """
    form mapping from close filter names to wavelengths for mastcam-family
    """
    filter_dict = make_xcam_filter_dict(abbreviation)
    filter_pairs = make_xcam_filter_pairs(abbreviation)
    return {
        pair[0]
        + "_"
        + pair[1]: floor(mean([filter_dict[pair[0]], filter_dict[pair[1]]]))
        for pair in filter_pairs
    }


def make_virtual_filter_mapping(abbreviation):
    """
    form mapping from close filter names to filter pairs for mastcam-family
    """
    return {
        pair[0] + "_" + pair[1]: pair
        for pair in make_xcam_filter_pairs(abbreviation)
    }


def make_canonical_averaged_filters(abbreviation):
    filter_dict = make_xcam_filter_dict(abbreviation)
    virtual_filters = make_virtual_filters(abbreviation)
    virtual_filter_mapping = make_virtual_filter_mapping(abbreviation)
    retained_filters = {
        filt: filter_dict[filt]
        for filt in filter_dict
        if filt not in chain.from_iterable(virtual_filter_mapping.values())
    }
    caf = {**retained_filters, **virtual_filters}
    return {filt: caf[filt] for filt in sorted(caf, key=lambda x: caf[x])}


XCAM_ABBREVIATIONS = ["MCAM", "ZCAM", "CCAM"]
DERIVED_CAM_DICT = {
    abbrev: {
        "filters": make_xcam_filter_dict(abbrev),
        "virtual_filters": make_virtual_filters(abbrev),
        "virtual_filter_mapping": make_virtual_filter_mapping(abbrev),
        "canonical_averaged_filters": make_canonical_averaged_filters(abbrev),
    }
    for abbrev in XCAM_ABBREVIATIONS
}


def polish_xcam_spectrum(
    spectrum: Mapping[str, float],
    cam_info: Mapping[str, dict],
    scale_to: Optional[Sequence[str, str]] = None,
    average_filters: bool = True,
):
    """
    scale and merge values of a spectrum according to MERSPECT-style rules
    scale_to: None or tuple of (lefteye filter name, righteye filter name)
    """
    values = {}
    lefteye_scale = 1
    righteye_scale = 1
    # don't scale eyes to a value that doesn't exist or if you're asked not to
    if scale_to not in [None, "None"]:
        if all([spectrum.get(comp) for comp in scale_to]):
            scales = (spectrum[scale_to[0]], spectrum[scale_to[1]])
            filter_mean = mean(scales)
            lefteye_scale = filter_mean / scales[0]
            righteye_scale = filter_mean / scales[1]

    real_filters_to_use = list(cam_info["filters"].keys())
    if average_filters is True:
        # construct dictionary of averaged filter values
        for v_filter, comps in cam_info["virtual_filter_mapping"].items():
            # do not attempt to average filters if both filters of
            # a pair are not present
            if not all([spectrum.get(comp) for comp in comps]):
                continue
            [real_filters_to_use.remove(comp) for comp in comps]
            values[v_filter] = {
                "wave": cam_info["virtual_filters"][v_filter],
                "mean": mean(
                    (
                        spectrum[comps[0]] * lefteye_scale,
                        spectrum[comps[1]] * righteye_scale,
                    ),
                ),
            }
            if all([comp + "_ERR" in spectrum.keys() for comp in comps]):
                values[v_filter]["err"] = (
                    spectrum[comps[0] + "_ERR"] ** 2
                    + spectrum[comps[1] + "_ERR"] ** 2
                ) ** 0.5
    # construct dictionary of leftover real filter values
    for real_filter in real_filters_to_use:
        mean_value = spectrum.get(real_filter)
        if mean_value is None:
            continue
        if real_filter.lower().startswith("r"):
            eye_scale = righteye_scale
        else:
            eye_scale = lefteye_scale
        values[real_filter] = {
            "wave": cam_info["filters"][real_filter],
            "mean": spectrum[real_filter] * eye_scale,
        }
        if real_filter + "_ERR" in spectrum.keys():
            values[real_filter]["err"] = (
                spectrum[real_filter + "_ERR"] * eye_scale
            )
    return dict(sorted(values.items(), key=lambda item: item[1]["wave"]))


INSTRUMENT_UNCERTAINTIES = {
    # table 7, hayes et al. 2021 https://doi.org/10.1007/s11214-021-00795-x
    # "ZCAM": {
    #     "L0R": 3.3,
    #     "L0G": 3.3,
    #     "L0B": 3.7,
    #     "L1": 1.4,
    #     "L2": 1.1,
    #     "L3": 0.2,
    #     "L4": 1.8,
    #     "L5": 1.6,
    #     "L6": 0.4,
    #     "R0R": 3.7,
    #     "R0G": 4.1,
    #     "R0B": 4.6,
    #     "R1": 0.4,
    #     "R2": 0.3,
    #     "R3": 0.6,
    #     "R4": 0.5,
    #     "R5": 0.8,
    #     "R6": 0.4,
    # },
    # hayes, p. comm, April 2021: use 3% for everything but the
    # bayers for now
    "ZCAM": {
        "L0R": 3.3,
        "L0G": 3.3,
        "L0B": 3.7,
        "L1": 3,
        "L2": 3,
        "L3": 3,
        "L4": 3,
        "L5": 3,
        "L6": 3,
        "R0R": 3.7,
        "R0G": 4.1,
        "R0B": 4.6,
        "R1": 3,
        "R2": 3,
        "R3": 3,
        "R4": 3,
        "R5": 3,
        "R6": 3,
    },
    # table 2, bell et al. 2017 https://doi.org/10.1002/2016EA000219
    "MCAM": {
        "L0R": 1.2,
        "L0G": 0.3,
        "L0B": 5.7,
        "L1": 4.3,
        "L2": 51.0,
        "L3": 0.3,
        "L4": 0.1,
        "L5": 0.3,
        "L6": 1.0,
        "R0R": 1.9,
        "R0G": 1.5,
        "R0B": 2.5,
        "R1": 3.7,
        "R2": 24.5,
        "R3": 3.4,
        "R4": 0.4,
        "R5": 0.5,
        "R6": 1.0,
    },
}

# see M20 Camera SIS
ZCAM_ZOOM_MOTOR_COUNT_TO_FOCAL_LENGTH = {
    0: 26,
    2448: 34,
    3834: 48,
    5196: 63,
    6720: 79,
    8652: 100,
    9600: 110,
}


def piecewise_interpolate_focal_length(zmc):
    z_values = ZCAM_ZOOM_MOTOR_COUNT_TO_FOCAL_LENGTH.keys()
    f_values = ZCAM_ZOOM_MOTOR_COUNT_TO_FOCAL_LENGTH.values()
    if zmc in z_values:
        return ZCAM_ZOOM_MOTOR_COUNT_TO_FOCAL_LENGTH[zmc]
    try:
        for z1z2, f1f2 in zip(windowed(z_values, 2), windowed(f_values, 2)):
            z1, z2 = z1z2
            f1, f2 = f1f2
            if not ((z1 < zmc) and (zmc < z2)):
                continue
            return round((f2 - f1) / (z2 - z1) * zmc + f1, 1)
    except StopIteration:
        raise ValueError(
            str(zmc) + " is outside the range of zoom"
            " motor counts I know how to deal with."
        )


BAND_TO_BAYER = {
    "ZCAM": {
        "L1": "red",
        "L2": "red",
        "L3": "red",
        "L4": "red",
        "L5": ("green_1", "green_2"),
        "L6": "blue",
        "R1": "red",
        "R2": None,
        "R3": None,
        "R4": None,
        "R5": None,
        "R6": None,
        "L0G": ("green_1", "green_2"),
        "L0B": "blue",
        "L0R": "red",
        "R0G": ("green_1", "green_2"),
        "R0B": "blue",
        "R0R": "red",
    },
    "MCAM": {
        "L1": ("green_1", "green_2"),
        "L2": "blue",
        "L3": "red",
        "L4": "red",
        "L5": None,
        "L6": None,
        "R1": ("green_1", "green_2"),
        "R2": "blue",
        "R3": "red",
        "R4": None,
        "R5": None,
        "R6": None,
        # these are apparently debayered onboard or always debayered
        # by the time we see reduced images
        "L0G": None,
        "L0B": None,
        "L0R": None,
        "R0G": None,
        "R0B": None,
        "R0R": None,
    },
}


def numeric_columns(data: pd.DataFrame) -> list[str]:
    return [
        col
        for col in data.columns
        if pandas.api.types.is_numeric_dtype(data[col])
    ]


# TODO: way too huge and messy.
def count_rois_on_xcam_images(
    roi_hdulist,
    xcam_image_dict,
    instrument,
    pixel_map_dict=None,
    bayer_pixel_dict=None,
):
    """
    takes an roi hdulist, a dict of xcam images, and returns a marslab data
    section dataframe.

    there are so many potential special cases here that utterly transform
    control flow that we've chosen to structure it differently from the
    quick imaging functions. perhaps this is wrong, though.

    """
    from marslab.imgops.debayer import RGGB_PATTERN, make_bayer
    from marslab.imgops.regions import count_rois_on_image, roi_stats

    roi_listing = []
    # unrolling for easier iteration
    roi_hdus = [roi_hdulist[hdu_ix] for hdu_ix in roi_hdulist]
    left_hdus = [
        hdu for hdu in roi_hdus if hdu.header["EYE"].upper() == "LEFT"
    ]
    right_hdus = [
        hdu for hdu in roi_hdus if hdu.header["EYE"].upper() == "RIGHT"
    ]
    left_hdu_arrays = [hdu.data for hdu in left_hdus]
    left_hdu_names = [hdu.header["NAME"] for hdu in left_hdus]
    right_hdu_arrays = [hdu.data for hdu in right_hdus]
    right_hdu_names = [hdu.header["NAME"] for hdu in right_hdus]
    if bayer_pixel_dict is None:
        bayer_pixel_dict = BAND_TO_BAYER[instrument]
    if not all([pixel is None for pixel in bayer_pixel_dict.values()]):
        bayer_masks = make_bayer(
            list(xcam_image_dict.values())[0].shape, RGGB_PATTERN
        )
    else:
        bayer_masks = None
    for filter_name in DERIVED_CAM_DICT[instrument]["filters"].keys():
        image = xcam_image_dict.get(filter_name)
        if image is None:
            continue
        # bayer-counting logic
        if bayer_pixel_dict[filter_name] is not None:
            detector_mask = np.full(image.shape, False)
            bayer_pixels = bayer_pixel_dict[filter_name]
            if isinstance(bayer_pixels, str):
                bayer_pixels = [bayer_pixels]
            for pixel in bayer_pixels:
                bayer_coords = bayer_masks[pixel]
                detector_mask[bayer_coords] = True
        else:
            detector_mask = np.full(image.shape, True)
        # forbidding saturated and otherwise bad pixels
        if pixel_map_dict:
            if filter_name[1] == "0":
                base_pixel_map = pixel_map_dict.get(filter_name[0:2])
            else:
                base_pixel_map = pixel_map_dict.get(filter_name)
            if base_pixel_map is not None:
                # masking bad, no-signal, and saturated pixels
                flag_mask = np.full(image.shape, True)
                flag_mask[np.where(np.isin(base_pixel_map, [1, 2, 4]))] = False
                detector_mask = np.logical_and(detector_mask, flag_mask)
        if filter_name.upper().startswith("L"):
            roi_arrays = left_hdu_arrays
            roi_names = left_hdu_names
        elif filter_name.upper().startswith("R"):
            roi_arrays = right_hdu_arrays
            roi_names = right_hdu_names
        else:
            raise ValueError(filter_name + " is a forbidden filter")
        roi_counts = count_rois_on_image(
            roi_arrays, roi_names, image, detector_mask, [0]
        )
        for roi_name, counts in roi_counts.items():
            roi_listing.append(
                {
                    "COLOR": roi_name,
                    filter_name: counts["mean"],
                    filter_name + "_ERR": counts["err"],
                    filter_name + "_MODE": counts["mode"][0],
                }
                | {
                    filter_name + "_" + stat.upper(): counts[stat]
                    for stat in counts.keys()
                }
            )
    roi_frame = pd.DataFrame(roi_listing)
    roi_frame.loc[:, numeric_columns(roi_frame)] = roi_frame.loc[
        :, numeric_columns(roi_frame)
    ].astype(np.float32)
    cubestat_frame = roi_frame.copy()
    cubestats = []
    for eye in ("LEFT", "RIGHT"):
        eye_values = cubestat_frame[
            [
                c
                for c in cubestat_frame.columns
                if "VALUES" in c and c.startswith(eye[0].upper())
            ]
        ]
        if len(eye_values.columns) == 0:
            continue
        eye_values.index = cubestat_frame["COLOR"]
        melted = pd.melt(eye_values, ignore_index=False).dropna()
        roi_index = left_hdu_names if eye == "LEFT" else right_hdu_names
        for roi_name in roi_index:
            cube_counts = roi_stats(
                np.hstack(melted.loc[roi_name]["value"].values)
            )
            cubestats.append(
                {
                    "COLOR": roi_name,
                    eye: cube_counts["mean"],
                    eye + "_ERR": cube_counts["err"],
                    eye + "_MODE": cube_counts["mode"][0],
                }
                | {
                    eye + "_" + stat.upper(): cube_counts[stat]
                    for stat in cube_counts.keys()
                }
            )
    cubestats = cubestats.copy()
    # note pivoting automatically destroys any columns with arraylikes
    base_df = (
        pd.concat(
            [
                pd.DataFrame(cubestats),
                pd.DataFrame(roi_listing),
            ]
        )
        .pivot_table(columns=["COLOR"])
        .T.reset_index()
    )
    # copying to defragment
    downcast = base_df[numeric_columns(base_df)].astype(np.float32)
    base_df.loc[:, numeric_columns(base_df)] = downcast.values
    # enter nan columns for err and mean only -- this is a format
    # standardization choice
    for filter_name in DERIVED_CAM_DICT[instrument]["filters"].keys():
        if filter_name not in base_df.columns:
            base_df[filter_name] = np.nan
            base_df[filter_name + "_ERR"] = np.nan
    return base_df
