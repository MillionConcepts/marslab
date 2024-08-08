"""
functions for handling the MERtools .sel format
"""
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Union

import numpy as np
import scipy.io
from astropy.io import fits

from marslab.compat.mertools import (
    MERSPECT_MSL_COLOR_MAPPINGS,
    MERSPECT_M20_COLOR_MAPPINGS,
    MERSPECT_COLOR_MAPPINGS,
)
from marslab.imgops.regions import select_roi_by_ix, make_roi_hdu


def is_sel_file(roi_path: Union[str, Path]) -> bool:
    """
    is this file a MERSpect .sel file?
    """
    # TODO: scipy.io.readsav does not close the buffer correctly
    #  on finding an invalid signature.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        try:
            sel = scipy.io.readsav(roi_path)
            assert set(sel.keys()).issubset(
                {
                    "erasecolor",
                    "lseltemp",
                    "rseltemp",
                    "region_info",
                    "left_pos",
                    "right_pos",
                    "sel_file_format_major_version",
                    "sel_file_format_minor_version",
                    "sel_file_format_date",
                }
            )
            return True
        except AssertionError:
            # this is apparently an IDL .sav file, but not a
            # properly-formatted .sel file
            return False
        # scipy.io.readsav uses general Exception
        except Exception as exception:
            if "Invalid SIGNATURE" in str(exception):
                # this is not an IDL .sav file but rather something else
                return False
            # presumably FileNotFoundError, you passed an integer, stuff
            # like that
            raise


def roi_color_ix_to_color_name(color_ix: int, instrument: str = "MCAM") -> str:
    """
    convert MERSpect ROI color index to color name
    """
    # this works because colors are listed in the dictionaries in
    # marslab.compat.mertools in the order
    # they are listed in the MERSpect IDL code. No explicit mapping
    # between MERSpect ROI color indices and actual colors exists.
    if instrument in ["MCAM", "PCAM"]:
        return list(MERSPECT_MSL_COLOR_MAPPINGS.keys())[color_ix]
    elif instrument == "ZCAM":
        # The unit offset to the color indices is because "0" is reserved
        # in MERspect as something like "blank canvas." The "erase"
        # capability has a different color associated with it.
        return list(MERSPECT_M20_COLOR_MAPPINGS.keys())[color_ix - 1]
    else:
        raise ValueError(
            "I don't have ROI color information about "
            + instrument
            + ". I know about PCAM, MCAM, and ZCAM."
        )


def roi_hdu_from_sel(
    sel_image: np.ndarray,
    color_ix: int,
    eye_name: str,
    instrument: str = "MCAM",
    sel_fn: str = None,
) -> fits.PrimaryHDU:
    """
    make an individual HDU for a marslab roi FITS file
    from an array generated from a MERSpect .sel file
    """
    roi_array = select_roi_by_ix(sel_image, color_ix)
    color_name = roi_color_ix_to_color_name(color_ix, instrument)
    merspect_metadata = {"EYE": eye_name, "SOURCEFN": Path(sel_fn).name}
    roi_hdu = make_roi_hdu(roi_array, color_name, merspect_metadata)
    roi_hdu.name = color_name + " " + eye_name
    return roi_hdu


def sel_to_roi(sel_fn: Union[str, Path], instrument: str) -> fits.HDUList:
    """
    convert a MERSpect .sel file to a marslab roi file -- a FITS file with a
    single HDU per eye per ROI
    """
    sel = scipy.io.readsav(sel_fn)
    sel_images = {}
    # add each eye if present
    for eye_name, sel_key in (("left", "lseltemp"), ("right", "rseltemp")):
        if isinstance(sel[sel_key], np.ndarray):
            # IDL y-axis indexing is flipped compared to numpy, so flip
            # images here
            sel_images[eye_name] = np.flipud(sel[sel_key])
    # TODO: decide if we'd actually prefer to call a generalized way of making
    #  ROI FITS in imgops -- but we do want to mess with the individual HDUs
    #  here
    roi_fits = fits.HDUList()
    # iterate over eyes and colors to create HDUs -- note that same colors
    # are not necessarily present in each eye
    for eye, sel_image in sel_images.items():
        color_indices = [
            ix for ix in np.unique(sel_image) if ix != sel["erasecolor"]
        ]
        for color_ix in color_indices:
            roi_fits.append(
                roi_hdu_from_sel(sel_image, color_ix, eye, instrument, sel_fn)
            )
    return roi_fits


def add_merspect_colors_to_edgemaps(edgemap_dict: Mapping):
    for roi_color_eye, _ in edgemap_dict.items():
        color = roi_color_eye.replace(" right", "").replace(" left", "")
        edgemap_dict[roi_color_eye]["color"] = MERSPECT_COLOR_MAPPINGS[color]
    return edgemap_dict
