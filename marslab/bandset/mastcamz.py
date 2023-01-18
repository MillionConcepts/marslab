"""
BandSet subclass for processing Mastcam-Z IOFs.

Note that this is closely related to asdf.zcam_bandset.ZcamBandset, but omits
many features of that implementation that only function in the context of the
full asdf workflow.
"""
import re
from functools import partial
from pathlib import Path

import pandas as pd
import pdr
from cytoolz import juxt

from marslab.bandset import BandSet
from marslab.compat.xcam import (
    DERIVED_CAM_DICT, BAND_TO_BAYER,
)
from marslab.imgops.debayer import RGGB_PATTERN
from marslab.imgops.loaders import pdr_load
import marslab.parse as mp

ZCAM_FN_PARSERS = {
    "SOL": mp.sol,
    "SITE": mp.site,
    "DRIVE": mp.drive,
    "SEQ_ID": mp.sequence,
    "CTIME": mp.secondary_timestamp,
    "ZOOM": mp.cam_specific,
    "FILTER": mp.color_filter,
    "PRODUCT_TYPE": mp.product_type,
    "THUMBNAIL": mp.thumbnail,
}


def parse_zcam_fn(path):
    """use mp.parse rules to get basic file identifiers"""
    filename = Path(path).name
    try:
        values = list(juxt(*ZCAM_FN_PARSERS.values())(filename))
        # chop off currently not-used-as-specified stereo counter
        values[5] = values[5][1:]
        # just keep filter name, not SIS-nominal wavelength
        values[6] = values[6][:2]
        parsed = {
            field: value
            for field, value in zip(ZCAM_FN_PARSERS.keys(), values)
        }
        parsed["PATH"] = str(path)
        return parsed
    except (KeyError, IndexError, ValueError):
        return None


def ls_zcam(root_dir):
    files = [file for file in Path(root_dir).iterdir()]
    matches = [
        file
        for file in files
        if not re.match(r".*\.(xml|lbl)$", str(file), re.I)
    ]
    products = tuple(filter(None, map(parse_zcam_fn, matches)))
    if products:
        return (
            pd.DataFrame(products)
            .sort_values(by="CTIME")
            .reset_index(drop=True)
        )
    return None


def setup_zcam_bandset_metadata(metadata):
    if "FILTER" in metadata.columns:
        metadata["BAND"] = metadata["FILTER"]
        metadata.drop("FILTER", axis=1)
    metadata.index = metadata["BAND"]
    # add references to the secret bands hidden inside the L0 and R0 images
    bayer_filter_rows = []
    for eye in ("L", "R"):
        if eye + "0" not in metadata.index:
            continue
        eye_row = metadata.loc[eye + "0"]
        for color in ("R", "G", "B"):
            eye_color_row = eye_row.copy()
            eye_color_row["BAND"] = eye + "0" + color
            eye_color_row.name = eye + "0" + color
            bayer_filter_rows.append(eye_color_row)
        metadata = metadata.drop(eye_row.name)
    if bayer_filter_rows:
        metadata = pd.concat(
            (metadata, pd.concat(bayer_filter_rows, axis=1).T)
        )
    # add wavelengths and bayer pixel mappings
    metadata["WAVELENGTH"] = pd.Series(DERIVED_CAM_DICT["ZCAM"]["filters"])[
        metadata["BAND"]
    ]
    metadata["BAYER_PIXEL"] = pd.Series(BAND_TO_BAYER["ZCAM"])[
        metadata["BAND"]
    ]
    return metadata.reset_index(drop=True)


class ZcamBandSet(BandSet):
    def __init__(self, files):
        load_method = partial(pdr_load, preserve_constants=[0])
        bayer_info = {"pattern": RGGB_PATTERN}
        super().__init__(
            metadata=files, load_method=load_method, bayer_info=bayer_info
        )
        self.metadata = setup_zcam_bandset_metadata(files)
        for path in self.metadata["PATH"].unique():
            self.precached[path] = pdr.Data(
                path, label_fn=path, skip_existence_check=True
            )
        self.check_onboard_debayer(fix_metadata=True)

    def check_onboard_debayer(self, *, fix_metadata=False):
        """
        if this observation was debayered onboard, drop all references to
        bayer pixels to avoid inappropriate debayering later. add
        references to additional bands of R0 / L0 image files (that don't
        exist in raw bayer images).
        """
        if tuple(self.precached.values())[0].metaget(
                'BAYER_METHOD') == 'RAW_BAYER':
            return False
        if fix_metadata:
            self.metadata["BAYER_PIXEL"] = None
            for color, ix in zip(("R", "G", "B"), (0, 1, 2)):
                self.metadata.loc[
                    self.metadata["BAND"].str.endswith(color), "IX"
                ] = ix
        return True
