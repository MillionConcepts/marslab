from functools import cache, partial
from operator import attrgetter
from pathlib import Path
from types import MappingProxyType

import pandas as pd
from dustgoggles.scrape import bulk_scrape_metadata, scrape_patterns

from marslab.compat.xcam import DERIVED_CAM_DICT, BAND_TO_BAYER
from marslab.imgops.bandset import BandSet
from marslab.imgops.debayer import RGGB_PATTERN
from marslab.imgops.loaders import pdr_load


def parse_mcam_fn(mcam_fn):
    return {
        "EYE": mcam_fn[5],
        "SEQ_ID": f"mcam{int(mcam_fn[6:12])}",
        "CAL": mcam_fn[26:30],
        "FILETYPE": mcam_fn[22:25],
    }


MCAM_METADATA_REGEX = MappingProxyType(
    {
        "FILTER": r"FILTER_NAME.*MASTCAM_([LR]\d)(?=_)",
        "CSEQ": r"COMMAND_SEQUENCE_NUMBER.*?(\d+)",
        "SOL": r"(?<=PLANET_DAY_NUMBER).*?(\d+)",
    }
)

scrape_mcam_metadata = cache(
    partial(scrape_patterns, metadata_regex=MCAM_METADATA_REGEX)
)


bulk_scrape_mcam_metadata = partial(
    bulk_scrape_metadata, pattern_scraper=scrape_mcam_metadata
)


def parse_mcam_files(file_paths):
    return tuple(
        map(parse_mcam_fn, map(attrgetter("name"), map(Path, file_paths)))
    )


def setup_mcam_bandset_metadata(metadata):
    if "FILTER" in metadata.columns:
        metadata = metadata.drop(
            metadata.loc[metadata.FILTER.isin(('L7', 'R7'))].index
        )
        metadata["BAND"] = metadata["FILTER"]
        metadata.drop("FILTER", axis=1)

    metadata.index = metadata["BAND"]
    # add references to the secret bands hidden inside the L0 and R0 images
    bayer_filter_rows = []
    for eye in ("L", "R"):
        if eye + "0" not in metadata.index:
            continue
        eye_row = metadata.loc[eye + "0"]
        for color, ix in zip(("R", "G", "B"), (0, 1, 2)):
            eye_color_row = eye_row.copy()
            eye_color_row["BAND"] = eye + "0" + color
            eye_color_row["IX"] = ix
            eye_color_row.name = eye + "0" + color
            bayer_filter_rows.append(eye_color_row)
        metadata = metadata.drop(eye_row.name)
    if bayer_filter_rows:
        metadata = pd.concat(
            (metadata, pd.concat(bayer_filter_rows, axis=1).T)
        )
    # add wavelengths and bayer pixel mappings
    metadata["WAVELENGTH"] = pd.Series(DERIVED_CAM_DICT["MCAM"]["filters"])[
        metadata["BAND"]
    ]
    metadata["BAYER_PIXEL"] = pd.Series(BAND_TO_BAYER["MCAM"])[
        metadata["BAND"]
    ]
    metadata.loc[metadata['FILETYPE'] != 'C00', 'BAYER_PIXEL'] = None
    return metadata.reset_index(drop=True)


class McamBandSet(BandSet):
    def __init__(self, observation, rois=None, threads=None):
        files = setup_mcam_bandset_metadata(observation)
        load_method = partial(pdr_load, preserve_constants=[4096, 999999])
        bayer_info = {"pattern": RGGB_PATTERN}
        super().__init__(
            metadata=files,
            load_method=load_method,
            bayer_info=bayer_info,
            rois=rois,
            threads=threads,
        )
