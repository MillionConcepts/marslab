"""skeleton BandSet for MER pancam multispectral observations"""

from functools import cache, partial
from operator import attrgetter
from pathlib import Path
import re
from string import digits, ascii_lowercase
from types import MappingProxyType
from typing import Literal, Optional

from dustgoggles.scrape import bulk_scrape_metadata, scrape_patterns
import pandas as pd

from marslab.bandset.bandset import BandSet
from marslab.compat.xcam import DERIVED_CAM_DICT
from marslab.imgops.loaders import pdr_load

B36_LEX = digits + ascii_lowercase

import re
from typing import Literal

RoverIdentifier = Literal[
    "A", "B", 1, 2, "1", "2", "MER-A", "MER-B", "MER-1", "MER-2",
    "MER_A", "MER_B", "MER_1", "MER_2"
]

ROVERPAT = re.compile(r"(?:MER)?[-_]?([AB12])")


def normalize_rover_id(rover: RoverIdentifier):
    try:
        if isinstance(rover, str):
            rover = ROVERPAT.match(rover.upper()).group(1)
        if rover in (1, "1"):
            rover = "B"
        elif rover in (2, "2"):
            rover = "A"
        if rover not in ("A", "B"):
            raise ValueError
        return rover
    except (AttributeError, ValueError) as e:
        raise ValueError(f"Unknown MER rover id representation '{rover}'")


def parse_pcam_36(chars: str):
    """
    PCAM uses a quasi-modulo 36 system for SITE and POS, but with an arbitrary
    three-branch ordering.
    """
    assert len(chars) == 2
    if chars in ("__", "##"):
        # "operations" and "archive" -- must be extracted from label
        return None
    if chars[0].isdigit():
        # 00 - 99 are base 10 as written
        if chars[1].isdigit():
            return int(chars)
        # 0A - 9Z -- with no digits in the second place -- are 1036-1295
        return 1036 + int(chars[0]) * 26 + ascii_lowercase.index(chars[1])
    # A0 - ZZ -- with no digits in the first place -- are 100 through 1035
    return 100 + ascii_lowercase.index(chars[0]) * 36 + B36_LEX.index(chars[1])


def parse_pcam_fn(pcam_fn):
    filename = Path(pcam_fn).name
    return {
        "ROVER": {"1": "B", "2": "A"}[filename[0]],
        "SCLK": int(filename[2:11]),
        "PRODUCT_TYPE": filename[11:14].upper(),
        "SITE": parse_pcam_36(filename[14:16]),
        "POS": parse_pcam_36(filename[16:18]),
        "SEQ_ID": filename[18:23].upper(),
        "FILTER": filename[23:25].upper(),
        "VERSION": filename[26],
    }


PCAM_METADATA_REGEX = MappingProxyType(
    {
        "CSEQ": r"COMMAND_SEQUENCE_NUMBER.*?(\d+)",
        "SOL": r"(?<=PLANET_DAY_NUMBER).*?(\d+)",
    }
)

scrape_pcam_metadata = cache(
    partial(scrape_patterns, metadata_regex=PCAM_METADATA_REGEX)
)

bulk_scrape_pcam_metadata = partial(
    bulk_scrape_metadata, pattern_scraper=scrape_pcam_metadata
)


def parse_pcam_files(file_paths):
    return tuple(
        map(parse_pcam_fn, map(attrgetter("name"), map(Path, file_paths)))
    )


def setup_pcam_bandset_metadata(metadata):
    if "FILTER" in metadata.columns:
        metadata = metadata.drop(
            metadata.loc[metadata.FILTER.isin(("L0", "L8", "R8"))].index
        )
        metadata["BAND"] = metadata["FILTER"]
        metadata.drop("FILTER", axis=1)

    metadata.index = metadata["BAND"]
    metadata["WAVELENGTH"] = pd.Series(DERIVED_CAM_DICT["PCAM"]["filters"])[metadata["BAND"]]
    return metadata.reset_index(drop=True)


class PcamBandSet(BandSet):
    def __init__(
        self,
        observation,
        rois=None,
        suffix: str = "",
        threads: Optional[dict] = None
    ):
        files = setup_pcam_bandset_metadata(observation)
        # MER pancam IOF object names are "IMAGE" in attached PDS3 labels.
        # The detached PDS4 labels use "Image_Object" instead.
        load_method = partial(pdr_load, object_name="IMAGE")
        super().__init__(
            metadata=files,
            load_method=load_method,
            bayer_info=None,
            rois=rois,
            threads=threads,
        )
        self.suffix = suffix
