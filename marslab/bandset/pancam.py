"""skeleton BandSet for MER pancam multispectral observations"""

from functools import cache, partial
from operator import attrgetter
from pathlib import Path
from types import MappingProxyType

import pandas as pd
from dustgoggles.scrape import bulk_scrape_metadata, scrape_patterns

from marslab.compat.xcam import DERIVED_CAM_DICT
from marslab.bandset.bandset import BandSet
from marslab.imgops.loaders import pdr_load


def parse_pcam_fn(pcam_fn):
    filename = Path(pcam_fn).name
    return {
        "ROVER": filename[0],
        "PRODUCT_TYPE": filename[11:14].upper(),
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
    def __init__(self, observation, rois=None, threads=None):
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
