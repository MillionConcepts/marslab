"""skeleton BandSet implementation for Clementine mosaics"""

from functools import partial
from pathlib import Path

import pandas as pd

from marslab.compat.clementine import (
    WAVELENGTH_TO_FILTER,
    MOSAIC_SPECIAL_CONSTANTS,
)
from marslab.bandset.bandset import BandSet
from marslab.imgops.loaders import pdr_load


def setup_clem_bandset_metadata(mosaic_file_path):
    dataset = Path(mosaic_file_path).name.split("_")[0]
    row_template = {"PATH": mosaic_file_path, "DATASET": dataset}
    rows = []
    for ix, wave_filter in enumerate(WAVELENGTH_TO_FILTER[dataset].items()):
        row = row_template.copy()
        row["BAND"] = wave_filter[1]
        row["WAVELENGTH"] = wave_filter[0]
        row["IX"] = ix
        rows.append(row)
    return pd.DataFrame(rows)


class ClemBandSet(BandSet):
    """
    a BandSet made from a tile from the Clementine NIR or UVVIS mosaics.
    """

    def __init__(self, tile_path, rois=None, threads=None):
        metadata = setup_clem_bandset_metadata(tile_path)
        load_method = pdr_load
        super().__init__(
            metadata=metadata,
            load_method=load_method,
            rois=rois,
            threads=threads,
        )
