"""
skeleton BandSet implementation for M3 images -- expects files that have been
processed by pdr.converter
"""

from functools import partial

import numpy as np
import pandas as pd

from marslab.compat.m3 import (
    parse_m3_fn,
    TARGET_WAVELENGTHS,
    GLOBAL_WAVELENGTHS,
)
from marslab.bandset.bandset import BandSet
from marslab.imgops.loaders import pdr_load


def setup_m3_bandset_metadata(file_path):
    fn_metadata = parse_m3_fn(str(file_path))
    metadata = pd.DataFrame()
    if fn_metadata["mode"] == "target":
        metadata["WAVELENGTH"] = TARGET_WAVELENGTHS
    elif fn_metadata["mode"] == "global":
        metadata["WAVELENGTH"] = GLOBAL_WAVELENGTHS
    else:
        raise ValueError("don't recognize this instrument mode")
    metadata["IX"] = np.array(range(len(metadata["WAVELENGTH"])))
    # 'channel' or 'channel #' in mission nomenclature
    metadata["BAND"] = metadata["IX"] + 1
    metadata["PATH"] = file_path
    return metadata


class M3BandSet(BandSet):
    def __init__(self, file_path, rois=None, threads=None):
        metadata = setup_m3_bandset_metadata(file_path)
        load_method = partial(
            pdr_load, preserve_constants=[-999.0], object_name='PRIMARY'
        )
        super().__init__(
            metadata=metadata,
            load_method=load_method,
            rois=rois,
            threads=threads,
        )
