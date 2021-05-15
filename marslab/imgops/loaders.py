"""
generic image-loading functions for multispectral ops
"""

from collections.abc import Callable, Sequence
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import rasterio
    import pdr


def cast_scale(
    array: np.ndarray,
    scale: float,
    offset: float,
    preserve_constants: Optional[Sequence[float]] = None,
    float_dtype: np.ScalarType = np.float32,
) -> np.ndarray:
    """
    utility function for scaled loaders in this module. actually apply the
    scale and offset specified in the file. if the array is an integer dtype,
    first cast it to a float dtype for better constant preservation and
    interoperability. could be made minimally more efficient, if it mattered.
    """
    if array.dtype.char in np.typecodes["AllInteger"]:
        scaled = array.astype(float_dtype).copy()
    else:
        scaled = array.copy()
    if preserve_constants is None:
        preserve_constants = []
    not_special = scaled[np.isin(scaled, preserve_constants, invert=True)]
    not_special *= scale
    not_special += offset
    scaled[np.isin(scaled, preserve_constants, invert=True)] = not_special
    return scaled


def rasterio_scaler(
    reader: "rasterio.DatasetReader",
    preserve_constants: Optional[Sequence[float]] = None,
    float_dtype: np.ScalarType = np.float32,
) -> Callable[[np.ndarray, int], np.ndarray]:
    """
    make a scaling function for a particular DatasetReader object
    """

    def scaler(array, band_ix):
        if reader.scales is None:
            scale = 1
        else:
            scale = reader.scales[band_ix]
        if reader.offsets is None:
            offset = 0
        else:
            offset = reader.offsets[band_ix]
        if scale == 1 and offset == 0:
            return array
        return cast_scale(
            array, scale, offset, preserve_constants, float_dtype
        )

    return scaler


def rasterio_load_scaled(
    path: str, band_df: "pd.DataFrame", bands: "pd.Series", **scale_kwargs
):
    """
    simple rasterio-based image loading function that reads an image in and
    scales it if applicable
    """
    import rasterio

    band_arrays = {}
    reader = rasterio.open(path)
    scaler = rasterio_scaler(reader, **scale_kwargs)
    for _, band in band_df.iterrows():
        if band["BAND"] not in bands.values:
            continue
        band_arrays[band["BAND"]] = scaler(
            reader.read(int(band["IX"] + 1)),
            band["IX"],
        )
    return band_arrays


def pdr_scaler(
    data: "pdr.Data",
    preserve_constants: Sequence[float] = None,
    float_dtype=np.float32,
) -> Callable[[np.ndarray, int], np.ndarray]:
    """
    make a scaling function for a particular DatasetReader object
    """

    def scaler(image, band_ix):
        if len(image.shape) == 3:
            image = image[band_ix].copy()
        else:
            image = image.copy()
        if "SCALING_FACTOR" not in data.LABEL["IMAGE"].keys():
            return image
        # leaving special constants as they are
        scale = data.LABEL["IMAGE"]["SCALING_FACTOR"]
        offset = data.LABEL["IMAGE"]["OFFSET"]
        return cast_scale(
            data.IMAGE, scale, offset, preserve_constants, float_dtype
        )

    return scaler


def pdr_load_scaled(path: str, band_df: "pd.DataFrame", bands: "pd.Series"):
    """
    simple pdr-based image loading function that reads an image in and
    scales it if applicable
    """
    import pdr

    band_arrays = {}
    data = pdr.read(path)
    scaler = pdr_scaler(data)
    for _, band in band_df.iterrows():
        if band["BAND"] not in bands.values:
            continue
        band_arrays[band["BAND"]] = scaler(data.IMAGE, band["IX"])
    return band_arrays