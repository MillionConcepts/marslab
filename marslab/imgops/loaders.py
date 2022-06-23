"""
generic image-loading functions for multispectral ops
"""
import sys
from pathlib import Path
from typing import (
    Optional, TYPE_CHECKING, Union, Callable, Sequence, Collection, Mapping
)

from dustgoggles.structures import enumerate_as_mapping
import numpy as np

if TYPE_CHECKING:
    import PIL.Image
    import rasterio
    import pdr
    import pandas as pd


def dont_scale(array, *_, **__):
    """identity function plus a black hole for args and kwargs"""
    return array


def cast_scale(
    array: np.ndarray,
    scale: float,
    offset: float,
    preserve_constants: Optional[Sequence[float]] = None,
    float_dtype: np.ScalarType = np.float32,
) -> np.ndarray:
    """
    utility function for scaled loaders in this module. actually apply the
    do_scale and offset specified in the file. if the array is an integer dtype,
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


def prep_scaled_loader(
    metadata, bands, image, scaler_factory, scale, **scale_kwargs
):
    """
    generic prep function for scaled loaders
    """
    if scale is False:
        scaler = dont_scale
    else:
        scaler = scaler_factory(image, **scale_kwargs)
        bands, metadata = unpack_pd_bands(bands, metadata)
    # if no band metadata is passed, just enumerate the bands
    if len(image.shape) != 3:
        band_count = 1
    else:
        band_count = image.shape[-1]
    if metadata is None:
        metadata = [{"BAND": ix, "IX": ix} for ix in range(band_count)]
    # if bands aren't passed, get everything
    if bands is None:
        bands = [ix for ix in range(band_count)]
    return scaler, metadata, bands


def unpack_pd_bands(bands, metadata):
    if "pandas" not in sys.modules:
        return bands, metadata
    import pandas as pd

    if isinstance(bands, pd.Series):
        bands = bands.values
    if isinstance(metadata, pd.DataFrame):
        metadata = metadata.to_dict(orient="records")
    return bands, metadata


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


def rasterio_load(
    path: str,
    metadata: Optional["pd.DataFrame"] = None,
    bands: Optional[Sequence[Union[str, int]]] = None,
    _precached=None,
    do_scale=True,
    **scale_kwargs
) -> dict[Union[int, str], np.ndarray]:
    """
    simple rasterio-based image loading function that reads an image in and
    scales it if applicable
    """
    import rasterio

    reader = rasterio.open(path)
    scaler_factory = rasterio_scaler
    scaler, metadata, bands = prep_scaled_loader(
        metadata, bands, reader, scaler_factory, do_scale, **scale_kwargs
    )
    band_arrays = {}
    for record in metadata:
        if record["BAND"] not in bands:
            continue
        band_arrays[record["BAND"]] = scaler(
            reader.read(int(record["IX"] + 1)),
            record["IX"],
        )
    return band_arrays


def pil_load_shell(
    image: "PIL.Image",
    metadata: Optional["pd.DataFrame"] = None,
    bands: Optional[Sequence[Union[str, int]]] = None,
):
    bands, metadata = unpack_pd_bands(bands, metadata)
    # different initialization rules from the scaled loaders
    if bands is None:
        bands = image.getbands()
    if metadata is None:
        metadata = [{"BAND": band, "IX": band} for band in image.getbands()]
    band_arrays = {}
    for record in metadata:
        if record["BAND"] not in bands:
            continue
        band_arrays[record["BAND"]] = np.asarray(
            image.getchannel(record["IX"])
        )
    return band_arrays


def pil_load(
    path: str,
    metadata: Optional["pd.DataFrame"] = None,
    bands: Optional[Sequence[Union[str, int]]] = None,
    _precached=None
) -> dict[Union[int, str], np.ndarray]:
    from PIL import Image

    image = Image.open(path)
    return pil_load_shell(image, metadata, bands)


# TODO: is this necessary?
def pil_load_rgb(
    path: Union[str, Path],
    metadata: Optional["pd.DataFrame"] = None,
    _bands=None,
    _precached=None
) -> dict[Union[int, str], np.ndarray]:
    from PIL import Image

    image = Image.open(path)
    image.convert("RGB")
    return pil_load_shell(image, metadata, ("R", "G", "B"))


def pdr_load(
    target: Union[str, "pdr.Data"],
    metadata: Collection = (),
    bands: Sequence[Union[str, int]] = (),
    precached: Optional[Mapping[str, "pdr.Data"]] = None,
    do_scale=True,
    object_name="IMAGE",
    preserve_constants=None,
    float_dtype=np.float32,
    **_scale_kwargs
) -> dict[Union[int, str], np.ndarray]:
    """
    simple pdr-based image loading function. reads an image; scales it using
    pdr's internal scaling functions if requested and applicable.

    target: either a fully-specified path to a local file or an
    already-initialized pdr.Data object
    metadata: metadata about bands that might be loaded from the file
    bands: names of bands to be loaded from the file
    do_scale: scale the image based on label metadata?
    object_name: name of the object, must be the same as its name in
     the pdr.Data object's index
    """
    import pdr

    data = None
    # did we directly pass a pdr.Data object? great
    if isinstance(target, pdr.Data):
        data = target
    # did we cache some pdr.Data objects? also great
    elif precached:
        if target in precached.keys():
            data = precached[target]
    # otherwise initialize pdr.Data object, treating target as a path
    if data is None:
        data = pdr.read(target)
    image = manage_pdr_scaling(
        data, float_dtype, object_name, preserve_constants, do_scale
    )
    bands, metadata = unpack_pd_bands(bands, metadata)
    band_arrays = {}
    for record in metadata:
        if record["BAND"] not in bands:
            continue
        if len(image.shape) > 2:
            band_arrays[record["BAND"]] = image[record["IX"]]
        else:
            band_arrays[record["BAND"]] = image
    return band_arrays


def manage_pdr_scaling(
    data: "pdr.Data",
    float_dtype: type,
    object_name: str,
    preserve_constants: Optional[Sequence[float]],
    do_scale: bool
) -> np.ndarray:
    if do_scale is False:
        return data[object_name]
    # TODO: PDS4 clause will probably need to be cut and/or modified later
    #  depending on pdr development exigencies
    if getattr(data, "standard", None) == "PDS4":
        return data.get_scaled(object_name)
    if preserve_constants is None:
        data.find_special_constants(object_name)
    else:
        data.specials[object_name] = enumerate_as_mapping(preserve_constants)
    return data.get_scaled(object_name, inplace=True, float_dtype=float_dtype)
