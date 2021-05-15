from collections.abc import Callable, Mapping, Collection, Sequence
from itertools import chain
from multiprocessing import Pool
from operator import methodcaller
from typing import Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from cytoolz.dicttoolz import merge, valfilter
from cytoolz.functoolz import compose_left, curry

import marslab.spectops
from marslab.imgops import (
    make_bayer,
    make_spectral_rapidlook,
    render_rgb_composite,
    decorrelation_stretch,
    render_overlay,
    debayer_upsample,
    absolutely_destroy,
    crop_all,
    broadcast_filter,
)

if TYPE_CHECKING:
    import pdr
    import rasterio


def get_from_all(key, mappings, default=None):
    if isinstance(mappings, Mapping):
        view = mappings.values()
    else:
        view = mappings
    return list(map(methodcaller("get", key, default), view))


def unpack_pipeline_specification(func_specification_dict: Optional[Mapping]):
    if func_specification_dict is None:
        return None
    func = curry(func_specification_dict.get("function"))
    params = func_specification_dict.get("params", {})
    return func(**params)


def unpack_look_overlay(overlay_specification_dict: Optional[Mapping]):
    if overlay_specification_dict is None:
        return None
    options = overlay_specification_dict.get("options", {})
    return curry(render_overlay)(**options)


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
    path: str, band_df: pd.DataFrame, bands: pd.Series, **scale_kwargs
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
            # in the context of ZCAM, iof_est / EPO-style
            # float32 IOFs based on RAF-type products
            return image
        # leaving special constants as they are
        scale = data.LABEL["IMAGE"]["SCALING_FACTOR"]
        offset = data.LABEL["IMAGE"]["OFFSET"]
        return cast_scale(
            data.IMAGE, scale, offset, preserve_constants, float_dtype
        )

    return scaler


def pdr_load_scaled(path: str, band_df: pd.DataFrame, bands: pd.Series):
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


def make_look_pipeline(
    looker=None,
    cropper=None,
    prefilter=None,
    postfilter=None,
    plotter=None,
    broadcast_prefilter=True,
):
    # by default, wrap the prefilter function so that it works on each channel;
    # some notional prefilters may of course not want this!
    if (broadcast_prefilter is True) and (prefilter is not None):
        prefilter = broadcast_filter(prefilter)
    pipeline = []
    for process in filter(
        None, (cropper, prefilter, looker, postfilter, plotter)
    ):
        pipeline.append(process)
    return compose_left(*pipeline)


def interpret_operation_name(operation):
    if operation in marslab.spectops.SPECTOP_NAMES:
        return curry(make_spectral_rapidlook)(
            spectop=getattr(marslab.spectops, operation)
        )
    elif operation in ("enhanced color", "true color", "composite"):
        return curry(render_rgb_composite)
    elif operation == "dcs":
        return curry(decorrelation_stretch)
    else:
        raise ValueError("unknown look operation " + operation)


# noinspection PyArgumentList
class BandSet:
    # TODO: do I need to allow more of these on init? like for copying? maybe?
    def __init__(
        self,
        metadata=None,
        rois=None,
        bayer_info=None,
        load_method=rasterio_load_scaled,
        name=None,
        threads=None,
    ):
        """
        :param metadata: dataframe containing at least "PATH", "BAND", "IX,
        " optionally others
        # :param raw: dictionary of NAME:ndarray or pdr.Data corresponding to
        # unprocessed images
        # :param debayered: dictionary of NAME:ndarray or pdr.Data
        # corresponding to debayered versions of raw images
        # :param looks = dictionary of str - Union[ndarray, mpl.figure.Figure,
        # PIL.Image] -- images generated by
        #     other methods
        # :param thumbs = dictionary of str - Union[ndarray, mpl.figure.Figure,
        # PIL.Image] -- thumbnails
        # :param extended -- pd.DataFrame, extended metadata perhaps plus
        # data, variably definable,
        #     but generally unpivoted metadata for both band level and
        #     region/pointing/spectrum level
        # :param compact -- pd.DataFrame, compact metadata perhaps plus data,
        # variably definable,
        #     but generally metadata pivoted on region/pointing/spectrum level
        # :param summary -- pd.DataFrame, just metadata pivoted on
        # pointing/spectrum level
        :param rois -- str, pathlike, or hdulist with 'regions of interest'
        drawn on images
        :param bayer_info: dict containing one or more of 'mask': full bayer
        masks fitting raw image sizes,
            'pattern' -- bayer pattern definition, 'row_column' -- row and
            column positions for
            bilinear_interpolate_subgrid
        :param load_method -- 'pdr' or 'rasterio' -- how to open files
        :param name -- str, designated name for this observation / cluster
        of bands / etc.
        # :param counts -- df of values for ROIs counted across bands
        :param threads -- dict of thread counts for different things
        """
        self.metadata = metadata
        self.raw = None
        self.debayered = None
        self.looks = None
        self.thumbs = None
        self.extended = None
        self.compact = None
        self.summary = None
        self.rois = rois
        self.bayer_info = bayer_info
        self.load_method = load_method
        self.name = name
        self.counts = None
        self.threads = threads
        if isinstance(metadata, pd.DataFrame):
            if "IX" not in metadata.columns:
                metadata["IX"] = 0
        for mapping in ("raw", "looks", "thumbs", "debayered", "threads"):
            if getattr(self, mapping) is None:
                setattr(self, mapping, {})
        self.cache_names = ("raw", "debayered", "looks")

    def setup_pool(self, thread_type):
        if self.threads.get(thread_type) is not None:
            return Pool(self.threads.get(thread_type))
        return None

    def load(
        self, bands: Collection[str], reload: bool = False, quiet: bool = False
    ):
        pool = self.setup_pool("load")
        load_df = self.metadata.copy()
        if bands == "all":
            bands = load_df["BAND"]
        else:
            bands = pd.Series(tuple(bands))
        if reload is False:
            bands = bands.loc[~bands.isin(self.raw.keys())]
        if (quiet is False) and not (bands.isin(load_df["BAND"]).all()):
            print("Not all requested bands are available.")
        # group bands by file -- for instruments like MASTCAM, this is a
        # one-row / band df per file; for RGB images, three rows per file; for
        # instruments like Kaguya or Supercam, dozens or hundreds per file.
        loading = load_df.loc[load_df["BAND"].isin(bands)]
        chunked_by_file = loading.dropna(subset=["PATH"]).groupby("PATH")
        band_results = []
        for path, band_df in chunked_by_file:
            if pool is not None:
                band_results.append(
                    pool.apply_async(self.load_method, (path, band_df, bands))
                )
            else:
                band_results.append(self.load_method(path, band_df, bands))
        if pool is not None:
            pool.close()
            pool.join()
            band_results = [result.get() for result in band_results]
        self.raw |= merge(band_results)

    def make_db_masks(self, shape: Sequence[int, int] = None, remake=False):
        if "masks" in self.bayer_info.keys():
            if remake is False:
                return
        if shape is None:
            try:
                shape = next(iter(self.raw.values())).shape
            except (AttributeError, StopIteration):
                raise ValueError(
                    "Need loaded images or an explicit shape to make debayer "
                    "masks."
                )
        self.bayer_info["masks"] = make_bayer(
            shape, self.bayer_info["pattern"]
        )
        self.bayer_info["row_column"] = {
            pixel: (np.unique(mask[0]), np.unique(mask[1]))
            for pixel, mask in self.bayer_info["masks"].items()
        }

    def bayer_pixel(self, band_name: str):
        try:
            return self.metadata.loc[
                self.metadata["BAND"] == band_name, "BAYER_PIXEL"
            ].iloc[0]
        except (KeyError, ValueError, AttributeError):
            return None

    def wavelength(self, band_names: Union[str, Sequence[str]]):
        wavelengths = []
        if isinstance(band_names, str):
            band_names = [band_names]
        for band_name in band_names:
            try:
                wavelengths.append(
                    self.metadata.loc[
                        self.metadata["BAND"] == band_name, "WAVELENGTH"
                    ].iloc[0]
                )
            except (KeyError, ValueError, AttributeError):
                continue
        return wavelengths

    def debayer_if_required(self, band_name: str, use_cache: bool = True):
        """
        return a debayered version of an image, if self.metadata["BAYER_PIXEL"]
        suggests that there's debayering to do and a raw image is available.
        optionally fetch from cache of debayered images.
        """
        pixel = self.bayer_pixel(band_name)
        if pixel is None:
            return None
        if use_cache and (band_name in self.debayered.keys()):
            return self.debayered[band_name]
        self.make_db_masks()
        return debayer_upsample(
            self.raw[band_name],
            pixel=pixel,
            masks=self.bayer_info["masks"],
            row_column=self.bayer_info["row_column"],
        )

    def bulk_debayer(self, bands: Collection[str]):
        """
        debayer all bands according to spec in self.metadata and
        self.bayer_info, asynchronously / multithreaded if
        bandset.threads["debayer"] is set; cache in self.debayered

        don't set None for non-debayered images: debayer availability
        should be visible by looking at bandset.debayered's keys
        """
        pool = self.setup_pool("debayer")
        if pool is not None:
            db = pool.map_async(self.debayer_if_required, bands)
            pool.close()
            pool.join()
            # this is like the unthreaded case
            # except that we get the debayer object as a chunk
            self.debayered = {
                band: array_or_none
                for band, array_or_none in zip(
                    filter(lambda x: x is not None, bands), db.get()
                )
            }
            return
        for band in bands:
            debayer = self.debayer_if_required(band)
            if debayer is not None:
                self.debayered[band] = debayer

    def get_band(self, band: str):
        """
        get the "most processed" (right now just meaning debayered if
        in a bayered band, raw if not in a bayered band) version of a cached
        image. does not 'intelligently' check for anything like
        BandSet.debayer_if_required() -- this grabs _only_ from cache.
        caveat emptor. TODO: consider unifying these anyway.
        """
        if band in self.debayered.keys():
            return self.debayered[band]
        return self.raw[band]

    def prep_look_set(
        self, instructions: Mapping[str, Mapping], autoload: bool
    ):
        """
        filter the instruction set we want for the images we have.
        if requested, also load/cache images, debayering as required.
        """
        # what bands do we want?
        desired_bands = set(
            chain.from_iterable(get_from_all("bands", instructions))
        )
        # try to get them, if we don't have them yet
        if (autoload is True) and (self.metadata is not None):
            self.load(desired_bands, quiet=True)
            # debayer them, if needed
            self.bulk_debayer(
                set(self.raw.keys()).intersection(desired_bands),
            )
        # what looks can we make with what we have?
        return valfilter(
            lambda value: set(value["bands"]).issubset(tuple(self.raw.keys())),
            instructions,
        )

    def make_look_set(
        self,
        instructions: Mapping[str, Mapping],
        autoload: bool = True,
    ):
        # load images and filter instruction set for unavailable bands
        available_instructions = self.prep_look_set(instructions, autoload)
        # TODO, maybe: print skipping messages
        look_cache = {}
        pool = self.setup_pool("look")
        for inst in available_instructions.values():
            # do we have a special name? TODO: make this more opinionated?
            op_name = inst.get("name")
            if op_name is None:
                op_name = inst["operation"]
            if not inst.get("no_band_names"):
                op_name += " " + "_".join(inst["bands"])
            print("generating " + op_name)
            op_images = [self.get_band(band).copy() for band in inst["bands"]]
            # get heart of pipeline: the look function (spectrum op, dcs, etc.)
            looker = interpret_operation_name(inst["operation"])
            # this is a curried function. bind options and auxiliary info:
            looker = looker(
                **inst.get("look_params", {}),
            )
            # add wavelength values to spectops -- others don't care as of now
            if looker.func.__name__ == "make_spectral_rapidlook":
                looker = looker(wavelengths=self.wavelength(inst["bands"]))
            # all of cropper, pre, post, overlay can potentially be absent --
            # these are _possible_ steps in the pipeline.
            cropper = curry(crop_all)(bounds=inst.get("crop"))
            pre = unpack_pipeline_specification(inst.get("prefilter"))
            post = unpack_pipeline_specification(inst.get("postfilter"))
            overlay = unpack_look_overlay(inst.get("overlay"))
            # grab base image layer if we're making an overlay
            if overlay is not None:
                base_image = cropper(
                    [self.get_band(inst["overlay"]["band"]).copy()]
                )[0]
                plotter = overlay(base_image=base_image)
            # finally, are we rendering a matplotlib image, and if so, how?
            # TODO: maybe this should be merged in some way with overlay,
            #  which is in and of itself a fancy matplotlib trick --
            #  or the overlay should be performed differently?
            else:
                plotter = unpack_pipeline_specification(
                    inst.get("mpl_settings")
                )
            pipeline = make_look_pipeline(looker, cropper, pre, post, plotter)

            if pool is not None:
                look_cache[op_name] = pool.apply_async(pipeline, op_images)
            else:
                look_cache[op_name] = pipeline(op_images)
        if pool is not None:
            pool.close()
            pool.join()
            look_cache = {
                look_name: look_result.get()
                for look_name, look_result in look_cache.items()
            }
        self.looks |= look_cache

    def purge(self, what: Optional[str] = None) -> None:
        if what is None:
            for cache_name in self.cache_names:
                absolutely_destroy(getattr(self, cache_name))
                setattr(self, cache_name, {})
        elif what in self.cache_names:
            absolutely_destroy(getattr(self, what))
            setattr(self, what, {})
        else:
            raise ValueError(str(what) + " is not a valid cache type.")
