"""
defines a class for organizing and performing bulk rendering operations on
multispectral image products
"""
import logging
from collections.abc import Mapping, Callable, Collection, Sequence
from itertools import chain
from typing import Optional, Union

# note: ignore complaints from static analyzers about this import. dill
# performs pickling magick at import.
from cytoolz.dicttoolz import merge, valfilter
import dill
from pathos.multiprocessing import ProcessPool
import numpy as np
import pandas as pd


from marslab.imgops.debayer import make_bayer, debayer_upsample
from marslab.imgops.imgutils import get_from_all, absolutely_destroy, mapfilter
from marslab.imgops.look import Look
from marslab.imgops.poolutils import wait_for_it

log = logging.getLogger(__name__)


class BandSet:
    """
    class for organizing and performing bulk rendering operations on
    multispectral image products
    """

    # TODO: do I need to allow more of these on init? like for copying? maybe?
    def __init__(
        self,
        metadata: pd.DataFrame = None,
        rois: Mapping = None,
        bayer_info: Mapping = None,
        load_method: Callable = None,
        name: str = None,
        threads: Mapping = None,
        raw: Mapping = None,
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
        self.raw = raw
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
        if self.load_method is None:
            from marslab.imgops.loaders import rasterio_load

            self.load_method = rasterio_load

    def setup_pool(self, thread_type):
        if self.threads.get(thread_type) is not None:
            log.info("... initializing worker pool ...")
            pool = ProcessPool(self.threads.get(thread_type))
            pool.restart()
            return pool

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
            log.info("Not all requested bands are available.")
        # group bands by file -- for instruments like MASTCAM, this is a
        # one-row / band df per file; for RGB images, three rows per file; for
        # instruments like Kaguya or Supercam, dozens or hundreds per file.
        loading = load_df.loc[load_df["BAND"].isin(bands)]
        chunked_by_file = loading.dropna(subset=["PATH"]).groupby("PATH")
        # TODO, maybe: dispatch single and multithreaded cases separately?
        if pool is None:
            results = []
            for path, band_df in chunked_by_file:
                results.append(self.load_method(path, band_df, bands))
                log.info("loaded " + path)
        else:
            results = {}
            # caution: dict comprehension does _not_ work well here
            for path, band_df in chunked_by_file:
                results[path] = pool.apipe(
                    self.load_method, path, band_df, bands
                )
            results = wait_for_it(pool, results, log)
        self.raw |= merge(results)

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
        if bands == "all":
            bands = self.metadata["BAND"].unique()
        for band in bands:
            debayer = self.debayer_if_required(band)
            if debayer is not None:
                self.debayered[band] = debayer
        # TODO: threading isn't working. this is not presently a performance
        #  concern because debayering is not a serious bottleneck for things
        #  we're doing
        # pool = self.setup_pool("debayer")
        # if pool is not None:
        #     db = pool.map_async(self.debayer_if_required, bands)
        #     pool.close()
        #     pool.join()
        #     # this is like the unthreaded case
        #     # except that we get the debayer object as a chunk
        #     self.debayered = {
        #         band: array_or_none
        #         for band, array_or_none in zip(
        #             filter(lambda x: x is not None, bands), db.get()
        #         )
        #     }
        #     return

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

    def prep_look_set(self, instructions: Sequence[Mapping], autoload: bool):
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
        available_looks = mapfilter(
            lambda bands: set(bands).issubset(tuple(self.raw.keys())),
            "bands",
            instructions,
        )
        for op in [op for op in instructions if op not in available_looks]:
            log.info(
                "skipping " + str(op.get("name")) + " due to missing bands"
            )
        return available_looks

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
        if pool is not None:
            log.info("... serializing arrays ...")
        for instruction in available_instructions:
            # do we have a special name? TODO: make this more opinionated?
            op_name = instruction.get("name")
            if op_name is None:
                op_name = (
                    instruction["look"] + "_" + "_".join(instruction["bands"])
                )
            op_images = [
                self.get_band(band).copy() for band in instruction["bands"]
            ]
            base_image = None
            if instruction.get("overlay") is not None:
                base_image = self.get_band(
                    instruction["overlay"]["band"]
                ).copy()
            # make processing pipeline fron instruction
            pipeline = Look.compile_from_instruction(
                instruction, metadata=self.metadata
            )
            # all of cropper, pre, post, overlay can potentially be absent --
            # these are _possible_ steps in the pipeline. note that wavelengths
            # for spectops are added automagically by the Look compiler.
            if pool is not None:
                look_cache[op_name] = pool.apipe(
                    pipeline.execute, op_images, base_image=base_image
                )
            else:
                look_cache[op_name] = pipeline.execute(
                    op_images, base_image=base_image
                )
                log.info("generated " + op_name)
        if pool is not None:
            look_cache = wait_for_it(
                pool, look_cache, log, message="generated ", as_dict=True
            )
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


class ImageBands(BandSet):
    """
    simple case of a bandset produced from a single multichannel image.
    """

    def __init__(self, path, load_method=None, **bandset_kwargs):
        if load_method is None:
            from marslab.imgops.loaders import pil_load

            load_method = pil_load
        metadata = pd.DataFrame()
        metadata["PATH"] = path
        super().__init__(
            metadata=metadata, load_method=load_method, **bandset_kwargs
        )
        # TODO: stuff about automatically coercing grayscale if desired
        self.raw = load_method(path)
        metadata["BAND"] = self.raw.keys()
        metadata["IX"] = self.raw.keys()
