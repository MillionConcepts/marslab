from itertools import chain

from multiprocessing import Pool

# Note: ignore any import optimization hints to remove `import dill`!
# it does magick on import that makes some kinds of serialization work.
# import dill
import numpy as np
import pandas as pd
from cytoolz import merge, valfilter

import marslab.spectops
from marslab.imgops import (
    make_bayer,
    make_spectral_rapidlook,
    render_enhanced,
    decorrelation_stretch,
    depth_stack,
    render_overlay,
    debayer_upsample,
    border_crop,
)


def rasterio_scaler(reader, preserve_constants=None, float_dtype=np.float32):
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
        if array.dtype.char in np.typecodes["AllInteger"]:
            scaled = array.astype(float_dtype).copy()
        else:
            scaled = array.copy()
        not_special = scaled[np.isin(scaled, preserve_constants, invert=True)]
        not_special += scale
        not_special *= offset
        return scaled

    return scaler


def rasterio_load_scaled(path, band_df, bands, **scale_kwargs):
    """
    simple rasterio-based image loading function that reads an image in and
    scales it if applicable
    """
    import rasterio

    band_arrays = {}
    reader = rasterio.open(path)
    scaler = rasterio_scaler(reader, **scale_kwargs)
    for _, band in band_df.iterrows():
        if band["BAND"] not in bands:
            continue
        band_arrays[band["BAND"]] = scaler(
            reader.read(int(band["IX"] + 1)), band["IX"]
        )
    return band_arrays


def pdr_scaler(data, preserve_constants=None, float_dtype=np.float32):
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
        if image.dtype.char in np.typecodes["AllInteger"]:
            scaled = image.astype(float_dtype).copy()
        else:
            scaled = image.copy()
        not_special = scaled[np.isin(scaled, preserve_constants, invert=True)]
        not_special += scale
        not_special *= offset
        return scaled

    return scaler


def pdr_load_scaled(path, band_df, bands):
    """
    simple pdr-based image loading function that reads an image in and
    scales it if applicable
    """
    import pdr

    band_arrays = {}
    data = pdr.read(path)
    scaler = pdr_scaler(data)
    for _, band in band_df.iterrows():
        if band["BAND"] not in bands:
            continue
        band_arrays[band["BAND"]] = scaler(data.IMAGE, band["IX"])
    return band_arrays


def make_look(
    operation,
    op_images,
    option_dict,
    overlay_dict=None,
    base_image=None,
    op_wavelengths=None,
):
    """
    primary execution function for handling look rendering. make an individual
    look from distinct image inputs. called by BandSet.make_look_set() as part
    of interpreting a full look markup dictionary, but can also be manually
    called for individual looks
    """
    if operation in marslab.spectops.SPECTOP_NAMES:
        look_image = make_spectral_rapidlook(
            spectop=getattr(marslab.spectops, operation),
            op_images=op_images,
            op_wavelengths=op_wavelengths,
            **option_dict
        )
    elif operation == "enhanced color":
        look_image = render_enhanced(op_images, **option_dict)
    elif operation == "dcs":
        look_image = decorrelation_stretch(
            depth_stack(op_images), **option_dict
        )
    else:
        raise ValueError("unknown look operation " + operation)
    if overlay_dict is not None:
        look_image = render_overlay(
            base_image=base_image, overlay_image=look_image, **overlay_dict
        )
    return look_image


# noinspection PyArgumentList
class BandSet:
    def __init__(
        self,
        metadata=None,
        raw=None,
        debayered=None,
        looks=None,
        thumbs=None,
        extended=None,
        compact=None,
        summary=None,
        rois=None,
        bayer_info=None,
        load_method=rasterio_load_scaled,
        name=None,
        counts=None,
    ):
        """
        :param metadata: dataframe containing at least "PATH", "BAND", "IX,
        " optionally others
        :param raw: dictionary of NAME:ndarray or pdr.Data corresponding to
        unprocessed images
        :param debayered: dictionary of NAME:ndarray or pdr.Data
        corresponding to debayered versions of raw images
        :param looks = dictionary of str - Union[ndarray, mpl.figure.Figure,
        PIL.Image] -- images generated by
            other methods
        :param thumbs = dictionary of str - Union[ndarray, mpl.figure.Figure,
        PIL.Image] -- thumbnails
        :param extended -- pd.DataFrame, extended metadata perhaps plus
        data, variably definable,
            but generally unpivoted metadata for both band level and
            region/pointing/spectrum level
        :param compact -- pd.DataFrame, compact metadata perhaps plus data,
        variably definable,
            but generally metadata pivoted on region/pointing/spectrum level
        :param summary -- pd.DataFrame, just metadata pivoted on
        pointing/spectrum level
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
        :param roi_counts -- df of values for ROIs counted across bands
        """
        self.metadata = metadata
        self.raw = raw
        self.debayered = debayered
        self.looks = looks
        self.thumbs = thumbs
        self.extended = extended
        self.compact = compact
        self.summary = summary
        self.rois = rois
        self.bayer_info = bayer_info
        self.load_method = load_method
        self.name = name
        self.counts = counts
        if isinstance(metadata, pd.DataFrame):
            if "IX" not in metadata.columns:
                metadata["IX"] = 0
        for image_set_attribute in ("raw", "looks", "thumbs", "debayered"):
            if getattr(self, image_set_attribute) is None:
                setattr(self, image_set_attribute, {})

    def load(self, bands, threads=None):
        if threads is not None:
            pool = Pool(threads)
        else:
            pool = None
        load_df = self.metadata.copy()

        if bands == "all":
            bands = load_df["BAND"].values
            relevant_metadata = load_df
        else:
            relevant_metadata = load_df.loc[load_df["BAND"].isin(bands)]
        # group bands by file -- for instruments like MASTCAM, this is a
        # one-row/band df per file;
        # for RGB images, it will be three rows per file;
        # for instruments like Kaguya or Supercam, it could be hundreds per
        # file.
        chunked_by_file = relevant_metadata.dropna(subset=["PATH"]).groupby(
            "PATH"
        )
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

    def make_db_masks(self, shape=None, remake=False):
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

    def bayer_pixel(self, band_name):
        if self.metadata is not None:
            if "BAYER_PIXEL" in self.metadata.columns:
                return self.metadata.loc[
                    self.metadata["BAND"] == band_name, "BAYER_PIXEL"
                ].iloc[0]
        return None

    def band_wavelength(self, band_name):
        return self.metadata.loc[
            self.metadata["BAND"] == band_name, "WAVELENGTH"
        ].iloc[0]

    def debayer_if_required(self, band_name, use_cache=True):
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

    def bulk_debayer(self, bands, threads=None):
        """
        debayer all bands according to spec in self.metadata and self.bayer_info,
        asynchronously / multithreaded if threads parameter is not None;
        cache in self.debayered
        """
        if threads is not None:
            pool = Pool(threads)
            db = pool.map_async(self.debayer_if_required, bands)
            pool.close()
            pool.join()
            self.debayered = {
                band: array_or_none
                for band, array_or_none in zip(
                    filter(lambda x: x is not None, bands), db.get()
                )
            }
        else:
            for band in bands:
                debayer = self.debayer_if_required(band)
                if debayer is not None:
                    self.debayered[band] = debayer

    def make_look_set(
        self,
        instructions,
        autoload=True,
        debayer_threads=None,
        look_threads=None,
    ):
        # don't mess with the literal from larger scope
        all_look_bands = set(
            chain.from_iterable(
                [instruction["bands"] for instruction in instructions.values()]
            )
        )
        if (autoload is True) and (self.metadata is not None):
            self.load(
                [
                    band
                    for band in all_look_bands
                    if (band in self.metadata["BAND"].unique())
                    and (band not in self.raw.keys())
                ]
            )

        available_instructions = valfilter(
            lambda value: set(value["bands"]).issubset(tuple(self.raw.keys())),
            instructions,
        )
        # TODO: make this (and all chatter) better
        print("debayering images (if required).")
        bands_at_hand = [
            band for band in all_look_bands if band in tuple(self.raw.keys())
        ]
        self.bulk_debayer(bands_at_hand, threads=debayer_threads)
        # TODO, maybe: print skipping messages
        look_cache = {}
        if look_threads is not None:
            pool = Pool(8)
        else:
            pool = None
        for instruction in available_instructions.values():
            bands = instruction["bands"]
            operation = instruction["operation"]
            option_dict = instruction["options"]
            if "name" in instruction.keys():
                op_name = instruction["name"]
            else:
                op_name = operation
            print("generating " + op_name + " " + str(bands))
            # TODO: if this is ever anything fancier than cropping,
            #  add a cache back in --
            #  and make sure this isn't doing anything mutable and upsetting
            op_images = []
            for band in bands:
                if band in self.debayered.keys():
                    op_image = self.debayered[band]
                else:
                    op_image = self.raw[band]
                if "crop" in instruction.keys():
                    op_images.append(
                        border_crop(op_image, *instruction["crop"])
                    )
                else:
                    op_images.append(op_image)
            # TODO: these are both sloppy. keeping self away from make_look
            #  is sort of a priority,
            #  but...
            op_wavelengths = None
            if self.metadata is not None:
                if "WAVELENGTH" in self.metadata.columns:
                    op_wavelengths = [
                        self.band_wavelength(band) for band in bands
                    ]
            if "overlay" in instruction.keys():
                base_image = self.raw[instruction["overlay"]["band"]]
                overlay_option_dict = instruction["overlay"]["options"]
            else:
                base_image = None
                overlay_option_dict = None
            # TODO: ugly
            # TODO: assess -- it seems like matplotlib is already doing some multithreading?
            if pool is not None:
                look_cache[op_name] = pool.apply_async(
                    make_look,
                    (
                        operation,
                        op_images,
                        option_dict,
                        overlay_option_dict,
                        base_image,
                        op_wavelengths,
                    ),
                )
            else:
                look_cache[op_name] = make_look(
                    operation,
                    op_images,
                    option_dict,
                    overlay_option_dict,
                    base_image,
                    op_wavelengths,
                )
        if pool is not None:
            pool.close()
            pool.join()
            look_cache = {
                look_name: look_result.get()
                for look_name, look_result in look_cache.items()
            }

        self.looks |= look_cache
        print(self.looks.keys())
