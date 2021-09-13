from functools import partial

import numpy as np
import pandas as pd
from cytoolz import merge
from numpy.random import default_rng

from marslab.bandset.bandset import log, BandSet

rng = default_rng()


def image_dict_loader(image_dict):
    def load_test_image(band_name):
        return image_dict[band_name]

    return load_test_image


def noise_image_loader(shape):
    def load_noise_image(_):
        return rng.poisson(lam=100, size=shape)

    return load_noise_image


def mock_load(self, generator, *_, **__):
    results = []
    for band_name in self.metadata["BAND"]:
        results.append({band_name: generator(band_name)})
        log.info(f"loaded test {band_name}")
    self.raw = merge(results)


def make_random_bandset():
    num_bands = rng.integers(5, 50)
    shape = rng.integers(1000, 2000, 2)
    band_names = list(range(num_bands))
    bandset = BandSet()
    bandset.metadata = pd.DataFrame()
    bandset.metadata["PATH"] = np.repeat("/fake/path/to/image.png", num_bands)
    bandset.metadata["BAND"] = band_names
    bandset.metadata["IX"] = 0
    bandset.load = partial(mock_load, bandset, noise_image_loader(shape))
    return bandset
