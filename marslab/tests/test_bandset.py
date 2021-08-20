import numpy as np

from marslab.tests.utilz.bandset_utilz import (
    make_random_bandset,
)


def test_random_bandset():
    bandset = make_random_bandset()
    bandset.load()
    assert len(bandset.raw.keys()) == len(bandset.metadata)
    assert np.all(bandset.get_band(1) == bandset.raw[1])
    bandset.debayer_if_required(1)
    bandset.bulk_debayer(list(bandset.metadata['BAND'].values))
    assert bandset.debayered == {}