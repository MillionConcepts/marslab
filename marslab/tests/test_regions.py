from functools import reduce
from operator import and_

import numpy as np

from marslab.imgops.regions import count_rois_on_image

rng = np.random.default_rng()


# TODO: parametrize, etc.


def test_with_trivial_case():
    length = rng.integers(25, 500)
    tile_size = (length // 2, length // 2)
    blackwhite_top = np.hstack([np.zeros(tile_size), np.ones(tile_size)])
    blackwhite = np.vstack([blackwhite_top, np.fliplr(blackwhite_top)])
    rois = {ix: np.full(blackwhite.shape, False) for ix in range(4)}
    rois[0][0 : length // 2, 0 : length // 2] = True
    rois[1][0 : length // 2, length // 2 :] = True
    rois[2][length // 2 :, 0 : length // 2] = True
    rois[3][length // 2 :, length // 2 :] = True
    blackwhite[length // 2 + 1, 0] = -9999
    counts = count_rois_on_image(
        rois.values(), rois.keys(), blackwhite, special_constants=[-9999]
    )
    assert reduce(
        and_,
        (
            counts[0]["total"] == 0,
            counts[1]["total"] == blackwhite.size / 4,
            counts[2]["total"] == blackwhite.size / 4 - 1,
            counts[3]["total"] == 0,
        ),
    )
