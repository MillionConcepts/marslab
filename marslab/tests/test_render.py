from collections import ChainMap

import numpy as np
from hypothesis import given, strategies as st, reject, assume
from hypothesis.errors import UnsatisfiedAssumption
from hypothesis.extra.numpy import (
    arrays,
    scalar_dtypes,
    array_shapes,
    integer_dtypes,
    floating_dtypes,
)
from more_itertools import all_equal

from marslab.imgops.render import decorrelation_stretch, eightbit

rng = np.random.default_rng()


# def check_dcs_input(channels):
#     if len(channels) < 2:
#         raise ValueError(
#             "At least two channels must be passed to this function"
#         )
#     if not all_equal([channel.shape for channel in channels]):
#         raise ValueError(
#             "arrays passed to this function must have equal shapes"
#         )
#     if not all_equal([channel.dtype for channel in channels]):
#         raise ValueError(
#             "arrays passed to this function must have equal dtypes"
#         )
#     if (
#         channels[0].dtype.char
#         not in np.typecodes["AllInteger"] + np.typecodes["AllFloat"]
#     ):
#         raise ValueError(
#             "arrays passed to this function must be integer or float arrays"
#         )

#
# def normal_array():
#     randarray = rng.normal(128, 16, (256, 256))
#     return eightbit(randarray)
#
#
# def test_decorrelation_stretch_1():
#     channels = [normal_array() for _ in range(3)]
#     stretched = decorrelation_stretch(channels, contrast_stretch=1)
#     assert np.std(np.dstack(channels)) < np.std(eightbit(stretched))
#

@given(
    channels=st.lists(
        arrays(
            dtype=st.one_of(integer_dtypes(), floating_dtypes()),
            shape=array_shapes(min_dims=2, max_dims=2),
        ),
        min_size=2,
    ),
    contrast_stretch=st.one_of(st.none(), st.floats(), st.lists(st.floats())),
    special_constants=st.one_of(
        st.none(),
        st.lists(st.floats()),
        st.sets(st.floats()),
        st.frozensets(st.floats()),
        st.dictionaries(keys=st.floats(), values=st.floats()),
        st.dictionaries(keys=st.floats(), values=st.none()).map(dict.keys),
        st.dictionaries(keys=st.integers(), values=st.floats()).map(
            dict.values
        ),
        st.dictionaries(keys=st.floats(), values=st.floats()).map(ChainMap),
    ),
    sigma=st.one_of(st.none(), st.floats()),
)
def test_fuzz_decorrelation_stretch(
    channels, contrast_stretch, special_constants, sigma
):
    try:
        assume(all_equal([channel.shape for channel in channels]))
        assume(all([np.all(~np.isnan(channel)) for channel in channels]))
        assume(all([channel.size > 1 for channel in channels]))
        decorrelation_stretch(
            channels=channels,
            contrast_stretch=contrast_stretch,
            special_constants=special_constants,
            sigma=sigma,
        )
    # we're wrapping this so that the pycharm debugger doesn't freeze on every
    # unsatisfied assumption
    except UnsatisfiedAssumption:
        return




