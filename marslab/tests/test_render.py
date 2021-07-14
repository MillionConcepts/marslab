from collections import ChainMap

import numpy as np
from hypothesis import (
    given,
    strategies as st,
    reject,
    assume,
    settings as hset,
    Verbosity,
)
from hypothesis.errors import UnsatisfiedAssumption
from hypothesis.extra.numpy import (
    arrays,
    scalar_dtypes,
    array_shapes,
    integer_dtypes,
    floating_dtypes,
    from_dtype,
)
from hypothesis.strategies import integers
from more_itertools import all_equal

from marslab.imgops.render import decorrelation_stretch, eightbit
from marslab.tests.utilz.utilz import (
    positive_finite_floats,
    finite_floats,
    finite_numbers,
)

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
def normal_array():
    randarray = rng.normal(128, 16, (256, 256))
    return eightbit(randarray)

def test_decorrelation_stretch_1():
    channels = [normal_array() for _ in range(3)]
    stretched = decorrelation_stretch(channels, contrast_stretch=1)
    assert np.std(np.dstack(channels)) < np.std(eightbit(stretched))

@given(
channels=st.lists(
        arrays(
            dtype=np.float32,
            shape=st.shared(
                array_shapes(min_dims=2, max_dims=2, min_side=2), key="hi"
            ),
            elements={
                "allow_nan": False,
                "allow_infinity": False,
                "min_value": -1E8,
                "max_value": 1E8,
            },
            fill=st.nothing(),
        ),
        min_size=2,
    ),
)
def test_decorrelation_stretch_2(channels):
    # single-valued arrays have undefined covariance
    assume(all([not len(np.unique(channel)) == 1 for channel in channels]))
    # ridiculously large inputs relative to dtype we expect to cause errors
    assume(all([not np.isinf(np.std(channel)) for channel in channels]))
    stretched = decorrelation_stretch(channels)
    chstd = np.std(np.dstack(channels))
    stretchstd = np.std(eightbit(stretched))
    if np.all(~np.isnan(stretchstd)):
        print(stretchstd, chstd)
        assert np.std(np.dstack(channels)) < np.std(eightbit(stretched))
#


# @hset(max_examples=500, verbosity=Verbosity.verbose)
# @given(
#     channels=st.lists(
#         arrays(
#             dtype=st.one_of(floating_dtypes(), integer_dtypes()),
#             shape=st.shared(
#                 array_shapes(min_dims=2, max_dims=2, min_side=2), key="hi"
#             ),
#             elements={
#                 "allow_nan": False,
#                 "allow_infinity": False,
#             },
#             fill=st.nothing(),
#         ),
#         min_size=2,
#     ),
#     contrast_stretch=st.one_of(
#         st.none(),
#         st.floats(min_value=0, max_value=100),
#         st.lists(
#             st.floats(min_value=0, max_value=100), max_size=2, min_size=2
#         ),
#     ),
#     special_constants=st.one_of(
#         st.none(),
#         st.lists(finite_floats),
#         st.sets(finite_floats),
#         st.frozensets(st.floats()),
#         st.dictionaries(keys=finite_floats, values=finite_floats),
#         st.dictionaries(keys=finite_floats, values=st.none()).map(dict.keys),
#         st.dictionaries(keys=st.integers(), values=finite_floats).map(
#             dict.values
#         ),
#         st.dictionaries(keys=finite_floats, values=finite_floats).map(
#             ChainMap
#         ),
#     ),
#     sigma=st.one_of(st.none(), positive_finite_floats),
# )
# def test_fuzz_decorrelation_stretch_nasty(
#     channels, contrast_stretch, special_constants, sigma
# ):
#     # try:
#     # single-valued arrays have undefined covariance
#     assume(all([not len(np.unique(channel)) == 1 for channel in channels]))
#     # ridiculously large inputs relative to dtype we expect to cause errors
#     assume(all([not np.isinf(np.std(channel)) for channel in channels]))
#
#     decorrelation_stretch(
#         channels=channels,
#         contrast_stretch=contrast_stretch,
#         special_constants=special_constants,
#         sigma=sigma,
#     )

@given(
    channels=st.lists(
        arrays(
            dtype=np.float32,
            shape=st.shared(
                array_shapes(min_dims=2, max_dims=2, min_side=2), key="hi"
            ),
            elements={
                "allow_nan": False,
                "allow_infinity": False,
                "min_value": -1E8,
                "max_value": 1E8,
            },
            fill=st.nothing(),
        ),
        min_size=2,
    ),
    contrast_stretch=st.one_of(
        st.none(),
        st.floats(min_value=0, max_value=100),
        st.lists(
            st.floats(min_value=0, max_value=100), max_size=2, min_size=2
        ),
    ),
    special_constants=st.one_of(
        st.none(),
        st.lists(finite_floats),
        st.sets(finite_floats),
        st.frozensets(st.floats()),
        st.dictionaries(keys=finite_floats, values=finite_floats),
        st.dictionaries(keys=finite_floats, values=st.none()).map(dict.keys),
        st.dictionaries(keys=st.integers(), values=finite_floats).map(
            dict.values
        ),
        st.dictionaries(keys=finite_floats, values=finite_floats).map(
            ChainMap
        ),
    ),
    sigma=st.one_of(st.none(), positive_finite_floats),
)
def test_fuzz_decorrelation_stretch_nice(
    channels, contrast_stretch, special_constants, sigma
):
    # try:
    # single-valued arrays have undefined covariance
    assume(all([not len(np.unique(channel)) == 1 for channel in channels]))
    # ridiculously large inputs relative to dtype we expect to cause errors
    assume(all([not np.isinf(np.std(channel)) for channel in channels]))

    decorrelation_stretch(
        channels=channels,
        contrast_stretch=contrast_stretch,
        special_constants=special_constants,
        sigma=sigma,
    )
