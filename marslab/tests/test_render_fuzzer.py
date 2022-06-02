from collections import ChainMap

import numpy as np
from hypothesis import given, strategies as st, assume
from hypothesis.extra.numpy import arrays, array_shapes

from marslab.imgops.render import decorrelation_stretch
from marslab.tests.utilz.utilz import positive_finite_floats, finite_floats


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
        st.frozensets(finite_floats),
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