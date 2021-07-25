import numpy as np
from hypothesis import (
    given,
    strategies as st,
    assume,
)
from hypothesis.extra.numpy import (
    arrays,
    array_shapes,
)

from marslab.imgops.imgutils import normalize_range
from marslab.imgops.render import decorrelation_stretch, eightbit
from marslab.tests.utilz.utilz import normal_array

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


def test_decorrelation_stretch_1():
    for _ in range(100):
        channels = [normal_array() for _ in range(3)]
        stretched = decorrelation_stretch(channels, contrast_stretch=1)
        print(np.std(normalize_range(np.dstack(channels))), np.std(stretched))
        assert np.std(normalize_range(np.dstack(channels))) < np.std(stretched)


@given(
channels=st.lists(
        arrays(
            dtype=np.float32,
            shape=st.shared(
                array_shapes(min_dims=2, max_dims=2, min_side=4), key="hi"
            ),
            elements={
                "allow_nan": False,
                "allow_infinity": False,
                "min_value": 0,
                "max_value": 256,
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
