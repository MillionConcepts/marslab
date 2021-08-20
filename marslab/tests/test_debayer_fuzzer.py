import numpy as np

import marslab.imgops.imgutils
from hypothesis import given, strategies as st, settings as hset, Verbosity

from hypothesis.extra.numpy import (
    arrays,
    floating_dtypes,
    integer_dtypes,
    array_shapes,
)

# def positive_integer_arrays(**array_kwargs):
#     return arrays(
#         dtype=integer_dtypes(),
#         elements={"min_value": 0},
#         **array_kwargs
#     )
#
# rng = np.random.default_rng()
# @hset(verbosity=Verbosity.verbose)
# @given(
#     input_array=arrays(
#         dtype=floating_dtypes(), shape=array_shapes(max_dims=2)
#     ),
# )
# def test_fuzz_bilinear_interpolate_subgrid(
#     input_array
# ):
#     rows = rng.choice(np.arange(0, input_array.shape[1] * 2,
#     cols = rng.choice(np.arange(0, input_array.shape[0], num_cols))
#     output_shape = (input_array.shape[0]*2, input_array.shape[1]*2)
#     marslab.imgops.imgutils.bilinear_interpolate_subgrid(
#         rows=rows,
#         columns=cols,
#         input_array=input_array,
#         output_shape=output_shape,
#     )
