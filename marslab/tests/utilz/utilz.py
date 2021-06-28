from hypothesis import given, strategies as st, reject, assume


positive_finite_floats = st.floats(
    min_value=0, allow_nan=False, allow_infinity=False
)

finite_floats = st.floats(allow_nan=False, allow_infinity=False)
