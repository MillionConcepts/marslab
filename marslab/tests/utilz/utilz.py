from io import BytesIO

from hypothesis import strategies as st
from matplotlib.figure import Figure
import numpy as np
from PIL import Image


def boring_reals(min_mag: int, max_mag: int):
    """
    hypothesis strategy that returns finite floats within a specified range of
    absolute magnitudes.
    """
    return st.floats(
        allow_infinity=False,
        allow_nan=False,
        allow_subnormal=False,
        min_value=-(10 ** max_mag),
        max_value=10 ** max_mag,
    ).filter(lambda x: x == 0 or min_mag <= x <= max_mag)


quaternions = st.lists(boring_reals(-4, 8), min_size=4, max_size=4)
finite_floats = st.floats(allow_nan=False, allow_infinity=False)
finite_numbers = st.one_of(st.integers(), finite_floats)
positive_finite_floats = st.floats(
    min_value=0, allow_nan=False, allow_infinity=False
)


def normal_array(loc=128, scale=16, size=(256, 256), dtype=np.int64):
    rng = np.random.default_rng()
    randarray = rng.normal(loc, scale, size)
    return randarray.astype(dtype)


def render_in_memory(fig: Figure) -> Image:
    outbuf = BytesIO()
    fig.savefig(outbuf)
    outbuf.seek(0)
    return Image.open(outbuf)
