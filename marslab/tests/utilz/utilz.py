from io import BytesIO

from PIL import Image
from hypothesis import given, strategies as st, reject, assume
from matplotlib.figure import Figure
import numpy as np

from marslab.imgops.imgutils import eightbit


positive_finite_floats = st.floats(
    min_value=0, allow_nan=False, allow_infinity=False
)

finite_floats = st.floats(allow_nan=False, allow_infinity=False)

finite_numbers = st.one_of(st.integers(), finite_floats)


def normal_array(loc=128, scale=16, size=(256, 256), dtype=np.int64):
    rng = np.random.default_rng()
    randarray = rng.normal(loc, scale, size)
    return randarray.astype(dtype)


def render_in_memory(fig: Figure) -> Image:
    outbuf = BytesIO()
    fig.savefig(outbuf)
    outbuf.seek(0)
    return Image.open(outbuf)
