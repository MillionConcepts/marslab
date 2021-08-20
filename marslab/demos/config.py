from scipy.ndimage import gaussian_filter

from marslab.imgops.imgutils import std_clip
from marslab.imgops.render import colormapped_plot


def gen_spectop_defaults(special_constants=None):
    return {
        "params": {"special_constants": special_constants},
        "limiter": {"function": std_clip},
        "postfilter": {"function": gaussian_filter, "params": {"sigma": 2}},
        "plotter": {
            "function": colormapped_plot,
            "params": {
                "cmap": "orange_teal",
                "render_colorbar": True,
                "special_constants": special_constants,
            },
        },
    }
